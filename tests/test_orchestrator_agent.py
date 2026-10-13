"""Focused tests for the standalone Agentverse chat coordinator."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from uagents_core.contrib.protocols.chat import ChatMessage, TextContent

from agents import orchestrator_agent as coordinator
from agents import action_agent, evidence_agent
from scripts.sync_agent_models import forbidden_imports
from skillbridge.schemas import AssessmentRecord, ExperienceLevel
from tests.test_orchestrator import opportunity_result


class MemoryStorage:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value

    def remove(self, key):
        self.values.pop(key, None)


class FakeContext:
    def __init__(self):
        self.storage = MemoryStorage()


def make_delta(**changes):
    values = {
        "intent": "continue",
        "target_role": None,
        "country": None,
        "state": None,
        "city": None,
        "experience_level": None,
        "preferred_companies": None,
        "keywords": None,
        "remote_only": None,
        "learning_hours_per_week": None,
        "claimed_skills": None,
        "study_techniques": None,
        "study_notes": None,
        "assessments": None,
        "practical_tasks": None,
        "projects": None,
    }
    return coordinator.ChatIntakeDelta.model_validate({**values, **changes})


def make_message(text):
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )


def run_async(coroutine):
    try:
        return asyncio.run(coroutine)
    finally:
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())


def test_hosted_agent_has_no_shared_package_imports():
    source = open(coordinator.__file__, encoding="utf-8").read()
    assert forbidden_imports(source) == []
    assert "agent.include(chat_proto, publish_manifest=True)" in source


def test_state_serializes_evidence_and_merges_distinct_turns():
    sender = "agent1qtest"
    candidate_id = coordinator._candidate_id_for_sender(sender)
    record = AssessmentRecord(
        record_id="assessment-1",
        candidate_id=candidate_id,
        skills_evaluated=["python"],
        score=0.8,
        taken_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        provider="user-supplied record",
    )
    storage = MemoryStorage()
    state = coordinator.ConversationState(storage, sender)
    record_data = record.model_dump(mode="json")
    state.update(make_delta(claimed_skills=["Python"], assessments=[record_data]))
    state.update(make_delta(claimed_skills=["Python", "SQL"], assessments=[record_data]))

    reloaded = coordinator.ConversationState(storage, sender)
    assert reloaded.values["claimed_skills"] == ["Python", "SQL"]
    assert reloaded.values["assessments"][0]["record_id"] == "assessment-1"
    assert json.dumps(reloaded.values)

    conflicting = record.model_copy(update={"score": 0.9})
    with pytest.raises(ValueError, match="reused with different data"):
        reloaded.update(make_delta(assessments=[conflicting.model_dump(mode="json")]))


def test_chat_gathers_fields_then_runs_once_without_replaying_stale_output(monkeypatch):
    context = FakeContext()
    sender = "agent1qstudent"
    record = AssessmentRecord(
        record_id="assessment-1",
        candidate_id=coordinator._candidate_id_for_sender(sender),
        skills_evaluated=["SIEM"],
        score=0.75,
        taken_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        provider="candidate-supplied assessment",
    )
    partial = make_delta(target_role="SOC Analyst")
    complete = make_delta(
        target_role="SOC Analyst",
        country="US",
        city="Austin",
        experience_level=ExperienceLevel.ENTRY_LEVEL,
        learning_hours_per_week=5,
        claimed_skills=["SIEM"],
        assessments=[record.model_dump(mode="json")],
    )
    monkeypatch.setattr(
        coordinator, "_extract_chat_intake",
        lambda _text, _candidate_id=None: partial if _text == "role" else complete,
    )
    from unittest.mock import AsyncMock

    run = AsyncMock(return_value=object())
    monkeypatch.setattr(coordinator, "_run_workflow", run)
    monkeypatch.setattr(coordinator, "_format_workflow", lambda _result: "Workflow completed.")

    first = run_async(coordinator.answer_chat(context, sender, make_message("role")))
    second = run_async(coordinator.answer_chat(context, sender, make_message("details")))
    third = run_async(coordinator.answer_chat(context, sender, make_message("same details")))

    assert "still need" in first
    assert second == "Workflow completed."
    assert "already ran this exact" in third
    assert run.await_count == 1
    saved = next(iter(context.storage.values.values()))
    assert saved["assessments"][0]["candidate_id"] == coordinator._candidate_id_for_sender(sender)


def test_hosted_workflow_validates_and_routes_through_all_three_agents(monkeypatch):
    sender = "agent1qworkflow"
    values = {
        "target_role": "SOC Analyst",
        "country": "IN",
        "city": "Remote",
        "experience_level": "entry_level",
        "learning_hours_per_week": 5,
        "claimed_skills": ["SIEM"],
    }
    addresses = {
        coordinator.AgentName.OPPORTUNITY_INTELLIGENCE: "opportunity-address",
        coordinator.AgentName.CANDIDATE_EVIDENCE: "evidence-address",
        coordinator.AgentName.CAREER_ACTION: "action-address",
    }
    monkeypatch.setattr(coordinator, "_address_config", lambda: addresses)
    calls = []

    async def reply(_ctx, _address, request, expected_sender, expected_type):
        calls.append((expected_sender, request.message_type))
        if expected_sender == coordinator.AgentName.OPPORTUNITY_INTELLIGENCE:
            payload = opportunity_result().model_dump(mode="json")
        elif expected_sender == coordinator.AgentName.CANDIDATE_EVIDENCE:
            query = evidence_agent.EvidenceProfileRequest.model_validate(request.payload)
            payload = evidence_agent.build_evidence_profile(query).model_dump(mode="json")
        else:
            query = action_agent.ActionPlanRequest.model_validate(request.payload)
            payload = action_agent.build_action_plan(query).model_dump(mode="json")
        return coordinator.AgentResponse(
            request_id=request.request_id,
            sender=expected_sender,
            recipient=coordinator.AgentName.ORCHESTRATOR,
            message_type=expected_type,
            created_at=request.created_at,
            status="success",
            payload=payload,
        )

    monkeypatch.setattr(coordinator, "_call_agent", reply)
    result = run_async(
        coordinator._run_workflow(FakeContext(), sender, SimpleNamespace(values=values))
    )

    assert result.workflow_status == coordinator.WorkflowStatus.SUCCEEDED
    assert {agent for agent, _ in calls} == {
        coordinator.AgentName.OPPORTUNITY_INTELLIGENCE,
        coordinator.AgentName.CANDIDATE_EVIDENCE,
        coordinator.AgentName.CAREER_ACTION,
    }
    assert result.action_plan.ranked_opportunities[0].job_id == "fixture-job-1"
    assert result.opportunities.contains_fixture_data is True


def test_asi_intake_validates_structured_response_without_live_network(monkeypatch):
    monkeypatch.setenv("ASI1_API_KEY", "test-key")
    monkeypatch.delenv("ASI_ONE_API_KEY", raising=False)
    sample = make_delta(intent="continue", country="US").model_dump_json()
    captured = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": sample}}]}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(coordinator.requests, "post", post)
    result = coordinator._extract_chat_intake("I want jobs in the US", "candidate-test")

    assert result.country == "US"
    assert captured["url"] == "https://api.asi1.ai/v1/chat/completions"
    assert captured["timeout"] == coordinator.ASI_TIMEOUT_SECONDS
    assert "candidate-test" in captured["json"]["messages"][0]["content"]


def test_asi_intake_rejects_malformed_structured_response(monkeypatch):
    monkeypatch.setenv("ASI1_API_KEY", "test-key")

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"intent":"continue"}'}}]}

    monkeypatch.setattr(coordinator.requests, "post", lambda *_args, **_kwargs: Response())
    with pytest.raises(coordinator.ASIProviderError, match="ASI:One intake failed"):
        coordinator._extract_chat_intake("hello", "candidate-test")


def test_asi_provider_timeout_raises_provider_error(monkeypatch):
    monkeypatch.setenv("ASI1_API_KEY", "test-key")

    def timeout(*_args, **_kwargs):
        raise coordinator.requests.Timeout()

    monkeypatch.setattr(coordinator.requests, "post", timeout)
    monkeypatch.setattr(coordinator.time, "sleep", lambda _duration: None)

    with pytest.raises(coordinator.ASIProviderError):
        coordinator._extract_chat_intake("hello", "candidate-test")
