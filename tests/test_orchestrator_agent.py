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
        "project_details": None,
        "question_topics": None,
        "question_role": None,
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
    third = run_async(coordinator.answer_chat(context, sender, make_message("no projects")))
    fourth = run_async(coordinator.answer_chat(context, sender, make_message("same details")))

    assert "Thanks, that helps" in first
    assert "Before I search" in second and "projects" in second
    assert third == "Workflow completed."
    assert "already ran this exact" in fourth
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

    async def reply(_ctx, _address, request, expected_sender, expected_type, **_kwargs):
        calls.append((expected_sender, request.message_type))
        if request.message_type == coordinator.CAREER_GUIDANCE_MESSAGE:
            raise RuntimeError("guidance unavailable")
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
    # Guidance is supplementary: its failure must not fail the job-based plan.
    assert result.guidance is None


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
    assert captured["timeout"] == coordinator._http_timeout(coordinator.ASI_TIMEOUT_SECONDS)
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


class FakeGuidanceClient:
    def __init__(self, *_args, **_kwargs):
        pass

    def role_advice(self, _request):
        return self.role_guidance(None, None)

    def role_guidance(self, candidate, evidence_profile, project_notes=(), job_market_context=()):
        make = action_agent.ASISkillSuggestion
        self.project_notes = list(project_notes)
        self.job_market_context = list(job_market_context)
        return action_agent.ASISkillSuggestions(
            certifications=[action_agent.CertificationSuggestion(
                name="CompTIA Security+", issuer="CompTIA", why_relevant="Common entry-level SOC credential.")],
            projects=[action_agent.ProjectSuggestion(
                title="Home SIEM lab", description="Ingest logs and write detections.",
                skills_practiced=["SIEM"], deliverable="Documented detections")],
            skills=[
            make(skill="SIEM", why_it_matters="Core SOC tooling.", evidence_to_produce="A lab write-up.",
                 completion_criteria="Triage three alerts."),
            make(skill="Security Information and Event Management", why_it_matters="dup",
                 evidence_to_produce="dup", completion_criteria="dup"),
            make(skill="Incident Response", why_it_matters="Handling incidents.",
                 evidence_to_produce="A playbook.", completion_criteria="Playbook reviewed."),
        ])


def _empty_workflow(monkeypatch, guidance_ok, extra=None):
    addresses = {name: f"{name.value}-address" for name in coordinator.AgentName
                 if name != coordinator.AgentName.ORCHESTRATOR}
    monkeypatch.setattr(coordinator, "_address_config", lambda: addresses)
    monkeypatch.setenv("ASI1_API_KEY", "test-key")
    monkeypatch.setattr(action_agent, "ASIOneClient", FakeGuidanceClient)
    calls = []

    async def reply(_ctx, _address, request, expected_sender, expected_type, **_kwargs):
        calls.append(request.message_type)
        if expected_sender == coordinator.AgentName.OPPORTUNITY_INTELLIGENCE:
            payload = opportunity_result(records=[]).model_dump(mode="json")
        elif expected_sender == coordinator.AgentName.CANDIDATE_EVIDENCE:
            query = evidence_agent.EvidenceProfileRequest.model_validate(request.payload)
            payload = evidence_agent.build_evidence_profile(query).model_dump(mode="json")
        else:
            if not guidance_ok:
                raise RuntimeError("career action agent did not respond")
            wire = action_agent.handle_action_request(request)
            assert wire.status == "success", wire.error
            payload = wire.payload
        return coordinator.AgentResponse(
            request_id=request.request_id, sender=expected_sender,
            recipient=coordinator.AgentName.ORCHESTRATOR, message_type=expected_type,
            created_at=request.created_at, status="success", payload=payload)

    monkeypatch.setattr(coordinator, "_call_agent", reply)
    values = {"target_role": "SOC Analyst", "country": "IN", "experience_level": "entry_level",
              "learning_hours_per_week": 5, "claimed_skills": ["SIEM"], **(extra or {})}
    result = run_async(coordinator._run_workflow(
        FakeContext(), "agent1qempty", SimpleNamespace(values=values)))
    return result, calls


def test_no_jobs_returns_labeled_guidance_from_career_action_agent(monkeypatch):
    result, calls = _empty_workflow(monkeypatch, guidance_ok=True)

    assert coordinator.CAREER_GUIDANCE_MESSAGE in calls
    assert result.action_plan is None
    assert result.guidance.guidance_source == "model_generated_unverified"
    assert result.guidance.verified_against_job_listings is False
    # The duplicate alias is dropped; status comes from evidence, not the model.
    assert [item.skill for item in result.guidance.skills] == ["SIEM", "Incident Response"]
    assert result.guidance.skills[0].current_status == "claimed_without_evidence"
    assert result.guidance.skills[1].current_status == "no_evidence_provided"
    text = coordinator._format_workflow(result)
    assert "NOT from job" in text and "Incident Response" in text
    assert "CompTIA Security+" in text and "Home SIEM lab" in text and "not employer requirements" in text
    assert "comptia.org/en-us/certifications/security" in text
    assert "No current requirements are cached" in text


def test_no_jobs_and_guidance_failure_is_reported_not_invented(monkeypatch):
    result, _ = _empty_workflow(monkeypatch, guidance_ok=False)

    assert result.guidance is None
    assert result.action_stage.status == coordinator.StageStatus.SKIPPED
    assert "couldn't produce skill, certification and project guidance" in coordinator._format_workflow(result)


def test_guidance_timeout_covers_action_agent_provider_retries():
    assert coordinator.GUIDANCE_TIMEOUT_SECONDS > (
        action_agent.ASI_TIMEOUT_SECONDS * action_agent.ASI_MAX_ATTEMPTS
    )


def test_guidance_uses_skills_from_returned_custom_job_records():
    opportunities = opportunity_result()
    signals = coordinator._guidance_job_signals(opportunities)

    assert signals
    assert opportunities.records[0].job.title in signals[0]
    requirement = opportunities.records[0].job.required_skills
    if requirement:
        assert requirement[0].skill in signals[0]


def test_guidance_request_forwards_compact_job_signals_to_career_action(monkeypatch):
    candidate = coordinator.CandidateProfile(
        candidate_id="candidate-guidance",
        target_role="SOC Analyst",
        preferred_location="India",
        experience_level=ExperienceLevel.ENTRY_LEVEL,
        learning_hours_per_week=5,
        claimed_skills=["SIEM"],
    )
    evidence = coordinator.EvidenceProfileResult(
        candidate_id=candidate.candidate_id,
        as_of=coordinator._now(),
        skills=[],
    )
    opportunities = opportunity_result()
    captured = {}

    async def reply(_ctx, _address, request, expected_sender, expected_type, **_kwargs):
        captured.update(request=request)
        guidance = coordinator.GuidanceResult(
            candidate_id=candidate.candidate_id,
            target_role=candidate.target_role,
            skills=[],
            job_market_context_used=True,
        )
        return coordinator.AgentResponse(
            request_id=request.request_id,
            sender=expected_sender,
            recipient=coordinator.AgentName.ORCHESTRATOR,
            message_type=expected_type,
            created_at=request.created_at,
            status="success",
            payload=guidance.model_dump(mode="json"),
        )

    monkeypatch.setattr(coordinator, "_call_agent", reply)
    addresses = {coordinator.AgentName.CAREER_ACTION: "action-address"}
    guidance, _note = run_async(coordinator._request_guidance(
        FakeContext(), addresses, "request-guidance", candidate, evidence, [], opportunities,
    ))

    forwarded = captured["request"].payload["job_market_context"]
    assert forwarded == coordinator._guidance_job_signals(opportunities)
    assert guidance.job_market_context_used is True


def test_guidance_agent_delivery_retries_transient_transport_error():
    request = coordinator._make_request(
        "request-1", coordinator.AgentName.CAREER_ACTION,
        coordinator.CAREER_GUIDANCE_MESSAGE, {},
    )
    expected = coordinator.AgentResponse(
        request_id=request.request_id,
        sender=coordinator.AgentName.CAREER_ACTION,
        recipient=coordinator.AgentName.ORCHESTRATOR,
        message_type=coordinator.CAREER_GUIDANCE_RESULT_MESSAGE,
        created_at=request.created_at,
        status="success",
        payload={},
    )

    class RetryContext:
        calls = 0

        async def send_and_receive(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise OSError("temporary delivery failure")
            return (
                coordinator.response_to_message(expected),
                SimpleNamespace(status=coordinator.DeliveryStatus.DELIVERED, detail=None),
            )

    context = RetryContext()
    response = run_async(coordinator._call_agent(
        context, "action-address", request,
        coordinator.AgentName.CAREER_ACTION,
        coordinator.CAREER_GUIDANCE_RESULT_MESSAGE,
        timeout=coordinator.GUIDANCE_TIMEOUT_SECONDS,
        max_attempts=2,
        retry_timeouts=False,
    ))

    assert response.request_id == request.request_id
    assert context.calls == 2


def test_guidance_without_asi_key_returns_error_response(monkeypatch):
    monkeypatch.delenv("ASI1_API_KEY", raising=False)
    monkeypatch.delenv("ASI_ONE_API_KEY", raising=False)
    result, _ = None, None
    request = action_agent.AgentRequest(
        request_id="r:career-guidance", sender=action_agent.AgentName.ORCHESTRATOR,
        recipient=action_agent.AgentName.CAREER_ACTION,
        message_type=action_agent.CAREER_GUIDANCE_MESSAGE, created_at=datetime.now(timezone.utc),
        payload={"bad": True})
    assert action_agent.handle_action_request(request).error.code == "invalid_request"


def test_project_details_become_unverified_records_and_reach_guidance(monkeypatch):
    sender = "agent1qprojects"
    candidate_id = coordinator._candidate_id_for_sender(sender)
    details = [
        {"name": "EDR lab", "status": "in_progress", "summary": "Detections for Windows logs",
         "skills_used": ["Python", "Sysmon"], "worth": "Used by my college club", "link": None},
        {"name": "Idea only", "status": "idea", "summary": None, "skills_used": [], "worth": None, "link": None},
    ]
    context = FakeContext()
    state = coordinator.ConversationState(context.storage, sender)
    state.update(make_delta(project_details=details))
    state.update(make_delta(project_details=[{**details[0], "status": "completed"}]))
    assert [d["status"] for d in state.values["project_details"]] == ["completed", "idea"]

    record = coordinator._project_record(candidate_id, state.values["project_details"][0], coordinator._now())
    assert record.authorship_verified is None and record.model_interpreted is False
    assert record.verification_status == "unverified" and record.candidate_id == candidate_id
    assert "worth: Used by my college club" in record.description
    # Skill-less ideas are noted for guidance but are never evidence.
    assert coordinator._project_record(candidate_id, state.values["project_details"][1], coordinator._now()) is None

    result, calls = _empty_workflow(monkeypatch, guidance_ok=True, extra={"project_details": details})
    profile = result.evidence_profile
    projects = {skill.skill_id: skill.evidence_classes for skill in profile.skills}
    assert projects["python"][0].value == "project_backed" and "unknown:sysmon" in projects
    assert all(c.value != "independently_verified" for classes in projects.values() for c in classes)


def _fake_post(content=None, raises=None, captured=None):
    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": content}}]}

    def post(url, **kwargs):
        if captured is not None:
            captured.append(kwargs["json"])
        if raises:
            raise raises
        return Response()

    return post


def test_converse_uses_llm_wording_and_falls_back_on_any_failure(monkeypatch):
    monkeypatch.setenv("ASI1_API_KEY", "test-key")
    sent = []
    monkeypatch.setattr(coordinator.requests, "post", _fake_post("Happy to help! Which city?", captured=sent))
    assert coordinator._converse("ask_missing", {"ask_for": ["city"]}, "fallback") == "Happy to help! Which city?"
    assert "untrusted" in sent[0]["messages"][0]["content"]

    monkeypatch.setattr(coordinator.requests, "post", _fake_post(raises=coordinator.requests.Timeout()))
    assert coordinator._converse("ask_missing", {}, "fallback") == "fallback"
    monkeypatch.setattr(coordinator.requests, "post", _fake_post(" "))
    assert coordinator._converse("ask_missing", {}, "fallback") == "fallback"
    monkeypatch.setattr(coordinator.requests, "post", _fake_post("x" * 1300))
    assert coordinator._converse("ask_missing", {}, "fallback") == "fallback"
    monkeypatch.delenv("ASI1_API_KEY")
    assert coordinator._converse("ask_missing", {}, "fallback") == "fallback"


def test_small_talk_gets_a_conversational_reply_and_changes_no_state(monkeypatch):
    context, sender = FakeContext(), "agent1qchat"
    monkeypatch.setattr(coordinator, "_extract_chat_intake", lambda _t, _c=None: make_delta())
    seen = []

    def converse(situation, facts, fallback):
        seen.append((situation, facts))
        return "Hi! I'm doing well. What role are you aiming for?"

    monkeypatch.setattr(coordinator, "_converse", converse)
    reply = run_async(coordinator.answer_chat(context, sender, make_message("hey, how are you?")))

    assert reply.startswith("Hi!")
    assert seen[0][0] == "small_talk" and "the role you are targeting" in seen[0][1]["ask_for"]
    assert not any(context.storage.values.values())


def test_missing_info_and_project_prompts_are_gentle_without_llm(monkeypatch):
    monkeypatch.delenv("ASI1_API_KEY", raising=False)
    monkeypatch.delenv("ASI_ONE_API_KEY", raising=False)
    assert "Thanks, that helps" in coordinator._ask_for_missing(["your city"])
    assert "just say \"no projects\"" in coordinator.PROJECT_QUESTION


def test_listing_gaps_are_disclosed_not_guessed():
    result = opportunity_result()
    text = coordinator._format_workflow(coordinator.CandidateWorkflowResult(
        request_id="r", workflow_status=coordinator.WorkflowStatus.PARTIAL,
        opportunity_stage=coordinator.WorkflowStage(status=coordinator.StageStatus.SUCCEEDED),
        evidence_stage=coordinator.WorkflowStage(status=coordinator.StageStatus.SKIPPED, message="not needed"),
        action_stage=coordinator.WorkflowStage(status=coordinator.StageStatus.SKIPPED, message="not needed"),
        opportunities=result.model_dump(mode="json"), completed_at=coordinator._now()))
    assert "the job source didn't provide" in text and "I haven't guessed them" in text


def _advice_setup(monkeypatch, ask_delta, fail=False):
    from skillbridge.schemas import GuidanceResult as CanonicalGuidance

    monkeypatch.setattr(coordinator, "_address_config",
                        lambda: {coordinator.AgentName.CAREER_ACTION: "action-address"})
    monkeypatch.setattr(coordinator, "_extract_chat_intake", lambda _t, _c=None: ask_delta)
    monkeypatch.setattr(coordinator, "_converse", lambda _s, _f, fallback: fallback)
    monkeypatch.setenv("ASI1_API_KEY", "test-key")
    monkeypatch.setattr(action_agent, "ASIOneClient", FakeGuidanceClient)
    seen = []

    async def reply(_ctx, _address, request, expected_sender, expected_type, **_kwargs):
        seen.append(request)
        if fail:
            raise RuntimeError("career action agent did not respond")
        wire = action_agent.handle_action_request(request)
        assert wire.status == "success", wire.error
        return coordinator.AgentResponse(
            request_id=request.request_id, sender=expected_sender,
            recipient=coordinator.AgentName.ORCHESTRATOR, message_type=expected_type,
            created_at=request.created_at, status="success", payload=wire.payload)

    monkeypatch.setattr(coordinator, "_call_agent", reply)
    CanonicalGuidance  # imported to confirm the canonical model exists
    return seen


def test_user_can_ask_for_projects_and_certifications_for_a_role_without_full_intake(monkeypatch):
    delta = make_delta(intent="ask", question_topics=["certifications", "projects"],
                       question_role="Cloud Engineer", claimed_skills=["AWS"])
    seen = _advice_setup(monkeypatch, delta)
    context = FakeContext()
    reply = run_async(coordinator.answer_chat(context, "agent1qask", make_message("certs and projects for cloud?")))

    assert seen[0].message_type == coordinator.CAREER_ADVICE_MESSAGE
    assert seen[0].payload["target_role"] == "Cloud Engineer"
    assert seen[0].payload["topics"] == ["certifications", "projects"]
    assert seen[0].payload["experience_level"] is None and seen[0].payload["learning_hours_per_week"] is None
    assert "CompTIA Security+" in reply and "Home SIEM lab" in reply
    assert "Incident Response" not in reply  # skills were not requested
    assert "I don't know your experience level or weekly learning hours" in reply
    assert "NOT from job" in reply
    # Details shared alongside the question are still remembered.
    assert next(iter(context.storage.values.values()))["claimed_skills"] == ["AWS"]


def test_question_uses_saved_role_asks_when_role_unknown_and_reports_failure(monkeypatch):
    delta = make_delta(intent="ask", question_topics=["skills"])
    seen = _advice_setup(monkeypatch, delta)
    asked = run_async(coordinator.answer_chat(FakeContext(), "agent1qnorole", make_message("what skills?")))
    assert "Which role" in asked and not seen

    context = FakeContext()
    state = coordinator.ConversationState(context.storage, "agent1qsaved")
    state.update(make_delta(target_role="SOC Analyst", experience_level="student"))
    answered = run_async(coordinator.answer_chat(context, "agent1qsaved", make_message("what skills?")))
    assert seen[0].payload["target_role"] == "SOC Analyst" and "SIEM" in answered

    delta = make_delta(intent="ask", question_topics=["skills"], question_role="SOC Analyst")
    _advice_setup(monkeypatch, delta, fail=True)
    failed = run_async(coordinator.answer_chat(FakeContext(), "agent1qfail", make_message("skills for SOC")))
    assert "couldn't get that advice" in failed


def test_advice_filters_to_requested_topics_and_rejects_empty_results(monkeypatch):
    monkeypatch.setenv("ASI1_API_KEY", "test-key")
    request = action_agent.AdviceRequest(candidate_id="c", target_role="SOC Analyst", topics=["certifications"])
    result = action_agent.build_advice(request, client=FakeGuidanceClient())
    assert result.skills == [] and result.suggested_projects == [] and len(result.certifications) == 1

    class Empty(FakeGuidanceClient):
        def role_advice(self, _request):
            return action_agent.ASISkillSuggestions(skills=[], certifications=[], projects=[])

    with pytest.raises(RuntimeError):
        action_agent.build_advice(request, client=Empty())


def test_greeting_after_intake_chats_instead_of_running_workflow(monkeypatch):
    from unittest.mock import AsyncMock

    context = FakeContext()
    sender = "agent1qhey"
    state = coordinator.ConversationState(context.storage, sender)
    state.update(make_delta(
        target_role="SOC Analyst", country="IN", experience_level="student",
        learning_hours_per_week=10, remote_only=True))
    state.mark_projects_asked()
    monkeypatch.setattr(coordinator, "_extract_chat_intake", lambda _t, _c=None: make_delta())
    run = AsyncMock(return_value=object())
    monkeypatch.setattr(coordinator, "_run_workflow", run)
    monkeypatch.delenv("ASI1_API_KEY", raising=False)
    reply = run_async(coordinator.answer_chat(context, sender, make_message("hey")))
    assert run.await_count == 0 and "go ahead" in reply


def test_source_notes_report_connection_and_warn_about_thin_or_unreachable_sources():
    from skillbridge.schemas import SourceKind, SourceReport, SourceStatus

    ok = SourceReport(source_id="remotive", source_kind=SourceKind.LIVE, status=SourceStatus.EMPTY,
                      attempts=1, duration_ms=1200)
    down = SourceReport(source_id="remotive", source_kind=SourceKind.LIVE, status=SourceStatus.ERROR,
                        attempts=1, error_code="timeout", error_message="request timed out")
    connected = " ".join(coordinator._source_notes([ok]))
    assert "connected" in connected and "not a good or complete" in connected and "1.2s" in connected
    failed = " ".join(coordinator._source_notes([down]))
    assert "could NOT be reached (timeout)" in failed and "connected" not in failed


def test_bare_go_runs_the_workflow_once_details_are_complete(monkeypatch):
    from unittest.mock import AsyncMock

    context = FakeContext()
    sender = "agent1qgo"
    state = coordinator.ConversationState(context.storage, sender)
    state.update(make_delta(
        target_role="Front End Developer", country="IN", experience_level="student",
        learning_hours_per_week=10, remote_only=True))
    state.mark_projects_asked()
    monkeypatch.setattr(coordinator, "_extract_chat_intake", lambda _t, _c=None: make_delta())
    run = AsyncMock(return_value=object())
    monkeypatch.setattr(coordinator, "_run_workflow", run)
    monkeypatch.setattr(coordinator, "_format_workflow", lambda _r: "done")
    assert run_async(coordinator.answer_chat(context, sender, make_message("go"))) == "done"
    assert run.await_count == 1
