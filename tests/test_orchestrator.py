"""Workflow tests use real schemas and mocked uAgents request/reply handlers."""

import asyncio
from functools import wraps
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from uagents_core.types import DeliveryStatus, MsgStatus

from agents import action_agent as action
from agents import evidence_agent as evidence
from agents import opportunity_agent as opportunity
from skillbridge import orchestrator
from skillbridge.orchestration import (
    CandidateWorkflowRequest,
    CandidateWorkflowResult,
    StageStatus,
    WorkflowStatus,
)
from skillbridge.schemas import (
    AgentName,
    AgentRequest,
    AgentResponse,
    AssessmentRecord,
    CandidateProfile,
    DeclaredSkill,
    EvidenceProfileRequest,
    EvidenceProfileResult,
    ExtractionMethod,
    JobOpportunity,
    OpportunityRecord,
    OpportunitySearchRequest,
    OpportunitySearchResult,
    SearchStatus,
    SkillRequirement,
    SourceKind,
    SourceReport,
    SourceStatus,
    missing_job_fields,
)
from skillbridge.transport import (
    AgentRequestMessage,
    AgentResponseMessage,
    message_to_request,
    message_to_response,
    request_to_message,
    response_to_message,
)
from skillbridge.workflow_store import SQLiteWorkflowStore

NOW = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
ADDRESSES = orchestrator.AgentAddresses(
    opportunity=opportunity.agent.address,
    evidence=evidence.agent.address,
    career_action=action.agent.address,
)


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return asyncio.run(function(*args, **kwargs))
        finally:
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
    return wrapper


def workflow_request(**changes):
    candidate = CandidateProfile(
        candidate_id="cand-1",
        target_role="SOC Analyst",
        preferred_location="Remote",
        experience_level="entry_level",
        claimed_skills=["SIEM"],
        learning_hours_per_week=5,
    )
    data = {
        "candidate": candidate,
        "job_search": OpportunitySearchRequest(
            target_role="SOC Analyst",
            country="US",
            experience_level="entry_level",
        ),
        "declared_skills": [DeclaredSkill(skill="SIEM")],
        "assessments": [
            AssessmentRecord(
                record_id="assessment-1",
                candidate_id="cand-1",
                skills_evaluated=["SIEM"],
                score=0.95,
                taken_at=NOW,
                provider="controlled-fixture",
            )
        ],
        "as_of": NOW,
    }
    data.update(changes)
    return CandidateWorkflowRequest.model_validate(data)


def opportunity_result(status=SearchStatus.COMPLETE, records=None):
    job = JobOpportunity(
        job_id="fixture-job-1",
        title="SOC Analyst",
        employer="Fixture Employer",
        location="Remote",
        required_skills=[SkillRequirement(skill="SIEM")],
        preferred_skills=[],
        eligibility_requirements=[],
        source_url="https://example.com/jobs/fixture-job-1",
        source_id="fixture-source",
        retrieved_at=NOW,
    )
    record = OpportunityRecord(
        job=job,
        source_kind=SourceKind.FIXTURE,
        original_title=job.title,
        normalized_title="soc analyst",
        description="Clearly labeled test fixture.",
        remote=True,
        skills_extraction=ExtractionMethod.STRUCTURED_SOURCE,
        eligibility_extraction=ExtractionMethod.STRUCTURED_SOURCE,
        missing_fields=missing_job_fields(job, "Clearly labeled test fixture.", True),
    )
    entries = [record] if records is None else records
    reports = [
        SourceReport(
            source_id="fixture-source",
            source_kind=SourceKind.FIXTURE,
            status=SourceStatus.OK,
            attempts=1,
            retrieved_count=len(entries),
            accepted_count=len(entries),
        )
    ]
    if status == SearchStatus.PARTIAL:
        reports.append(SourceReport(
            source_id="failed-source",
            source_kind=SourceKind.LIVE,
            status=SourceStatus.ERROR,
            attempts=2,
            error_code="timeout",
            error_message="source timed out",
        ))
    return OpportunitySearchResult(
        search_status=status,
        records=entries,
        source_reports=reports,
        contains_fixture_data=any(item.source_kind == SourceKind.FIXTURE for item in entries),
    )


def evidence_result(request=None):
    request = request or workflow_request()
    ev_request = EvidenceProfileRequest(
        profile=request.candidate,
        declared_skills=request.declared_skills,
        assessments=request.assessments,
        practical_tasks=request.practical_tasks,
        projects=request.projects,
        as_of=request.as_of,
    )
    local_request = evidence.EvidenceProfileRequest.model_validate(ev_request.model_dump(mode="json"))
    return EvidenceProfileResult.model_validate(
        evidence.build_evidence_profile(local_request).model_dump(mode="json")
    )


class MockContext:
    def __init__(self, fail=None, invalid=None):
        self.fail = set(fail or ())
        self.invalid = set(invalid or ())
        self.calls = []
        self.sent = []

    async def send(self, destination, message):
        self.sent.append((destination, message))

    async def send_and_receive(self, destination, message, response_type, timeout):
        request = message_to_request(message)
        self.calls.append((request.recipient, request, timeout))
        if request.recipient in self.fail:
            status = MsgStatus(
                status=DeliveryStatus.FAILED,
                detail="Timeout waiting for response",
                destination=destination,
                endpoint="",
            )
            return None, status
        request_id = "wrong-id" if request.recipient in self.invalid else request.request_id
        if request.recipient == AgentName.OPPORTUNITY_INTELLIGENCE:
            payload = opportunity_result().model_dump(mode="json")
            sender, kind = AgentName.OPPORTUNITY_INTELLIGENCE, opportunity.OPPORTUNITY_RESULT_MESSAGE
        elif request.recipient == AgentName.CANDIDATE_EVIDENCE:
            query = evidence.EvidenceProfileRequest.model_validate(request.payload)
            local_result = evidence.build_evidence_profile(query)
            payload = local_result.model_dump(mode="json")
            sender, kind = AgentName.CANDIDATE_EVIDENCE, evidence.EVIDENCE_RESULT_MESSAGE
        else:
            query = action.ActionPlanRequest.model_validate(request.payload)
            local_result = action.build_action_plan(query)
            payload = local_result.model_dump(mode="json")
            sender, kind = AgentName.CAREER_ACTION, action.CAREER_ACTION_RESULT_MESSAGE
        response = AgentResponse(
            request_id=request_id,
            sender=sender,
            recipient=AgentName.ORCHESTRATOR,
            message_type=kind,
            created_at=NOW,
            status="success",
            payload=payload,
        )
        status = MsgStatus(
            status=DeliveryStatus.DELIVERED,
            detail="delivered",
            destination=destination,
            endpoint="",
        )
        return response_to_message(response), status


def setup(tmp_path, **context_options):
    return (
        MockContext(**context_options),
        orchestrator.OrchestratorSettings(timeout_seconds=1, max_attempts=1,
                                          database_path=str(tmp_path / "workflows.sqlite3")),
        SQLiteWorkflowStore(tmp_path / "workflows.sqlite3"),
    )


async def run(ctx, request, settings, store, request_id="workflow-1"):
    return await orchestrator.run_candidate_workflow(
        ctx, request_id, request, ADDRESSES, settings, store,
    )


@async_test
async def test_successful_workflow_calls_three_agents_and_persists_traceable_result(tmp_path):
    ctx, settings, store = setup(tmp_path)
    result = await run(ctx, workflow_request(), settings, store)

    assert result.workflow_status == WorkflowStatus.SUCCEEDED
    assert [item[0] for item in ctx.calls] == [
        AgentName.OPPORTUNITY_INTELLIGENCE,
        AgentName.CANDIDATE_EVIDENCE,
        AgentName.CAREER_ACTION,
    ]
    assert result.action_plan.ranked_opportunities[0].job_id == "fixture-job-1"
    assert result.action_plan.ranked_opportunities[0].source_id == "fixture-source"
    assert result.action_plan.ranked_opportunities[0].match.breakdown[0].evidence_refs == ["assessment-1"]
    assert result.opportunities.records[0].job.retrieved_at == NOW
    assert result.action_plan.rubric.required_weight == 3.0

    with store._connect() as connection:
        row = connection.execute(
            "SELECT state, result_json FROM skillbridge_workflows WHERE request_id = ?",
            ("workflow-1",),
        ).fetchone()
    assert row["state"] == "completed"
    persisted = CandidateWorkflowResult.model_validate_json(row["result_json"])
    assert str(persisted.action_plan.ranked_opportunities[0].source_url) == (
        "https://example.com/jobs/fixture-job-1"
    )


@pytest.mark.parametrize("failure,expected_stage", [
    (AgentName.OPPORTUNITY_INTELLIGENCE, "opportunity_stage"),
    (AgentName.CANDIDATE_EVIDENCE, "evidence_stage"),
])
@async_test
async def test_upstream_failures_are_explicit_and_do_not_call_action(tmp_path, failure, expected_stage):
    ctx, settings, store = setup(tmp_path, fail={failure})

    result = await run(ctx, workflow_request(), settings, store)

    assert result.workflow_status == WorkflowStatus.PARTIAL
    assert getattr(result, expected_stage).status == StageStatus.FAILED
    assert result.action_stage.status == StageStatus.SKIPPED
    assert AgentName.CAREER_ACTION not in [item[0] for item in ctx.calls]


@async_test
async def test_career_action_failure_is_reported_after_upstream_stages_succeed(tmp_path):
    ctx, settings, store = setup(tmp_path, fail={AgentName.CAREER_ACTION})

    result = await run(ctx, workflow_request(), settings, store)

    assert result.workflow_status == WorkflowStatus.PARTIAL
    assert result.opportunity_stage.status == StageStatus.SUCCEEDED
    assert result.evidence_stage.status == StageStatus.SUCCEEDED
    assert result.action_stage.error_code == "agent_timeout"


@async_test
async def test_invalid_agent_response_is_rejected(tmp_path):
    ctx, settings, store = setup(tmp_path, invalid={AgentName.OPPORTUNITY_INTELLIGENCE})

    result = await run(ctx, workflow_request(), settings, store)

    assert result.opportunity_stage.error_code == "invalid_agent_response"
    assert result.action_stage.status == StageStatus.SKIPPED


@async_test
async def test_timeout_retries_are_bounded_and_reuse_the_step_request_id(tmp_path):
    ctx, settings, store = setup(tmp_path, fail={AgentName.OPPORTUNITY_INTELLIGENCE})
    settings = orchestrator.OrchestratorSettings(
        timeout_seconds=1, max_attempts=2, database_path=settings.database_path,
    )

    result = await run(ctx, workflow_request(), settings, store)

    retries = [item for item in ctx.calls if item[0] == AgentName.OPPORTUNITY_INTELLIGENCE]
    assert len(retries) == 2
    assert retries[0][1].request_id == retries[1][1].request_id == "workflow-1:opportunity"
    assert result.opportunity_stage.error_code == "agent_timeout"


@async_test
async def test_partial_job_source_results_are_labeled_and_still_reach_action_agent(tmp_path):
    ctx, settings, store = setup(tmp_path)

    async def partial_handler(destination, message, response_type, timeout):
        request = message_to_request(message)
        if request.recipient == AgentName.OPPORTUNITY_INTELLIGENCE:
            ctx.calls.append((request.recipient, request, timeout))
            result = opportunity_result(SearchStatus.PARTIAL)
            response = AgentResponse(
                request_id=request.request_id,
                sender=AgentName.OPPORTUNITY_INTELLIGENCE,
                recipient=AgentName.ORCHESTRATOR,
                message_type=opportunity.OPPORTUNITY_RESULT_MESSAGE,
                created_at=NOW,
                status="success",
                payload=result.model_dump(mode="json"),
            )
            return response_to_message(response), MsgStatus(
                status=DeliveryStatus.DELIVERED, detail="delivered",
                destination=destination, endpoint="",
            )
        return await MockContext.send_and_receive(ctx, destination, message, response_type, timeout)

    ctx.send_and_receive = partial_handler
    result = await run(ctx, workflow_request(), settings, store)

    assert result.workflow_status == WorkflowStatus.PARTIAL
    assert result.opportunity_stage.status == StageStatus.PARTIAL
    assert result.opportunities.search_status == SearchStatus.PARTIAL
    assert result.action_plan is not None


@async_test
async def test_all_job_sources_failed_is_preserved_without_action_call(tmp_path):
    ctx, settings, store = setup(tmp_path)
    failed_result = OpportunitySearchResult(
        search_status=SearchStatus.FAILED,
        records=[],
        source_reports=[
            SourceReport(
                source_id="live-source", source_kind=SourceKind.LIVE,
                status=SourceStatus.ERROR, attempts=2,
                error_code="timeout", error_message="source request timed out",
            )
        ],
        contains_fixture_data=False,
    )

    async def failed_job_handler(destination, message, response_type, timeout):
        request = message_to_request(message)
        ctx.calls.append((request.recipient, request, timeout))
        if request.recipient != AgentName.OPPORTUNITY_INTELLIGENCE:
            return await MockContext.send_and_receive(ctx, destination, message, response_type, timeout)
        response = AgentResponse(
            request_id=request.request_id,
            sender=AgentName.OPPORTUNITY_INTELLIGENCE,
            recipient=AgentName.ORCHESTRATOR,
            message_type=opportunity.OPPORTUNITY_RESULT_MESSAGE,
            created_at=NOW,
            status="error",
            error={"code": "all_sources_failed", "message": "all configured sources failed"},
            payload=failed_result.model_dump(mode="json"),
        )
        return response_to_message(response), MsgStatus(
            status=DeliveryStatus.DELIVERED, detail="delivered",
            destination=destination, endpoint="",
        )

    ctx.send_and_receive = failed_job_handler
    result = await run(ctx, workflow_request(), settings, store)

    assert result.opportunity_stage.status == StageStatus.FAILED
    assert result.opportunities.search_status == SearchStatus.FAILED
    assert result.workflow_status == WorkflowStatus.PARTIAL
    assert result.action_stage.status == StageStatus.SKIPPED


@async_test
async def test_repeated_request_returns_labeled_replay_without_agent_calls(tmp_path):
    ctx, settings, store = setup(tmp_path)
    request = workflow_request()
    first = await run(ctx, request, settings, store)
    call_count = len(ctx.calls)

    replay = await run(ctx, request, settings, store)

    assert not first.replayed
    assert replay.replayed
    assert replay.completed_at == first.completed_at
    assert len(ctx.calls) == call_count


@async_test
async def test_request_id_reuse_with_different_payload_is_rejected(tmp_path):
    ctx, settings, store = setup(tmp_path)
    await run(ctx, workflow_request(), settings, store)
    with pytest.raises(orchestrator.WorkflowRequestError, match="different workflow input"):
        await run(ctx, workflow_request(include_asi_summary=True), settings, store)


def test_workflow_request_validates_candidate_search_and_evidence_integrity():
    base = workflow_request().model_dump(mode="python")
    with pytest.raises(ValidationError, match="target_role"):
        CandidateWorkflowRequest.model_validate({
            **base,
            "job_search": {**base["job_search"], "target_role": "Other"},
        })
    with pytest.raises(ValidationError, match="different candidate"):
        CandidateWorkflowRequest.model_validate({
            **base,
            "assessments": [{**base["assessments"][0], "candidate_id": "other"}],
        })


def test_config_requires_valid_distinct_agent_addresses(monkeypatch):
    monkeypatch.setenv("SKILLBRIDGE_OPPORTUNITY_AGENT_ADDRESS", ADDRESSES.opportunity)
    monkeypatch.setenv("SKILLBRIDGE_EVIDENCE_AGENT_ADDRESS", ADDRESSES.evidence)
    monkeypatch.setenv("SKILLBRIDGE_ACTION_AGENT_ADDRESS", ADDRESSES.career_action)
    assert orchestrator.AgentAddresses.from_env() == ADDRESSES
    monkeypatch.setenv("SKILLBRIDGE_ACTION_AGENT_ADDRESS", "not-an-agent")
    with pytest.raises(ValueError, match="valid uAgents address"):
        orchestrator.AgentAddresses.from_env()


def test_workflow_agent_registers_only_verified_request_handler():
    from uagents import Model

    protocol = orchestrator.agent._protocol
    digest = Model.build_schema_digest(AgentRequestMessage)
    assert digest in protocol._signed_message_handlers
    assert Model.build_schema_digest(AgentResponseMessage) in protocol._replies[digest]


@async_test
async def test_registered_handler_returns_structured_workflow_to_api_sender(tmp_path, monkeypatch):
    ctx = MockContext()
    monkeypatch.setenv("SKILLBRIDGE_OPPORTUNITY_AGENT_ADDRESS", ADDRESSES.opportunity)
    monkeypatch.setenv("SKILLBRIDGE_EVIDENCE_AGENT_ADDRESS", ADDRESSES.evidence)
    monkeypatch.setenv("SKILLBRIDGE_ACTION_AGENT_ADDRESS", ADDRESSES.career_action)
    monkeypatch.setenv("SKILLBRIDGE_WORKFLOW_DB", str(tmp_path / "handler.sqlite3"))
    request = AgentRequest(
        request_id="handler-workflow",
        sender=AgentName.ORCHESTRATOR,
        recipient=AgentName.ORCHESTRATOR,
        message_type=orchestrator.WORKFLOW_MESSAGE,
        created_at=NOW,
        payload=workflow_request().model_dump(mode="json"),
    )
    wire_request = request_to_message(request)

    await orchestrator.on_workflow_request(ctx, "verified-api-client-address", wire_request)

    assert len(ctx.sent) == 1
    assert ctx.sent[0][0] == "verified-api-client-address"
    response = message_to_response(ctx.sent[0][1])
    assert response.status.value == "success"
    result = CandidateWorkflowResult.model_validate(response.payload)
    assert result.workflow_status == WorkflowStatus.SUCCEEDED
