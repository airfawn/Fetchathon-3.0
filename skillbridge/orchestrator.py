"""Deterministic uAgents workflow coordinator; not a specialized AI agent."""

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import ValidationError
from uagents import Agent, Context
from uagents_core.identity import is_valid_address
from uagents_core.types import DeliveryStatus

from skillbridge.orchestration import (
    CandidateWorkflowRequest,
    CandidateWorkflowResult,
    StageStatus,
    WorkflowStage,
    WorkflowStatus,
)
from skillbridge.schemas import (
    ActionPlanRequest,
    ActionPlanResult,
    AgentError,
    AgentName,
    AgentRequest,
    AgentResponse,
    CAREER_ACTION_MESSAGE,
    CAREER_ACTION_RESULT_MESSAGE,
    EVIDENCE_PROFILE_MESSAGE,
    EVIDENCE_RESULT_MESSAGE,
    EvidenceProfileRequest,
    EvidenceProfileResult,
    OpportunitySearchResult,
    OPPORTUNITY_RESULT_MESSAGE,
    OPPORTUNITY_SEARCH_MESSAGE,
    ResponseStatus,
    SearchStatus,
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


AGENT_NAME = "skillbridge_orchestrator"
WORKFLOW_MESSAGE = "candidate.workflow.run"
WORKFLOW_RESULT_MESSAGE = "candidate.workflow.result"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRIES = 2
logger = logging.getLogger(AGENT_NAME)


class WorkflowCallError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class WorkflowRequestError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AgentAddresses:
    opportunity: str
    evidence: str
    career_action: str

    def __post_init__(self):
        for label, address in (
            ("opportunity", self.opportunity),
            ("evidence", self.evidence),
            ("career_action", self.career_action),
        ):
            if not is_valid_address(address):
                raise ValueError(f"{label} agent address is not a valid uAgents address")
        if len({self.opportunity, self.evidence, self.career_action}) != 3:
            raise ValueError("each specialized agent must have a distinct configured address")

    @classmethod
    def from_env(cls, env=os.environ) -> "AgentAddresses":
        names = {
            "opportunity": "SKILLBRIDGE_OPPORTUNITY_AGENT_ADDRESS",
            "evidence": "SKILLBRIDGE_EVIDENCE_AGENT_ADDRESS",
            "career_action": "SKILLBRIDGE_ACTION_AGENT_ADDRESS",
        }
        missing = [variable for variable in names.values() if not env.get(variable)]
        if missing:
            raise ValueError("missing agent address configuration: " + ", ".join(missing))
        return cls(**{field: env[variable] for field, variable in names.items()})


@dataclass(frozen=True)
class OrchestratorSettings:
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_RETRIES
    database_path: str = "skillbridge_workflows.sqlite3"

    def __post_init__(self):
        if not 1 <= self.timeout_seconds <= 300:
            raise ValueError("workflow timeout must be between 1 and 300 seconds")
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("workflow max_attempts must be between 1 and 3")
        if not self.database_path.strip():
            raise ValueError("workflow database path must not be empty")

    @classmethod
    def from_env(cls, env=os.environ) -> "OrchestratorSettings":
        try:
            timeout = int(env.get("SKILLBRIDGE_WORKFLOW_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
            attempts = int(env.get("SKILLBRIDGE_WORKFLOW_MAX_ATTEMPTS", DEFAULT_RETRIES))
        except (TypeError, ValueError) as error:
            raise ValueError("workflow timeout and max attempts must be integers") from error
        return cls(
            timeout_seconds=timeout,
            max_attempts=attempts,
            database_path=env.get("SKILLBRIDGE_WORKFLOW_DB", "skillbridge_workflows.sqlite3"),
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _log(event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, json.dumps({"event": event, **fields}, default=str, sort_keys=True))


def _request_hash(request: CandidateWorkflowRequest) -> str:
    canonical = json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stage_failed(code: str, message: str) -> WorkflowStage:
    return WorkflowStage(status=StageStatus.FAILED, error_code=code, message=message)


def _failed_workflow(request_id: str, code: str, message: str) -> CandidateWorkflowResult:
    failed = _stage_failed(code, message)
    return CandidateWorkflowResult(
        request_id=request_id,
        workflow_status=WorkflowStatus.FAILED,
        opportunity_stage=failed,
        evidence_stage=failed,
        action_stage=WorkflowStage(
            status=StageStatus.SKIPPED,
            message="action stage was not run because workflow request validation or dispatch failed",
        ),
        completed_at=_now(),
    )


def _build_agent_request(
    request_id: str,
    recipient: AgentName,
    message_type: str,
    payload: dict,
) -> AgentRequest:
    return AgentRequest(
        request_id=request_id,
        sender=AgentName.ORCHESTRATOR,
        recipient=recipient,
        message_type=message_type,
        created_at=_now(),
        payload=payload,
    )


async def _exchange(
    ctx: Context,
    address: str,
    outgoing: AgentRequest,
    expected_sender: AgentName,
    expected_message_type: str,
    settings: OrchestratorSettings,
) -> AgentResponse:
    wire_request = request_to_message(outgoing)
    last_error = None
    for attempt in range(1, settings.max_attempts + 1):
        try:
            response, delivery = await asyncio.wait_for(
                ctx.send_and_receive(
                    address,
                    wire_request,
                    response_type=AgentResponseMessage,
                    timeout=settings.timeout_seconds,
                ),
                timeout=settings.timeout_seconds + 2,
            )
            if response is None or delivery.status != DeliveryStatus.DELIVERED:
                detail = delivery.detail or "no response received"
                last_error = WorkflowCallError(
                    "agent_timeout" if "timeout" in detail.lower() else "agent_delivery_failed",
                    detail,
                    retryable=True,
                )
                raise last_error
            validated = message_to_response(response)
            if (
                validated.request_id != outgoing.request_id
                or validated.sender != expected_sender
                or validated.recipient != AgentName.ORCHESTRATOR
                or validated.message_type != expected_message_type
            ):
                raise WorkflowCallError(
                    "invalid_agent_response",
                    "response identity or message type did not match the request",
                )
            return validated
        except asyncio.TimeoutError:
            last_error = WorkflowCallError("agent_timeout", "timed out waiting for agent response", True)
        except ValidationError as error:
            raise WorkflowCallError(
                "invalid_agent_response",
                f"response failed envelope validation ({len(error.errors())} validation error(s))",
            ) from error
        except WorkflowCallError as error:
            last_error = error
            if not error.retryable:
                raise
        except Exception as error:
            last_error = WorkflowCallError("agent_transport_error", type(error).__name__, True)
        _log(
            "agent_request_retry",
            logging.WARNING,
            request_id=outgoing.request_id,
            agent=expected_sender.value,
            attempt=attempt,
            error_code=last_error.code,
        )
        if attempt < settings.max_attempts:
            await asyncio.sleep(0.1 * attempt)
    raise last_error or WorkflowCallError("agent_call_failed", "agent call failed")


def _parse_opportunity_response(
    response: AgentResponse,
) -> tuple[OpportunitySearchResult | None, WorkflowStage]:
    result = None
    if response.payload:
        try:
            result = OpportunitySearchResult.model_validate(response.payload)
        except ValidationError as error:
            return None, _stage_failed(
                "invalid_opportunity_result",
                f"job result failed schema validation ({len(error.errors())} validation error(s))",
            )
    if response.status == ResponseStatus.ERROR:
        return result, _stage_failed(
            response.error.code,
            response.error.message,
        )
    if result is None:
        return None, _stage_failed("missing_opportunity_result", "agent returned no job search result")
    reported_sources = {report.source_id for report in result.source_reports}
    unreported_sources = {
        source_id
        for record in result.records
        for source_id in (
            record.job.source_id,
            *(alternate.source_id for alternate in record.alternate_sources),
        )
        if source_id not in reported_sources
    }
    if unreported_sources:
        return None, _stage_failed(
            "invalid_opportunity_provenance",
            "job records refer to sources absent from the source reports",
        )
    if result.search_status == SearchStatus.FAILED:
        return result, _stage_failed("job_sources_failed", "all configured job sources failed")
    if result.search_status == SearchStatus.PARTIAL:
        return result, WorkflowStage(status=StageStatus.PARTIAL)
    return result, WorkflowStage(status=StageStatus.SUCCEEDED)


def _parse_evidence_response(
    response: AgentResponse,
    request: CandidateWorkflowRequest,
) -> tuple[EvidenceProfileResult | None, WorkflowStage]:
    if response.status == ResponseStatus.ERROR:
        return None, _stage_failed(response.error.code, response.error.message)
    try:
        result = EvidenceProfileResult.model_validate(response.payload)
    except ValidationError as error:
        return None, _stage_failed(
            "invalid_evidence_result",
            f"evidence result failed schema validation ({len(error.errors())} validation error(s))",
        )
    if result.candidate_id != request.candidate.candidate_id:
        return None, _stage_failed(
            "candidate_mismatch",
            "evidence result belongs to a different candidate",
        )
    record_ids = {
        item.record_id for item in (*request.assessments, *request.practical_tasks, *request.projects)
    }
    invalid_matching_refs = {
        item.evidence_ref
        for item in result.matching_evidence
        if item.evidence_ref is not None and item.evidence_ref not in record_ids
    }
    if invalid_matching_refs:
        return None, _stage_failed(
            "invalid_evidence_provenance",
            "matching evidence references a record not supplied in the workflow request",
        )
    if any(item.candidate_id != request.candidate.candidate_id for item in result.matching_evidence):
        return None, _stage_failed(
            "candidate_mismatch",
            "matching evidence contains a different candidate id",
        )
    return result, WorkflowStage(status=StageStatus.SUCCEEDED)


def _parse_action_response(
    response: AgentResponse,
    candidate_id: str,
    opportunities: OpportunitySearchResult,
    evidence: EvidenceProfileResult,
) -> tuple[ActionPlanResult | None, WorkflowStage]:
    if response.status == ResponseStatus.ERROR:
        return None, _stage_failed(response.error.code, response.error.message)
    try:
        result = ActionPlanResult.model_validate(response.payload)
    except ValidationError as error:
        return None, _stage_failed(
            "invalid_action_result",
            f"action plan failed schema validation ({len(error.errors())} validation error(s))",
        )
    if result.candidate_id != candidate_id:
        return None, _stage_failed("candidate_mismatch", "action plan belongs to a different candidate")
    expected_jobs = {item.job.job_id: item.job for item in opportunities.records}
    actual_job_ids = [item.job_id for item in result.ranked_opportunities]
    if len(actual_job_ids) != len(set(actual_job_ids)) or set(actual_job_ids) != set(expected_jobs):
        return None, _stage_failed(
            "job_provenance_mismatch",
            "action plan jobs do not match the validated opportunity response",
        )
    for recommendation in result.ranked_opportunities:
        source_job = expected_jobs[recommendation.job_id]
        if (
            recommendation.source_id != source_job.source_id
            or str(recommendation.source_url) != str(source_job.source_url)
            or recommendation.match.job_id != recommendation.job_id
            or recommendation.match.candidate_id != candidate_id
            or recommendation.match.eligibility != recommendation.eligibility
        ):
            return None, _stage_failed(
                "job_provenance_mismatch",
                "action plan recommendation changed job provenance or matching identity",
            )
    evidence_refs = {
        item.evidence_ref
        for item in evidence.matching_evidence
        if item.evidence_ref is not None
    }
    action_refs = {
        reference
        for recommendation in result.ranked_opportunities
        for contribution in recommendation.match.breakdown
        for reference in contribution.evidence_refs
    }
    if not action_refs <= evidence_refs:
        return None, _stage_failed(
            "evidence_provenance_mismatch",
            "action plan cites evidence outside the validated evidence profile",
        )
    return result, WorkflowStage(status=StageStatus.SUCCEEDED)


async def run_candidate_workflow(
    ctx: Context,
    request_id: str,
    request: CandidateWorkflowRequest,
    addresses: AgentAddresses,
    settings: OrchestratorSettings,
    store: SQLiteWorkflowStore,
) -> CandidateWorkflowResult:
    """Coordinate all agents through signed uAgents request/reply messaging."""
    request_hash = _request_hash(request)
    claim = await asyncio.to_thread(store.claim, request_id, request_hash, _now().isoformat())
    if claim.state == "conflict":
        raise WorkflowRequestError(
            "request_id_conflict",
            "request_id was already used with different workflow input",
        )
    if claim.state == "in_progress":
        raise WorkflowRequestError("request_in_progress", "a matching workflow request is already running")
    if claim.state == "completed":
        replay = CandidateWorkflowResult.model_validate(
            claim.result.model_dump(mode="python") | {"replayed": True}
        )
        _log("workflow_replayed", request_id=request_id, status=replay.workflow_status.value)
        return replay

    opp_request = _build_agent_request(
        f"{request_id}:opportunity",
        AgentName.OPPORTUNITY_INTELLIGENCE,
        OPPORTUNITY_SEARCH_MESSAGE,
        request.job_search.model_dump(mode="json"),
    )
    evidence_request_data = EvidenceProfileRequest(
        profile=request.candidate,
        declared_skills=request.declared_skills,
        assessments=request.assessments,
        practical_tasks=request.practical_tasks,
        projects=request.projects,
        as_of=request.as_of,
    )
    evidence_request = _build_agent_request(
        f"{request_id}:evidence",
        AgentName.CANDIDATE_EVIDENCE,
        EVIDENCE_PROFILE_MESSAGE,
        evidence_request_data.model_dump(mode="json"),
    )

    _log("workflow_started", request_id=request_id)
    opp_raw, evidence_raw = await asyncio.gather(
        _exchange(
            ctx, addresses.opportunity, opp_request,
            AgentName.OPPORTUNITY_INTELLIGENCE, OPPORTUNITY_RESULT_MESSAGE, settings,
        ),
        _exchange(
            ctx, addresses.evidence, evidence_request,
            AgentName.CANDIDATE_EVIDENCE, EVIDENCE_RESULT_MESSAGE, settings,
        ),
        return_exceptions=True,
    )

    opportunity_result, opportunity_stage = (None, None)
    if isinstance(opp_raw, AgentResponse):
        opportunity_result, opportunity_stage = _parse_opportunity_response(opp_raw)
    else:
        error = opp_raw
        code = error.code if isinstance(error, WorkflowCallError) else "opportunity_call_failed"
        message = str(error) if isinstance(error, WorkflowCallError) else type(error).__name__
        opportunity_stage = _stage_failed(code, message)

    evidence_result, evidence_stage = (None, None)
    if isinstance(evidence_raw, AgentResponse):
        evidence_result, evidence_stage = _parse_evidence_response(evidence_raw, request)
    else:
        error = evidence_raw
        code = error.code if isinstance(error, WorkflowCallError) else "evidence_call_failed"
        message = str(error) if isinstance(error, WorkflowCallError) else type(error).__name__
        evidence_stage = _stage_failed(code, message)

    action_result = None
    if opportunity_result is None or evidence_result is None:
        action_stage = WorkflowStage(
            status=StageStatus.SKIPPED,
            message="action stage requires validated opportunity and evidence results",
        )
    elif not opportunity_result.records:
        action_stage = WorkflowStage(
            status=StageStatus.SKIPPED,
            message="opportunity search returned no job records to evaluate",
        )
    else:
        try:
            action_request_data = ActionPlanRequest(
                candidate=request.candidate,
                opportunities=opportunity_result.records,
                evidence_profile=evidence_result,
                matching_config=request.matching_config,
                decision_policy=request.decision_policy,
                improvement_metadata=request.improvement_metadata,
                include_asi_summary=request.include_asi_summary,
            )
        except ValidationError as error:
            action_stage = _stage_failed(
                "invalid_action_request",
                f"validated agent outputs cannot form an action request ({len(error.errors())} validation error(s))",
            )
        else:
            action_request = _build_agent_request(
                f"{request_id}:career-action",
                AgentName.CAREER_ACTION,
                CAREER_ACTION_MESSAGE,
                action_request_data.model_dump(mode="json"),
            )
            try:
                action_response = await _exchange(
                    ctx, addresses.career_action, action_request,
                    AgentName.CAREER_ACTION, CAREER_ACTION_RESULT_MESSAGE, settings,
                )
                action_result, action_stage = _parse_action_response(
                    action_response,
                    request.candidate.candidate_id,
                    opportunity_result,
                    evidence_result,
                )
            except WorkflowCallError as error:
                action_stage = _stage_failed(error.code, str(error))

    stages = (opportunity_stage, evidence_stage, action_stage)
    successes = sum(stage.status in {StageStatus.SUCCEEDED, StageStatus.PARTIAL} for stage in stages)
    if all(stage.status == StageStatus.SUCCEEDED for stage in stages):
        workflow_status = WorkflowStatus.SUCCEEDED
    elif successes == 0:
        workflow_status = WorkflowStatus.FAILED
    else:
        workflow_status = WorkflowStatus.PARTIAL

    result = CandidateWorkflowResult(
        request_id=request_id,
        workflow_status=workflow_status,
        opportunity_stage=opportunity_stage,
        evidence_stage=evidence_stage,
        action_stage=action_stage,
        opportunities=opportunity_result,
        evidence_profile=evidence_result,
        action_plan=action_result,
        completed_at=_now(),
    )
    await asyncio.to_thread(store.complete, result)
    _log(
        "workflow_completed",
        request_id=request_id,
        status=workflow_status.value,
        opportunity_count=len(opportunity_result.records) if opportunity_result else 0,
        evidence_skill_count=len(evidence_result.skills) if evidence_result else 0,
        recommendation_count=len(action_result.ranked_opportunities) if action_result else 0,
    )
    return result


agent = Agent(
    name=AGENT_NAME,
    seed=os.environ.get("SKILLBRIDGE_ORCHESTRATOR_SEED"),
    port=int(os.environ.get("SKILLBRIDGE_ORCHESTRATOR_PORT", "8000")),
)


@agent.on_message(model=AgentRequestMessage, replies={AgentResponseMessage}, allow_unverified=False)
async def on_workflow_request(ctx: Context, sender: str, message: AgentRequestMessage):
    reply_to = AgentName.ORCHESTRATOR
    try:
        envelope = message_to_request(message)
        reply_to = envelope.sender
        if envelope.recipient != AgentName.ORCHESTRATOR or envelope.message_type != WORKFLOW_MESSAGE:
            raise WorkflowRequestError("unsupported_message", "request is not a candidate workflow request")
        request = CandidateWorkflowRequest.model_validate(envelope.payload)
        settings = OrchestratorSettings.from_env()
        addresses = AgentAddresses.from_env()
        store = SQLiteWorkflowStore(settings.database_path)
        result = await run_candidate_workflow(
            ctx, envelope.request_id, request, addresses, settings, store,
        )
        status = ResponseStatus.ERROR if result.workflow_status == WorkflowStatus.FAILED else ResponseStatus.SUCCESS
        response = AgentResponse(
            request_id=envelope.request_id,
            sender=AgentName.ORCHESTRATOR,
            recipient=envelope.sender,
            message_type=WORKFLOW_RESULT_MESSAGE,
            created_at=_now(),
            status=status,
            payload=result.model_dump(mode="json"),
            error=(
                AgentError(code="workflow_failed", message="required workflow stages failed")
                if status == ResponseStatus.ERROR else None
            ),
        )
    except ValidationError as error:
        response = AgentResponse(
            request_id=message.request_id,
            sender=AgentName.ORCHESTRATOR,
            recipient=reply_to,
            message_type=WORKFLOW_RESULT_MESSAGE,
            created_at=_now(),
            status=ResponseStatus.ERROR,
            error=AgentError(
                code="invalid_workflow_request",
                message=f"request failed validation ({len(error.errors())} validation error(s))",
            ),
        )
    except WorkflowRequestError as error:
        response = AgentResponse(
            request_id=message.request_id,
            sender=AgentName.ORCHESTRATOR,
            recipient=reply_to,
            message_type=WORKFLOW_RESULT_MESSAGE,
            created_at=_now(),
            status=ResponseStatus.ERROR,
            error=AgentError(code=error.code, message=str(error)),
        )
    except Exception as error:
        _log("workflow_handler_failed", logging.ERROR, error=type(error).__name__)
        response = AgentResponse(
            request_id=message.request_id,
            sender=AgentName.ORCHESTRATOR,
            recipient=reply_to,
            message_type=WORKFLOW_RESULT_MESSAGE,
            created_at=_now(),
            status=ResponseStatus.ERROR,
            error=AgentError(code="workflow_internal_error", message="workflow could not be completed"),
        )
    await ctx.send(sender, response_to_message(response))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent.run()
