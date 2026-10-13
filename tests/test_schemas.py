from datetime import datetime

import pytest
from pydantic import ValidationError

from skillbridge.schemas import (
    AgentError,
    AgentRequest,
    AgentResponse,
    CareerAction,
    EligibilityRequirement,
    EligibilityResult,
    MatchResult,
    SCHEMA_VERSION,
)
from tests.conftest import NOW, make_candidate, make_evidence, make_job


def envelope(**overrides):
    data = {
        "request_id": "req-1",
        "sender": "orchestrator",
        "recipient": "opportunity_intelligence",
        "message_type": "find_jobs",
        "created_at": NOW,
    }
    return {**data, **overrides}


# --- candidate / evidence


def test_valid_candidate():
    candidate = make_candidate(claimed_skills=["  SIEM "])
    assert candidate.claimed_skills == ["SIEM"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_role": "  "},
        {"learning_hours_per_week": -1},
        {"learning_hours_per_week": 169},
        {"experience_level": "wizard"},
        {"surprise": 1},
    ],
)
def test_invalid_candidate(overrides):
    with pytest.raises(ValidationError):
        make_candidate(**overrides)


def test_assessment_requires_score_and_reference():
    with pytest.raises(ValidationError):
        make_evidence(kind="assessment", assessment_score=None)
    with pytest.raises(ValidationError):
        make_evidence(kind="assessment", evidence_ref=None)


def test_assessment_score_bounded():
    with pytest.raises(ValidationError):
        make_evidence(kind="assessment", assessment_score=1.5)


def test_project_requires_reference_and_rejects_score():
    with pytest.raises(ValidationError):
        make_evidence(kind="project", evidence_ref=None)
    with pytest.raises(ValidationError):
        make_evidence(kind="project", assessment_score=0.5)


def test_self_reported_cannot_be_verified():
    with pytest.raises(ValidationError):
        make_evidence(kind="self_reported", verification_status="verified")


def test_evidence_timestamp_must_be_timezone_aware():
    with pytest.raises(ValidationError):
        make_evidence(recorded_at=datetime(2026, 1, 1))


# --- job


def test_valid_job_preserves_source_metadata():
    job = make_job()
    assert str(job.source_url) == "https://example.com/jobs/1"
    assert job.source_id == "fixture-source"
    assert job.retrieved_at == NOW
    assert job.application_deadline is None


def test_job_unknown_fields_default_to_none_not_empty():
    job = make_job(required_skills=None, eligibility_requirements=None, location=None)
    assert job.required_skills is None
    assert job.eligibility_requirements is None
    assert job.location is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_url": "not a url"},
        {"source_url": "ftp://example.com/x"},
        {"retrieved_at": datetime(2026, 1, 1)},
        {"source_id": ""},
        {"title": " "},
    ],
)
def test_invalid_job(overrides):
    with pytest.raises(ValidationError):
        make_job(**overrides)


def test_job_requires_source_fields():
    with pytest.raises(ValidationError):
        make_job(source_id=None)


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "min_experience_level", "expected": "guru"},
        {"kind": "min_experience_level", "expected": "junior", "fact_key": "x"},
        {"kind": "fact_equals", "expected": "IN"},
        {"kind": "fact_equals", "expected": ["IN"], "fact_key": "wa"},
        {"kind": "fact_one_of", "expected": [], "fact_key": "wa"},
        {"kind": "fact_one_of", "expected": "IN", "fact_key": "wa"},
        {"kind": "fact_min_number", "expected": "3", "fact_key": "gpa"},
        {"kind": "fact_min_number", "expected": True, "fact_key": "gpa"},
    ],
)
def test_invalid_eligibility_requirement(bad):
    with pytest.raises(ValidationError):
        EligibilityRequirement(requirement_id="r", description="d", **bad)


# --- eligibility result / match result


def test_eligibility_result_status_must_match_checks():
    with pytest.raises(ValidationError, match="inconsistent"):
        EligibilityResult(
            job_id="j", candidate_id="c", requirements_known=True,
            status="eligible_on_available_information",
            checks=[{"requirement_id": "r", "outcome": "fail", "reason": "no"}],
        )


def test_unknown_requirements_cannot_be_eligible():
    with pytest.raises(ValidationError):
        EligibilityResult(
            job_id="j", candidate_id="c", requirements_known=False,
            status="eligible_on_available_information",
        )


def test_match_result_score_presence_rule():
    eligibility = EligibilityResult(
        job_id="j", candidate_id="c", requirements_known=False, status="unknown"
    )
    with pytest.raises(ValidationError):
        MatchResult(job_id="j", candidate_id="c", rubric_version="v", score_status="scored",
                    eligibility=eligibility)
    with pytest.raises(ValidationError):
        MatchResult(job_id="j", candidate_id="c", rubric_version="v",
                    score_status="insufficient_job_data", skill_alignment_score=50,
                    eligibility=eligibility)


# --- career action


def test_career_action_valid_and_rules():
    action = CareerAction(action_id="a", job_id="j", action_type="apply_now",
                          rationale="Meets requirements", priority=1)
    assert action.resource_url is None
    with pytest.raises(ValidationError):
        CareerAction(action_id="a", job_id="j", action_type="build_project",
                     rationale="r", priority=1)
    with pytest.raises(ValidationError):
        CareerAction(action_id="a", job_id="j", action_type="apply_now",
                     rationale="r", priority=0)


# --- envelopes


def test_request_defaults_to_current_schema_version():
    assert AgentRequest(**envelope()).schema_version == SCHEMA_VERSION


@pytest.mark.parametrize("version", ["2.0", "abc", "1", "0.9"])
def test_unsupported_schema_version_rejected(version):
    with pytest.raises(ValidationError):
        AgentRequest(**envelope(schema_version=version))


def test_minor_version_bump_accepted():
    assert AgentRequest(**envelope(schema_version="1.7")).schema_version == "1.7"


def test_unknown_agent_rejected():
    with pytest.raises(ValidationError):
        AgentRequest(**envelope(recipient="fourth_ai_agent"))


def test_response_error_consistency():
    ok = AgentResponse(**envelope(status="success"))
    assert ok.error is None
    with pytest.raises(ValidationError):
        AgentResponse(**envelope(status="error"))
    with pytest.raises(ValidationError):
        AgentResponse(**envelope(status="success", error=AgentError(code="x", message="y")))
    failed = AgentResponse(**envelope(status="error", error={"code": "timeout", "message": "slow"}))
    assert failed.error.code == "timeout"
