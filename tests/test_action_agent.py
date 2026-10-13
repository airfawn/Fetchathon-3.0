"""Deterministic Career Action Agent tests; ASI:One is always mocked."""

import asyncio
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from agents import action_agent as aa
from skillbridge.matching import evaluate_match as shared_evaluate_match
from skillbridge.schemas import (
    ActionPlanRequest,
    AgentName,
    AgentRequest,
    EvidenceProfileResult,
    ExtractionMethod,
    ExperienceRequirement,
    OpportunityRecord,
    SkillEvidenceProfile,
    SkillEvidenceProfile,
    SourceKind,
    missing_job_fields,
)
from tests.conftest import make_candidate, make_evidence, make_job

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
ECLASS = aa.EvidenceClass
DECISION = aa.OpportunityDecision
STATUS = aa.EligibilityStatus


def profile(candidate_id="cand-1", evidence=()):
    return EvidenceProfileResult(candidate_id=candidate_id, as_of=NOW, skills=[],
                                 matching_evidence=list(evidence))


def job_record(job=None, *, job_id="job-1", required=("SIEM",), preferred=(),
               eligibility=(), location="Remote", source_id="fixture-source", remote=True):
    job = job or make_job(
        job_id=job_id,
        required_skills=None if required is None else [{"skill": s} for s in required],
        preferred_skills=None if preferred is None else [{"skill": s} for s in preferred],
        eligibility_requirements=None if eligibility is None else list(eligibility),
        location=location,
        source_id=source_id,
    )
    description = "Fixture job description."
    return OpportunityRecord(
        job=job, source_kind=SourceKind.FIXTURE, original_title=job.title,
        normalized_title=job.title.casefold(), description=description, remote=remote,
        skills_extraction=ExtractionMethod.STRUCTURED_SOURCE,
        eligibility_extraction=ExtractionMethod.STRUCTURED_SOURCE,
        missing_fields=missing_job_fields(job, description, remote),
    )


def request(candidate=None, records=None, evidence=(), metadata=(), **kwargs):
    candidate = candidate or make_candidate(claimed_skills=["SIEM"])
    return ActionPlanRequest(
        candidate=candidate,
        opportunities=list(records or [job_record()]),
        evidence_profile=profile(candidate.candidate_id, evidence),
        improvement_metadata=list(metadata),
        **kwargs,
    )


def run(*args, **kwargs):
    return aa.build_action_plan(request(*args, **kwargs))


def test_strong_eligible_candidate_is_apply_now():
    result = run(evidence=[make_evidence("SIEM", "assessment", assessment_score=0.95)])
    recommendation = result.ranked_opportunities[0]
    assert recommendation.decision == DECISION.APPLY_NOW
    assert recommendation.eligibility.status == STATUS.ELIGIBLE_ON_AVAILABLE_INFORMATION
    assert recommendation.match.skill_alignment_score == 95
    assert recommendation.supported_competencies == ["siem"]
    assert "not employment probabilities" in result.notes[0]


def test_eligible_candidate_missing_preferred_skill_is_not_delayed():
    result = run(records=[job_record(required=("SIEM",), preferred=("Python",))],
                 evidence=[make_evidence("SIEM", "assessment", assessment_score=0.9)])
    recommendation = result.ranked_opportunities[0]
    assert recommendation.decision == DECISION.APPLY_NOW
    assert any(i.skill_id == "python" for i in result.prioritized_improvements)


def test_confirmed_mandatory_failure_deprioritizes_even_with_high_score():
    from tests.conftest import experience_requirement

    result = run(
        records=[job_record(eligibility=[experience_requirement("senior")])],
        candidate=make_candidate(experience_level="entry_level", claimed_skills=["SIEM"]),
        evidence=[make_evidence("SIEM", "assessment", assessment_score=1.0)],
    )
    recommendation = result.ranked_opportunities[0]
    assert recommendation.match.skill_alignment_score == 100
    assert recommendation.eligibility.status == STATUS.INELIGIBLE
    assert recommendation.decision == DECISION.DEPRIORITIZE


def test_unknown_eligibility_requires_more_information():
    from skillbridge.schemas import EligibilityKind, EligibilityRequirement

    requirement = EligibilityRequirement(
        requirement_id="auth", description="Work authorization in IN",
        kind=EligibilityKind.FACT_EQUALS, fact_key="work_authorization", expected="IN")
    recommendation = run(records=[job_record(eligibility=[requirement])]).ranked_opportunities[0]
    assert recommendation.eligibility.status == STATUS.UNKNOWN
    assert recommendation.decision == DECISION.NEEDS_MORE_INFORMATION


def test_explicit_years_requirement_with_categorical_only_profile_needs_more_information():
    record = job_record()
    record = record.model_copy(update={
        "experience_requirement": ExperienceRequirement(minimum_years=2, statement="At least 2 years")
    })
    record = record.model_copy(update={
        "missing_fields": [name for name in record.missing_fields if name != "experience_requirement"]
    })
    recommendation = run(records=[record],
                         evidence=[make_evidence("SIEM", "assessment", assessment_score=0.95)]).ranked_opportunities[0]
    assert recommendation.eligibility.status == STATUS.ELIGIBLE_ON_AVAILABLE_INFORMATION
    assert recommendation.decision == DECISION.NEEDS_MORE_INFORMATION
    assert any("numeric years-experience" in note for note in recommendation.constraint_notes)


def test_unsupported_claim_is_exposed_and_strengthen_first():
    result = run(candidate=make_candidate(claimed_skills=["SIEM"]))
    recommendation = result.ranked_opportunities[0]
    assert recommendation.unsupported_claims == ["siem"]
    assert recommendation.decision == DECISION.STRENGTHEN_FIRST
    assert recommendation.match.breakdown[0].evidence_basis == aa.EvidenceBasis.SELF_REPORTED


def test_declared_claim_from_candidate_evidence_profile_is_not_lost():
    upstream = EvidenceProfileResult(
        candidate_id="cand-1", as_of=NOW,
        skills=[SkillEvidenceProfile(
            skill_id="siem", recognized=True, declared=True,
            evidence_classes=["self_reported"], confidence="self_report_only",
            needs_assessment=True, recommended_assessment="Complete an assessment for SIEM",
            explanation="Candidate declared SIEM.",
        )],
    )
    plan_request = ActionPlanRequest(
        candidate=make_candidate(claimed_skills=[]),
        opportunities=[job_record()],
        evidence_profile=upstream.model_dump(mode="json"),
    )
    recommendation = aa.build_action_plan(plan_request).ranked_opportunities[0]
    assert recommendation.match.breakdown[0].evidence_basis == aa.EvidenceBasis.SELF_REPORTED
    assert recommendation.unsupported_claims == ["siem"]


def test_incomplete_required_job_information_needs_more_information():
    recommendation = run(records=[job_record(required=None)]).ranked_opportunities[0]
    assert recommendation.match.score_status == aa.ScoreStatus.INSUFFICIENT_JOB_DATA
    assert recommendation.eligibility.status == STATUS.ELIGIBLE_ON_AVAILABLE_INFORMATION
    assert recommendation.decision == DECISION.NEEDS_MORE_INFORMATION


def test_shared_matching_engine_has_identical_result():
    candidate = make_candidate(claimed_skills=["SIEM"])
    record = job_record(required=("SIEM",), preferred=("Python",))
    evidence_sets = [
        [make_evidence("SIEM", "assessment", assessment_score=0.83)],
        [make_evidence("SIEM", "practical_task", evidence_ref="task-1", assessment_score=0.76)],
        [make_evidence("SIEM", "project", verification_status="unverified")],
    ]
    normalizer = aa.SkillNormalizer.from_config(aa.AliasConfig.model_validate(aa.SKILL_ALIAS_DATA))
    for evidence in evidence_sets:
        shared = shared_evaluate_match(candidate, evidence, record.job)
        local = aa.evaluate_match(candidate, evidence, record.job, normalizer, aa.MatchRubric())
        assert local.model_dump(mode="json") == shared.model_dump(mode="json")


def test_candidate_evidence_agent_profile_feeds_action_agent_without_reconstructing_scores():
    from agents import evidence_agent as evidence_agent

    evidence_request = evidence_agent.EvidenceProfileRequest.model_validate({
        "profile": {
            "candidate_id": "cand-1", "target_role": "SOC Analyst", "preferred_location": "Remote",
            "experience_level": "entry_level", "claimed_skills": ["SIEM"], "learning_hours_per_week": 10,
        },
        "as_of": NOW.isoformat(),
        "assessments": [{
            "record_id": "real-assessment-1", "candidate_id": "cand-1",
            "skills_evaluated": ["SIEM"], "score": 0.91, "taken_at": NOW.isoformat(),
            "provider": "stored-provider", "verification_status": "verified", "verified_by": "reviewer-1",
        }],
    })
    upstream_profile = evidence_agent.build_evidence_profile(evidence_request)
    plan_request = ActionPlanRequest(
        candidate=make_candidate(claimed_skills=["SIEM"]),
        opportunities=[job_record()],
        evidence_profile=upstream_profile.model_dump(mode="json"),
    )
    plan = aa.build_action_plan(plan_request)
    recommendation = plan.ranked_opportunities[0]
    assert recommendation.match.skill_alignment_score == 91
    assert recommendation.decision == DECISION.APPLY_NOW
    assert recommendation.match.breakdown[0].evidence_refs == ["real-assessment-1"]


def test_repeated_skill_improvement_is_aggregated_across_jobs():
    records = [
        job_record(job_id="job-a", required=("SIEM", "Linux administration")),
        job_record(job_id="job-b", required=("Linux system administration", "Python")),
    ]
    result = run(records=records, candidate=make_candidate(claimed_skills=["SIEM"]))
    linux = next(item for item in result.prioritized_improvements if item.skill_id == "linux-administration")
    assert len(linux.requirements_addressed) == 2
    assert {ref.job_id for ref in linux.requirements_addressed} == {"job-a", "job-b"}
    assert linux.estimated_hours is None and linux.resources == []


def test_metadata_provides_effort_completion_and_real_resource_refs():
    metadata = [{
        "skill_id": "SIEM", "estimated_hours": 6,
        "evidence_to_produce": "Complete a stored SIEM assessment.",
        "completion_criteria": "Assessment record reaches 0.7 and includes its provider.",
        "resources": [{"title": "Vendor course", "url": "https://example.com/course", "source_id": "source-1"}],
    }]
    result = run(metadata=metadata)
    improvement = result.prioritized_improvements[0]
    assert improvement.estimated_hours == 6
    assert improvement.estimated_weeks_at_available_hours == 0.6
    assert improvement.evidence_to_produce == "Complete a stored SIEM assessment."
    assert improvement.resources[0].source_id == "source-1"


def test_ranking_is_stable_for_identical_inputs():
    records = [job_record(job_id="b", required=("SIEM",)), job_record(job_id="a", required=("SIEM",))]
    candidate = make_candidate(claimed_skills=["SIEM"])
    first = run(candidate=candidate, records=records).model_dump_json()
    second = run(candidate=candidate, records=records).model_dump_json()
    assert first == second
    assert [x.job_id for x in run(candidate=candidate, records=records).ranked_opportunities] == ["a", "b"]


def test_invalid_upstream_payload_is_rejected():
    candidate = make_candidate()
    with pytest.raises(Exception):
        ActionPlanRequest.model_validate({
            "candidate": candidate.model_dump(mode="json"),
            "opportunities": [{"job": {"job_id": "bad"}}],
            "evidence_profile": {"candidate_id": "different", "as_of": NOW.isoformat(), "skills": []},
        })


def test_mismatched_upstream_candidate_is_rejected():
    with pytest.raises(Exception, match="different candidate"):
        ActionPlanRequest.model_validate(request().model_dump(mode="json") | {
                "evidence_profile": {"candidate_id": "someone-else", "as_of": NOW.isoformat(), "skills": []}
            })


def envelope(payload, recipient=AgentName.CAREER_ACTION, message_type=aa.CAREER_ACTION_MESSAGE):
    return AgentRequest(
        request_id="request-1", sender=AgentName.ORCHESTRATOR, recipient=recipient,
        message_type=message_type, created_at=NOW, payload=payload)


def test_envelope_handler_returns_validated_plan():
    req = envelope(request().model_dump(mode="json"))
    response = aa.handle_action_request(req)
    assert response.status == "success"
    assert response.message_type == aa.CAREER_ACTION_RESULT_MESSAGE
    assert aa.ActionPlanResult.model_validate(response.payload).candidate_id == "cand-1"


@pytest.mark.parametrize(("recipient", "message_type", "code"), [
    (AgentName.OPPORTUNITY_INTELLIGENCE, aa.CAREER_ACTION_MESSAGE, "wrong_recipient"),
    (AgentName.CAREER_ACTION, "other", "unsupported_message_type"),
])
def test_envelope_rejects_wrong_route(recipient, message_type, code):
    response = aa.handle_action_request(envelope({}, recipient, message_type))
    assert response.status == "error" and response.error.code == code


def test_invalid_upstream_envelope_payload_returns_controlled_error():
    response = aa.handle_action_request(envelope({"candidate": {}}))
    assert response.status == "error" and response.error.code == "invalid_request"


def test_agentverse_file_is_self_contained_and_imports_isolated(tmp_path):
    from scripts.sync_agent_models import REPO_ROOT, forbidden_imports

    source = (REPO_ROOT / "agents" / "action_agent.py").read_text()
    assert forbidden_imports(source) == []
    (tmp_path / "isolated_action.py").write_text(source)
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         f"import sys; sys.path.insert(0, {str(tmp_path)!r}); import isolated_action as m; print(m.AGENT_NAME)"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, env={"PATH": ""},
    )
    assert "skillbridge_career_action" in result.stdout, result.stderr


def test_uagents_message_handler_registered():
    from uagents import Model

    protocol = aa.agent._protocol
    digest = Model.build_schema_digest(aa.AgentRequestMessage)
    assert digest in protocol._signed_message_handlers
    assert Model.build_schema_digest(aa.AgentResponseMessage) in protocol._replies[digest]


def test_asi_one_configuration_uses_agentverse_environment(monkeypatch):
    monkeypatch.setenv("ASI1_API_KEY", "agentverse-secret")
    monkeypatch.setenv("ASI1_BASE_URL", "https://api.asi1.ai/v1")
    monkeypatch.setenv("SKILLBRIDGE_LLM_MODEL", "asi1")
    settings = aa.ASIOneSettings.from_env()
    assert settings.api_key == "agentverse-secret"
    assert settings.base_url == "https://api.asi1.ai/v1"
    assert settings.model == "asi1"


def test_asi_one_rejects_non_https_base_url(monkeypatch):
    monkeypatch.setenv("ASI1_BASE_URL", "http://asi.example/v1")
    with pytest.raises(ValueError, match="HTTPS"):
        aa.ASIOneSettings.from_env()
