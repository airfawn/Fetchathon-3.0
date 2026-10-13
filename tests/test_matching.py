import pytest
from pydantic import ValidationError

from skillbridge.matching import MatchRubric, evaluate_match
from skillbridge.schemas import (
    EligibilityStatus,
    EvidenceBasis,
    RequirementLevel,
    ScoreStatus,
    SkillRequirement,
)
from tests.conftest import NOW, experience_requirement, make_candidate, make_evidence, make_job


def req(*names):
    return [SkillRequirement(skill=n) for n in names]


def by_skill(result):
    return {c.skill_id: c for c in result.breakdown}


def test_verified_assessment_scores_its_result():
    result = evaluate_match(make_candidate(), [make_evidence(assessment_score=0.9)], make_job())
    assert result.score_status == ScoreStatus.SCORED
    assert result.skill_alignment_score == 90.0
    assert result.breakdown[0].evidence_basis == EvidenceBasis.VERIFIED_ASSESSMENT


def test_evidence_tiers_are_ordered():
    candidate = make_candidate(claimed_skills=["SIEM"])
    job = make_job()
    assessed = evaluate_match(candidate, [make_evidence(assessment_score=0.95)], job)
    project = evaluate_match(candidate, [make_evidence(kind="project")], job)
    unverified = evaluate_match(
        candidate, [make_evidence(kind="project", verification_status="unverified")], job
    )
    claimed = evaluate_match(candidate, [], job)
    scores = [r.skill_alignment_score for r in (assessed, project, unverified, claimed)]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == 4


def test_unsupported_claim_gets_only_self_reported_credit():
    result = evaluate_match(make_candidate(claimed_skills=["SIEM"]), [], make_job())
    item = result.breakdown[0]
    assert item.evidence_basis == EvidenceBasis.SELF_REPORTED
    assert item.credit == 0.25
    assert "no supporting evidence" in item.explanation


def test_missing_evidence_is_unevaluated_not_incompetence():
    result = evaluate_match(make_candidate(), [], make_job())
    item = result.breakdown[0]
    assert item.evidence_basis == EvidenceBasis.NO_EVIDENCE
    assert item.credit == 0
    assert "not evidence that the candidate lacks it" in item.explanation


def test_low_verified_assessment_is_not_masked_by_claim():
    candidate = make_candidate(claimed_skills=["SIEM"])
    result = evaluate_match(candidate, [make_evidence(assessment_score=0.1)], make_job())
    assert result.breakdown[0].evidence_basis == EvidenceBasis.VERIFIED_ASSESSMENT
    assert result.skill_alignment_score == 10.0


def test_rejected_evidence_is_ignored():
    evidence = make_evidence(verification_status="rejected")
    result = evaluate_match(make_candidate(), [evidence], make_job())
    assert result.breakdown[0].evidence_basis == EvidenceBasis.NO_EVIDENCE
    assert "1 rejected" in result.breakdown[0].explanation


def test_latest_verified_assessment_wins():
    old = make_evidence(assessment_score=0.2, evidence_ref="old")
    new = make_evidence(assessment_score=0.8, evidence_ref="new",
                        recorded_at=NOW.replace(year=2027))
    result = evaluate_match(make_candidate(), [old, new], make_job())
    assert result.skill_alignment_score == 80.0
    assert result.breakdown[0].evidence_refs == ["new"]


def test_required_outweighs_preferred():
    job = make_job(required_skills=req("SIEM"), preferred_skills=req("Python"))
    siem_only = evaluate_match(make_candidate(), [make_evidence("SIEM", assessment_score=1.0)], job)
    python_only = evaluate_match(make_candidate(), [make_evidence("Python", assessment_score=1.0)], job)
    assert siem_only.skill_alignment_score == 75.0
    assert python_only.skill_alignment_score == 25.0
    assert by_skill(siem_only)["siem"].weight > by_skill(siem_only)["python"].weight


def test_breakdown_sums_to_score_and_max_points_to_100():
    job = make_job(required_skills=req("SIEM", "Linux administration"),
                   preferred_skills=req("Python"))
    evidence = [make_evidence("SIEM", assessment_score=0.5), make_evidence("Python", kind="project")]
    result = evaluate_match(make_candidate(), evidence, job)
    assert result.skill_alignment_score == pytest.approx(sum(c.points for c in result.breakdown), abs=0.01)
    assert sum(c.max_points for c in result.breakdown) == pytest.approx(100, abs=0.01)


def test_alias_in_evidence_and_claim_matches_job_requirement():
    job = make_job(required_skills=req("Security Information and Event Management"))
    result = evaluate_match(
        make_candidate(claimed_skills=["siem"]), [make_evidence("SIEM", assessment_score=0.6)], job
    )
    assert len(result.breakdown) == 1
    assert result.skill_alignment_score == 60.0


def test_unknown_skill_is_scored_and_flagged():
    job = make_job(required_skills=req("Quantum Basket Weaving"))
    result = evaluate_match(make_candidate(claimed_skills=["quantum basket weaving"]), [], job)
    item = result.breakdown[0]
    assert item.recognized_skill is False
    assert item.skill_id.startswith("unknown:")
    assert item.evidence_basis == EvidenceBasis.SELF_REPORTED


def test_duplicate_requirement_counted_once_as_required():
    job = make_job(required_skills=req("SIEM"),
                   preferred_skills=req("Security information and event management"))
    result = evaluate_match(make_candidate(), [], job)
    assert len(result.breakdown) == 1
    assert result.breakdown[0].level == RequirementLevel.REQUIRED


def test_high_skill_score_does_not_override_ineligibility():
    job = make_job(eligibility_requirements=[experience_requirement("senior")])
    result = evaluate_match(
        make_candidate(experience_level="student"), [make_evidence(assessment_score=1.0)], job
    )
    assert result.skill_alignment_score == 100.0
    assert result.eligibility.status == EligibilityStatus.INELIGIBLE


def test_unknown_eligibility_reported_with_score():
    job = make_job(eligibility_requirements=None)
    result = evaluate_match(make_candidate(), [make_evidence()], job)
    assert result.score_status == ScoreStatus.SCORED
    assert result.eligibility.status == EligibilityStatus.UNKNOWN


def test_missing_required_skills_yields_no_score():
    result = evaluate_match(make_candidate(), [make_evidence()], make_job(required_skills=None))
    assert result.score_status == ScoreStatus.INSUFFICIENT_JOB_DATA
    assert result.skill_alignment_score is None
    assert result.breakdown == []


def test_empty_requirements_yield_no_score_rather_than_perfect_score():
    result = evaluate_match(make_candidate(), [], make_job(required_skills=[], preferred_skills=[]))
    assert result.score_status == ScoreStatus.INSUFFICIENT_JOB_DATA
    assert result.skill_alignment_score is None


def test_unknown_preferred_skills_are_noted_and_ignored():
    result = evaluate_match(make_candidate(), [make_evidence()], make_job(preferred_skills=None))
    assert result.skill_alignment_score == 90.0
    assert any("preferred" in n for n in result.notes)


def test_scoring_is_reproducible_and_order_independent():
    job = make_job(required_skills=req("SIEM", "Linux administration"))
    evidence = [make_evidence("SIEM", assessment_score=0.7),
                make_evidence("Linux administration", kind="project")]
    candidate = make_candidate(claimed_skills=["SIEM"])
    first = evaluate_match(candidate, evidence, job)
    second = evaluate_match(candidate, list(reversed(evidence)), job)
    assert first.model_dump() == second.model_dump() == evaluate_match(candidate, evidence, job).model_dump()


def test_result_never_claims_employment_probability():
    result = evaluate_match(make_candidate(), [], make_job())
    assert "not a probability of employment" in result.score_meaning
    assert not any("probab" in field for field in type(result).model_fields)


def test_evidence_for_another_candidate_is_rejected():
    with pytest.raises(ValueError, match="other candidates"):
        evaluate_match(make_candidate(), [make_evidence(candidate_id="someone-else")], make_job())


def test_custom_rubric_changes_weights_and_version():
    rubric = MatchRubric(version="custom", required_weight=1.5, preferred_weight=1.0)
    job = make_job(required_skills=req("SIEM"), preferred_skills=req("Python"))
    result = evaluate_match(make_candidate(), [make_evidence("SIEM", assessment_score=1.0)], job,
                            rubric=rubric)
    assert result.rubric_version == "custom"
    assert result.skill_alignment_score == 60.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"required_weight": 1.0, "preferred_weight": 1.0},
        {"self_reported_credit": 0.9, "unverified_credit": 0.3},
        {"verified_project_credit": 1.2},
    ],
)
def test_invalid_rubric_rejected(kwargs):
    with pytest.raises(ValidationError):
        MatchRubric(**kwargs)
