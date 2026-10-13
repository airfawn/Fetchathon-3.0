from datetime import datetime, timezone

import pytest

from skillbridge.schemas import (
    CandidateProfile,
    EligibilityKind,
    EligibilityRequirement,
    EvidenceType,
    JobOpportunity,
    SkillEvidence,
    SkillRequirement,
    VerificationStatus,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_candidate(**overrides) -> CandidateProfile:
    data = {
        "candidate_id": "cand-1",
        "target_role": "SOC Analyst",
        "preferred_location": "Remote",
        "experience_level": "entry_level",
        "claimed_skills": [],
        "learning_hours_per_week": 10,
    }
    return CandidateProfile(**{**data, **overrides})


def make_job(**overrides) -> JobOpportunity:
    data = {
        "job_id": "job-1",
        "title": "SOC Analyst",
        "employer": "Example Corp",
        "location": "Remote",
        "required_skills": [SkillRequirement(skill="SIEM")],
        "preferred_skills": [],
        "eligibility_requirements": [],
        "source_url": "https://example.com/jobs/1",
        "source_id": "fixture-source",
        "retrieved_at": NOW,
    }
    return JobOpportunity(**{**data, **overrides})


def make_evidence(skill="SIEM", kind="assessment", **overrides) -> SkillEvidence:
    data = {
        "candidate_id": "cand-1",
        "skill_id": skill,
        "evidence_type": EvidenceType(kind),
        "recorded_at": NOW,
    }
    if kind == "assessment":
        data.update(evidence_ref="assess-1", assessment_score=0.9,
                    verification_status=VerificationStatus.VERIFIED)
    elif kind == "project":
        data.update(evidence_ref="https://example.com/repo",
                    verification_status=VerificationStatus.VERIFIED)
    return SkillEvidence(**{**data, **overrides})


def experience_requirement(level="junior") -> EligibilityRequirement:
    return EligibilityRequirement(
        requirement_id="exp",
        description=f"At least {level}",
        kind=EligibilityKind.MIN_EXPERIENCE_LEVEL,
        expected=level,
    )


@pytest.fixture
def candidate():
    return make_candidate()


@pytest.fixture
def job():
    return make_job()
