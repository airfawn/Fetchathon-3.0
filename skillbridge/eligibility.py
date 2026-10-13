"""Deterministic evaluation of mandatory eligibility requirements.

Outcomes per requirement are pass, fail (confirmed unmet) or unknown
(information missing or not comparable). Unknown is never a pass.
"""

from skillbridge.normalization import canonical_text
from skillbridge.schemas import (
    CandidateProfile,
    CheckOutcome,
    EligibilityKind,
    EligibilityRequirement,
    EligibilityResult,
    ExperienceLevel,
    FactValue,
    JobOpportunity,
    RequirementCheck,
    derive_eligibility_status,
)


def _equals(actual: FactValue, expected: FactValue) -> bool | None:
    """True/False when comparable, None when the types are not comparable."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        if isinstance(actual, bool) and isinstance(expected, bool):
            return actual == expected
        return None
    if isinstance(actual, str) and isinstance(expected, str):
        return canonical_text(actual) == canonical_text(expected)
    if isinstance(actual, str) or isinstance(expected, str):
        return None
    return actual == expected


def _check(
    outcome: CheckOutcome, requirement: EligibilityRequirement, reason: str
) -> RequirementCheck:
    return RequirementCheck(
        requirement_id=requirement.requirement_id, outcome=outcome, reason=reason
    )


def _lookup_fact(candidate: CandidateProfile, key: str) -> FactValue | None:
    wanted = canonical_text(key)
    for name, value in candidate.eligibility_facts.items():
        if canonical_text(name) == wanted:
            return value
    return None


def _check_experience(
    candidate: CandidateProfile, requirement: EligibilityRequirement
) -> RequirementCheck:
    minimum = ExperienceLevel(requirement.expected)
    if candidate.experience_level.rank >= minimum.rank:
        return _check(
            CheckOutcome.PASS, requirement, f"experience level meets minimum {minimum.value}"
        )
    return _check(
        CheckOutcome.FAIL,
        requirement,
        f"experience level {candidate.experience_level.value} is below required {minimum.value}",
    )


def _check_fact(
    candidate: CandidateProfile, requirement: EligibilityRequirement
) -> RequirementCheck:
    key = requirement.fact_key
    actual = _lookup_fact(candidate, key)
    if actual is None:
        return _check(CheckOutcome.UNKNOWN, requirement, f"candidate fact {key!r} is not provided")

    if requirement.kind == EligibilityKind.FACT_MIN_NUMBER:
        if isinstance(actual, bool) or not isinstance(actual, int | float):
            return _check(
                CheckOutcome.UNKNOWN, requirement, f"candidate fact {key!r} is not numeric"
            )
        outcome = CheckOutcome.PASS if actual >= requirement.expected else CheckOutcome.FAIL
        return _check(outcome, requirement, f"{key} is {actual}; minimum is {requirement.expected}")

    expected_values = (
        requirement.expected
        if requirement.kind == EligibilityKind.FACT_ONE_OF
        else [requirement.expected]
    )
    comparisons = [_equals(actual, expected) for expected in expected_values]
    if any(result is True for result in comparisons):
        return _check(CheckOutcome.PASS, requirement, f"{key} satisfies the requirement")
    if any(result is None for result in comparisons):
        return _check(
            CheckOutcome.UNKNOWN, requirement, f"candidate fact {key!r} is not comparable"
        )
    return _check(CheckOutcome.FAIL, requirement, f"{key} does not satisfy the requirement")


def evaluate_eligibility(candidate: CandidateProfile, job: JobOpportunity) -> EligibilityResult:
    requirements = job.eligibility_requirements
    if requirements is None:
        return EligibilityResult(
            job_id=job.job_id,
            candidate_id=candidate.candidate_id,
            requirements_known=False,
            status=derive_eligibility_status([], False),
        )
    checks = [
        _check_experience(candidate, r)
        if r.kind == EligibilityKind.MIN_EXPERIENCE_LEVEL
        else _check_fact(candidate, r)
        for r in requirements
    ]
    return EligibilityResult(
        job_id=job.job_id,
        candidate_id=candidate.candidate_id,
        requirements_known=True,
        status=derive_eligibility_status(checks, True),
        checks=checks,
    )
