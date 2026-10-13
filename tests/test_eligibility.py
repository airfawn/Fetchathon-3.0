import pytest

from skillbridge.eligibility import evaluate_eligibility
from skillbridge.schemas import EligibilityKind, EligibilityRequirement, EligibilityStatus
from tests.conftest import experience_requirement, make_candidate, make_job


def fact_requirement(kind, expected, key="work_authorization", rid="fact"):
    return EligibilityRequirement(
        requirement_id=rid, description="fact requirement", kind=kind,
        fact_key=key, expected=expected,
    )


def evaluate(candidate, *requirements):
    job = make_job(eligibility_requirements=list(requirements))
    return evaluate_eligibility(candidate, job)


def test_eligible_when_all_requirements_pass():
    result = evaluate(
        make_candidate(experience_level="junior", eligibility_facts={"work_authorization": "IN"}),
        experience_requirement("entry_level"),
        fact_requirement(EligibilityKind.FACT_EQUALS, "in"),
    )
    assert result.status == EligibilityStatus.ELIGIBLE_ON_AVAILABLE_INFORMATION


def test_no_mandatory_requirements_is_eligible():
    assert evaluate(make_candidate()).status == EligibilityStatus.ELIGIBLE_ON_AVAILABLE_INFORMATION


def test_experience_shortfall_is_confirmed_failure():
    result = evaluate(make_candidate(experience_level="student"), experience_requirement("mid"))
    assert result.status == EligibilityStatus.INELIGIBLE
    assert result.checks[0].outcome == "fail"


def test_missing_fact_is_unknown_not_pass():
    result = evaluate(make_candidate(), fact_requirement(EligibilityKind.FACT_EQUALS, "IN"))
    assert result.status == EligibilityStatus.UNKNOWN
    assert result.checks[0].outcome == "unknown"


def test_explicit_none_fact_is_unknown():
    candidate = make_candidate(eligibility_facts={"work_authorization": None})
    result = evaluate(candidate, fact_requirement(EligibilityKind.FACT_EQUALS, "IN"))
    assert result.status == EligibilityStatus.UNKNOWN


def test_failure_beats_unknown():
    result = evaluate(
        make_candidate(experience_level="student"),
        fact_requirement(EligibilityKind.FACT_EQUALS, "IN"),
        experience_requirement("senior"),
    )
    assert result.status == EligibilityStatus.INELIGIBLE


def test_unknown_requirements_list_is_unknown():
    job = make_job(eligibility_requirements=None)
    result = evaluate_eligibility(make_candidate(), job)
    assert result.status == EligibilityStatus.UNKNOWN
    assert result.requirements_known is False


def test_one_of_and_mismatch():
    candidate = make_candidate(eligibility_facts={"work_authorization": "US"})
    ok = evaluate(candidate, fact_requirement(EligibilityKind.FACT_ONE_OF, ["IN", "US"]))
    bad = evaluate(candidate, fact_requirement(EligibilityKind.FACT_ONE_OF, ["IN", "UK"]))
    assert ok.status == EligibilityStatus.ELIGIBLE_ON_AVAILABLE_INFORMATION
    assert bad.status == EligibilityStatus.INELIGIBLE


@pytest.mark.parametrize(
    ("gpa", "expected_status"),
    [(3.5, EligibilityStatus.ELIGIBLE_ON_AVAILABLE_INFORMATION),
     (2.9, EligibilityStatus.INELIGIBLE),
     ("high", EligibilityStatus.UNKNOWN),
     (True, EligibilityStatus.UNKNOWN)],
)
def test_min_number(gpa, expected_status):
    candidate = make_candidate(eligibility_facts={"gpa": gpa})
    result = evaluate(candidate, fact_requirement(EligibilityKind.FACT_MIN_NUMBER, 3.0, key="gpa"))
    assert result.status == expected_status


def test_incomparable_types_are_unknown_not_failure():
    candidate = make_candidate(eligibility_facts={"work_authorization": 5})
    result = evaluate(candidate, fact_requirement(EligibilityKind.FACT_EQUALS, "IN"))
    assert result.status == EligibilityStatus.UNKNOWN


def test_fact_keys_are_case_insensitive():
    candidate = make_candidate(eligibility_facts={"Work Authorization": "IN"})
    result = evaluate(candidate, fact_requirement(EligibilityKind.FACT_EQUALS, "IN"))
    assert result.status == EligibilityStatus.ELIGIBLE_ON_AVAILABLE_INFORMATION
