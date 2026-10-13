"""Versioned Pydantic domain models and inter-agent envelopes.

Conventions:
* Models reject unknown fields so schema drift fails loudly.
* ``None`` on a job field means "not provided by the source" (unknown),
  which is different from an empty list ("source confirms there are none").
* Unknown is never treated as a confirmed pass.
"""

import re
from datetime import date
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.0"
SUPPORTED_MAJOR_VERSION = 1
SCORE_MEANING = (
    "Alignment with the defined requirements under the stated rubric; "
    "not a probability of employment."
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FactValue = str | int | float | bool


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------- candidate


class ExperienceLevel(StrEnum):
    STUDENT = "student"
    ENTRY_LEVEL = "entry_level"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"

    @property
    def rank(self) -> int:
        return list(ExperienceLevel).index(self)


class CandidateProfile(StrictModel):
    candidate_id: NonEmptyStr
    target_role: NonEmptyStr
    preferred_location: NonEmptyStr
    experience_level: ExperienceLevel
    claimed_skills: list[NonEmptyStr] = Field(default_factory=list)
    learning_hours_per_week: float = Field(ge=0, le=168)
    # Facts used for mandatory eligibility checks. A missing key or a None
    # value means the fact is unknown.
    eligibility_facts: dict[NonEmptyStr, FactValue | None] = Field(default_factory=dict)


class EvidenceType(StrEnum):
    SELF_REPORTED = "self_reported"
    ASSESSMENT = "assessment"
    PRACTICAL_TASK = "practical_task"
    PROJECT = "project"


class VerificationStatus(StrEnum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"


class SkillEvidence(StrictModel):
    candidate_id: NonEmptyStr
    skill_id: NonEmptyStr
    evidence_type: EvidenceType
    evidence_ref: NonEmptyStr | None = None
    # Normalized result in [0, 1]; only meaningful for assessments.
    assessment_score: float | None = Field(default=None, ge=0, le=1)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    recorded_at: AwareDatetime

    @model_validator(mode="after")
    def _check_type_rules(self) -> "SkillEvidence":
        kind = self.evidence_type
        if kind in {EvidenceType.ASSESSMENT, EvidenceType.PRACTICAL_TASK}:
            if self.assessment_score is None or self.evidence_ref is None:
                raise ValueError("scored evidence requires assessment_score and evidence_ref")
        elif self.assessment_score is not None:
            raise ValueError("assessment_score is only valid for assessment evidence")
        if kind == EvidenceType.PROJECT and self.evidence_ref is None:
            raise ValueError("project evidence requires evidence_ref")
        if (
            kind == EvidenceType.SELF_REPORTED
            and self.verification_status != VerificationStatus.UNVERIFIED
        ):
            raise ValueError("self-reported evidence cannot be marked verified or rejected")
        return self


# --------------------------------------------------------------------- job


class SkillRequirement(StrictModel):
    skill: NonEmptyStr
    description: NonEmptyStr | None = None


class EligibilityKind(StrEnum):
    MIN_EXPERIENCE_LEVEL = "min_experience_level"
    FACT_EQUALS = "fact_equals"
    FACT_ONE_OF = "fact_one_of"
    FACT_MIN_NUMBER = "fact_min_number"


class EligibilityRequirement(StrictModel):
    requirement_id: NonEmptyStr
    description: NonEmptyStr
    kind: EligibilityKind
    fact_key: NonEmptyStr | None = None
    expected: FactValue | list[FactValue]

    @model_validator(mode="after")
    def _check_kind_rules(self) -> "EligibilityRequirement":
        kind, expected = self.kind, self.expected
        if kind == EligibilityKind.MIN_EXPERIENCE_LEVEL:
            if self.fact_key is not None:
                raise ValueError("min_experience_level does not use fact_key")
            if not isinstance(expected, str) or expected not in {e.value for e in ExperienceLevel}:
                raise ValueError("expected must be a valid ExperienceLevel value")
            return self
        if self.fact_key is None:
            raise ValueError(f"{kind.value} requires fact_key")
        if kind == EligibilityKind.FACT_ONE_OF:
            if not isinstance(expected, list) or not expected:
                raise ValueError("fact_one_of requires a non-empty list")
        elif isinstance(expected, list):
            raise ValueError(f"{kind.value} requires a scalar expected value")
        if kind == EligibilityKind.FACT_MIN_NUMBER and (
            isinstance(expected, bool) or not isinstance(expected, int | float)
        ):
            raise ValueError("fact_min_number requires a numeric expected value")
        return self


class JobOpportunity(StrictModel):
    job_id: NonEmptyStr
    title: NonEmptyStr
    employer: NonEmptyStr
    location: NonEmptyStr | None = None
    required_skills: list[SkillRequirement] | None = None
    preferred_skills: list[SkillRequirement] | None = None
    eligibility_requirements: list[EligibilityRequirement] | None = None
    source_url: HttpUrl
    source_id: NonEmptyStr
    retrieved_at: AwareDatetime
    # Only set when the source states a deadline.
    application_deadline: date | None = None


# ------------------------------------------------------------- eligibility


class CheckOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class EligibilityStatus(StrEnum):
    ELIGIBLE_ON_AVAILABLE_INFORMATION = "eligible_on_available_information"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"


class RequirementCheck(StrictModel):
    requirement_id: NonEmptyStr
    outcome: CheckOutcome
    reason: NonEmptyStr


def derive_eligibility_status(
    checks: list[RequirementCheck], requirements_known: bool
) -> EligibilityStatus:
    """A confirmed failure wins; any unknown prevents an eligible verdict."""
    if any(c.outcome == CheckOutcome.FAIL for c in checks):
        return EligibilityStatus.INELIGIBLE
    if not requirements_known or any(c.outcome == CheckOutcome.UNKNOWN for c in checks):
        return EligibilityStatus.UNKNOWN
    return EligibilityStatus.ELIGIBLE_ON_AVAILABLE_INFORMATION


class EligibilityResult(StrictModel):
    job_id: NonEmptyStr
    candidate_id: NonEmptyStr
    requirements_known: bool
    status: EligibilityStatus
    checks: list[RequirementCheck] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_matches_checks(self) -> "EligibilityResult":
        if self.status != derive_eligibility_status(self.checks, self.requirements_known):
            raise ValueError("status is inconsistent with checks")
        return self


# ------------------------------------------------------------------- match


class RequirementLevel(StrEnum):
    REQUIRED = "required"
    PREFERRED = "preferred"


class MatchRubric(StrictModel):
    version: NonEmptyStr = "rubric-1"
    required_weight: float = Field(default=3.0, gt=0)
    preferred_weight: float = Field(default=1.0, gt=0)
    verified_project_credit: float = Field(default=0.8, ge=0, le=1)
    unverified_credit: float = Field(default=0.3, ge=0, le=1)
    self_reported_credit: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode="after")
    def _ordering(self) -> "MatchRubric":
        if self.required_weight <= self.preferred_weight:
            raise ValueError("required_weight must exceed preferred_weight")
        if not (
            self.self_reported_credit
            <= self.unverified_credit
            <= self.verified_project_credit
        ):
            raise ValueError(
                "credits must satisfy self_reported <= unverified <= verified_project"
            )
        return self

    def weight_for(self, level: RequirementLevel) -> float:
        return self.required_weight if level == RequirementLevel.REQUIRED else self.preferred_weight


class EvidenceBasis(StrEnum):
    VERIFIED_ASSESSMENT = "verified_assessment"
    VERIFIED_PRACTICAL_TASK = "verified_practical_task"
    VERIFIED_PROJECT = "verified_project"
    UNVERIFIED_ASSESSMENT = "unverified_assessment"
    UNVERIFIED_PRACTICAL_TASK = "unverified_practical_task"
    UNVERIFIED_PROJECT = "unverified_project"
    SELF_REPORTED = "self_reported"
    NO_EVIDENCE = "no_evidence"


class ScoreStatus(StrEnum):
    SCORED = "scored"
    INSUFFICIENT_JOB_DATA = "insufficient_job_data"


class RequirementContribution(StrictModel):
    skill_id: NonEmptyStr
    skill_name: NonEmptyStr
    recognized_skill: bool
    level: RequirementLevel
    weight: float = Field(gt=0)
    credit: float = Field(ge=0, le=1)
    points: float = Field(ge=0)
    max_points: float = Field(ge=0)
    evidence_basis: EvidenceBasis
    evidence_refs: list[NonEmptyStr] = Field(default_factory=list)
    explanation: NonEmptyStr


class MatchResult(StrictModel):
    job_id: NonEmptyStr
    candidate_id: NonEmptyStr
    rubric_version: NonEmptyStr
    score_status: ScoreStatus
    skill_alignment_score: float | None = Field(default=None, ge=0, le=100)
    breakdown: list[RequirementContribution] = Field(default_factory=list)
    # Reported separately; never merged into or overridden by the score.
    eligibility: EligibilityResult
    notes: list[NonEmptyStr] = Field(default_factory=list)
    score_meaning: Literal[
        "Alignment with the defined requirements under the stated rubric; "
        "not a probability of employment."
    ] = SCORE_MEANING

    @model_validator(mode="after")
    def _score_presence(self) -> "MatchResult":
        scored = self.score_status == ScoreStatus.SCORED
        if scored != (self.skill_alignment_score is not None):
            raise ValueError("skill_alignment_score must be set if and only if status is scored")
        return self


# ------------------------------------------------------------------ action


class ActionType(StrEnum):
    APPLY_NOW = "apply_now"
    COMPLETE_ASSESSMENT = "complete_assessment"
    BUILD_PROJECT = "build_project"
    CLARIFY_ELIGIBILITY = "clarify_eligibility"
    ADDRESS_ELIGIBILITY_GAP = "address_eligibility_gap"


class CareerAction(StrictModel):
    action_id: NonEmptyStr
    job_id: NonEmptyStr
    action_type: ActionType
    skill_id: NonEmptyStr | None = None
    rationale: NonEmptyStr
    priority: int = Field(ge=1)
    estimated_hours: float | None = Field(default=None, ge=0)
    # Only a resource supplied by a real source; never invented.
    resource_url: HttpUrl | None = None

    @model_validator(mode="after")
    def _skill_actions_need_skill(self) -> "CareerAction":
        needs_skill = {ActionType.COMPLETE_ASSESSMENT, ActionType.BUILD_PROJECT}
        if self.action_type in needs_skill and self.skill_id is None:
            raise ValueError(f"{self.action_type.value} requires skill_id")
        return self


class OpportunityDecision(StrEnum):
    APPLY_NOW = "apply_now"
    APPLY_WHILE_IMPROVING = "apply_while_improving"
    STRENGTHEN_FIRST = "strengthen_first"
    DEPRIORITIZE = "deprioritize"
    NEEDS_MORE_INFORMATION = "needs_more_information"


class ConstraintFit(StrEnum):
    FITS = "fits"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class DecisionPolicy(StrictModel):
    version: NonEmptyStr = "career-action-1"
    apply_now_score: float = Field(default=70, ge=0, le=100)
    apply_while_improving_score: float = Field(default=40, ge=0, le=100)
    max_improvements: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def _thresholds(self) -> "DecisionPolicy":
        if self.apply_while_improving_score > self.apply_now_score:
            raise ValueError("apply_while_improving_score must not exceed apply_now_score")
        return self


class ImprovementResource(StrictModel):
    title: NonEmptyStr
    url: HttpUrl
    source_id: NonEmptyStr


class ImprovementMetadata(StrictModel):
    """Optional measured task metadata; no effort or resource is inferred."""

    skill_id: NonEmptyStr
    estimated_hours: float | None = Field(default=None, ge=0)
    evidence_to_produce: NonEmptyStr | None = None
    completion_criteria: NonEmptyStr | None = None
    resources: list[ImprovementResource] = Field(default_factory=list)


class RequirementReference(StrictModel):
    job_id: NonEmptyStr
    skill_id: NonEmptyStr
    level: RequirementLevel
    source_id: NonEmptyStr
    source_url: HttpUrl


class PrioritizedImprovement(StrictModel):
    priority: int = Field(ge=1)
    skill_id: NonEmptyStr
    reason: NonEmptyStr
    requirements_addressed: list[RequirementReference] = Field(min_length=1)
    estimated_hours: float | None = Field(default=None, ge=0)
    estimated_weeks_at_available_hours: float | None = Field(default=None, ge=0)
    evidence_to_produce: NonEmptyStr
    completion_criteria: NonEmptyStr
    resources: list[ImprovementResource] = Field(default_factory=list)


class OpportunityRecommendation(StrictModel):
    rank: int = Field(ge=1)
    job_id: NonEmptyStr
    title: NonEmptyStr
    employer: NonEmptyStr
    source_id: NonEmptyStr
    source_url: HttpUrl
    eligibility: EligibilityResult
    match: MatchResult
    decision: OpportunityDecision
    constraint_fit: ConstraintFit
    constraint_notes: list[NonEmptyStr] = Field(default_factory=list)
    supported_competencies: list[NonEmptyStr] = Field(default_factory=list)
    unsupported_claims: list[NonEmptyStr] = Field(default_factory=list)
    missing_evidence: list[NonEmptyStr] = Field(default_factory=list)
    important_skill_gaps: list[NonEmptyStr] = Field(default_factory=list)
    explanation: NonEmptyStr


# --------------------------------------------------------------- envelopes


class AgentName(StrEnum):
    ORCHESTRATOR = "orchestrator"
    OPPORTUNITY_INTELLIGENCE = "opportunity_intelligence"
    CANDIDATE_EVIDENCE = "candidate_evidence"
    CAREER_ACTION = "career_action"


class _Envelope(StrictModel):
    schema_version: str = SCHEMA_VERSION
    request_id: NonEmptyStr
    sender: AgentName
    recipient: AgentName
    message_type: NonEmptyStr
    created_at: AwareDatetime
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _supported_version(cls, value: str) -> str:
        if not re.fullmatch(r"\d+\.\d+", value):
            raise ValueError("schema_version must look like 'MAJOR.MINOR'")
        if int(value.split(".")[0]) != SUPPORTED_MAJOR_VERSION:
            raise ValueError(f"unsupported schema major version in {value!r}")
        return value


class AgentRequest(_Envelope):
    pass


class ResponseStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class AgentError(StrictModel):
    code: NonEmptyStr
    message: NonEmptyStr


class AgentResponse(_Envelope):
    status: ResponseStatus
    error: AgentError | None = None

    @model_validator(mode="after")
    def _error_iff_failed(self) -> "AgentResponse":
        failed = self.status == ResponseStatus.ERROR
        if failed != (self.error is not None):
            raise ValueError("error must be provided if and only if status is error")
        return self


# ------------------------------------------------- opportunity search payloads
# Payloads of the opportunity-search message pair. They travel inside the
# AgentRequest/AgentResponse ``payload`` and are validated on both ends.

FIXTURE_SOURCE_PREFIX = "fixture"
OPPORTUNITY_SEARCH_MESSAGE = "opportunity.search"
OPPORTUNITY_RESULT_MESSAGE = "opportunity.search.result"
CAREER_ACTION_MESSAGE = "career.action.plan"
CAREER_ACTION_RESULT_MESSAGE = "career.action.plan.result"

CountryCode = Annotated[
    str, StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"^[A-Za-z]{2}$")
]


class SourceKind(StrEnum):
    LIVE = "live"
    FIXTURE = "fixture"


class StudyPreferences(StrictModel):
    """Study techniques the user prefers, passed on to the Career Action Agent."""

    techniques: list[NonEmptyStr] = Field(default_factory=list, max_length=20)
    notes: NonEmptyStr | None = None


class JobFilters(StrictModel):
    keywords: list[NonEmptyStr] = Field(default_factory=list, max_length=20)
    # When true, only listings known to be non-remote are excluded; unknown stays.
    remote_only: bool = False
    max_results: int = Field(default=20, ge=1, le=100)


class OpportunitySearchRequest(StrictModel):
    target_role: NonEmptyStr
    country: CountryCode
    state: NonEmptyStr | None = None
    city: NonEmptyStr | None = None
    experience_level: ExperienceLevel
    preferred_companies: list[NonEmptyStr] = Field(default_factory=list, max_length=20)
    filters: JobFilters = Field(default_factory=JobFilters)
    study_preferences: StudyPreferences | None = None


class ExtractionMethod(StrEnum):
    STRUCTURED_SOURCE = "structured_source"
    DETERMINISTIC_TEXT = "deterministic_text"
    LLM_VALIDATED = "llm_validated"
    NOT_EXTRACTED = "not_extracted"


class SourceReference(StrictModel):
    source_id: NonEmptyStr
    source_url: HttpUrl
    source_job_id: NonEmptyStr | None = None


class WorkArrangement(StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"


class ExperienceRequirement(StrictModel):
    """Experience a listing states; minimum_years is None when no number is stated."""

    minimum_years: int | None = Field(default=None, ge=0, le=50)
    statement: NonEmptyStr


def missing_job_fields(
    job: JobOpportunity,
    description: str | None,
    remote: bool | None,
    experience_requirement: "ExperienceRequirement | None" = None,
) -> list[str]:
    """Names of fields the source did not provide (unknown, not "none")."""
    candidates = {
        "location": job.location,
        "required_skills": job.required_skills,
        "preferred_skills": job.preferred_skills,
        "eligibility_requirements": job.eligibility_requirements,
        "application_deadline": job.application_deadline,
        "description": description,
        "remote": remote,
        "experience_requirement": experience_requirement,
    }
    return [name for name, value in candidates.items() if value is None]


class OpportunityRecord(StrictModel):
    job: JobOpportunity
    source_kind: SourceKind
    original_title: NonEmptyStr
    normalized_title: NonEmptyStr
    # Original source text, preserved verbatim.
    description: NonEmptyStr | None = None
    remote: bool | None = None
    work_arrangement: WorkArrangement | None = None
    experience_requirement: ExperienceRequirement | None = None
    preferred_company: bool = False
    skills_extraction: ExtractionMethod
    eligibility_extraction: ExtractionMethod
    missing_fields: list[NonEmptyStr] = Field(default_factory=list)
    alternate_sources: list[SourceReference] = Field(default_factory=list)
    notes: list[NonEmptyStr] = Field(default_factory=list)
    # Fields whose value came from validated LLM extraction rather than the source or rules.
    llm_derived_fields: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "OpportunityRecord":
        is_fixture_id = self.job.source_id.startswith(FIXTURE_SOURCE_PREFIX)
        if (self.source_kind == SourceKind.FIXTURE) != is_fixture_id:
            raise ValueError("fixture records must use a fixture source_id and vice versa")
        expected = missing_job_fields(
            self.job, self.description, self.remote, self.experience_requirement)
        if sorted(self.missing_fields) != sorted(expected):
            raise ValueError(f"missing_fields must be exactly {expected}")
        return self


class SourceStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"


class SourceReport(StrictModel):
    source_id: NonEmptyStr
    source_kind: SourceKind
    status: SourceStatus
    attempts: int = Field(ge=0)
    retrieved_count: int = Field(default=0, ge=0)
    accepted_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    error_code: NonEmptyStr | None = None
    error_message: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _error_fields_iff_error(self) -> "SourceReport":
        failed = self.status == SourceStatus.ERROR
        if failed != (self.error_code is not None and self.error_message is not None):
            raise ValueError("error_code and error_message are required iff status is error")
        if not failed and (self.error_code or self.error_message):
            raise ValueError("error details are only valid when status is error")
        return self


class RejectedRecord(StrictModel):
    source_id: NonEmptyStr
    source_job_ref: NonEmptyStr | None = None
    reasons: list[NonEmptyStr] = Field(min_length=1)


class SearchStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


def derive_search_status(reports: list[SourceReport]) -> SearchStatus:
    failures = sum(r.status == SourceStatus.ERROR for r in reports)
    if reports and failures == len(reports):
        return SearchStatus.FAILED
    return SearchStatus.PARTIAL if failures else SearchStatus.COMPLETE


class OpportunitySearchResult(StrictModel):
    search_status: SearchStatus
    records: list[OpportunityRecord] = Field(default_factory=list)
    source_reports: list[SourceReport] = Field(min_length=1)
    rejected_records: list[RejectedRecord] = Field(default_factory=list)
    duplicates_merged: int = Field(default=0, ge=0)
    contains_fixture_data: bool = False
    # Echoed for the Career Action Agent; None means the user gave none.
    study_preferences: StudyPreferences | None = None
    notes: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "OpportunitySearchResult":
        if self.search_status != derive_search_status(self.source_reports):
            raise ValueError("search_status is inconsistent with source_reports")
        if self.search_status == SearchStatus.FAILED and self.records:
            raise ValueError("a failed search cannot contain records")
        has_fixture = any(r.source_kind == SourceKind.FIXTURE for r in self.records)
        if self.contains_fixture_data != has_fixture:
            raise ValueError("contains_fixture_data must reflect the records")
        return self


# ------------------------------------------------- candidate evidence payloads

EVIDENCE_PROFILE_MESSAGE = "evidence.profile"
EVIDENCE_RESULT_MESSAGE = "evidence.profile.result"


class ProficiencyLevel(StrEnum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class RecordKind(StrEnum):
    ASSESSMENT = "assessment"
    PRACTICAL_TASK = "practical_task"
    PROJECT = "project"


class DeclaredSkill(StrictModel):
    """A skill the candidate claims, optionally with proficiency and the record ids they cite."""

    skill: NonEmptyStr
    proficiency: ProficiencyLevel | None = None
    cited_record_ids: list[NonEmptyStr] = Field(default_factory=list, max_length=20)


class _EvidenceRecord(StrictModel):
    record_id: NonEmptyStr
    candidate_id: NonEmptyStr
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    # Who verified it (a person or system other than the candidate).
    verified_by: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _verified_needs_verifier(self) -> "_EvidenceRecord":
        if self.verification_status == VerificationStatus.VERIFIED and self.verified_by is None:
            raise ValueError("verified records must name verified_by")
        return self


class AssessmentRecord(_EvidenceRecord):
    """A stored assessment result. Scores must come from real results or test fixtures."""

    skills_evaluated: list[NonEmptyStr] = Field(min_length=1, max_length=20)
    score: float = Field(ge=0, le=1)
    taken_at: AwareDatetime
    provider: NonEmptyStr


class PracticalTaskRecord(_EvidenceRecord):
    skills_evaluated: list[NonEmptyStr] = Field(min_length=1, max_length=20)
    # None means the task was completed but no outcome was recorded.
    outcome_score: float | None = Field(default=None, ge=0, le=1)
    completed_at: AwareDatetime
    evaluator: NonEmptyStr | None = None


class ProjectRecord(_EvidenceRecord):
    reference: NonEmptyStr
    skills_demonstrated: list[NonEmptyStr] = Field(min_length=1, max_length=20)
    description: NonEmptyStr | None = None
    submitted_at: AwareDatetime
    # None means authorship was never checked.
    authorship_verified: bool | None = None
    # True when the skills list was produced by a model reading the description (unverified).
    model_interpreted: bool = False


class EvidenceProfileRequest(StrictModel):
    profile: CandidateProfile
    declared_skills: list[DeclaredSkill] = Field(default_factory=list, max_length=100)
    assessments: list[AssessmentRecord] = Field(default_factory=list, max_length=200)
    practical_tasks: list[PracticalTaskRecord] = Field(default_factory=list, max_length=200)
    projects: list[ProjectRecord] = Field(default_factory=list, max_length=200)
    # Extra competencies to report on (for example a job's requirements) even if unclaimed.
    requested_skills: list[NonEmptyStr] = Field(default_factory=list, max_length=100)
    # Evaluation time; recency rules use it, so results are reproducible.
    as_of: AwareDatetime


class EvidenceClass(StrEnum):
    SELF_REPORTED = "self_reported"
    ASSESSMENT_BACKED = "assessment_backed"
    PRACTICAL_TASK_BACKED = "practical_task_backed"
    PROJECT_BACKED = "project_backed"
    INDEPENDENTLY_VERIFIED = "independently_verified"
    NO_EVIDENCE = "no_evidence"


class ConfidenceLabel(StrEnum):
    CORROBORATED = "corroborated"
    SINGLE_SOURCE = "single_source"
    SELF_REPORT_ONLY = "self_report_only"
    NONE = "none"


class EvidenceRole(StrEnum):
    SUPPORTS = "supports"
    INSUFFICIENT = "insufficient"
    IGNORED = "ignored"


class EvidenceReference(StrictModel):
    record_id: NonEmptyStr
    record_kind: RecordKind
    role: EvidenceRole
    note: NonEmptyStr


class SkillEvidenceProfile(StrictModel):
    skill_id: NonEmptyStr
    recognized: bool
    declared: bool
    declared_proficiency: ProficiencyLevel | None = None
    # Every class the evidence supports; no single ranking of reliability is assumed.
    evidence_classes: list[EvidenceClass] = Field(min_length=1)
    confidence: ConfidenceLabel
    references: list[EvidenceReference] = Field(default_factory=list)
    limitations: list[NonEmptyStr] = Field(default_factory=list)
    missing_evidence: list[NonEmptyStr] = Field(default_factory=list)
    needs_assessment: bool
    recommended_assessment: NonEmptyStr | None = None
    explanation: NonEmptyStr

    @model_validator(mode="after")
    def _check_consistency(self) -> "SkillEvidenceProfile":
        none = EvidenceClass.NO_EVIDENCE
        if none in self.evidence_classes and len(self.evidence_classes) > 1:
            raise ValueError("no_evidence cannot be combined with other classes")
        if self.needs_assessment != (self.recommended_assessment is not None):
            raise ValueError("recommended_assessment must be set exactly when needs_assessment is true")
        return self


class EvidenceProfileResult(StrictModel):
    candidate_id: NonEmptyStr
    as_of: AwareDatetime
    skills: list[SkillEvidenceProfile]
    # Flat evidence consumed by the deterministic matching rubric. Contains only
    # records classified as supporting; raw scores/references remain traceable.
    matching_evidence: list[SkillEvidence] = Field(default_factory=list)
    # Problems found in the supplied records (missing, mismatched, duplicate, invalid).
    issues: list[NonEmptyStr] = Field(default_factory=list)
    notes: list[NonEmptyStr] = Field(default_factory=list)


class ActionPlanRequest(StrictModel):
    candidate: CandidateProfile
    opportunities: list[OpportunityRecord] = Field(min_length=1, max_length=100)
    evidence_profile: EvidenceProfileResult
    matching_config: MatchRubric = Field(default_factory=MatchRubric)
    decision_policy: DecisionPolicy = Field(default_factory=DecisionPolicy)
    improvement_metadata: list[ImprovementMetadata] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def _same_candidate(self) -> "ActionPlanRequest":
        if self.evidence_profile.candidate_id != self.candidate.candidate_id:
            raise ValueError("evidence profile belongs to a different candidate")
        foreign = {e.candidate_id for e in self.evidence_profile.matching_evidence} - {self.candidate.candidate_id}
        if foreign:
            raise ValueError("matching evidence belongs to a different candidate")
        if len({o.job.job_id for o in self.opportunities}) != len(self.opportunities):
            raise ValueError("opportunity job_id values must be unique")
        evidence_by_skill = {}
        for item in self.evidence_profile.matching_evidence:
            evidence_by_skill.setdefault(item.skill_id, set()).add(item.evidence_type)
        required_types = {
            EvidenceClass.ASSESSMENT_BACKED: EvidenceType.ASSESSMENT,
            EvidenceClass.PRACTICAL_TASK_BACKED: EvidenceType.PRACTICAL_TASK,
            EvidenceClass.PROJECT_BACKED: EvidenceType.PROJECT,
        }
        for skill in self.evidence_profile.skills:
            available = evidence_by_skill.get(skill.skill_id, set())
            for evidence_class, evidence_type in required_types.items():
                if evidence_class in skill.evidence_classes and evidence_type not in available:
                    raise ValueError(
                        f"evidence profile for {skill.skill_id!r} is missing its {evidence_type.value} record"
                    )
            if EvidenceClass.INDEPENDENTLY_VERIFIED in skill.evidence_classes and not any(
                item.skill_id == skill.skill_id and item.verification_status == VerificationStatus.VERIFIED
                for item in self.evidence_profile.matching_evidence
            ):
                raise ValueError(f"evidence profile for {skill.skill_id!r} is missing its verified record")
        return self


class ActionPlanResult(StrictModel):
    candidate_id: NonEmptyStr
    rubric: MatchRubric
    decision_policy: DecisionPolicy
    ranked_opportunities: list[OpportunityRecommendation]
    prioritized_improvements: list[PrioritizedImprovement] = Field(default_factory=list)
    notes: list[NonEmptyStr] = Field(default_factory=list)
