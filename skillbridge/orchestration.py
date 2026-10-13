"""Validated request and result models for the deterministic agent workflow."""

from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from skillbridge.schemas import (
    ActionPlanResult,
    AssessmentRecord,
    CandidateProfile,
    CountryCode,
    DecisionPolicy,
    DeclaredSkill,
    ExperienceLevel,
    EvidenceProfileResult,
    GuidanceResult,
    ImprovementMetadata,
    MatchRubric,
    NonEmptyStr,
    OpportunitySearchRequest,
    OpportunitySearchResult,
    PracticalTaskRecord,
    ProjectRecord,
    StrictModel,
)


class CandidateWorkflowRequest(StrictModel):
    candidate: CandidateProfile
    job_search: OpportunitySearchRequest
    declared_skills: list[DeclaredSkill] = Field(default_factory=list, max_length=100)
    assessments: list[AssessmentRecord] = Field(default_factory=list, max_length=200)
    practical_tasks: list[PracticalTaskRecord] = Field(default_factory=list, max_length=200)
    projects: list[ProjectRecord] = Field(default_factory=list, max_length=200)
    matching_config: MatchRubric = Field(default_factory=MatchRubric)
    decision_policy: DecisionPolicy = Field(default_factory=DecisionPolicy)
    improvement_metadata: list[ImprovementMetadata] = Field(default_factory=list, max_length=100)
    include_asi_summary: bool = False
    as_of: AwareDatetime

    @model_validator(mode="after")
    def _candidate_matches_search(self) -> "CandidateWorkflowRequest":
        if self.job_search.target_role != self.candidate.target_role:
            raise ValueError("job search target_role must match candidate target_role")
        if self.job_search.experience_level != self.candidate.experience_level:
            raise ValueError("job search experience_level must match candidate profile")
        record_ids = [
            item.record_id
            for item in (*self.assessments, *self.practical_tasks, *self.projects)
        ]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("evidence record ids must be unique across all record types")
        foreign = {
            item.candidate_id
            for item in (*self.assessments, *self.practical_tasks, *self.projects)
        } - {self.candidate.candidate_id}
        if foreign:
            raise ValueError("evidence records belong to a different candidate")
        return self


class ProjectStatus(StrEnum):
    IDEA = "idea"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEPLOYED = "deployed"


class ProjectDetail(StrictModel):
    """A project as the candidate describes it in chat; self-reported, never verified."""

    name: NonEmptyStr = Field(max_length=120)
    status: ProjectStatus | None
    summary: NonEmptyStr | None = Field(max_length=500)
    skills_used: list[NonEmptyStr] = Field(max_length=20)
    # The candidate's own statement of its value or impact (users, results, purpose).
    worth: NonEmptyStr | None = Field(max_length=300)
    link: NonEmptyStr | None = Field(max_length=300)


class ChatIntakeDelta(StrictModel):
    """Only facts explicitly present in a user chat turn; null means not supplied."""

    intent: Literal["continue", "reset", "capabilities", "ask"]
    target_role: NonEmptyStr | None
    country: CountryCode | None
    state: NonEmptyStr | None
    city: NonEmptyStr | None
    experience_level: ExperienceLevel | None
    preferred_companies: list[NonEmptyStr] | None
    keywords: list[NonEmptyStr] | None
    remote_only: bool | None
    learning_hours_per_week: float | None = Field(ge=0, le=168)
    claimed_skills: list[NonEmptyStr] | None
    study_techniques: list[NonEmptyStr] | None
    study_notes: NonEmptyStr | None
    assessments: list[AssessmentRecord] | None
    practical_tasks: list[PracticalTaskRecord] | None
    projects: list[ProjectRecord] | None
    project_details: list[ProjectDetail] | None
    # Set only when intent is "ask": what the user wants to know, and about which role.
    question_topics: list[Literal["skills", "certifications", "projects"]] | None
    question_role: NonEmptyStr | None


class WorkflowStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class StageStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStage(StrictModel):
    status: StageStatus
    error_code: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def _failure_has_details(self) -> "WorkflowStage":
        if self.status == StageStatus.FAILED:
            if not self.error_code or not self.message:
                raise ValueError("failed stages require an error_code and message")
        elif self.status == StageStatus.SKIPPED:
            if not self.message or self.error_code is not None:
                raise ValueError("skipped stages require a reason and cannot have an error_code")
        elif self.error_code is not None or self.message is not None:
            raise ValueError("only failed or skipped stages may contain details")
        return self


class CandidateWorkflowResult(StrictModel):
    request_id: str
    workflow_status: WorkflowStatus
    opportunity_stage: WorkflowStage
    evidence_stage: WorkflowStage
    action_stage: WorkflowStage
    opportunities: OpportunitySearchResult | None = None
    evidence_profile: EvidenceProfileResult | None = None
    action_plan: ActionPlanResult | None = None
    guidance: GuidanceResult | None = None
    completed_at: AwareDatetime
    replayed: bool = False

    @model_validator(mode="after")
    def _outputs_match_stages(self) -> "CandidateWorkflowResult":
        if self.opportunities is None and self.opportunity_stage.status in {
            StageStatus.SUCCEEDED, StageStatus.PARTIAL,
        }:
            raise ValueError("successful opportunity stages require a validated result")
        if self.opportunities is not None and self.opportunity_stage.status == StageStatus.SKIPPED:
            raise ValueError("skipped opportunity stages cannot contain a result")
        if self.evidence_profile is None and self.evidence_stage.status in {
            StageStatus.SUCCEEDED, StageStatus.PARTIAL,
        }:
            raise ValueError("successful evidence stages require a validated result")
        if self.evidence_profile is not None and self.evidence_stage.status == StageStatus.SKIPPED:
            raise ValueError("skipped evidence stages cannot contain a result")
        if self.action_plan is None and self.action_stage.status == StageStatus.SUCCEEDED:
            raise ValueError("a successful action stage requires a validated plan")
        if self.action_plan is not None and self.action_stage.status != StageStatus.SUCCEEDED:
            raise ValueError("action plans are only valid when the action stage succeeded")
        if self.workflow_status == WorkflowStatus.SUCCEEDED and any(
            stage.status != StageStatus.SUCCEEDED
            for stage in (self.opportunity_stage, self.evidence_stage, self.action_stage)
        ):
            raise ValueError("a succeeded workflow requires all stages to succeed")
        return self
