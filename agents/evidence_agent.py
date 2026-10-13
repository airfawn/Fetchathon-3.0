"""SkillBridge Candidate Evidence Agent (self-contained Agentverse file).

Responsibility: build an evidence-backed profile of a candidate's skills, keeping
what the candidate CLAIMS separate from what stored records SUPPORT. It does not
search jobs, match jobs or decide whether to apply (Career Action Agent). No LLM
is used here, so nothing is "verified by a model"; a project whose skills were
listed by a model reading its description must arrive with model_interpreted=true
and is treated as unverified and insufficient.

The block between the GENERATED markers is produced by
``scripts/sync_agent_models.py`` from the canonical SkillBridge schemas; edit the
canonical sources, never the block. Everything else is hand-written. This file
must not import the shared ``skillbridge`` package.

Configuration (environment variables; none are secrets):
    SKILLBRIDGE_EVIDENCE_SEED   stable agent identity seed (optional)
    SKILLBRIDGE_EVIDENCE_PORT   local port (default 8002)

ASI:One / Agentverse chat is exposed through the Agent Chat Protocol. Chat accepts
the JSON form of EvidenceProfileRequest and invokes the same deterministic
profiling service as the uAgents message handler. ASI:One is not used to create
or interpret evidence: candidate assertions, scores and verification must come
from the submitted records.

Rules (deterministic; ``EvidenceRules`` holds the two tunable numbers). They are
documented defaults, not claims about real-world reliability:
* Skill names go through the shared normalizer. Unknown skills stay identifiable
  (exact normalized text only) and are flagged.
* A record counts for a skill only if the record itself lists that skill, belongs
  to this candidate, is not dated after ``as_of`` and was not rejected.
* Assessments / practical tasks support a skill when scored at or above
  ``pass_threshold`` and not older than ``max_age_days``. A low score or missing
  outcome is "insufficient", never proof the skill is absent.
* A project supports a skill only as ``project_backed``: submission alone is not
  proof of authorship or independent competence. A failed authorship check
  ignores it; model-interpreted skill lists are insufficient.
* ``independently_verified`` needs status VERIFIED, a named verifier who is not
  the candidate and, for projects, ``authorship_verified`` true.
* Duplicate records (same id, or same project reference / assessment details)
  are counted once and reported.
* Every class a skill earns is listed; no universal ranking between classes is
  assumed. Confidence labels follow one rule: >=2 distinct supporting record
  kinds -> corroborated; exactly one -> single_source; claim only ->
  self_report_only; nothing -> none.
* ``needs_assessment`` is true unless a passing assessment, a passing practical
  task or independent verification exists. The recommendation names the skill
  only; it never invents a specific test, provider or learning resource.
"""

import asyncio
import json
import logging
import os
from uuid import uuid4
from datetime import timezone

from pydantic import ValidationError
import uagents
from uagents import Agent, Context
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

# BEGIN GENERATED SKILLBRIDGE MODELS (scripts/sync_agent_models.py; do not edit by hand)
import re
import unicodedata
try:
    from enum import StrEnum
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
)
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from datetime import datetime
from uagents import Model


SCHEMA_VERSION = "1.0"
SUPPORTED_MAJOR_VERSION = 1
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FactValue = str | int | float | bool


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


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


UNKNOWN_PREFIX = "unknown:"
_CANONICAL_ID = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")


class AliasEntry(StrictModel):
    id: NonEmptyStr
    aliases: list[NonEmptyStr] = Field(default_factory=list)
    note: NonEmptyStr


class KeepSeparate(StrictModel):
    terms: list[NonEmptyStr] = Field(min_length=2)
    reason: NonEmptyStr


class AliasConfig(StrictModel):
    version: Literal[1]
    skills: list[AliasEntry]
    keep_separate: list[KeepSeparate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "AliasConfig":
        ids = [entry.id for entry in self.skills]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate skill id in alias configuration")
        return self


@dataclass(frozen=True)
class NormalizedSkill:
    original: str
    skill_id: str
    recognized: bool


def canonical_text(raw: str) -> str:
    """Form of a skill name insensitive to case, spacing, hyphens, underscores,
    '&' vs 'and', and spaces around '/'."""
    text = unicodedata.normalize("NFKC", raw).casefold().replace("&", " and ")
    text = re.sub(r"\s*/\s*", "/", text)
    return re.sub(r"[\s_\-]+", " ", text).strip()


class SkillNormalizer:
    def __init__(
        self,
        aliases: Mapping[str, Iterable[str]],
        keep_separate: Iterable[Iterable[str]] = (),
    ) -> None:
        self._lookup: dict[str, str] = {}
        for canonical_id, spellings in aliases.items():
            if not _CANONICAL_ID.fullmatch(canonical_id):
                raise ValueError(f"invalid canonical skill id: {canonical_id!r}")
            for spelling in [canonical_id, *spellings]:
                self._register(canonical_text(spelling), canonical_id)
        for group in keep_separate:
            terms = list(group)
            if len({self.normalize(t).skill_id for t in terms}) != len(terms):
                raise ValueError(f"terms must stay separate but were merged: {terms}")

    @classmethod
    def from_config(cls, config: AliasConfig) -> "SkillNormalizer":
        return cls(
            {entry.id: entry.aliases for entry in config.skills},
            [group.terms for group in config.keep_separate],
        )

    def _register(self, text: str, canonical_id: str) -> None:
        if not text:
            raise ValueError(f"blank alias for {canonical_id!r}")
        existing = self._lookup.setdefault(text, canonical_id)
        if existing != canonical_id:
            raise ValueError(
                f"alias {text!r} maps to both {existing!r} and {canonical_id!r}"
            )

    def normalize(self, raw: str) -> NormalizedSkill:
        text = canonical_text(raw)
        if not text:
            raise ValueError("skill name must not be blank")
        canonical_id = self._lookup.get(text)
        if canonical_id is not None:
            return NormalizedSkill(raw, canonical_id, True)
        return NormalizedSkill(raw, UNKNOWN_PREFIX + text.replace(" ", "-"), False)


SKILL_ALIAS_DATA = {'version': 1,
 'skills': [{'id': 'siem',
             'aliases': ['Security Information and Event Management'],
             'note': 'Acronym and its expansion name the same technology category.'},
            {'id': 'linux-administration',
             'aliases': ['Linux system administration'],
             'note': 'Administration vs system administration is a wording difference only. Plain '
                     "'Linux' (usage/familiarity) is deliberately not merged."},
            {'id': 'python',
             'aliases': ['Python programming', 'Python programming language'],
             'note': "Same language. 'Python scripting' is deliberately kept separate because it "
                     'describes a narrower scope of use.'},
            {'id': 'tcp-ip-networking',
             'aliases': ['TCP/IP', 'TCP IP', 'TCPIP'],
             'note': 'Spelling variants of the TCP/IP protocol suite. Deliberately NOT merged with '
                     "'computer networking', which is a broader field."},
            {'id': 'computer-networking',
             'aliases': ['Computer networks'],
             'note': 'Broad networking competency. Kept separate from TCP/IP networking so a '
                     'narrow protocol skill is not credited as the whole field, or the reverse.'},
            {'id': 'web-application-security',
             'aliases': ['Web security', 'Web app security'],
             'note': "Equivalent phrasing in job postings. Broader 'application security' and "
                     "'network security' are deliberately not merged."},
            {'id': 'incident-response',
             'aliases': ['Security incident response', 'Cyber incident response'],
             'note': "Security qualifier is implied in this context. ITSM 'incident management' is "
                     'a different discipline and is not merged.'},
            {'id': 'digital-forensics',
             'aliases': ['Digital forensic analysis'],
             'note': "Wording variants only. 'Malware analysis' and 'memory forensics' are "
                     'specialisations and are not merged.'},
            {'id': 'aws',
             'aliases': ['Amazon Web Services'],
             'note': 'Acronym and company name for the same platform. Individual services such as '
                     "'AWS IAM' are not merged into this general skill."},
            {'id': 'iam',
             'aliases': ['Identity and Access Management'],
             'note': "General IAM discipline. Bare 'IAM' can also mean the AWS service, so 'AWS "
                     "IAM' is deliberately kept separate."},
            {'id': 'ci-cd',
             'aliases': ['CI/CD',
                         'CI CD',
                         'CICD',
                         'Continuous Integration and Continuous Delivery',
                         'Continuous Integration/Continuous Delivery'],
             'note': "Delivery expansion only. 'Continuous deployment' and standalone 'continuous "
                     "integration' are narrower or different practices and are not merged."},
            {'id': 'javascript',
             'aliases': ['JS'],
             'note': 'Common abbreviation. Unrelated to Java, which stays separate.'},
            {'id': 'sql',
             'aliases': ['Structured Query Language'],
             'note': 'Acronym and expansion. NoSQL is a different family of databases and is not '
                     'merged.'},
            {'id': 'kubernetes',
             'aliases': ['K8s'],
             'note': 'Standard abbreviation. Docker is a different tool and stays separate.'},
            {'id': 'penetration-testing',
             'aliases': ['Pen testing', 'Pentesting'],
             'note': "Wording variants. 'Vulnerability assessment' and 'ethical hacking' are not "
                     'merged.'},
            {'id': 'rest-api',
             'aliases': ['RESTful API', 'REST APIs', 'RESTful APIs'],
             'note': "Wording and plural variants. Generic 'API' and 'GraphQL' are not merged."},
            {'id': 'object-oriented-programming',
             'aliases': ['OOP'],
             'note': 'Acronym and expansion.'}],
 'keep_separate': [{'terms': ['TCP/IP networking', 'computer networking'],
                    'reason': 'Protocol suite vs the broader networking field.'},
                   {'terms': ['IAM', 'AWS IAM'],
                    'reason': 'General discipline vs a specific cloud service.'},
                   {'terms': ['Linux', 'Linux administration', 'Windows administration'],
                    'reason': 'Using an OS, administering it, and administering a different OS are '
                              'different competencies.'},
                   {'terms': ['Java', 'JavaScript'],
                    'reason': 'Unrelated languages with similar names.'},
                   {'terms': ['SQL', 'NoSQL'],
                    'reason': 'Relational query language vs a different database family.'},
                   {'terms': ['Python', 'Python scripting'],
                    'reason': 'General programming vs a narrower use.'},
                   {'terms': ['CI/CD', 'continuous integration', 'continuous deployment'],
                    'reason': 'CI/CD is the combined pipeline practice; the other terms name '
                              'narrower practices.'},
                   {'terms': ['web application security',
                              'application security',
                              'network security'],
                    'reason': 'Different security domains.'},
                   {'terms': ['penetration testing', 'vulnerability assessment'],
                    'reason': 'Active exploitation vs identification of weaknesses.'},
                   {'terms': ['digital forensics', 'malware analysis'],
                    'reason': 'Malware analysis is a specialisation, not a synonym.'},
                   {'terms': ['incident response', 'incident management'],
                    'reason': 'Security response vs ITSM process.'},
                   {'terms': ['Kubernetes', 'Docker'],
                    'reason': 'Orchestrator vs container runtime and tooling.'},
                   {'terms': ['AWS', 'Azure', 'cloud computing'],
                    'reason': 'A specific vendor platform is not the same as other platforms or '
                              'the general field.'}]}


class AgentErrorMessage(Model):
    code: str
    message: str


class AgentRequestMessage(Model):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    sender: str
    recipient: str
    message_type: str
    created_at: datetime
    payload: dict[str, Any] = {}


class AgentResponseMessage(Model):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    sender: str
    recipient: str
    message_type: str
    created_at: datetime
    payload: dict[str, Any] = {}
    status: str
    error: Optional[AgentErrorMessage] = None


def _as_dict(message: Model) -> dict[str, Any]:
    dump = getattr(message, "model_dump", None) or message.dict
    return dump()


def request_to_message(request: AgentRequest) -> AgentRequestMessage:
    return AgentRequestMessage(**request.model_dump(mode="json"))


def message_to_request(message: AgentRequestMessage) -> AgentRequest:
    """Validate a received message with the canonical model."""
    return AgentRequest.model_validate(_as_dict(message))


def response_to_message(response: AgentResponse) -> AgentResponseMessage:
    return AgentResponseMessage(**response.model_dump(mode="json"))


def message_to_response(message: AgentResponseMessage) -> AgentResponse:
    """Validate a received message with the canonical model."""
    return AgentResponse.model_validate(_as_dict(message))
# END GENERATED SKILLBRIDGE MODELS


AGENT_NAME = "skillbridge_candidate_evidence"
logger = logging.getLogger(AGENT_NAME)


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    """One JSON object per line; never pass record contents or personal data."""
    logger.log(level, json.dumps({"event": event, **fields}, default=str, sort_keys=True))


# ------------------------------------------------------------------- rules


@dataclass(frozen=True)
class EvidenceRules:
    pass_threshold: float = 0.6
    max_age_days: int = 730


DEFAULT_RULES = EvidenceRules()
RESULT_NOTES = [
    "Classifications describe the supplied records only; they are not a hiring decision.",
    "Missing evidence is not proof of incompetence.",
]


class AssessmentRegistry:
    """Interface for registering stored assessment results. Scores must come from real
    stored results or controlled fixtures. In-memory: hosted agents should pass records in
    each request, because process state is not guaranteed to persist."""

    def __init__(self) -> None:
        self._records: dict[str, AssessmentRecord] = {}

    def register(self, record) -> AssessmentRecord:
        """Validate and store a result; raises ValueError if the id exists with different content."""
        record = AssessmentRecord.model_validate(record)
        existing = self._records.get(record.record_id)
        if existing is not None and existing != record:
            raise ValueError(f"assessment {record.record_id!r} is already registered with different content")
        self._records[record.record_id] = record
        return record

    def for_candidate(self, candidate_id: str) -> list[AssessmentRecord]:
        return [r for r in self._records.values() if r.candidate_id == candidate_id]


# ------------------------------------------------------ record intake


@dataclass
class _Entry:
    kind: RecordKind
    record: object
    skills: set[str]
    when: datetime
    score: float | None


def _kinds_and_records(request, registered: Iterable[AssessmentRecord]):
    for r in (*registered, *request.assessments):
        yield RecordKind.ASSESSMENT, r, r.skills_evaluated, r.taken_at, r.score
    for r in request.practical_tasks:
        yield RecordKind.PRACTICAL_TASK, r, r.skills_evaluated, r.completed_at, r.outcome_score
    for r in request.projects:
        yield RecordKind.PROJECT, r, r.skills_demonstrated, r.submitted_at, None


def _content_key(kind: RecordKind, record, skills: set[str], when: datetime):
    if kind == RecordKind.PROJECT:
        return (kind, canonical_text(record.reference))
    owner = record.provider if kind == RecordKind.ASSESSMENT else (record.evaluator or "")
    return (kind, canonical_text(owner), when, frozenset(skills))


def _intake(request, registered, normalizer, issues: list[str]):
    """Usable entries by record id, all known ids, and skill recognition."""
    entries: dict[str, _Entry] = {}
    seen: dict[object, str] = {}
    all_ids: set[str] = set()
    recognized: dict[str, bool] = {}

    def norm(raw: str) -> str:
        found = normalizer.normalize(raw)
        recognized[found.skill_id] = found.recognized
        return found.skill_id

    for kind, record, raw_skills, when, score in _kinds_and_records(request, registered):
        all_ids.add(record.record_id)
        label = f"{kind.value} {record.record_id!r}"
        if record.candidate_id != request.profile.candidate_id:
            issues.append(f"{label} belongs to a different candidate and was ignored")
            continue
        if when > request.as_of:
            issues.append(f"{label} is dated after as_of and was ignored")
            continue
        skills = {norm(s) for s in raw_skills}
        key = _content_key(kind, record, skills, when)
        if record.record_id in entries:
            same = entries[record.record_id].record == record
            issues.append(f"{label} appears more than once" + (" (identical; counted once)" if same
                                                              else " with different content (later copy ignored)"))
            continue
        if key in seen:
            issues.append(f"{label} duplicates {seen[key]!r} (same underlying evidence; counted once)")
            continue
        seen[key] = record.record_id
        entries[record.record_id] = _Entry(kind, record, skills, when, score)
    return entries, all_ids, recognized, norm


# ---------------------------------------------------- per-record judgment

_CLASS_FOR_KIND = {RecordKind.ASSESSMENT: EvidenceClass.ASSESSMENT_BACKED,
                   RecordKind.PRACTICAL_TASK: EvidenceClass.PRACTICAL_TASK_BACKED,
                   RecordKind.PROJECT: EvidenceClass.PROJECT_BACKED}


def _independently_verified(entry: _Entry, candidate_id: str) -> bool:
    record = entry.record
    ok = (record.verification_status == VerificationStatus.VERIFIED
          and record.verified_by is not None and record.verified_by != candidate_id)
    if entry.kind == RecordKind.PROJECT:
        ok = ok and record.authorship_verified is True
    return ok


def _judge(entry: _Entry, candidate_id: str, rules: EvidenceRules, as_of: datetime):
    """-> (role, classes earned, note, limitations)."""
    record = entry.record
    if record.verification_status == VerificationStatus.REJECTED:
        return EvidenceRole.IGNORED, [], "verification was rejected", ["a rejected record was ignored"]
    if entry.kind == RecordKind.PROJECT and record.authorship_verified is False:
        return EvidenceRole.IGNORED, [], "authorship check failed", ["a project failed its authorship check and was ignored"]
    age_days = (as_of - entry.when).days
    if age_days > rules.max_age_days:
        return (EvidenceRole.INSUFFICIENT, [], f"recorded {entry.when.date()}, older than {rules.max_age_days} days",
                ["evidence is older than the recency limit and may not reflect current ability"])
    if entry.kind == RecordKind.PROJECT:
        if record.model_interpreted:
            return (EvidenceRole.INSUFFICIENT, [], "skills were listed by model interpretation (unverified)",
                    ["some project skills were model-interpreted and are unverified"])
        limits = []
    else:
        if entry.score is None:
            return (EvidenceRole.INSUFFICIENT, [], "task completed but no outcome was recorded",
                    ["a practical task has no recorded outcome"])
        if entry.score < rules.pass_threshold:
            return (EvidenceRole.INSUFFICIENT, [],
                    f"score {entry.score:.2f} is below the pass threshold {rules.pass_threshold:.2f}",
                    ["a below-threshold result does not support the claim (it is not proof the skill is absent)"])
        limits = []
    classes = [_CLASS_FOR_KIND[entry.kind]]
    if _independently_verified(entry, candidate_id):
        classes.append(EvidenceClass.INDEPENDENTLY_VERIFIED)
        return EvidenceRole.SUPPORTS, classes, "supports the skill; independently verified", limits
    if record.verified_by == candidate_id:
        limits.append("the named verifier is the candidate, so this is not independent verification")
    if entry.kind == RecordKind.PROJECT:
        limits.append("a project submission alone does not prove authorship or independent competence")
    return EvidenceRole.SUPPORTS, classes, "supports the skill; not independently verified", limits


# ------------------------------------------------------ skill profiles


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _confidence(supporting_kinds: set, declared: bool) -> ConfidenceLabel:
    if len(supporting_kinds) >= 2:
        return ConfidenceLabel.CORROBORATED
    if supporting_kinds:
        return ConfidenceLabel.SINGLE_SOURCE
    return ConfidenceLabel.SELF_REPORT_ONLY if declared else ConfidenceLabel.NONE


def _profile_skill(skill_id, recognized, declarations, entries, all_ids, candidate_id, rules, as_of):
    declared = bool(declarations)
    proficiencies = _unique(d.proficiency.value for d in declarations if d.proficiency)
    limitations, missing, references = [], [], []
    if not recognized:
        limitations.append("skill is not in the alias vocabulary; matched by exact normalized text only")
    if len(proficiencies) > 1:
        limitations.append(f"conflicting declared proficiencies {proficiencies}; the first was kept")
    earned: set = set()
    supporting_kinds: set = set()
    covering = {rid: e for rid, e in entries.items() if skill_id in e.skills}
    for rid, entry in sorted(covering.items()):
        role, classes, note, limits = _judge(entry, candidate_id, rules, as_of)
        references.append(EvidenceReference(record_id=rid, record_kind=entry.kind, role=role, note=note))
        limitations.extend(limits)
        if role == EvidenceRole.SUPPORTS:
            earned.update(classes)
            supporting_kinds.add(entry.kind)
    for cited in sorted({c for d in declarations for c in d.cited_record_ids}):
        if cited in covering:
            continue
        if cited in entries:
            references.append(EvidenceReference(
                record_id=cited, record_kind=entries[cited].kind, role=EvidenceRole.INSUFFICIENT,
                note="cited record does not evaluate this skill"))
            limitations.append(f"cited record {cited!r} does not evaluate this skill")
        elif cited in all_ids:
            missing.append(f"cited record {cited!r} could not be used (see issues)")
        else:
            missing.append(f"cited record {cited!r} does not exist")
    classes = [c for c in EvidenceClass if c in earned and c != EvidenceClass.NO_EVIDENCE]
    if declared:
        classes.insert(0, EvidenceClass.SELF_REPORTED)
    if not classes:
        classes = [EvidenceClass.NO_EVIDENCE]
    backed = bool({EvidenceClass.ASSESSMENT_BACKED, EvidenceClass.PRACTICAL_TASK_BACKED,
                   EvidenceClass.INDEPENDENTLY_VERIFIED} & earned)
    if not backed:
        missing.append("no passing, current assessment or practical-task result on record")
    if EvidenceClass.PROJECT_BACKED in earned and EvidenceClass.INDEPENDENTLY_VERIFIED not in earned:
        missing.append("independent verification of project authorship")
    if declared and not earned:
        limitations.append("the claim is self-reported only; no supporting evidence was found")
    confidence = _confidence(supporting_kinds, declared)
    needs = not backed
    return SkillEvidenceProfile(
        skill_id=skill_id, recognized=recognized, declared=declared,
        declared_proficiency=ProficiencyLevel(proficiencies[0]) if proficiencies else None,
        evidence_classes=classes, confidence=confidence, references=references,
        limitations=_unique(limitations), missing_evidence=_unique(missing), needs_assessment=needs,
        recommended_assessment=(f"Complete an assessment or practical task that evaluates {skill_id}" if needs else None),
        explanation=(f"{skill_id}: {'declared' if declared else 'not declared'}; classes "
                     f"{', '.join(c.value for c in classes)}; confidence {confidence.value} "
                     f"({len(supporting_kinds)} distinct supporting record kind(s))."),
    )


def _matching_evidence_from_profile(skills, entries, candidate_id):
    """Expose only current supporting source records to the shared match rubric."""
    supported_ids = {
        reference.record_id
        for profile in skills
        for reference in profile.references
        if reference.role == EvidenceRole.SUPPORTS
    }
    result = []
    for record_id in sorted(supported_ids):
        entry = entries[record_id]
        record = entry.record
        verified = _independently_verified(entry, candidate_id)
        status = VerificationStatus.VERIFIED if verified else VerificationStatus.UNVERIFIED
        for skill_id in sorted(entry.skills):
            if entry.kind == RecordKind.ASSESSMENT:
                result.append(SkillEvidence(
                    candidate_id=candidate_id, skill_id=skill_id, evidence_type=EvidenceType.ASSESSMENT,
                    evidence_ref=record_id, assessment_score=entry.score,
                    verification_status=status, recorded_at=entry.when))
            elif entry.kind == RecordKind.PRACTICAL_TASK:
                if entry.score is None:
                    continue
                result.append(SkillEvidence(
                    candidate_id=candidate_id, skill_id=skill_id, evidence_type=EvidenceType.PRACTICAL_TASK,
                    evidence_ref=record_id, assessment_score=entry.score,
                    verification_status=status, recorded_at=entry.when))
            elif entry.kind == RecordKind.PROJECT:
                result.append(SkillEvidence(
                    candidate_id=candidate_id, skill_id=skill_id, evidence_type=EvidenceType.PROJECT,
                    evidence_ref=record.reference, verification_status=status, recorded_at=entry.when))
    return result


# ------------------------------------------------------ service function


def build_evidence_profile(
    request: EvidenceProfileRequest,
    registry: AssessmentRegistry | None = None,
    rules: EvidenceRules = DEFAULT_RULES,
    normalizer: SkillNormalizer | None = None,
) -> EvidenceProfileResult:
    """Deterministic evidence profile for one candidate. Pure given its inputs."""
    normalizer = normalizer or SkillNormalizer.from_config(AliasConfig.model_validate(SKILL_ALIAS_DATA))
    issues: list[str] = []
    registered = registry.for_candidate(request.profile.candidate_id) if registry else []
    entries, all_ids, recognized, norm = _intake(request, registered, normalizer, issues)
    declarations: dict[str, list[DeclaredSkill]] = {}
    declared_inputs = [*request.declared_skills, *(DeclaredSkill(skill=s) for s in request.profile.claimed_skills)]
    for declared in declared_inputs:
        declarations.setdefault(norm(declared.skill), []).append(declared)
    requested = {norm(s) for s in request.requested_skills}
    skill_ids = set(declarations) | requested | {s for e in entries.values() for s in e.skills}
    skills = [
        _profile_skill(sid, recognized[sid], declarations.get(sid, []), entries, all_ids,
                       request.profile.candidate_id, rules, request.as_of)
        for sid in sorted(skill_ids)
    ]
    matching_evidence = _matching_evidence_from_profile(skills, entries, request.profile.candidate_id)
    log_event("evidence_profile_built", skills=len(skills), issues=len(issues))
    return EvidenceProfileResult(
        candidate_id=request.profile.candidate_id, as_of=request.as_of, skills=skills,
        matching_evidence=matching_evidence,
        issues=_unique(issues), notes=RESULT_NOTES)


# ------------------------------------------------------ message handler


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _error_response(request: AgentRequest, code: str, message: str) -> AgentResponse:
    return AgentResponse(
        request_id=request.request_id, sender=AgentName.CANDIDATE_EVIDENCE, recipient=request.sender,
        message_type=EVIDENCE_RESULT_MESSAGE, created_at=_now(), status=ResponseStatus.ERROR,
        error=AgentError(code=code, message=message))


registry = AssessmentRegistry()


def handle_evidence_request(request: AgentRequest, registry: AssessmentRegistry | None = None,
                            **options) -> AgentResponse:
    """Callable entry point for the orchestrator: validated request in, validated response out."""
    if request.recipient != AgentName.CANDIDATE_EVIDENCE:
        return _error_response(request, "wrong_recipient", "request is not addressed to this agent")
    if request.message_type != EVIDENCE_PROFILE_MESSAGE:
        return _error_response(request, "unsupported_message_type", f"expected {EVIDENCE_PROFILE_MESSAGE!r}")
    try:
        query = EvidenceProfileRequest.model_validate(request.payload)
    except ValidationError as error:
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                            for e in error.errors(include_input=False, include_url=False))
        return _error_response(request, "invalid_request", details)
    result = build_evidence_profile(query, registry=registry, **options)
    return AgentResponse(
        request_id=request.request_id, sender=AgentName.CANDIDATE_EVIDENCE, recipient=request.sender,
        message_type=EVIDENCE_RESULT_MESSAGE, created_at=_now(), status=ResponseStatus.SUCCESS,
        payload=result.model_dump(mode="json"))


# -------------------------------------------------------- Agentverse chat

CHAT_HELP = (
    "I can build an evidence profile from records you provide. Send one JSON object "
    "matching EvidenceProfileRequest, including profile and as_of. Include declared_skills, "
    "assessments, practical_tasks and projects when available. Scores and verification "
    "must be actual recorded values; I will not invent them. Example shape:\n"
    '{"profile":{"candidate_id":"candidate-1","target_role":"SOC Analyst",'
    '"preferred_location":"Remote","experience_level":"entry_level",'
    '"claimed_skills":["Python"],"learning_hours_per_week":5},'
    '"as_of":"2026-10-14T00:00:00Z","declared_skills":[],"assessments":[],'
    '"practical_tasks":[],"projects":[]}'
)
CHAT_ERROR_TEXT = (
    "I couldn't build the evidence profile. No evidence was changed or inferred; "
    "please check the JSON and try again."
)


def answer_chat(text: str) -> str:
    """Validate a JSON profile request and call the deterministic evidence service."""
    try:
        query = EvidenceProfileRequest.model_validate_json(text)
    except (ValidationError, ValueError, TypeError):
        return (
            "I couldn't validate that evidence-profile request. Send a JSON object matching "
            "EvidenceProfileRequest; missing or invalid evidence is never treated as a pass.\n\n"
            + CHAT_HELP
        )
    try:
        result = build_evidence_profile(query, registry=registry)
        return result.model_dump_json(indent=2)
    except Exception as error:
        log_event("chat_evidence_profile_failed", logging.ERROR, error=type(error).__name__)
        return CHAT_ERROR_TEXT


def chat_text(message: ChatMessage) -> str:
    return " ".join(item.text for item in message.content if isinstance(item, TextContent)).strip()


# ------------------------------------------------------------------- uAgents

agent = Agent(
    name=AGENT_NAME,
    seed=os.environ.get("SKILLBRIDGE_EVIDENCE_SEED"),
    port=int(os.environ.get("SKILLBRIDGE_EVIDENCE_PORT", "8002")),
)


@agent.on_message(model=AgentRequestMessage, replies={AgentResponseMessage})
async def on_evidence_request(ctx: Context, sender: str, message: AgentRequestMessage):
    try:
        request = message_to_request(message)
    except ValueError:
        ctx.logger.warning(f"rejected invalid request {message.request_id}")
        await ctx.send(sender, AgentResponseMessage(
            request_id=message.request_id, sender=AgentName.CANDIDATE_EVIDENCE,
            recipient=AgentName.ORCHESTRATOR, message_type=EVIDENCE_RESULT_MESSAGE,
            created_at=_now(), status=ResponseStatus.ERROR,
            error=AgentErrorMessage(code="invalid_request", message="envelope failed canonical validation")))
        return
    response = await asyncio.to_thread(handle_evidence_request, request, registry)
    await ctx.send(sender, response_to_message(response))


# Agentverse-hosted agents do not receive Agent Chat Protocol support automatically.
chat_proto = uagents.Protocol(spec=chat_protocol_spec)


@chat_proto.on_message(ChatMessage)
async def on_chat_message(ctx: Context, sender: str, message: ChatMessage):
    try:
        await ctx.send(sender, ChatAcknowledgement(
            timestamp=_now(), acknowledged_msg_id=message.msg_id))
    except Exception:
        ctx.logger.warning("could not acknowledge chat message")
    try:
        text = chat_text(message)
        reply = CHAT_HELP if not text else await asyncio.to_thread(answer_chat, text)
    except Exception as error:
        ctx.logger.error(f"chat handling failed: {type(error).__name__}")
        reply = CHAT_ERROR_TEXT
    await ctx.send(sender, ChatMessage(
        timestamp=_now(), msg_id=uuid4(),
        content=[TextContent(type="text", text=reply)]))


@chat_proto.on_message(ChatAcknowledgement)
async def on_chat_acknowledgement(ctx: Context, sender: str, message: ChatAcknowledgement):
    ctx.logger.info(f"chat message {message.acknowledged_msg_id} acknowledged")


agent.include(chat_proto, publish_manifest=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent.run()
