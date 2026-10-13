"""Weighted, rubric-driven skill alignment.

Formula (all values come from ``MatchRubric``):

    weight_i   = required_weight or preferred_weight
    credit_i   = credit for the strongest evidence tier held for skill i
    score      = 100 * sum(weight_i * credit_i) / sum(weight_i)

Evidence tiers, strongest first, and their credit:
    verified assessment    -> the normalized assessment score (0..1)
    verified project       -> rubric.verified_project_credit
    unverified evidence    -> rubric.unverified_credit
    self-reported claim    -> rubric.self_reported_credit
    nothing                -> 0 credit, labelled NO_EVIDENCE

Only the strongest tier counts, so a low verified assessment is not masked by
a self-reported claim. Rejected evidence is ignored. NO_EVIDENCE means the
skill is unevaluated, not that the candidate lacks it. The score measures
alignment with the listed requirements; it is not a hiring probability.
Eligibility is evaluated separately and never changes the score.
"""

from dataclasses import dataclass

from skillbridge.eligibility import evaluate_eligibility
from skillbridge.normalization import DEFAULT_NORMALIZER, NormalizedSkill, SkillNormalizer
from skillbridge.schemas import (
    CandidateProfile,
    EligibilityResult,
    EvidenceBasis,
    EvidenceType,
    JobOpportunity,
    MatchRubric,
    MatchResult,
    RequirementContribution,
    RequirementLevel,
    ScoreStatus,
    SkillEvidence,
    VerificationStatus,
)

_BASIS_RANK = {
    EvidenceBasis.NO_EVIDENCE: 0,
    EvidenceBasis.SELF_REPORTED: 1,
    EvidenceBasis.UNVERIFIED_PROJECT: 2,
    EvidenceBasis.UNVERIFIED_ASSESSMENT: 2,
    EvidenceBasis.UNVERIFIED_PRACTICAL_TASK: 2,
    EvidenceBasis.VERIFIED_PROJECT: 3,
    EvidenceBasis.VERIFIED_ASSESSMENT: 4,
    EvidenceBasis.VERIFIED_PRACTICAL_TASK: 4,
}


DEFAULT_RUBRIC = MatchRubric()


@dataclass(frozen=True)
class _Requirement:
    skill: NormalizedSkill
    level: RequirementLevel


def _collect_requirements(
    job: JobOpportunity, normalizer: SkillNormalizer
) -> dict[str, _Requirement]:
    """Dedupe by canonical id; a skill listed as required stays required."""
    collected: dict[str, _Requirement] = {}
    groups = (
        (RequirementLevel.REQUIRED, job.required_skills or []),
        (RequirementLevel.PREFERRED, job.preferred_skills or []),
    )
    for level, items in groups:
        for item in items:
            skill = normalizer.normalize(item.skill)
            collected.setdefault(skill.skill_id, _Requirement(skill, level))
    return collected


def _basis_of(evidence: SkillEvidence) -> EvidenceBasis | None:
    """None for rejected evidence, which must not count."""
    if evidence.verification_status == VerificationStatus.REJECTED:
        return None
    if evidence.evidence_type == EvidenceType.SELF_REPORTED:
        return EvidenceBasis.SELF_REPORTED
    verified = evidence.verification_status == VerificationStatus.VERIFIED
    if evidence.evidence_type == EvidenceType.ASSESSMENT:
        return (
            EvidenceBasis.VERIFIED_ASSESSMENT if verified else EvidenceBasis.UNVERIFIED_ASSESSMENT
        )
    if evidence.evidence_type == EvidenceType.PRACTICAL_TASK:
        return (EvidenceBasis.VERIFIED_PRACTICAL_TASK if verified
                else EvidenceBasis.UNVERIFIED_PRACTICAL_TASK)
    return EvidenceBasis.VERIFIED_PROJECT if verified else EvidenceBasis.UNVERIFIED_PROJECT


def _credit_for(
    basis: EvidenceBasis, tier: list[SkillEvidence], rubric: MatchRubric
) -> tuple[float, list[str]]:
    if basis in {EvidenceBasis.VERIFIED_ASSESSMENT, EvidenceBasis.VERIFIED_PRACTICAL_TASK}:
        latest = max(tier, key=lambda e: (e.recorded_at, e.evidence_ref))
        return latest.assessment_score, [latest.evidence_ref]
    refs = sorted({e.evidence_ref for e in tier if e.evidence_ref})
    if basis == EvidenceBasis.VERIFIED_PROJECT:
        return rubric.verified_project_credit, refs
    if basis in (EvidenceBasis.UNVERIFIED_PROJECT, EvidenceBasis.UNVERIFIED_ASSESSMENT,
                 EvidenceBasis.UNVERIFIED_PRACTICAL_TASK):
        return rubric.unverified_credit, refs
    return rubric.self_reported_credit, refs


def _explain(basis: EvidenceBasis, credit: float, rejected: int) -> str:
    text = {
        EvidenceBasis.VERIFIED_ASSESSMENT: f"Verified assessment result {credit:.2f}.",
        EvidenceBasis.VERIFIED_PRACTICAL_TASK: f"Verified practical-task result {credit:.2f}.",
        EvidenceBasis.VERIFIED_PROJECT: "Verified project evidence.",
        EvidenceBasis.UNVERIFIED_ASSESSMENT: "Assessment evidence not yet verified.",
        EvidenceBasis.UNVERIFIED_PRACTICAL_TASK: "Practical-task evidence not yet verified.",
        EvidenceBasis.UNVERIFIED_PROJECT: "Project evidence not yet verified.",
        EvidenceBasis.SELF_REPORTED: "Self-reported only; no supporting evidence.",
        EvidenceBasis.NO_EVIDENCE: (
            "No evidence provided; this skill is unevaluated, "
            "not evidence that the candidate lacks it."
        ),
    }[basis]
    return f"{text} {rejected} rejected evidence item(s) ignored." if rejected else text


def _contribution(
    requirement: _Requirement,
    evidence: list[SkillEvidence],
    claimed: bool,
    total_weight: float,
    rubric: MatchRubric,
) -> RequirementContribution:
    weight = rubric.weight_for(requirement.level)
    rejected = 0
    tiers: dict[EvidenceBasis, list[SkillEvidence]] = {}
    for item in evidence:
        basis = _basis_of(item)
        if basis is None:
            rejected += 1
        else:
            tiers.setdefault(basis, []).append(item)
    if claimed:
        tiers.setdefault(EvidenceBasis.SELF_REPORTED, [])

    basis = max(tiers, key=_BASIS_RANK.__getitem__, default=EvidenceBasis.NO_EVIDENCE)
    if basis == EvidenceBasis.NO_EVIDENCE:
        credit, refs = 0.0, []
    else:
        credit, refs = _credit_for(basis, tiers[basis], rubric)
    return RequirementContribution(
        skill_id=requirement.skill.skill_id,
        skill_name=requirement.skill.original,
        recognized_skill=requirement.skill.recognized,
        level=requirement.level,
        weight=weight,
        credit=credit,
        points=round(100 * weight * credit / total_weight, 4),
        max_points=round(100 * weight / total_weight, 4),
        evidence_basis=basis,
        evidence_refs=refs,
        explanation=_explain(basis, credit, rejected),
    )


def _insufficient(
    job: JobOpportunity,
    candidate: CandidateProfile,
    eligibility: EligibilityResult,
    rubric: MatchRubric,
    note: str,
) -> MatchResult:
    return MatchResult(
        job_id=job.job_id,
        candidate_id=candidate.candidate_id,
        rubric_version=rubric.version,
        score_status=ScoreStatus.INSUFFICIENT_JOB_DATA,
        eligibility=eligibility,
        notes=[note],
    )


def evaluate_match(
    candidate: CandidateProfile,
    evidence: list[SkillEvidence],
    job: JobOpportunity,
    normalizer: SkillNormalizer = DEFAULT_NORMALIZER,
    rubric: MatchRubric = DEFAULT_RUBRIC,
) -> MatchResult:
    foreign = {e.candidate_id for e in evidence} - {candidate.candidate_id}
    if foreign:
        raise ValueError(f"evidence belongs to other candidates: {sorted(foreign)}")

    eligibility = evaluate_eligibility(candidate, job)
    if job.required_skills is None:
        return _insufficient(
            job, candidate, eligibility, rubric, "Job source did not provide required skills."
        )
    requirements = _collect_requirements(job, normalizer)
    if not requirements:
        return _insufficient(
            job, candidate, eligibility, rubric, "Job lists no skill requirements to score."
        )

    evidence_by_skill: dict[str, list[SkillEvidence]] = {}
    for item in evidence:
        skill_id = normalizer.normalize(item.skill_id).skill_id
        evidence_by_skill.setdefault(skill_id, []).append(item)
    claimed = {normalizer.normalize(s).skill_id for s in candidate.claimed_skills}

    total_weight = sum(rubric.weight_for(r.level) for r in requirements.values())
    breakdown = [
        _contribution(
            r, evidence_by_skill.get(skill_id, []), skill_id in claimed, total_weight, rubric
        )
        for skill_id, r in requirements.items()
    ]
    notes = []
    if job.preferred_skills is None:
        notes.append("Job source did not provide preferred skills; only required skills scored.")
    return MatchResult(
        job_id=job.job_id,
        candidate_id=candidate.candidate_id,
        rubric_version=rubric.version,
        score_status=ScoreStatus.SCORED,
        skill_alignment_score=round(sum(c.points for c in breakdown), 2),
        breakdown=breakdown,
        eligibility=eligibility,
        notes=notes,
    )
