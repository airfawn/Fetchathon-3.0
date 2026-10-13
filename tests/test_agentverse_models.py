"""Verify that models inlined into hosted Agentverse files match the canonical ones."""

import os
import subprocess
import sys
from datetime import date

import pytest
from pydantic import BaseModel as V2BaseModel
from pydantic.v1 import BaseModel as V1BaseModel

import skillbridge.schemas as canonical
from scripts.sync_agent_models import (
    AGENT_MODELS,
    BEGIN_MARKER,
    END_MARKER,
    canonical_definition,
    check_agent_file,
    compare_behavior,
    compare_block,
    extract_block,
    forbidden_imports,
    generate_block,
    inject_block,
    load_block,
    main,
)
from skillbridge.matching import evaluate_match
from skillbridge.schemas import (
    ExtractionMethod,
    EvidenceProfileResult,
    OpportunityRecord,
    SkillRequirement,
    SourceKind,
    missing_job_fields,
)
from tests.conftest import (
    NOW,
    experience_requirement,
    make_candidate,
    make_evidence,
    make_job,
)


def dump(model):
    return model.model_dump(mode="json")


def changed(base, **changes):
    return {**base, **changes}


def without(base, *keys):
    return {k: v for k, v in base.items() if k not in keys}


CANDIDATE = dump(make_candidate(claimed_skills=["SIEM"], eligibility_facts={"work_authorization": "IN"}))
JOB = dump(make_job(
    required_skills=[SkillRequirement(skill="SIEM")],
    preferred_skills=[SkillRequirement(skill="Python", description="scripting")],
    eligibility_requirements=[experience_requirement("junior")],
    application_deadline=date(2026, 6, 1),
))
ASSESSMENT = dump(make_evidence())
PROJECT = dump(make_evidence(kind="project"))
SELF_REPORTED = dump(make_evidence(kind="self_reported"))
MATCH = dump(evaluate_match(
    make_candidate(claimed_skills=["SIEM"]), [make_evidence()], make_job(
        eligibility_requirements=[experience_requirement("senior")]),
))
ELIGIBILITY = MATCH["eligibility"]
ENVELOPE = {
    "request_id": "req-1", "sender": "orchestrator", "recipient": "opportunity_intelligence",
    "message_type": "find_jobs", "created_at": NOW.isoformat(), "payload": {"k": [1, "x"]},
}
ACTION = {"action_id": "a", "job_id": "j", "action_type": "apply_now",
          "rationale": "Meets requirements", "priority": 1}
ACTION_JOB = make_job(
    job_id="action-job", source_id="fixture-source",
    required_skills=[SkillRequirement(skill="SIEM")],
)
ACTION_OPPORTUNITY = dump(OpportunityRecord(
    job=ACTION_JOB, source_kind=SourceKind.FIXTURE, original_title=ACTION_JOB.title,
    normalized_title=ACTION_JOB.title.casefold(), description="Fixture description.",
    remote=True, skills_extraction=ExtractionMethod.STRUCTURED_SOURCE,
    eligibility_extraction=ExtractionMethod.STRUCTURED_SOURCE,
    missing_fields=missing_job_fields(ACTION_JOB, "Fixture description.", True),
))
ACTION_PROFILE = {"candidate_id": "cand-1", "as_of": NOW.isoformat(), "skills": [],
                  "matching_evidence": [ASSESSMENT]}
ACTION_PLAN_REQUEST = {
    "candidate": CANDIDATE,
    "opportunities": [ACTION_OPPORTUNITY],
    "evidence_profile": ACTION_PROFILE,
}
ACTION_PLAN_RESULT = {
    "candidate_id": "cand-1", "rubric": {}, "decision_policy": {},
    "ranked_opportunities": [],
}

BEHAVIOR_CASES = {
    "CandidateProfile": [
        CANDIDATE,
        without(CANDIDATE, "claimed_skills", "eligibility_facts"),
        changed(CANDIDATE, target_role="  "),
        changed(CANDIDATE, learning_hours_per_week=-1),
        changed(CANDIDATE, learning_hours_per_week=169),
        changed(CANDIDATE, experience_level="wizard"),
        changed(CANDIDATE, surprise=1),
        without(CANDIDATE, "target_role"),
    ],
    "ActionPlanRequest": [
        ACTION_PLAN_REQUEST,
        without(ACTION_PLAN_REQUEST, "candidate"),
        changed(ACTION_PLAN_REQUEST, evidence_profile={**ACTION_PROFILE, "candidate_id": "other"}),
        changed(ACTION_PLAN_REQUEST, opportunities=[]),
        changed(ACTION_PLAN_REQUEST, opportunities=[ACTION_OPPORTUNITY, ACTION_OPPORTUNITY]),
        changed(ACTION_PLAN_REQUEST, evidence_profile={
            **ACTION_PROFILE,
            "matching_evidence": [changed(ASSESSMENT, candidate_id="other")],
        }),
        changed(ACTION_PLAN_REQUEST, matching_config={"required_weight": 1, "preferred_weight": 1}),
    ],
    "ActionPlanResult": [
        ACTION_PLAN_RESULT,
        without(ACTION_PLAN_RESULT, "ranked_opportunities"),
        changed(ACTION_PLAN_RESULT, asi_summary="summary"),
        changed(ACTION_PLAN_RESULT, decision_policy={"apply_now_score": -1}),
    ],
    "SkillEvidence": [
        ASSESSMENT, PROJECT, SELF_REPORTED,
        changed(ASSESSMENT, evidence_type="practical_task"),
        changed(ASSESSMENT, assessment_score=None),
        changed(ASSESSMENT, evidence_ref=None),
        changed(ASSESSMENT, assessment_score=1.5),
        changed(PROJECT, evidence_ref=None),
        changed(PROJECT, assessment_score=0.5),
        changed(SELF_REPORTED, verification_status="verified"),
        changed(ASSESSMENT, recorded_at="2026-01-01T12:00:00"),
        changed(ASSESSMENT, evidence_type="practical_task", evidence_ref=None),
        without(ASSESSMENT, "verification_status"),
    ],
    "JobOpportunity": [
        JOB,
        changed(JOB, required_skills=None, preferred_skills=None,
                eligibility_requirements=None, location=None, application_deadline=None),
        changed(JOB, source_url="not a url"),
        changed(JOB, source_url="ftp://example.com/x"),
        changed(JOB, retrieved_at="2026-01-01T12:00:00"),
        changed(JOB, source_id=""),
        changed(JOB, application_deadline="not a date"),
        without(JOB, "source_id"),
        changed(JOB, surprise=True),
    ],
    "EligibilityResult": [
        ELIGIBILITY,
        changed(ELIGIBILITY, status="eligible_on_available_information"),
        {"job_id": "j", "candidate_id": "c", "requirements_known": False, "status": "unknown"},
        {"job_id": "j", "candidate_id": "c", "requirements_known": False,
         "status": "eligible_on_available_information"},
    ],
    "MatchResult": [
        MATCH,
        changed(MATCH, skill_alignment_score=None),
        changed(MATCH, score_status="insufficient_job_data"),
        changed(MATCH, skill_alignment_score=101),
        changed(MATCH, score_meaning="Probability of getting hired."),
        without(MATCH, "eligibility"),
    ],
    "CareerAction": [
        ACTION,
        changed(ACTION, action_type="build_project", skill_id="siem"),
        changed(ACTION, action_type="build_project"),
        changed(ACTION, action_type="complete_assessment"),
        changed(ACTION, priority=0),
        changed(ACTION, resource_url="not a url"),
        changed(ACTION, resource_url="https://example.com/course"),
        changed(ACTION, estimated_hours=-1),
    ],
    "AgentRequest": [
        ENVELOPE,
        without(ENVELOPE, "payload"),
        changed(ENVELOPE, schema_version="1.7"),
        changed(ENVELOPE, schema_version="2.0"),
        changed(ENVELOPE, schema_version="abc"),
        changed(ENVELOPE, recipient="fourth_ai_agent"),
        changed(ENVELOPE, created_at="2026-01-01T12:00:00"),
    ],
    "AgentResponse": [
        changed(ENVELOPE, status="success"),
        changed(ENVELOPE, status="error"),
        changed(ENVELOPE, status="success", error={"code": "x", "message": "y"}),
        changed(ENVELOPE, status="error", error={"code": "timeout", "message": "slow"}),
        changed(ENVELOPE, status="error", error={"code": "", "message": "slow"}),
        changed(ENVELOPE, status="maybe"),
    ],
}


OPP_REQUEST = {"target_role": "SOC Analyst", "country": "IN", "experience_level": "entry_level",
               "study_preferences": {"techniques": ["videos"], "notes": "evenings"}}
OPP_REF = {"source_id": "s", "source_url": "https://x.example.com/1", "source_job_id": "j"}
OPP_REPORT = {"source_id": "s", "source_kind": "live", "status": "ok", "retrieved_count": 1,
              "accepted_count": 1, "rejected_count": 0, "attempts": 1}
OPP_RESULT = {"search_status": "complete", "source_reports": [OPP_REPORT], "records": [],
              "contains_fixture_data": False}
EV_PROFILE = {"candidate_id": "c1", "target_role": "SOC Analyst", "preferred_location": "Remote",
              "experience_level": "entry_level", "claimed_skills": ["python"], "learning_hours_per_week": 5}
EV_REQUEST = {"profile": EV_PROFILE, "as_of": "2025-01-01T00:00:00Z",
              "declared_skills": [{"skill": "python", "proficiency": "beginner"}],
              "assessments": [{"record_id": "a1", "candidate_id": "c1", "skills_evaluated": ["python"],
                               "score": 0.8, "taken_at": "2024-12-01T00:00:00Z", "provider": "p"}]}
EV_SKILL = {"skill_id": "python", "recognized": True, "declared": True,
            "evidence_classes": ["self_reported"], "confidence": "self_report_only",
            "needs_assessment": True, "recommended_assessment": "Take one", "explanation": "e"}
EV_RESULT = {"candidate_id": "c1", "as_of": "2025-01-01T00:00:00Z", "skills": [EV_SKILL]}
BEHAVIOR_CASES.update({
    "EvidenceProfileRequest": [
        EV_REQUEST, without(EV_REQUEST, "as_of"), changed(EV_REQUEST, as_of="2025-01-01T00:00:00"),
        changed(EV_REQUEST, surprise=1), without(EV_REQUEST, "declared_skills"),
        changed(EV_REQUEST, declared_skills=[{"skill": " "}]),
        changed(EV_REQUEST, assessments=[{**EV_REQUEST["assessments"][0], "score": 1.5}]),
        changed(EV_REQUEST, assessments=[{**EV_REQUEST["assessments"][0], "verification_status": "verified"}]),
        changed(EV_REQUEST, projects=[{"record_id": "p", "candidate_id": "c1", "reference": "r",
                                       "skills_demonstrated": [], "submitted_at": "2024-01-01T00:00:00Z"}]),
    ],
    "EvidenceProfileResult": [
        EV_RESULT, without(EV_RESULT, "skills"), changed(EV_RESULT, skills=[]),
        changed(EV_RESULT, skills=[{**EV_SKILL, "needs_assessment": False}]),
        changed(EV_RESULT, skills=[{**EV_SKILL, "evidence_classes": ["no_evidence", "self_reported"]}]),
        changed(EV_RESULT, skills=[{**EV_SKILL, "confidence": "certain"}]),
        changed(EV_RESULT, skills=[{**EV_SKILL, "evidence_classes": []}]),
    ],
    "OpportunitySearchRequest": [
        OPP_REQUEST, without(OPP_REQUEST, "study_preferences"), changed(OPP_REQUEST, country="in"),
        changed(OPP_REQUEST, country="India"), changed(OPP_REQUEST, target_role=" "),
        changed(OPP_REQUEST, filters={"max_results": 0}), changed(OPP_REQUEST, filters={"max_results": 101}),
        changed(OPP_REQUEST, surprise=1), changed(OPP_REQUEST, experience_level="wizard"),
    ],
    "StudyPreferences": [{"techniques": ["videos"]}, {"techniques": []}, {"techniques": [" "]},
                         {"techniques": ["a"] * 21}, {"notes": "x" * 5000}, {"extra": 1}],
    "JobFilters": [{}, {"remote_only": True, "max_results": 100}, {"max_results": 0}, {"keywords": [" "]},
                   {"remote_only": "maybe"}, {"extra": 1}],
    "SourceReference": [OPP_REF, changed(OPP_REF, source_url="nope"), changed(OPP_REF, source_job_id=" "),
                        without(OPP_REF, "source_id")],
    "SourceReport": [OPP_REPORT, changed(OPP_REPORT, status="error"),
                     changed(OPP_REPORT, status="error", error_code="timeout", error_message="m"),
                     changed(OPP_REPORT, error_code="timeout", error_message="m"),
                     changed(OPP_REPORT, attempts=-1), changed(OPP_REPORT, source_kind="mock")],
    "OpportunitySearchResult": [
        OPP_RESULT, changed(OPP_RESULT, search_status="failed"), changed(OPP_RESULT, contains_fixture_data=True),
        changed(OPP_RESULT, source_reports=[]), changed(OPP_RESULT, search_status="nonsense"),
        changed(OPP_RESULT, extra=1),
    ],
    "ExperienceRequirement": [
        {"minimum_years": 3, "statement": "3 years"}, {"statement": "some experience"},
        {"minimum_years": -1, "statement": "x"}, {"minimum_years": 51, "statement": "x"},
        {"minimum_years": 3}, {"statement": " "}, {"statement": "x", "extra": 1},
    ],
    "RejectedRecord": [{"source_id": "s", "reasons": ["bad url"]}, {"source_id": "s", "reasons": []},
                       {"source_id": "s"}, {"reasons": ["x"]}],
})

# Wire models check types only; canonical rules are applied by the converters.
WIRE_REQUEST = {"request_id": "r1", "sender": "orchestrator", "recipient": "opportunity_intelligence",
                "message_type": "opportunity.search", "created_at": "2026-01-01T00:00:00+00:00"}
BEHAVIOR_CASES["AgentRequestMessage"] = [
    WIRE_REQUEST, {**WIRE_REQUEST, "payload": {"a": 1}}, without(WIRE_REQUEST, "request_id"),
    changed(WIRE_REQUEST, created_at="not a date"), changed(WIRE_REQUEST, payload="x"),
]
BEHAVIOR_CASES["AgentResponseMessage"] = [
    {**WIRE_REQUEST, "status": "success"},
    {**WIRE_REQUEST, "status": "error", "error": {"code": "c", "message": "m"}},
    WIRE_REQUEST, {**WIRE_REQUEST, "status": "success", "error": {"code": "c"}},
]

AGENT_IDS = list(AGENT_MODELS)


@pytest.fixture(scope="module")
def blocks():
    return {agent: generate_block(names) for agent, names in AGENT_MODELS.items()}


def behavior_diffs(block):
    module = load_block(block)
    defined = {n for n in BEHAVIOR_CASES if hasattr(module, n)}
    diffs = []
    for name in sorted(defined):
        for payload in BEHAVIOR_CASES[name]:
            diffs += compare_behavior(canonical_definition(name), getattr(module, name), payload)
    return diffs


# ------------------------------------------------- generated blocks vs canonical


@pytest.mark.parametrize("agent", AGENT_IDS)
def test_inlined_definitions_match_canonical(blocks, agent):
    assert compare_block(blocks[agent]) == []


@pytest.mark.parametrize("agent", AGENT_IDS)
def test_inlined_behavior_matches_canonical(blocks, agent):
    assert behavior_diffs(blocks[agent]) == []


@pytest.mark.parametrize("agent", AGENT_IDS)
def test_every_required_model_has_behavior_cases(agent):
    for name in AGENT_MODELS[agent]:
        definition = canonical_definition(name)
        if not (isinstance(definition, type) and issubclass(definition, (V2BaseModel, V1BaseModel))):
            continue  # functions/constants/plain classes have dedicated tests
        assert len(BEHAVIOR_CASES.get(name, [])) >= 2, f"add behavior cases for {name}"


def test_behavior_cases_contain_both_valid_and_invalid_payloads():
    for name, payloads in BEHAVIOR_CASES.items():
        model = canonical_definition(name)
        outcomes = set()
        for payload in payloads:
            try:
                model.model_validate(payload)
                outcomes.add("valid")
            except ValueError:
                outcomes.add("invalid")
        assert outcomes == {"valid", "invalid"}, name


@pytest.mark.parametrize("agent", AGENT_IDS)
def test_blocks_are_self_contained_and_importable_in_isolation(blocks, agent, tmp_path):
    assert forbidden_imports(blocks[agent]) == []
    script = tmp_path / "isolated.py"
    script.write_text(blocks[agent] + "\nprint('isolated-ok')\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "isolated-ok" in result.stdout


def test_blocks_include_only_what_each_agent_needs(blocks):
    evidence = blocks["agents/evidence_agent.py"]
    assert "class EvidenceProfileRequest" in evidence and "class CandidateProfile" in evidence
    for unneeded in ("class JobOpportunity", "class MatchResult", "class CareerAction"):
        assert unneeded not in evidence
    action = blocks["agents/action_agent.py"]
    assert "class ActionPlanRequest" in action and "class CandidateProfile" in action
    for required in ("class OpportunityRecord", "class EvidenceProfileResult", "class MatchResult",
                     "class MatchRubric"):
        assert required in action
    opportunity = blocks["agents/opportunity_agent.py"]
    assert "class JobOpportunity" in opportunity and "class SkillEvidence" not in opportunity


def test_generation_is_deterministic():
    names = AGENT_MODELS["agents/action_agent.py"]
    assert generate_block(names) == generate_block(names)


def test_unknown_model_name_fails_generation():
    with pytest.raises(KeyError):
        generate_block(("DoesNotExist",))


# -------------------------------------------- the verifier detects divergence


EVIDENCE_AGENT = "agents/evidence_agent.py"


def tampered(blocks, old, new, agent=EVIDENCE_AGENT):
    assert old in blocks[agent], f"fixture text {old!r} not found"
    return blocks[agent].replace(old, new, 1)


@pytest.mark.parametrize(
    ("old", "new", "label"),
    [
        ("le=168", "le=100", "constraint"),
        ('STUDENT = "student"', 'STUDENT = "pupil"', "enum value"),
        ("claimed_skills: list[NonEmptyStr] = Field(default_factory=list)",
         "claimed_skills: list[NonEmptyStr]", "required vs optional"),
        ("learning_hours_per_week: float = Field(ge=0, le=168)",
         "learning_hours_per_week: int = Field(ge=0, le=168)", "field type"),
        ('verification_status: VerificationStatus = VerificationStatus.UNVERIFIED',
         'verification_status: VerificationStatus = VerificationStatus.VERIFIED', "default"),
        ('extra="forbid"', 'extra="ignore"', "model config"),
    ],
)
def test_definition_drift_is_detected(blocks, old, new, label):
    block = tampered(blocks, old, new)
    assert compare_block(block), f"{label} drift went undetected"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('raise ValueError("verified records must name verified_by")', "pass"),
        ('raise ValueError("no_evidence cannot be combined with other classes")', "pass"),
        ('if int(value.split(".")[0]) != SUPPORTED_MAJOR_VERSION:', "if False:"),
    ],
)
def test_validator_logic_drift_is_detected_by_behavior_tests(blocks, old, new):
    # JSON Schema cannot see this class of difference; behavioral cases must.
    assert behavior_diffs(tampered(blocks, old, new)) != []


# -------------------------------------------------- hosted file verification


def make_agent_file(tmp_path, names, header="# agent code\n"):
    path = tmp_path / "agent.py"
    path.write_text(
        f"{header}{BEGIN_MARKER}\nstale\n{END_MARKER}\n\nprint('agent body')\n", encoding="utf-8"
    )
    path.write_text(inject_block(path.read_text(), generate_block(names)), encoding="utf-8")
    return path


def test_injected_file_passes_verification(tmp_path):
    names = AGENT_MODELS[EVIDENCE_AGENT]
    path = make_agent_file(tmp_path, names)
    assert check_agent_file(path, names) == []
    assert "print('agent body')" in path.read_text()


def test_inject_is_idempotent_and_preserves_surrounding_code(tmp_path):
    names = AGENT_MODELS[EVIDENCE_AGENT]
    path = make_agent_file(tmp_path, names, header="# before\n")
    once = path.read_text()
    assert inject_block(once, generate_block(names)) == once
    assert once.startswith("# before\n") and once.rstrip().endswith("print('agent body')")


def test_hand_edited_block_is_reported(tmp_path):
    names = AGENT_MODELS[EVIDENCE_AGENT]
    path = make_agent_file(tmp_path, names)
    path.write_text(path.read_text().replace("le=168", "le=100"), encoding="utf-8")
    problems = check_agent_file(path, names)
    assert any("out of date" in p for p in problems)
    assert any("CandidateProfile" in p for p in problems)


def test_import_of_shared_package_is_reported(tmp_path):
    names = AGENT_MODELS[EVIDENCE_AGENT]
    path = make_agent_file(tmp_path, names, header="from skillbridge.schemas import SkillEvidence\n")
    assert any("skillbridge" in p for p in check_agent_file(path, names))


@pytest.mark.parametrize(
    "source",
    ["import skillbridge", "from skillbridge import schemas", "from . import x",
     "import tests.conftest", "from scripts.sync_agent_models import main"],
)
def test_forbidden_imports_detected(source):
    assert forbidden_imports(source)


def test_allowed_imports_pass():
    assert forbidden_imports("import re\nfrom pydantic import BaseModel\nfrom uagents import Agent") == []


def test_missing_or_duplicate_markers_reported(tmp_path):
    names = AGENT_MODELS[EVIDENCE_AGENT]
    path = tmp_path / "agent.py"
    path.write_text("print('no block')\n", encoding="utf-8")
    assert any("exactly one" in p for p in check_agent_file(path, names))
    block = generate_block(names)
    path.write_text(f"{block}\n{block}\n", encoding="utf-8")
    assert any("exactly one" in p for p in check_agent_file(path, names))
    with pytest.raises(ValueError):
        extract_block("nothing here")


# ------------------------------- agent-file discovery and the no-skip policy

from scripts.sync_agent_models import (  # noqa: E402
    REPO_ROOT,
    discover_agent_files,
    verify_agents,
    write_agents,
)

EXPECTED_AGENT_FILES = {
    "agents/opportunity_agent.py",
    "agents/evidence_agent.py",
    "agents/action_agent.py",
}


def test_exactly_the_three_expected_agents_are_registered():
    assert set(AGENT_MODELS) == EXPECTED_AGENT_FILES


def test_generation_into_temporary_directory_creates_verifiable_files(tmp_path):
    written = write_agents(tmp_path, create_missing=True)
    assert set(written) == EXPECTED_AGENT_FILES
    assert set(discover_agent_files(tmp_path)) == EXPECTED_AGENT_FILES
    report = verify_agents(tmp_path, require_all=True)
    assert report.ok and set(report.checked) == EXPECTED_AGENT_FILES and report.pending == []
    for relative in EXPECTED_AGENT_FILES:
        assert forbidden_imports((tmp_path / relative).read_text()) == []


def test_write_without_create_missing_creates_nothing(tmp_path):
    assert write_agents(tmp_path) == []
    assert discover_agent_files(tmp_path) == []


def test_missing_files_are_pending_not_ok_and_fail_when_required(tmp_path):
    report = verify_agents(tmp_path)
    assert report.ok and report.checked == []
    assert set(report.pending) == EXPECTED_AGENT_FILES
    strict = verify_agents(tmp_path, require_all=True)
    assert not strict.ok
    assert set(strict.problems) == EXPECTED_AGENT_FILES


def test_one_missing_file_fails_when_required_even_if_others_exist(tmp_path):
    write_agents(tmp_path, create_missing=True)
    (tmp_path / "agents/action_agent.py").unlink()
    assert verify_agents(tmp_path).pending == ["agents/action_agent.py"]
    strict = verify_agents(tmp_path, require_all=True)
    assert list(strict.problems) == ["agents/action_agent.py"]
    assert set(strict.checked) == EXPECTED_AGENT_FILES - {"agents/action_agent.py"}


def test_unregistered_agent_file_is_an_error(tmp_path):
    write_agents(tmp_path, create_missing=True)
    (tmp_path / "agents/fourth_agent.py").write_text("print('x')\n")
    report = verify_agents(tmp_path)
    assert "agents/fourth_agent.py" in report.problems


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("le=168", "le=100"),
        ('extra="forbid"', 'extra="ignore"'),
        ("min_length=1", "min_length=0"),
        ('SUCCESS = "success"', 'SUCCESS = "ok"'),
    ],
)
def test_drift_in_a_discovered_file_fails_verification(tmp_path, old, new):
    write_agents(tmp_path, create_missing=True)
    path = tmp_path / "agents/evidence_agent.py"
    text = path.read_text()
    if old not in text:
        path = tmp_path / "agents/action_agent.py"
        text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new, 1))
    report = verify_agents(tmp_path)
    assert not report.ok and path.name in "".join(report.problems)


def test_repository_agent_files_are_verified_or_explicitly_pending():
    report = verify_agents(REPO_ROOT)
    assert report.problems == {}, report.problems
    # Every registered agent is accounted for: verified or reported pending.
    assert set(report.checked) | set(report.pending) == EXPECTED_AGENT_FILES
    assert set(report.checked) == set(discover_agent_files(REPO_ROOT))


def test_cli_reports_pending_then_strict_mode_fails(capsys, tmp_path):
    assert main(["--check"], root=tmp_path) == 0
    assert "PENDING" in capsys.readouterr().out
    assert main(["--check", "--require-all"], root=tmp_path) == 1
    assert "expected agent file is missing" in capsys.readouterr().out


def test_cli_env_var_makes_missing_files_fail(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLBRIDGE_REQUIRE_AGENTS", "1")
    assert main(["--check"], root=tmp_path) == 1


def test_cli_write_create_missing_then_check_passes(capsys, tmp_path):
    assert main(["--write", "--create-missing"], root=tmp_path) == 0
    assert main(["--check", "--require-all"], root=tmp_path) == 0
    assert "PENDING" not in capsys.readouterr().out
