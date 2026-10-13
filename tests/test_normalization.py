import json

import pytest
from pydantic import ValidationError

from skillbridge.matching import evaluate_match
from skillbridge.normalization import (
    DEFAULT_ALIAS_CONFIG,
    DEFAULT_SKILL_ALIASES,
    AliasConfig,
    SkillNormalizer,
    load_alias_config,
)
from skillbridge.schemas import SkillRequirement
from tests.conftest import make_candidate, make_evidence, make_job


@pytest.fixture
def normalizer():
    return SkillNormalizer(DEFAULT_SKILL_ALIASES)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SIEM", "siem"),
        ("Security Information and Event Management", "siem"),
        ("Security Information & Event Management", "siem"),
        ("Linux system administration", "linux-administration"),
        ("Python", "python"),
        ("python programming", "python"),
        ("TCP/IP", "tcp-ip-networking"),
        ("TCP / IP", "tcp-ip-networking"),
        ("tcp-ip networking", "tcp-ip-networking"),
        ("Computer networks", "computer-networking"),
        ("Web security", "web-application-security"),
        ("Web-Application_Security", "web-application-security"),
        ("Security incident response", "incident-response"),
        ("Digital Forensic Analysis", "digital-forensics"),
        ("AWS", "aws"),
        ("AMAZON WEB SERVICES", "aws"),
        ("Identity and Access Management", "iam"),
        ("Identity & Access Management", "iam"),
        ("IAM", "iam"),
        ("CI/CD", "ci-cd"),
        ("CI / CD", "ci-cd"),
        ("ci-cd", "ci-cd"),
        ("Continuous Integration and Continuous Delivery", "ci-cd"),
        ("Continuous Integration / Continuous Delivery", "ci-cd"),
        ("k8s", "kubernetes"),
        ("Pentesting", "penetration-testing"),
        ("RESTful APIs", "rest-api"),
        ("OOP", "object-oriented-programming"),
    ],
)
def test_configured_aliases_resolve_to_canonical_id(normalizer, raw, expected):
    result = normalizer.normalize(raw)
    assert (result.skill_id, result.recognized) == (expected, True)
    assert result.original == raw


@pytest.mark.parametrize(
    "terms",
    [
        ["TCP/IP networking", "computer networking"],
        ["IAM", "AWS IAM"],
        ["Java", "JavaScript"],
        ["SQL", "NoSQL"],
        ["Python", "Python scripting"],
        ["Linux", "Linux administration", "Windows administration"],
        ["CI/CD", "continuous integration", "continuous deployment"],
        ["web application security", "application security", "network security"],
        ["penetration testing", "vulnerability assessment"],
        ["digital forensics", "malware analysis"],
        ["incident response", "incident management"],
        ["Kubernetes", "Docker"],
        ["AWS", "Azure", "cloud computing"],
        ["SIEM", "SOAR"],
    ],
)
def test_ambiguous_or_different_skills_stay_separate(normalizer, terms):
    ids = [normalizer.normalize(t).skill_id for t in terms]
    assert len(set(ids)) == len(ids)


def test_every_keep_separate_group_in_config_holds(normalizer):
    assert DEFAULT_ALIAS_CONFIG.keep_separate
    for group in DEFAULT_ALIAS_CONFIG.keep_separate:
        ids = [normalizer.normalize(t).skill_id for t in group.terms]
        assert len(set(ids)) == len(ids), group.terms


def test_default_config_documents_every_mapping():
    assert all(entry.note for entry in DEFAULT_ALIAS_CONFIG.skills)
    assert all(group.reason for group in DEFAULT_ALIAS_CONFIG.keep_separate)


def test_default_config_has_no_ids_hidden_as_unknown(normalizer):
    for entry in DEFAULT_ALIAS_CONFIG.skills:
        for spelling in [entry.id, *entry.aliases]:
            result = normalizer.normalize(spelling)
            assert (result.skill_id, result.recognized) == (entry.id, True)


def test_unknown_skill_is_identifiable_and_stable(normalizer):
    first = normalizer.normalize("Quantum Basket Weaving")
    again = normalizer.normalize("quantum  basket-weaving")
    assert first.skill_id == "unknown:quantum-basket-weaving" == again.skill_id
    assert first.recognized is False
    assert first.original == "Quantum Basket Weaving"


def test_near_miss_spellings_are_not_fuzzily_merged(normalizer):
    for raw in ["SIEMs", "Amazon Web Service", "Pythonic", "Continuous Integration"]:
        assert normalizer.normalize(raw).recognized is False


def test_no_fuzzy_merging_of_related_skills(normalizer):
    ids = {normalizer.normalize(s).skill_id for s in
           ["Linux", "Linux administration", "SIEMs", "Windows administration"]}
    assert len(ids) == 4


def test_blank_skill_rejected(normalizer):
    with pytest.raises(ValueError):
        normalizer.normalize("   ")


def test_conflicting_alias_config_rejected():
    with pytest.raises(ValueError, match="maps to both"):
        SkillNormalizer({"a": ["shared"], "b": ["shared"]})


def test_invalid_canonical_id_rejected():
    with pytest.raises(ValueError, match="invalid canonical"):
        SkillNormalizer({"Bad Id": []})


def test_custom_configuration_is_honored():
    custom = SkillNormalizer({"python": ["py3"]})
    assert custom.normalize("PY3").skill_id == "python"
    assert custom.normalize("SIEM").recognized is False


def test_keep_separate_violation_rejected():
    with pytest.raises(ValueError, match="stay separate"):
        SkillNormalizer({"networking": ["tcp/ip", "computer networking"]},
                        keep_separate=[["TCP/IP", "computer networking"]])


def write_config(tmp_path, data):
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_external_config_file_can_extend_aliases(tmp_path):
    path = write_config(tmp_path, {
        "version": 1,
        "skills": [{"id": "golang", "aliases": ["Go language"], "note": "Same language."}],
    })
    custom = SkillNormalizer.from_config(load_alias_config(path))
    assert custom.normalize("go LANGUAGE").skill_id == "golang"


@pytest.mark.parametrize(
    "data",
    [
        {"version": 2, "skills": []},
        {"version": 1, "skills": [{"id": "x", "aliases": []}]},
        {"version": 1, "skills": [{"id": "x", "aliases": [], "note": "n"},
                                  {"id": "x", "aliases": [], "note": "n"}]},
        {"version": 1, "skills": [], "keep_separate": [{"terms": ["only one"], "reason": "r"}]},
        {"version": 1, "skills": [], "surprise": True},
    ],
)
def test_malformed_config_rejected(tmp_path, data):
    with pytest.raises(ValidationError):
        load_alias_config(write_config(tmp_path, data))


def test_config_that_merges_separated_terms_fails_to_build(tmp_path):
    path = write_config(tmp_path, {
        "version": 1,
        "skills": [{"id": "java", "aliases": ["JavaScript"], "note": "wrong"}],
        "keep_separate": [{"terms": ["Java", "JavaScript"], "reason": "Unrelated."}],
    })
    with pytest.raises(ValueError, match="stay separate"):
        SkillNormalizer.from_config(load_alias_config(path))


def test_matching_uses_aliases_without_engine_changes():
    job = make_job(required_skills=[SkillRequirement(skill="Amazon Web Services")])
    result = evaluate_match(
        make_candidate(), [make_evidence("AWS", assessment_score=0.8)], job
    )
    assert result.skill_alignment_score == 80.0
    assert result.breakdown[0].recognized_skill is True


def test_matching_does_not_credit_separate_skills():
    job = make_job(required_skills=[SkillRequirement(skill="computer networking")])
    result = evaluate_match(
        make_candidate(), [make_evidence("TCP/IP networking", assessment_score=1.0)], job
    )
    assert result.skill_alignment_score == 0.0
    assert result.breakdown[0].evidence_basis == "no_evidence"
