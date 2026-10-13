"""Candidate Evidence Agent tests: deterministic fixtures, no network, no LLM."""

import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from agents import evidence_agent as ea
from skillbridge.schemas import AgentName, AgentRequest

AS_OF = datetime(2025, 6, 1, tzinfo=timezone.utc)
RECENT = AS_OF - timedelta(days=30)
OLD = AS_OF - timedelta(days=1000)
EC, CF = ea.EvidenceClass, ea.ConfidenceLabel


def profile(claimed=("python",)):
    return {"candidate_id": "c1", "target_role": "SOC Analyst", "preferred_location": "Remote",
            "experience_level": "entry_level", "claimed_skills": list(claimed), "learning_hours_per_week": 5}


def assessment(rid="a1", skills=("python",), score=0.8, when=RECENT, candidate="c1", **extra):
    return {"record_id": rid, "candidate_id": candidate, "skills_evaluated": list(skills), "score": score,
            "taken_at": when.isoformat(), "provider": "fixture-provider", **extra}


def task(rid="t1", skills=("python",), score=0.9, when=RECENT, **extra):
    return {"record_id": rid, "candidate_id": "c1", "skills_evaluated": list(skills),
            "outcome_score": score, "completed_at": when.isoformat(), **extra}


def project(rid="p1", skills=("python",), ref="https://git.example.com/a", when=RECENT, **extra):
    return {"record_id": rid, "candidate_id": "c1", "reference": ref, "skills_demonstrated": list(skills),
            "submitted_at": when.isoformat(), **extra}


def run(claimed=("python",), declared=(), assessments=(), tasks=(), projects=(), requested=(), registry=None):
    request = ea.EvidenceProfileRequest.model_validate({
        "profile": profile(claimed), "declared_skills": list(declared), "assessments": list(assessments),
        "practical_tasks": list(tasks), "projects": list(projects), "requested_skills": list(requested),
        "as_of": AS_OF.isoformat()})
    return ea.build_evidence_profile(request, registry=registry)


def skill(result, sid):
    return next(s for s in result.skills if s.skill_id == sid)


def test_unsupported_claim_is_self_report_only_and_needs_assessment():
    s = skill(run(), "python")
    assert s.evidence_classes == [EC.SELF_REPORTED] and s.confidence == CF.SELF_REPORT_ONLY
    assert s.needs_assessment and "python" in s.recommended_assessment
    assert any("self-reported only" in x for x in s.limitations)


def test_passing_assessment_backs_claim():
    s = skill(run(assessments=[assessment()]), "python")
    assert s.evidence_classes == [EC.SELF_REPORTED, EC.ASSESSMENT_BACKED]
    assert s.confidence == CF.SINGLE_SOURCE and not s.needs_assessment and s.recommended_assessment is None


def test_multiple_evidence_types_are_corroborated_without_ranking():
    s = skill(run(assessments=[assessment()], tasks=[task()], projects=[project()]), "python")
    assert {EC.ASSESSMENT_BACKED, EC.PRACTICAL_TASK_BACKED, EC.PROJECT_BACKED} <= set(s.evidence_classes)
    assert s.confidence == CF.CORROBORATED and len(s.references) == 3


def test_project_alone_is_not_independent_proof():
    s = skill(run(projects=[project()]), "python")
    assert EC.PROJECT_BACKED in s.evidence_classes and EC.INDEPENDENTLY_VERIFIED not in s.evidence_classes
    assert s.needs_assessment
    assert any("authorship" in x for x in s.limitations + s.missing_evidence)


def test_verified_project_with_verified_authorship_is_independent():
    p = project(verification_status="verified", verified_by="mentor-1", authorship_verified=True)
    s = skill(run(projects=[p]), "python")
    assert EC.INDEPENDENTLY_VERIFIED in s.evidence_classes and not s.needs_assessment


def test_verified_project_without_authorship_check_is_not_independent():
    p = project(verification_status="verified", verified_by="mentor-1")
    assert EC.INDEPENDENTLY_VERIFIED not in skill(run(projects=[p]), "python").evidence_classes


def test_self_verification_is_not_independent():
    s = skill(run(assessments=[assessment(verification_status="verified", verified_by="c1")]), "python")
    assert EC.INDEPENDENTLY_VERIFIED not in s.evidence_classes
    assert any("not independent" in x for x in s.limitations)


def test_failed_authorship_project_is_ignored():
    s = skill(run(projects=[project(authorship_verified=False)]), "python")
    assert EC.PROJECT_BACKED not in s.evidence_classes
    assert s.references[0].role == ea.EvidenceRole.IGNORED


def test_model_interpreted_project_is_insufficient():
    s = skill(run(projects=[project(model_interpreted=True)]), "python")
    assert EC.PROJECT_BACKED not in s.evidence_classes
    assert s.references[0].role == ea.EvidenceRole.INSUFFICIENT and "unverified" in s.references[0].note


def test_below_threshold_assessment_is_insufficient_not_incompetence():
    s = skill(run(assessments=[assessment(score=0.3)]), "python")
    assert EC.ASSESSMENT_BACKED not in s.evidence_classes and s.needs_assessment
    assert any("not proof the skill is absent" in x for x in s.limitations)


def test_stale_assessment_is_insufficient():
    s = skill(run(assessments=[assessment(when=OLD)]), "python")
    assert EC.ASSESSMENT_BACKED not in s.evidence_classes and "older than" in s.references[0].note


def test_task_without_outcome_is_insufficient():
    s = skill(run(tasks=[task(score=None)]), "python")
    assert EC.PRACTICAL_TASK_BACKED not in s.evidence_classes


def test_rejected_record_is_ignored():
    s = skill(run(assessments=[assessment(verification_status="rejected")]), "python")
    assert s.references[0].role == ea.EvidenceRole.IGNORED and s.needs_assessment


def test_evidence_for_other_skill_is_not_attributed():
    result = run(assessments=[assessment(skills=("sql",))])
    assert EC.ASSESSMENT_BACKED not in skill(result, "python").evidence_classes
    assert EC.ASSESSMENT_BACKED in skill(result, "sql").evidence_classes


def test_cited_record_that_does_not_cover_skill_is_flagged():
    declared = [{"skill": "python", "cited_record_ids": ["a1"]}]
    s = skill(run(declared=declared, assessments=[assessment(skills=("sql",))]), "python")
    assert any("does not evaluate this skill" in x for x in s.limitations)


def test_missing_cited_record_is_reported():
    s = skill(run(declared=[{"skill": "python", "cited_record_ids": ["nope"]}]), "python")
    assert any("'nope' does not exist" in x for x in s.missing_evidence)


def test_other_candidates_record_is_rejected_as_invalid_reference():
    result = run(declared=[{"skill": "python", "cited_record_ids": ["a1"]}],
                 assessments=[assessment(candidate="someone-else")])
    s = skill(result, "python")
    assert EC.ASSESSMENT_BACKED not in s.evidence_classes
    assert any("different candidate" in i for i in result.issues)
    assert any("could not be used" in x for x in s.missing_evidence)


def test_future_dated_record_is_ignored():
    result = run(assessments=[assessment(when=AS_OF + timedelta(days=5))])
    assert any("after as_of" in i for i in result.issues)


def test_identical_duplicate_counts_once():
    result = run(assessments=[assessment(), assessment()])
    assert len(skill(result, "python").references) == 1
    assert any("identical; counted once" in i for i in result.issues)


def test_conflicting_duplicate_id_is_flagged():
    result = run(assessments=[assessment(), assessment(score=0.2)])
    assert any("different content" in i for i in result.issues)
    assert skill(result, "python").evidence_classes[-1] == EC.ASSESSMENT_BACKED


def test_same_underlying_evidence_under_two_ids_counts_once():
    result = run(assessments=[assessment("a1"), assessment("a2")])
    assert len(skill(result, "python").references) == 1
    assert any("duplicates 'a1'" in i for i in result.issues)


def test_aliases_are_normalized_and_unknown_skills_stay_identifiable():
    result = run(claimed=("Amazon Web Services", "Quantum Basket Weaving"),
                 assessments=[assessment(skills=("AWS",))])
    assert EC.ASSESSMENT_BACKED in skill(result, "aws").evidence_classes
    unknown = next(s for s in result.skills if not s.recognized)
    assert "quantum" in unknown.skill_id and any("alias vocabulary" in x for x in unknown.limitations)


def test_requested_unclaimed_skill_reports_no_evidence():
    s = skill(run(claimed=(), requested=("sql",)), "sql")
    assert s.evidence_classes == [EC.NO_EVIDENCE] and s.confidence == CF.NONE and s.needs_assessment
    assert not s.declared


def test_declared_proficiency_is_preserved():
    s = skill(run(declared=[{"skill": "python", "proficiency": "advanced"}]), "python")
    assert s.declared_proficiency == ea.ProficiencyLevel.ADVANCED


def test_results_are_reproducible():
    args = dict(assessments=[assessment()], projects=[project()])
    assert run(**args).model_dump_json() == run(**args).model_dump_json()


# ------------------------------------------------------------- registry


def test_registry_registers_and_feeds_the_profile():
    reg = ea.AssessmentRegistry()
    reg.register(assessment())
    s = skill(run(registry=reg), "python")
    assert EC.ASSESSMENT_BACKED in s.evidence_classes
    assert reg.for_candidate("other") == []


def test_registry_rejects_invalid_and_conflicting_records():
    reg = ea.AssessmentRegistry()
    with pytest.raises(Exception):
        reg.register(assessment(score=2))
    reg.register(assessment())
    reg.register(assessment())  # identical re-registration is idempotent
    with pytest.raises(ValueError):
        reg.register(assessment(score=0.1))


# ------------------------------------------------- schema validation


def test_verified_record_must_name_verifier():
    with pytest.raises(Exception):
        ea.AssessmentRecord.model_validate(assessment(verification_status="verified"))


def test_naive_as_of_and_extra_fields_rejected():
    with pytest.raises(Exception):
        ea.EvidenceProfileRequest.model_validate({"profile": profile(), "as_of": "2025-01-01T00:00:00"})
    with pytest.raises(Exception):
        ea.EvidenceProfileRequest.model_validate({"profile": profile(), "as_of": AS_OF.isoformat(), "x": 1})


# --------------------------------------------------- envelope handler


def envelope(payload, recipient=AgentName.CANDIDATE_EVIDENCE, message_type=ea.EVIDENCE_PROFILE_MESSAGE):
    return AgentRequest(request_id="r1", sender=AgentName.ORCHESTRATOR, recipient=recipient,
                        message_type=message_type, created_at=AS_OF, payload=payload)


def good_payload():
    return {"profile": profile(), "as_of": AS_OF.isoformat(), "assessments": [assessment()]}


def test_handler_success_returns_validated_result():
    response = ea.handle_evidence_request(ea.message_to_request(ea.request_to_message(envelope(good_payload()))))
    assert response.status == "success" and response.message_type == ea.EVIDENCE_RESULT_MESSAGE
    result = ea.EvidenceProfileResult.model_validate(response.payload)
    assert result.candidate_id == "c1"


@pytest.mark.parametrize("kwargs,code", [
    ({"recipient": AgentName.OPPORTUNITY_INTELLIGENCE}, "wrong_recipient"),
    ({"message_type": "opportunity.search"}, "unsupported_message_type"),
    ({"payload": {"profile": {}}}, "invalid_request"),
])
def test_handler_errors_are_controlled(kwargs, code):
    args = {"payload": good_payload(), **kwargs}
    response = ea.handle_evidence_request(envelope(**args))
    assert response.status == "error" and response.error.code == code and not response.payload


# ------------------------------------------------------------- uAgents


def test_agent_registers_handler_for_shared_wire_messages():
    from uagents import Model

    protocol = ea.agent._protocol
    digest = Model.build_schema_digest(ea.AgentRequestMessage)
    assert digest in protocol._signed_message_handlers
    assert Model.build_schema_digest(ea.AgentResponseMessage) in protocol._replies[digest]


def test_agent_file_is_self_contained_and_imports_in_isolation(tmp_path):
    from scripts.sync_agent_models import REPO_ROOT, forbidden_imports

    source = (REPO_ROOT / "agents" / "evidence_agent.py").read_text()
    assert forbidden_imports(source) == []
    (tmp_path / "isolated_agent.py").write_text(source)
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         f"import sys; sys.path.insert(0, {str(tmp_path)!r}); import isolated_agent as m; print('OK', m.AGENT_NAME)"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, env={"PATH": ""})
    assert "OK skillbridge_candidate_evidence" in result.stdout, result.stderr


def test_wire_handler_replies_through_ctx():
    import asyncio

    sent = []

    class Ctx:
        class logger:
            warning = staticmethod(lambda *_: None)

        async def send(self, to, msg):
            sent.append((to, msg))

    wire = ea.request_to_message(envelope(good_payload()))
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(ea.on_evidence_request(Ctx(), "agent1xyz", wire))
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())
    assert sent[0][0] == "agent1xyz"
    assert ea.message_to_response(sent[0][1]).status == "success"


# ---------------------------------------------------------- Agentverse chat


def chat_payload():
    import json

    return json.dumps(good_payload())


def test_chat_request_invokes_deterministic_evidence_service(monkeypatch):
    calls = []
    original = ea.build_evidence_profile

    def tracked(request, **kwargs):
        calls.append(request.profile.candidate_id)
        return original(request, **kwargs)

    monkeypatch.setattr(ea, "build_evidence_profile", tracked)
    reply = ea.answer_chat(chat_payload())
    assert calls == ["c1"]
    assert '"candidate_id": "c1"' in reply and '"assessment_backed"' in reply
    assert "not a hiring decision" in reply


def test_chat_invalid_json_and_plain_language_do_not_fabricate_results():
    malformed = ea.answer_chat("{not json")
    plain = ea.answer_chat("I am great at Python, please verify me")
    assert "couldn't validate" in malformed
    assert "couldn't validate" in plain
    assert "Scores and verification" in malformed


def test_chat_returns_controlled_error_if_service_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("internal detail")

    monkeypatch.setattr(ea, "build_evidence_profile", fail)
    reply = ea.answer_chat(chat_payload())
    assert reply == ea.CHAT_ERROR_TEXT
    assert "internal detail" not in reply


def test_chat_protocol_is_registered_on_installed_uagents():
    from uagents import Model
    from uagents_core.contrib.protocols.chat import (
        ChatAcknowledgement,
        ChatMessage,
        chat_protocol_spec,
    )

    assert ea.chat_proto.spec.name == chat_protocol_spec.name == "AgentChatProtocol"
    assert chat_protocol_spec.version == ea.chat_proto.spec.version
    assert chat_protocol_spec.name in {p.name for p in ea.agent.protocols.values()}
    handlers = ea.chat_proto._signed_message_handlers
    assert Model.build_schema_digest(ChatMessage) in handlers
    assert Model.build_schema_digest(ChatAcknowledgement) in handlers


def test_chat_handler_acknowledges_and_replies_with_profile():
    import asyncio
    from uuid import uuid4
    from uagents_core.contrib.protocols.chat import ChatAcknowledgement, ChatMessage, TextContent

    sent = []

    class Ctx:
        class logger:
            warning = staticmethod(lambda *_: None)
            error = staticmethod(lambda *_: None)
            info = staticmethod(lambda *_: None)

        async def send(self, destination, message):
            sent.append((destination, message))

    incoming = ChatMessage(
        timestamp=AS_OF,
        msg_id=uuid4(),
        content=[TextContent(type="text", text=chat_payload())],
    )
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(ea.on_chat_message(Ctx(), "agent1qchat", incoming))
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())

    assert len(sent) == 2 and all(destination == "agent1qchat" for destination, _ in sent)
    acknowledgement, reply = sent[0][1], sent[1][1]
    assert isinstance(acknowledgement, ChatAcknowledgement)
    assert acknowledgement.acknowledged_msg_id == incoming.msg_id
    assert isinstance(reply, ChatMessage) and isinstance(reply.content[0], TextContent)
    assert '"candidate_id": "c1"' in reply.content[0].text


def test_chat_handler_survives_service_exception():
    import asyncio
    from uuid import uuid4
    from uagents_core.contrib.protocols.chat import ChatMessage, TextContent

    sent = []

    class Ctx:
        class logger:
            warning = staticmethod(lambda *_: None)
            error = staticmethod(lambda *_: None)

        async def send(self, destination, message):
            sent.append(message)

    incoming = ChatMessage(
        timestamp=AS_OF, msg_id=uuid4(),
        content=[TextContent(type="text", text="{bad json")],
    )
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(ea.on_chat_message(Ctx(), "agent1qchat", incoming))
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())
    assert len(sent) == 2 and isinstance(sent[-1], ChatMessage)
    assert "couldn't validate" in sent[-1].content[0].text
