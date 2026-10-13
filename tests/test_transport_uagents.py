"""Transport (uagents.Model) wire models against the canonical domain models and uAgents."""

import json
import subprocess
import sys
import textwrap

import pytest
from pydantic.v1 import ValidationError as V1ValidationError
from uagents import Model
from uagents_core.models import Model as CoreModel

import skillbridge.transport as transport
from scripts.sync_agent_models import (
    AGENT_MODELS,
    REPO_ROOT,
    generate_block,
    load_block,
)
from skillbridge.schemas import AgentRequest, AgentResponse, ResponseStatus
from skillbridge.transport import (
    AgentRequestMessage,
    AgentResponseMessage,
    message_to_request,
    message_to_response,
    request_to_message,
    response_to_message,
)
from tests.conftest import NOW
from tests.test_agentverse_models import BEHAVIOR_CASES

SCHEMAS_SOURCE = (REPO_ROOT / "skillbridge" / "schemas.py").read_text()
REQUEST = AgentRequest(
    request_id="req-1", sender="orchestrator", recipient="candidate_evidence",
    message_type="assess", created_at=NOW, payload={"skills": ["siem"], "n": 1},
)
ERROR_RESPONSE = AgentResponse(
    request_id="req-1", sender="candidate_evidence", recipient="orchestrator",
    message_type="assess", created_at=NOW, status="error",
    error={"code": "timeout", "message": "inference timed out"},
)
OK_RESPONSE = AgentResponse(
    request_id="req-1", sender="candidate_evidence", recipient="orchestrator",
    message_type="assess", created_at=NOW, status=ResponseStatus.SUCCESS,
)


# ------------------------------------------------------ separation of concerns


def test_domain_schemas_do_not_import_uagents_or_transport():
    assert "uagents" not in SCHEMAS_SOURCE
    assert "transport" not in SCHEMAS_SOURCE


def test_domain_models_are_pydantic_v2_not_uagents_models():
    assert not issubclass(AgentRequest, Model)
    assert hasattr(AgentRequest, "model_fields")


def test_matching_logic_does_not_depend_on_transport():
    for module in ("matching", "eligibility", "normalization"):
        source = (REPO_ROOT / "skillbridge" / f"{module}.py").read_text()
        assert "transport" not in source and "uagents" not in source


# --------------------------------------------------- real uAgents integration


@pytest.mark.parametrize("model", [AgentRequestMessage, AgentResponseMessage])
def test_wire_models_are_uagents_models_with_stable_digests(model):
    assert issubclass(model, Model) and issubclass(model, CoreModel)
    digest = Model.build_schema_digest(model)
    assert digest.startswith("model:") and digest == Model.build_schema_digest(model)


def test_request_and_response_have_distinct_digests():
    assert Model.build_schema_digest(AgentRequestMessage) != Model.build_schema_digest(
        AgentResponseMessage
    )


@pytest.mark.parametrize("agent", list(AGENT_MODELS))
def test_inlined_wire_models_have_identical_digests_to_canonical(agent):
    module = load_block(generate_block(AGENT_MODELS[agent]))
    for name in ("AgentRequestMessage", "AgentResponseMessage"):
        assert Model.build_schema_digest(getattr(module, name)) == Model.build_schema_digest(
            getattr(transport, name)
        )


def test_agent_registers_handler_for_wire_model():
    from uagents import Agent

    agent = Agent(name="probe", seed="probe-seed-registration")
    agent.on_message(model=AgentRequestMessage, replies={AgentResponseMessage})(
        lambda ctx, sender, msg: None
    )
    protocol = agent._protocol
    request_digest = Model.build_schema_digest(AgentRequestMessage)
    assert request_digest in protocol._signed_message_handlers
    assert Model.build_schema_digest(AgentResponseMessage) in protocol._replies[request_digest]


EXCHANGE_SCRIPT = textwrap.dedent(
    """
    import asyncio, json, sys
    from datetime import datetime, timezone
    from pathlib import Path
    from uagents import Agent, Bureau, Context
    sys.path.insert(0, {repo!r})
    import skillbridge.transport as canonical
    from scripts.sync_agent_models import AGENT_MODELS, generate_block, load_block

    inlined = load_block(generate_block(AGENT_MODELS["agents/action_agent.py"]))
    server = Agent(name="server", seed="exchange-server-seed")
    client = Agent(name="client", seed="exchange-client-seed")
    received = []

    # server uses the inlined (hosted) classes; the client uses canonical ones.
    @server.on_message(model=inlined.AgentRequestMessage, replies={{inlined.AgentResponseMessage}})
    async def handle(ctx: Context, sender: str, msg):
        response = inlined.AgentResponseMessage(
            request_id=msg.request_id, sender="career_action", recipient="orchestrator",
            message_type=msg.message_type, created_at=datetime.now(timezone.utc),
            status="success", payload={{"echo": msg.payload}})
        await ctx.send(sender, response)

    @client.on_message(model=canonical.AgentResponseMessage)
    async def got(ctx: Context, sender: str, msg):
        received.append(msg.dict()["payload"])

    @client.on_event("startup")
    async def start(ctx: Context):
        await ctx.send(server.address, canonical.AgentRequestMessage(
            request_id="r1", sender="orchestrator", recipient="career_action",
            message_type="assess", created_at=datetime.now(timezone.utc),
            payload={{"skills": ["siem"]}}))

    async def main():
        bureau = Bureau(port=0)
        bureau.add(server); bureau.add(client)
        task = asyncio.create_task(bureau.run_async())
        for _ in range(100):
            if received:
                break
            await asyncio.sleep(0.1)
        task.cancel()
        print("RESULT " + json.dumps(received))

    asyncio.run(main())
    """
)


def test_real_uagents_exchange_between_canonical_and_inlined_models(tmp_path):
    script = tmp_path / "exchange.py"
    script.write_text(EXCHANGE_SCRIPT.format(repo=str(REPO_ROOT)), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=90
    )
    line = next((x for x in result.stdout.splitlines() if x.startswith("RESULT ")), None)
    assert line is not None, result.stdout + result.stderr
    assert json.loads(line[7:]) == [{"echo": {"skills": ["siem"]}}]


# ---------------------------------------- wire models vs canonical (behavior)


def pipeline(wire, convert, payload):
    """Raw payload -> wire model -> canonical model, as a received message travels."""
    return convert(wire.parse_obj(payload))


@pytest.mark.parametrize(
    ("domain", "wire", "convert", "name"),
    [(AgentRequest, AgentRequestMessage, message_to_request, "AgentRequest"),
     (AgentResponse, AgentResponseMessage, message_to_response, "AgentResponse")],
)
def test_wire_pipeline_accepts_and_rejects_like_the_domain_model(domain, wire, convert, name):
    for payload in BEHAVIOR_CASES[name]:
        if set(payload) - set(domain.model_fields):
            continue  # unknown keys are dropped by the wire model; covered below
        try:
            expected = domain.model_validate(payload)
        except ValueError:
            with pytest.raises(ValueError):
                pipeline(wire, convert, payload)
        else:
            assert pipeline(wire, convert, payload) == expected


def test_wire_models_use_only_plain_annotations_and_defaults():
    for wire in (AgentRequestMessage, AgentResponseMessage):
        for info in wire.__fields__.values():
            assert info.field_info.min_length is None and info.field_info.max_length is None


def test_domain_and_wire_field_names_and_requiredness_match():
    for domain, wire in ((AgentRequest, AgentRequestMessage), (AgentResponse, AgentResponseMessage)):
        assert list(domain.model_fields) == list(wire.__fields__)
        for name, info in domain.model_fields.items():
            assert info.is_required() == (wire.__fields__[name].required is True), name


def test_domain_wire_domain_round_trip_preserves_everything():
    for original in (REQUEST,):
        assert message_to_request(request_to_message(original)) == original
    for original in (ERROR_RESPONSE, OK_RESPONSE):
        assert message_to_response(response_to_message(original)) == original


def test_json_round_trip_through_the_wire_format():
    wire = request_to_message(REQUEST)
    restored = AgentRequestMessage.parse_raw(wire.json())
    assert restored == wire and message_to_request(restored) == REQUEST
    wire = response_to_message(ERROR_RESPONSE)
    restored = AgentResponseMessage.parse_raw(wire.json())
    assert message_to_response(restored) == ERROR_RESPONSE


def test_domain_json_is_accepted_by_wire_model_and_vice_versa():
    assert AgentRequestMessage.parse_raw(REQUEST.model_dump_json()).dict() == REQUEST.model_dump()
    wire_json = request_to_message(REQUEST).json()
    assert AgentRequest.model_validate_json(wire_json) == REQUEST


def test_tz_offsets_survive_conversion_as_same_instant():
    wire = AgentRequestMessage.parse_raw(request_to_message(REQUEST).json())
    assert wire.created_at == REQUEST.created_at and wire.created_at.utcoffset() is not None


@pytest.mark.parametrize(
    "bad",
    [
        {"schema_version": "2.0"},
        {"schema_version": "x"},
        {"request_id": "   "},
        {"recipient": "fourth_ai_agent"},
        {"created_at": "2026-01-01T12:00:00"},
    ],
)
def test_converter_rejects_invalid_messages_that_the_wire_model_accepts(bad):
    payload = {**REQUEST.model_dump(mode="json"), **bad}
    message = AgentRequestMessage.parse_obj(payload)
    with pytest.raises(ValueError):
        message_to_request(message)


def test_wire_model_drops_unknown_keys_and_missing_fields_fail():
    payload = {**REQUEST.model_dump(mode="json"), "unexpected": 1}
    assert message_to_request(AgentRequestMessage.parse_obj(payload)) == REQUEST
    with pytest.raises(V1ValidationError):
        AgentRequestMessage.parse_obj({k: v for k, v in payload.items() if k != "request_id"})


def test_converter_enforces_error_iff_status_error():
    base = {**OK_RESPONSE.model_dump(mode="json")}
    for bad in ({**base, "status": "error"}, {**base, "error": {"code": "x", "message": "y"}}):
        with pytest.raises(ValueError):
            message_to_response(AgentResponseMessage.parse_obj(bad))


def test_converter_revalidates_with_canonical_rules():
    wire = request_to_message(REQUEST)
    wire.schema_version = "9.9"
    with pytest.raises(ValueError):
        message_to_request(wire)


def test_whitespace_is_stripped_by_the_converter():
    payload = {**REQUEST.model_dump(mode="json"), "request_id": "  r1  "}
    assert message_to_request(AgentRequestMessage.parse_obj(payload)).request_id == "r1"
