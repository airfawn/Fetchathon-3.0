"""uAgents wire-format messages for the SkillBridge inter-agent envelopes.

Transport only: this module holds no business logic. The wire models are plain
``uagents.Model`` classes with type annotations and defaults only, so they work
with whichever pydantic generation the installed (or Agentverse-hosted) uagents
uses: no ``Field``, validators or ``Config``. All validation is done by the
canonical models in ``schemas.py`` when the converters run, so a received
message is never trusted as-is. ``tests/`` and ``scripts/sync_agent_models.py``
verify that the wire models stay in step with the canonical ones.

Everything here is inlined into hosted Agentverse agent files by the generator.
"""

from datetime import datetime
from typing import Any, Optional

from uagents import Model

from skillbridge.schemas import SCHEMA_VERSION, AgentRequest, AgentResponse


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
