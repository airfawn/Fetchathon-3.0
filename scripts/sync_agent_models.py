"""Generate and verify the code inlined into hosted Agentverse agent files.

Agentverse hosts each agent as one self-contained file, so agents cannot import
``skillbridge``. Each agent file instead carries a generated block, delimited
by BEGIN/END markers, holding only what that agent needs from two canonical
sources, extracted with ``ast`` (nothing is maintained by hand):

* ``skillbridge/schemas.py``   - pydantic v2 domain models and envelopes
* ``skillbridge/transport.py`` - explicit ``uagents.Model`` (pydantic v1) wire
  messages and the converters between them and the domain envelopes

Dependencies (enums, aliases, constants, helpers, imports) are resolved
automatically; ``from skillbridge...`` imports are dropped because the code
they name is inlined.

Usage (from the repository root):
    python -m scripts.sync_agent_models --write [--create-missing]
    python -m scripts.sync_agent_models --check [--require-all]

Synchronization rules:
* Every ``agents/*.py`` file is discovered automatically and verified. A file
  that is not registered in ``AGENT_MODELS`` is an error.
* A registered agent whose file is missing is reported as PENDING, never as ok.
  With ``--require-all`` (or SKILLBRIDGE_REQUIRE_AGENTS=1) it is an error.

Verification compares each inlined definition with its canonical one: fields,
types, requiredness, defaults, constraints, validators, config, enum members,
JSON Schema, uAgents schema digests, and (see ``compare_behavior``) real
validation and serialization results, because identical schemas do not prove
identical runtime behavior.
"""

import argparse
import ast
import importlib
import json
import os
import pprint
import re
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel as V2BaseModel
from pydantic.v1 import BaseModel as V1BaseModel
from pydantic.v1 import ValidationError as V1ValidationError
from pydantic_core import PydanticUndefined

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_PATH = REPO_ROOT / "skillbridge" / "schemas.py"
TRANSPORT_PATH = REPO_ROOT / "skillbridge" / "transport.py"
NORMALIZATION_PATH = REPO_ROOT / "skillbridge" / "normalization.py"
SOURCE_PATHS = (SCHEMAS_PATH, NORMALIZATION_PATH, TRANSPORT_PATH)
CANONICAL_MODULES = ("skillbridge.schemas", "skillbridge.normalization", "skillbridge.transport")
# Constants loaded from a data file; rendered as literals so hosted files need no file access.
DATA_CONSTANTS = {"SKILL_ALIAS_DATA": REPO_ROOT / "skillbridge" / "skill_aliases.json"}
BEGIN_MARKER = "# BEGIN GENERATED SKILLBRIDGE MODELS"
END_MARKER = "# END GENERATED SKILLBRIDGE MODELS"
FORBIDDEN_IMPORT_ROOTS = {"skillbridge", "scripts", "tests"}
AGENTS_DIR = "agents"
REQUIRE_ALL_ENV = "SKILLBRIDGE_REQUIRE_AGENTS"

_TRANSPORT_NAMES = (
    "AgentRequestMessage",
    "AgentResponseMessage",
    "_as_dict",
    "request_to_message",
    "message_to_request",
    "response_to_message",
    "message_to_response",
)

# Names each hosted agent file needs; dependencies are resolved automatically.
# Revise when an agent is implemented.
AGENT_MODELS: dict[str, tuple[str, ...]] = {
    "agents/opportunity_agent.py": (
        "OpportunitySearchRequest",
        "OpportunitySearchResult",
        "OPPORTUNITY_SEARCH_MESSAGE",
        "OPPORTUNITY_RESULT_MESSAGE",
        "SkillNormalizer",
        "SKILL_ALIAS_DATA",
        "canonical_text",
        *_TRANSPORT_NAMES,
    ),
    "agents/evidence_agent.py": (
        "EvidenceProfileRequest",
        "EvidenceProfileResult",
        "EVIDENCE_PROFILE_MESSAGE",
        "EVIDENCE_RESULT_MESSAGE",
        "SkillNormalizer",
        "SKILL_ALIAS_DATA",
        "canonical_text",
        *_TRANSPORT_NAMES,
    ),
    "agents/action_agent.py": ("MatchResult", "CareerAction", *_TRANSPORT_NAMES),
}

_BLOCK_PATTERN = re.compile(
    rf"^{re.escape(BEGIN_MARKER)}[^\n]*\n.*?^{re.escape(END_MARKER)}[^\n]*$",
    re.DOTALL | re.MULTILINE,
)
_MODULE_PREFIX = re.compile(r"\b(?:[A-Za-z_]\w*\.)+(?=[A-Za-z_])")


# ------------------------------------------------------------- generation


@dataclass(frozen=True)
class _Definition:
    names: tuple[str, ...]
    source: str
    refs: frozenset[str]
    is_assignment: bool


@dataclass(frozen=True)
class _Import:
    kind: str  # "import" or "from"
    module: str
    name: str
    alias: str | None

    @property
    def bound(self) -> str:
        return self.alias or self.name.split(".")[0]

    @property
    def is_internal(self) -> bool:
        return self.kind == "from" and self.module.split(".")[0] == "skillbridge"


def _assigned_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Assign):
        return tuple(t.id for t in node.targets if isinstance(t, ast.Name))
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return (node.target.id,)
    return ()


def _top_level_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.append(node.name)
        else:
            names.extend(_assigned_names(node))
    return names


def _make_definition(
    node: ast.AST, names: tuple[str, ...], lines: list[str], start: int, is_assignment: bool
) -> _Definition:
    refs = frozenset(n.id for n in ast.walk(node) if isinstance(n, ast.Name))
    source = "\n".join(lines[start - 1 : node.end_lineno])
    return _Definition(names, source, refs, is_assignment)


def _data_definition(name: str) -> _Definition:
    data = json.loads(DATA_CONSTANTS[name].read_text(encoding="utf-8"))
    literal = pprint.pformat(data, width=100, sort_dicts=False)
    return _Definition((name,), f"{name} = {literal}", frozenset(), True)


@lru_cache(maxsize=None)
def _parse_source(path: Path) -> tuple[tuple[_Import, ...], tuple[_Definition, ...]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    imports: list[_Import] = []
    definitions: list[_Definition] = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Import):
            imports += [_Import("import", a.name, a.name, a.asname) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports += [_Import("from", node.module, a.name, a.asname) for a in node.names]
        elif isinstance(node, ast.ClassDef | ast.FunctionDef):
            start = min([d.lineno for d in node.decorator_list], default=node.lineno)
            definitions.append(_make_definition(node, (node.name,), lines, start, False))
        elif names := _assigned_names(node):
            definition = _make_definition(node, names, lines, node.lineno, True)
            if names[0] in DATA_CONSTANTS:
                definition = _data_definition(names[0])
            definitions.append(definition)
    return tuple(imports), tuple(definitions)


def _collect_sources(
    paths: tuple[Path, ...],
) -> tuple[list[_Import], tuple[_Definition, ...]]:
    imports: list[_Import] = []
    definitions: list[_Definition] = []
    seen: set[str] = set()
    for path in paths:
        file_imports, file_definitions = _parse_source(path)
        imports += [i for i in file_imports if not i.is_internal and i not in imports]
        for definition in file_definitions:
            clash = seen.intersection(definition.names)
            if clash:
                raise ValueError(f"{path.name}: name defined in more than one source: {clash}")
            seen.update(definition.names)
            definitions.append(definition)
    return imports, tuple(definitions)


def _needed_definitions(
    definitions: tuple[_Definition, ...], wanted: tuple[str, ...]
) -> list[_Definition]:
    by_name = {name: d for d in definitions for name in d.names}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise KeyError(f"not defined in canonical sources: {unknown}")
    needed: set[str] = set()
    stack = list(wanted)
    while stack:
        name = stack.pop()
        if name in needed or name not in by_name:
            continue
        needed.add(name)
        stack.extend(by_name[name].refs)
    return [d for d in definitions if needed.intersection(d.names)]


def _render_imports(imports: list[_Import]) -> str:
    groups: dict[str, list[str]] = {}
    lines: list[str] = []
    for item in imports:
        label = item.name if item.alias is None else f"{item.name} as {item.alias}"
        if item.kind == "import":
            lines.append(f"import {label}")
        else:
            groups.setdefault(item.module, []).append(label)
    for module, labels in groups.items():
        if len(labels) == 1:
            lines.append(f"from {module} import {labels[0]}")
        else:
            body = "".join(f"    {label},\n" for label in labels)
            lines.append(f"from {module} import (\n{body})")
    return "\n".join(lines)


# Hosted runtimes older than Python 3.11 lack enum.StrEnum.
STRENUM_COMPAT = """try:
    from enum import StrEnum
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)"""


def generate_block(
    model_names: tuple[str, ...], source_paths: tuple[Path, ...] = SOURCE_PATHS
) -> str:
    imports, definitions = _collect_sources(tuple(source_paths))
    selected = _needed_definitions(definitions, tuple(model_names))
    refs = set().union(*(d.refs for d in selected))
    used_imports = [i for i in imports if i.bound in refs]
    body = selected[0].source
    for previous, current in zip(selected, selected[1:]):
        both_assignments = previous.is_assignment and current.is_assignment
        body += ("\n" if both_assignments else "\n\n\n") + current.source
    header = f"{BEGIN_MARKER} (scripts/sync_agent_models.py; do not edit by hand)"
    imports_text = _render_imports(used_imports)
    if "from enum import StrEnum" in imports_text:
        imports_text = imports_text.replace("from enum import StrEnum", STRENUM_COMPAT, 1)
    return f"{header}\n{imports_text}\n\n\n{body}\n{END_MARKER}"


def extract_block(source: str) -> str:
    matches = [m.group(0) for m in _BLOCK_PATTERN.finditer(source)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one generated model block, found {len(matches)}")
    return matches[0]


def inject_block(source: str, block: str) -> str:
    extract_block(source)
    return _BLOCK_PATTERN.sub(lambda _: block, source, count=1)


def new_agent_source(block: str) -> str:
    """Minimal file holding only the generated block; agent code is added separately."""
    return f'"""Hosted Agentverse agent (self-contained single file)."""\n\n{block}\n'


# ------------------------------------------------------------ verification


def forbidden_imports(source: str) -> list[str]:
    problems = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [("." * node.level) + (node.module or "")]
        else:
            continue
        for module in modules:
            if module.startswith(".") or module.split(".")[0] in FORBIDDEN_IMPORT_ROOTS:
                problems.append(f"hosted file must not import {module!r}")
    return problems


_module_counter = 0


def load_block(block: str) -> types.ModuleType:
    """Execute a block in an isolated module namespace."""
    global _module_counter
    _module_counter += 1
    module = types.ModuleType(f"agentverse_inlined_{_module_counter}")
    sys.modules[module.__name__] = module
    exec(compile(block, f"<{module.__name__}>", "exec"), module.__dict__)
    return module


def _norm(text: str) -> str:
    return _MODULE_PREFIX.sub("", text)


def _callable_names(items: Any) -> list[str]:
    names = []
    for item in items:
        func = item[1] if isinstance(item, tuple) else item
        names.append(getattr(getattr(func, "func", func), "__name__", repr(func)))
    return names


def _v2_field_summary(info: Any) -> dict[str, Any]:
    factory = info.default_factory
    return {
        "annotation": _norm(str(info.annotation)),
        "required": info.is_required(),
        "default": None if info.default is PydanticUndefined else repr(info.default),
        "default_factory_value": None if factory is None else repr(factory()),
        "metadata": _norm(repr(info.metadata)),
        "alias": (info.alias, info.validation_alias, info.serialization_alias),
        "description": info.description,
        "json_schema_extra": repr(info.json_schema_extra),
    }


def _v2_validator_summary(cls: type[V2BaseModel]) -> dict[str, Any]:
    decorators = cls.__pydantic_decorators__
    return {
        "model": sorted((k, v.info.mode) for k, v in decorators.model_validators.items()),
        "field": sorted(
            (k, tuple(v.info.fields), v.info.mode) for k, v in decorators.field_validators.items()
        ),
    }


_V1_CONSTRAINTS = ("min_length", "max_length", "gt", "ge", "lt", "le", "regex", "multiple_of")
_V1_CONFIG = ("extra", "anystr_strip_whitespace", "validate_assignment", "allow_mutation")


def _v1_field_summary(info: Any) -> dict[str, Any]:
    factory = info.default_factory
    return {
        "type": _norm(repr(info.outer_type_)),
        "required": info.required is True,
        "allow_none": info.allow_none,
        "default": repr(info.default),
        "default_factory_value": None if factory is None else repr(factory()),
        "constraints": {k: getattr(info.field_info, k, None) for k in _V1_CONSTRAINTS},
        "alias": info.alias,
    }


def _v1_validator_summary(cls: type[V1BaseModel]) -> dict[str, Any]:
    return {
        "field": sorted(
            (name, tuple(_callable_names(validators)))
            for name, validators in cls.__validators__.items()
        ),
        "pre_root": _callable_names(cls.__pre_root_validators__),
        "post_root": _callable_names(cls.__post_root_validators__),
    }


def _diff_checker(name: str, diffs: list[str]):
    def check(label: str, left: Any, right: Any) -> None:
        if left != right:
            diffs.append(f"{name}: {label} differs: canonical={left!r} inlined={right!r}")

    return check


def _compare_v2_model(name: str, canonical: type[V2BaseModel], inlined: type) -> list[str]:
    diffs: list[str] = []
    check = _diff_checker(name, diffs)
    check("bases", [c.__name__ for c in canonical.__mro__], [c.__name__ for c in inlined.__mro__])
    check("model_config", dict(canonical.model_config), dict(inlined.model_config))
    check("field names", list(canonical.model_fields), list(inlined.model_fields))
    for field_name in canonical.model_fields.keys() & inlined.model_fields.keys():
        check(
            f"field {field_name!r}",
            _v2_field_summary(canonical.model_fields[field_name]),
            _v2_field_summary(inlined.model_fields[field_name]),
        )
    check("validators", _v2_validator_summary(canonical), _v2_validator_summary(inlined))
    for mode in ("validation", "serialization"):
        check(
            f"{mode} JSON Schema",
            canonical.model_json_schema(mode=mode),
            inlined.model_json_schema(mode=mode),
        )
    return diffs


def _compare_v1_model(name: str, canonical: type[V1BaseModel], inlined: type) -> list[str]:
    diffs: list[str] = []
    check = _diff_checker(name, diffs)
    check("bases", [c.__name__ for c in canonical.__mro__], [c.__name__ for c in inlined.__mro__])
    check(
        "config",
        {k: getattr(canonical.__config__, k) for k in _V1_CONFIG},
        {k: getattr(inlined.__config__, k) for k in _V1_CONFIG},
    )
    check("field names", list(canonical.__fields__), list(inlined.__fields__))
    for field_name in canonical.__fields__.keys() & inlined.__fields__.keys():
        check(
            f"field {field_name!r}",
            _v1_field_summary(canonical.__fields__[field_name]),
            _v1_field_summary(inlined.__fields__[field_name]),
        )
    check("validators", _v1_validator_summary(canonical), _v1_validator_summary(inlined))
    check("JSON Schema", canonical.schema(), inlined.schema())
    if hasattr(canonical, "build_schema_digest"):
        check(
            "uAgents schema digest",
            canonical.build_schema_digest(canonical),
            inlined.build_schema_digest(inlined),
        )
    return diffs


def _compare_enum(name: str, canonical: type[Enum], inlined: type[Enum]) -> list[str]:
    diffs = []
    left = [(m.name, m.value) for m in canonical]
    right = [(m.name, m.value) for m in inlined]
    if left != right:
        diffs.append(f"{name}: enum members differ: canonical={left!r} inlined={right!r}")
    if [c.__name__ for c in canonical.__mro__] != [c.__name__ for c in inlined.__mro__]:
        diffs.append(f"{name}: enum base classes differ")
    return diffs


def canonical_definition(name: str) -> Any:
    for module_name in CANONICAL_MODULES:
        module = importlib.import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    return None


def compare_block(block: str) -> list[str]:
    """Compare every name defined by the block with its canonical definition."""
    module = load_block(block)
    diffs: list[str] = []
    for name in _top_level_names(ast.parse(block)):
        inlined = getattr(module, name)
        canonical = canonical_definition(name)
        if canonical is None:
            diffs.append(f"{name}: not defined in a canonical module")
        elif isinstance(inlined, type) and issubclass(inlined, V1BaseModel):
            diffs += _compare_v1_model(name, canonical, inlined)
        elif isinstance(inlined, type) and issubclass(inlined, V2BaseModel):
            diffs += _compare_v2_model(name, canonical, inlined)
        elif isinstance(inlined, type) and issubclass(inlined, Enum):
            diffs += _compare_enum(name, canonical, inlined)
        elif callable(inlined) and not isinstance(inlined, type):
            continue  # helper functions are verified through model behavior
        elif _norm(repr(canonical)) != _norm(repr(inlined)):
            diffs.append(f"{name}: value differs: canonical={canonical!r} inlined={inlined!r}")
    return diffs


def _error_summary(error: ValueError) -> list[tuple[Any, ...]]:
    if isinstance(error, V1ValidationError):
        return [(e["type"], tuple(e["loc"]), e["msg"]) for e in error.errors()]
    errors = error.errors(include_url=False, include_context=False, include_input=False)
    return [(e["type"], tuple(e["loc"]), e["msg"]) for e in errors]


def _validate(model: type, payload: dict[str, Any]) -> tuple[str, Any]:
    try:
        return "ok", model.model_validate(payload)
    except ValueError as error:  # both pydantic generations raise ValueError subclasses
        return "error", _error_summary(error)


def _json_data(instance: Any) -> Any:
    return json.loads(instance.model_dump_json())


def compare_behavior(canonical: type, inlined: type, payload: dict[str, Any]) -> list[str]:
    """Same generation of model: identical accept/reject, errors, JSON and round trips."""
    name = canonical.__name__
    left_status, left = _validate(canonical, payload)
    right_status, right = _validate(inlined, payload)
    if left_status != right_status:
        return [f"{name}: canonical {left_status} but inlined {right_status} for {payload!r}"]
    if left_status == "error":
        return [] if left == right else [f"{name}: validation errors differ: {left!r} vs {right!r}"]
    diffs = []
    if _json_data(left) != _json_data(right):
        diffs.append(f"{name}: serialized output differs for {payload!r}")
    if _json_data(inlined.model_validate_json(left.model_dump_json())) != _json_data(left):
        diffs.append(f"{name}: inlined model does not round-trip canonical JSON")
    if _json_data(canonical.model_validate_json(right.model_dump_json())) != _json_data(right):
        diffs.append(f"{name}: canonical model does not round-trip inlined JSON")
    return diffs


def compare_acceptance(domain: type, wire: type, payload: dict[str, Any]) -> list[str]:
    """Domain (pydantic v2) and wire (pydantic v1) models must accept and reject the
    same payloads and, when accepted, hold the same values."""
    label = f"{domain.__name__}/{wire.__name__}"
    domain_status, domain_value = _validate(domain, payload)
    wire_status, wire_value = _validate(wire, payload)
    if domain_status != wire_status:
        return [f"{label}: domain {domain_status} but wire {wire_status} for {payload!r}"]
    if domain_status == "ok" and domain_value.model_dump() != wire_value.dict():
        return [f"{label}: accepted values differ for {payload!r}"]
    return []


# ------------------------------------------------- agent files and reporting


@dataclass
class VerificationReport:
    checked: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    problems: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def add_problem(self, path: str, message: str) -> None:
        self.problems.setdefault(path, []).append(message)


def discover_agent_files(root: Path = REPO_ROOT) -> list[str]:
    directory = Path(root) / AGENTS_DIR
    if not directory.is_dir():
        return []
    return [
        f"{AGENTS_DIR}/{path.name}"
        for path in sorted(directory.glob("*.py"))
        if path.name != "__init__.py"
    ]


def check_agent_file(path: Path, model_names: tuple[str, ...]) -> list[str]:
    source = Path(path).read_text(encoding="utf-8")
    problems = forbidden_imports(source)
    try:
        block = extract_block(source)
    except ValueError as error:
        return problems + [str(error)]
    if block != generate_block(model_names):
        problems.append("generated model block is out of date; run --write")
    return problems + compare_block(block)


def verify_agents(
    root: Path = REPO_ROOT,
    manifest: dict[str, tuple[str, ...]] | None = None,
    require_all: bool = False,
) -> VerificationReport:
    manifest = AGENT_MODELS if manifest is None else manifest
    report = VerificationReport()
    discovered = discover_agent_files(root)
    for relative in discovered:
        if relative not in manifest:
            report.add_problem(relative, "agent file is not registered in AGENT_MODELS")
            continue
        report.checked.append(relative)
        for problem in check_agent_file(Path(root) / relative, manifest[relative]):
            report.add_problem(relative, problem)
    for relative in manifest:
        if relative in discovered:
            continue
        if require_all:
            report.add_problem(relative, "expected agent file is missing")
        else:
            report.pending.append(relative)
    return report


def write_agents(
    root: Path = REPO_ROOT,
    manifest: dict[str, tuple[str, ...]] | None = None,
    create_missing: bool = False,
) -> list[str]:
    """Refresh the generated block in existing agent files; optionally create missing ones."""
    manifest = AGENT_MODELS if manifest is None else manifest
    written = []
    for relative, names in manifest.items():
        path = Path(root) / relative
        block = generate_block(names)
        if path.exists():
            path.write_text(inject_block(path.read_text(encoding="utf-8"), block), encoding="utf-8")
        elif create_missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_agent_source(block), encoding="utf-8")
        else:
            continue
        written.append(relative)
    return written


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="refresh blocks in agent files")
    mode.add_argument("--check", action="store_true", help="verify blocks against canonical models")
    parser.add_argument("--create-missing", action="store_true",
                        help="with --write, create registered agent files that do not exist")
    parser.add_argument("--require-all", action="store_true",
                        help="treat missing registered agent files as errors")
    args = parser.parse_args(argv)
    require_all = args.require_all or os.environ.get(REQUIRE_ALL_ENV) == "1"

    if args.write:
        for relative in write_agents(root, create_missing=args.create_missing):
            print(f"written: {relative}")
    report = verify_agents(root, require_all=require_all)
    for relative in report.checked:
        print(f"{'FAILED' if relative in report.problems else 'ok'}: {relative}")
    for relative in report.pending:
        print(f"PENDING (file not present): {relative}")
    for relative, problems in report.problems.items():
        print(f"FAILED: {relative}")
        for problem in problems:
            print(f"  - {problem}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
