"""Controlled skill-alias normalization.

Only explicitly configured aliases are merged. There is no fuzzy matching,
so unrelated skills are never combined. Skills without a configured alias
keep a stable ``unknown:<text>`` identifier.

Aliases live in ``skill_aliases.json`` so they can be extended without
touching code. Every entry carries a note explaining the mapping, and
``keep_separate`` groups list terms that must never resolve to the same id;
loading fails if a configuration merges them.
"""

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from skillbridge.schemas import NonEmptyStr, StrictModel

UNKNOWN_PREFIX = "unknown:"
DEFAULT_ALIAS_FILE = Path(__file__).with_name("skill_aliases.json")
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


def load_alias_config(path: Path = DEFAULT_ALIAS_FILE) -> AliasConfig:
    return AliasConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


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


# Raw alias data; the generator inlines it into self-contained hosted agent files.
SKILL_ALIAS_DATA = json.loads(DEFAULT_ALIAS_FILE.read_text(encoding="utf-8"))
DEFAULT_ALIAS_CONFIG = AliasConfig.model_validate(SKILL_ALIAS_DATA)
# canonical id -> additional accepted spellings (the id itself is implicit).
DEFAULT_SKILL_ALIASES: dict[str, list[str]] = {
    entry.id: list(entry.aliases) for entry in DEFAULT_ALIAS_CONFIG.skills
}
DEFAULT_NORMALIZER = SkillNormalizer.from_config(DEFAULT_ALIAS_CONFIG)
