"""One catalog of the dataset sources that feed the training pipeline.

Both `scripts/audit_model_coverage.py` and `scripts/build_training_dataset.py`
read their list of sources from here. Before this module they each carried their
own copy, which meant the 99% coverage gate could be auditing a *different* set
of sources than the one being built -- and it was: RouterBench was audited at
401k model-name slots but exempted from the builder's gate.

`name_mode` is the field that decides both questions, so neither consumer has
to keep a hardcoded list of family names in sync with the other:

  columns            fixed model-name columns per row (arena battles)
  pkl_model_columns  model names are the wide pkl's column headers
  none               model-free: labels arrive already resolved, nothing to audit

A source is auditable, and coverage-gated, exactly when `name_mode != none`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FAMILY_ARENA = "arena"
FAMILY_ROUTERBENCH = "routerbench"
FAMILY_COMPLEXITY_ROUTER = "complexity_router"
FAMILIES: tuple[str, ...] = (FAMILY_ARENA, FAMILY_ROUTERBENCH, FAMILY_COMPLEXITY_ROUTER)

NAME_MODE_COLUMNS = "columns"
NAME_MODE_PKL_MODEL_COLUMNS = "pkl_model_columns"
NAME_MODE_NONE = "none"

LABEL_KIND_WINNER_TIER = "winner_tier"
LABEL_KIND_MIN_SOLVING_TIER = "min_solving_tier"
LABEL_KIND_ANNOTATED_LEVEL = "annotated_level"


@dataclass(frozen=True)
class Source:
    key: str
    family: str
    hf_id: str
    label_kind: str
    name_mode: str
    model_columns: tuple[str, ...] = ()
    note: str = ""
    #: Rows may not be checked into this repo (or otherwise redistributed) —
    #: access requires accepting terms on the Hub, and the license is not a
    #: redistribution right. Build them locally into the gitignored data files
    #: only. `tests/test_registry_rebuild.py` enforces this for the fixture.
    gated: bool = False

    @property
    def carries_model_names(self) -> bool:
        """True when this source's labels come through the registry resolver, so
        its raw model names are auditable and coverage-gated."""
        return self.name_mode != NAME_MODE_NONE


ARENA_HF_IDS = (
    ("arena_33k", "lmsys/chatbot_arena_conversations", "2023 Chatbot Arena", True),
    ("arena_55k", "lmarena-ai/arena-human-preference-55k",
     "2023-24 LMArena battles (csv, flat prompt)", False),
    ("arena_100k", "lmarena-ai/arena-human-preference-100k", "2024 LMArena battles", False),
    ("arena_140k", "lmarena-ai/arena-human-preference-140k", "2025 LMArena battles", False),
)

ROUTERBENCH_HF = "withmartian/routerbench"
ROUTERBENCH_FILE = "routerbench_0shot.pkl"
COMPLEXITY_ROUTER_HF = "RowRed/ComplexityRouter"

SOURCES: tuple[Source, ...] = (
    *(
        Source(key, FAMILY_ARENA, hf_id, LABEL_KIND_WINNER_TIER, NAME_MODE_COLUMNS,
               ("model_a", "model_b"), note, gated)
        for key, hf_id, note, gated in ARENA_HF_IDS
    ),
    Source("routerbench", FAMILY_ROUTERBENCH, ROUTERBENCH_HF,
           LABEL_KIND_MIN_SOLVING_TIER, NAME_MODE_PKL_MODEL_COLUMNS, (),
           "RouterBench 0-shot (pkl): 30k prompts x 11 models"),
    Source("complexity_router", FAMILY_COMPLEXITY_ROUTER, COMPLEXITY_ROUTER_HF,
           LABEL_KIND_ANNOTATED_LEVEL, NAME_MODE_NONE, (),
           "RowRed/ComplexityRouter: Qwen3.5-4B complexity levels 0-3, model-free"),
)

_BY_KEY: dict[str, Source] = {s.key: s for s in SOURCES}

#: Every source's labels were produced through the registry resolver.
AUDITABLE: tuple[Source, ...] = tuple(s for s in SOURCES if s.carries_model_names)

#: Source keys whose rows must never be committed or redistributed.
GATED: frozenset[str] = frozenset(s.key for s in SOURCES if s.gated)


def by_key(key: str) -> Source:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"unknown source {key!r}; known: {', '.join(sorted(_BY_KEY))}"
        ) from None


def for_families(families: list[str]) -> list[Source]:
    """Sources belonging to the given families, in catalog order.

    `arena` is one CLI family but four dataset generations, so a family expands
    to its sources rather than returning a single entry.
    """
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(
            f"unknown sources: {sorted(unknown)}; valid: {', '.join(FAMILIES)}"
        )
    wanted = set(families)
    return [s for s in SOURCES if s.family in wanted]


def parse_sources(raw: str | None, default: str = FAMILY_ARENA) -> list[str]:
    """Parse a `--sources a,b` value into validated, de-duplicated family names."""
    families = [s.strip() for s in (raw or default).split(",") if s.strip()]
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(
            f"unknown sources: {sorted(unknown)}; valid: {', '.join(FAMILIES)}"
        )
    seen: dict[str, None] = {}
    for f in families:
        seen.setdefault(f, None)
    return list(seen)


# --------------------------------------------------------------------------
# RouterBench's wide pkl layout, needed identically by the auditor and the
# builder. Duplicating it let the two disagree about which columns are models.

ROUTERBENCH_META_COLS = frozenset(
    {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}
)


def routerbench_model_columns(columns: list[Any]) -> list[Any]:
    """Wide pkl: quality lives in bare '<model>' columns; '<model>|suffix' is not.

    Columns are typed loosely because they come straight off a pandas frame;
    the `str(c)` guard below is what makes a non-string header survivable.
    """
    return [
        c
        for c in columns
        if "|" not in str(c) and str(c) not in ROUTERBENCH_META_COLS
    ]
