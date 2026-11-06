"""Dataset rows -> (prompt, tier, outcome) training pairs.

The dataset file holds two row shapes and this module is the **only** place that
dispatches between them:

  `candidate_tiers` is a list  -> min-solving-tier shape (routerbench and
      complexity_router). Solving tiers score 1.0, every other candidate 0.0.
      complexity_router reuses this shape with `solving_tiers=[label_tier]`, so
      its gold tier wins and the other four lose.
  otherwise                    -> arena battle shape, using
      `tier_a`/`tier_b`/`winner`.

Dispatch is on field presence, not on `source`, so a new source either emits one
of these two shapes or adds a branch *here* -- one branch, not one per caller.
`build_training_pairs.py` streams the file twice (strata counts, then emit) and
both passes must call this one function, or the sample and the counts disagree.

Outcome semantics that downstream training depends on: a decisive row contributes
exactly one 1.0, which is what makes `routing_accuracy` well-defined and what
`BCEWithLogitsLoss(pos_weight=4.0)` is balanced against (4 losers per winner).
Tie rows contribute two 0.5s instead -- revisit `pos_weight` if they come back.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Pair:
    """Field order is the serialized key order; don't reorder."""

    prompt: str
    tier: int
    outcome: float
    source: str

    def as_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairFilter:
    """Row-selection rules, all applied before any sampling."""

    keep_same_tier: bool = False
    min_tier_gap: int = 0
    drop_ties: bool = False


@dataclass
class PairStats:
    rows_in: int = 0
    pairs: int = 0
    same_tier_skipped: int = 0
    filter_skipped: int = 0
    ties_skipped: int = 0
    no_signal_skipped: int = 0
    outcome_hist: Counter[str] = field(default_factory=Counter)
    tier_hist: Counter[int] = field(default_factory=Counter)


def arena_pairs(
    row: dict[str, Any], flt: PairFilter, stats: PairStats
) -> list[Pair]:
    """Pairs from a human-preference battle.

    Same-tier decisive battles carry no complexity gradient (an equal-tier vote
    self-cancels in training), so they are dropped unless --keep-same-tier.
    """
    tier_a = row["tier_a"]
    tier_b = row["tier_b"]
    winner = row["winner"]
    if not isinstance(tier_a, int) or not isinstance(tier_b, int):
        stats.no_signal_skipped += 1
        return []
    if tier_a == tier_b:
        if not flt.keep_same_tier or winner not in ("a", "b"):
            stats.same_tier_skipped += 1
            return []
    if flt.min_tier_gap and abs(tier_a - tier_b) < flt.min_tier_gap:
        stats.filter_skipped += 1
        return []
    if winner in ("tie", "both_bad") and flt.drop_ties:
        stats.ties_skipped += 1
        return []
    prompt = row["prompt"]
    source = row["source"]
    if winner == "a":
        outs: list[tuple[Any, float]] = [(tier_a, 1.0), (tier_b, 0.0)]
    elif winner == "b":
        outs = [(tier_b, 1.0), (tier_a, 0.0)]
    elif winner in ("tie", "both_bad"):
        outs = [(tier_a, 0.5), (tier_b, 0.5)]
    else:
        stats.no_signal_skipped += 1
        return []
    return [Pair(prompt=prompt, tier=t, outcome=o, source=source) for t, o in outs]


def solving_tier_pairs(row: dict[str, Any], stats: PairStats) -> list[Pair]:
    """Pairs from a min-solving-tier row: solving tiers win, other candidates lose."""
    candidates = row.get("candidate_tiers")
    solving = row.get("solving_tiers")
    if solving is None:  # dataset built before the field existed
        label = row.get("label_tier")
        solving = [label] if isinstance(label, int) else []
    if not isinstance(candidates, list) or not candidates or not solving:
        stats.no_signal_skipped += 1
        return []
    prompt = row["prompt"]
    return [
        Pair(
            prompt=prompt,
            tier=t,
            outcome=1.0 if t in solving else 0.0,
            source=row["source"],
        )
        for t in candidates
    ]


def pairs_from_row(row: dict[str, Any], flt: PairFilter, stats: PairStats) -> list[Pair]:
    """The row-shape dispatch. See the module docstring for the two shapes."""
    if isinstance(row.get("candidate_tiers"), list):
        return solving_tier_pairs(row, stats)
    return arena_pairs(row, flt, stats)


def strata_key(pair: Pair) -> tuple[str, int]:
    """The sampling stratum: (source, tier).

    Stratifying on source as well as tier is what stops a --max-pairs cap from
    collapsing the mix onto the largest source, and stratifying on tier stops it
    from dropping the rare frontier tiers.
    """
    return pair.source, pair.tier


def allocate(counts: Mapping[tuple[str, int], int],
             cap: int) -> dict[tuple[str, int], int]:
    """Proportional allocation with largest-remainder rounding, sum == cap."""
    total = sum(counts.values())
    if cap >= total:
        return dict(counts)
    alloc: dict[tuple[str, int], int] = {}
    remainders: list[tuple[float, tuple[str, int]]] = []
    assigned = 0
    for key, n in counts.items():
        exact = cap * n / total
        alloc[key] = int(exact)
        assigned += int(exact)
        remainders.append((exact - int(exact), key))
    remainders.sort(reverse=True)
    for _frac, key in remainders[: cap - assigned]:
        alloc[key] += 1
    return alloc
