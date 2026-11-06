"""Tier scoring: canonical model groups -> absolute complexity tier 1-5.

The tier is *absolute capability*, not cost or preference: Opus-class = 5,
flash/mini = 1. Every rule below exists because one of the raw signals lies in a
specific, knowable way:

  price               commodity hosting prices legacy open models for scarcity,
                      not capability, and free mirrors say nothing at all
  intelligence index  today's index is era-biased (gpt-4o scores 8.4 in 2026),
                      so it may only DOWNGRADE models released >= ERA_MODERN
  design-arena ELO    sparse and self-selected, so it may only LIFT a model to
                      frontier (>= ELO_FRONTIER), never demote one
  parameter count     the fallback for unpriced models; legacy (pre-o1) models
                      are cut on a lower scale because 405B was 2024's frontier

These are the invariants AGENTS.md calls "don't break silently". They live in
named functions now, so each has a test (`tests/test_scoring.py`) instead of
being buried in one 130-line function where a change quietly re-tiers 3,500
models and every training row derived from them.

`overrides.yaml` wins over everything here, including the era caps.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from router.catalogs import OPEN_PARAMS_VENDORS, Canonical

try:
    import yaml
except ImportError:  # overrides are optional without pyyaml
    yaml = None  # type: ignore[assignment]

PRICE_EDGES = [
    (0.05, 1.0),
    (0.15, 1.7),
    (0.70, 2.7),
    (4.0, 3.9),
    (15.0, 4.9),
    (30.0, 5.0),
]
II_CUTS = [(45.0, 5), (28.0, 4), (14.0, 3), (7.0, 2), (0.0, 1)]
ELO_FRONTIER = 1300.0  # design-arena ELO only used to lift, never to downgrade
PARAMS_CUTS_MODERN = [(300.0, 5), (67.0, 4), (13.0, 3), (3.0, 2), (0.0, 1)]
PARAMS_CUTS_LEGACY = [(300.0, 4), (70.0, 3), (7.0, 2), (0.0, 1)]

# absolute-capability era windows (see build_report for calibration notes)
ERA_PRE_FRONTIER = dt.date(2023, 3, 15)  # before gpt-4: everything ~ tier 2
ERA_CAP_4_END = dt.date(2024, 9, 1)  # before o1: nothing is tier 5
ERA_MODERN = dt.date(2025, 1, 1)  # before this, today's intelligence
# index is not a fair downgrade signal

TINY_WORDS = re.compile(
    r"\b(?:tiny|mini|micro|nano|slim|small|lite|flash|instant|1b|2b|3b|4b|5b|6b|7b|8b|9b)\b|-(?:1|2|3)b\b"
)
HUGE_WORDS = re.compile(r"\b(?:opus|ultra|max|large|405b|671b|200b| frontier)\b")

OVERRIDE_SIGNAL_KEYS = (
    "price_in_per_m",
    "price_out_per_m",
    "elo",
    "intelligence_index",
    "params_b",
)


def interp_price(pin: float | None, pout: float | None) -> float | None:
    """Continuous tier prior from blended $/1M tokens, log-interpolated.

    0.75 input + 0.25 output: input tokens dominate most routing workloads and
    most catalogs price them, so output price is the weaker signal.
    """
    if pin is None and pout is None:
        return None
    pin = pin or 0.0
    pout = pout or 0.0
    if pin == 0.0 and pout == 0.0:
        return 1.0  # free tier
    blended = 0.75 * pin + 0.25 * pout
    if blended <= PRICE_EDGES[0][0]:
        return 1.0
    if blended >= PRICE_EDGES[-1][0]:
        return 5.0
    for (p0, s0), (p1, s1) in zip(PRICE_EDGES, PRICE_EDGES[1:]):
        if p0 < blended <= p1:
            frac = (blended - p0) / (p1 - p0)
            return s0 + frac * (s1 - s0)
    return None


def cut_tier(value: float | None, cuts: list[tuple[float, int]]) -> int | None:
    """Bucket a signal onto its tier by descending thresholds."""
    if value is None:
        return None
    for threshold, tier in cuts:
        if value >= threshold:
            return tier
    return cuts[-1][1]


def era_of(released: str | None) -> dt.date | None:
    if not released:
        return None
    try:
        return dt.date.fromisoformat(released[:10])
    except ValueError:
        return None


def best_price(prices: Sequence[tuple[str, float | None, float | None]]
               ) -> tuple[float | None, float | None]:
    """The priciest non-free hoster's (in, out) — list price is the capability signal."""
    best: tuple[float, float] | None = None
    best_blended = -1.0
    for _source, p_in, p_out in prices:
        if not (p_in or p_out):
            continue  # free mirrors carry no capability information
        blended = 0.75 * (p_in or 0.0) + 0.25 * (p_out or 0.0)
        if blended > best_blended:
            best_blended = blended
            best = (p_in or 0.0, p_out or 0.0)
    return best if best else (None, None)


def quality_tier(
    ii: float | None, elo: float | None
) -> tuple[int | None, int | None]:
    """(intelligence-index tier, overall quality tier).

    ELO is a lift-only signal: it fires when the index is missing and the model
    is frontier-graded, and it can never pull a tier down.
    """
    ii_tier = cut_tier(ii, II_CUTS)
    quality = ii_tier
    if quality is None and elo and elo >= ELO_FRONTIER:
        quality = 5
    return ii_tier, quality


def quality_adjust(
    base_tier: int,
    quality: int | None,
    quality_ii: int | None,
    era: dt.date | None,
) -> int:
    """One step toward measured quality: -1, 0 or +1.

    Lifts are cheap (a strong index on a cheap-priced model is usually a pricing
    artifact), so +1 applies when quality is >= 2 tiers above, or 1 tier above
    and the disagreement is at either extreme. Downgrades are trusted only when
    both the index exists and the model is modern (>= ERA_MODERN) — a 2023 model
    scored against today's index is not evidence that it was worse than its price.
    """
    if quality is None:
        return 0
    diff = quality - base_tier
    if diff >= 2 or (diff == 1 and (quality == 5 or base_tier <= 2)):
        return 1
    if diff <= -2 and quality_ii is not None and era is not None and era >= ERA_MODERN:
        return -1
    return 0


def apply_era_caps(tier: int, era: dt.date | None) -> int:
    """A model cannot outrank the frontier that existed when it shipped."""
    if era is not None:
        if era < ERA_PRE_FRONTIER:
            return min(tier, 3)
        elif era < ERA_CAP_4_END:
            return min(tier, 4)
    return tier


def score_group(g: Canonical) -> dict[str, Any] | None:
    """One canonical group -> a registry entry, or None if unscoreable."""
    f = g.fields
    pin, pout = best_price(f.get("prices") or [])
    price_cont = interp_price(pin, pout)
    released = (f.get("released") or (None, None))[1]
    era = era_of(released)

    quality_ii, quality = quality_tier(f.get("ii"), f.get("elo"))

    params_b = f.get("params_b")
    legacy = era is not None and era < ERA_CAP_4_END
    params_tier = cut_tier(
        params_b, PARAMS_CUTS_LEGACY if legacy else PARAMS_CUTS_MODERN
    )

    nm = g.key.split("/", 1)[1]
    vendor = g.key.split("/", 1)[0]
    tiny = bool(TINY_WORDS.search(nm))
    huge = bool(HUGE_WORDS.search(nm))

    # --- base tier selection: era floor, then params for legacy open models,
    # then price, then params, then name keywords, then a guess
    basis: str
    base_tier: int
    pre_frontier = era is not None and era < ERA_PRE_FRONTIER
    legacy_open = (
        legacy
        and params_tier is not None
        and vendor in OPEN_PARAMS_VENDORS
        and not pre_frontier
    )
    if pre_frontier and not nm.startswith("gpt-4"):
        # pre-gpt-4 models were priced for scarcity, not capability
        base_tier, basis = 2, "era(pre-2023-03)"
    elif legacy_open and params_tier is not None:
        # commodity hosting prices legacy open models for scarcity, not capability
        base_tier, basis = params_tier, f"params({params_b:g}B, legacy open)"
    elif price_cont is not None:
        base_tier = max(1, min(5, int(price_cont + 0.5)))  # half-up
        basis = (
            f"price({pin:.2f}/{pout:.2f}$ per M)" if (pin or pout) else "price(free)"
        )
    elif params_tier is not None:
        base_tier, basis = params_tier, f"params({params_b:g}B)"
    elif tiny:
        base_tier, basis = 1, "keyword(tiny)"
    elif huge:
        base_tier, basis = 4, "keyword(huge)"
    else:
        base_tier, basis = 2, "guess"

    adj = quality_adjust(base_tier, quality, quality_ii, era)
    tier = apply_era_caps(base_tier + adj, era)

    tier = max(1, min(5, int(tier)))
    score_cont = float(tier)
    if price_cont is not None and basis.startswith("price"):
        score_cont = price_cont
    if adj:
        score_cont = min(
            max(score_cont + adj * 0.5, float(tier) - 0.5), float(tier) + 0.5
        )
    score_cont = min(max(score_cont, 1.0), 5.0)

    notes = []
    if basis == "guess":
        notes.append("unpriced, no params found - review")
    if adj:
        notes.append(f"quality adj {adj:+d} (ii={f.get('ii')}, elo={f.get('elo')})")
    if released is None and tier == 5:
        notes.append("tier 5 with no release date - review")
    if (
        price_cont is not None
        and quality is not None
        and abs(quality - round(price_cont)) >= 2
    ):
        notes.append("price/quality disagree >=2 tiers")

    # display name: longest alias, capitalized by the source that had it
    raws = [r for r in g.raws if "/" not in r or r.count("/") == 1]

    return {
        "canonical_id": g.key,
        "display_name": max((r for r in raws), key=len) if raws else g.key,
        "aliases": sorted(a for a in g.aliases if a and a != g.key.split("/", 1)[1]),
        "released": released,
        "mode": "chat",
        "complexity": tier,
        "score": round(score_cont, 2),
        "signals": {
            "price_in_per_m": round(pin, 4) if pin is not None else None,
            "price_out_per_m": round(pout, 4) if pout is not None else None,
            "elo": round(f["elo"]) if f.get("elo") else None,
            "intelligence_index": f.get("ii"),
            "params_b": params_b,
        },
        "basis": basis,
        "notes": "; ".join(notes),
        "_quality": quality,
    }


def score_all(
    groups: dict[str, Canonical], excludes: set[str] | None = None
) -> list[dict[str, Any]]:
    """Score every group, skipping junk merge groups excluded by name.

    An excluded group is not a real model (a bad alias merge); the real model is
    recreated through `overrides.yaml` instead.
    """
    excluded = excludes or set()
    built: list[dict[str, Any]] = []
    for g in groups.values():
        if g.key in excluded:
            continue
        entry = score_group(g)
        if entry is not None:
            built.append(entry)
    return built


def load_overrides(path: Path) -> dict[str, Any]:
    """Hand edits from overrides.yaml, or {} when absent or pyyaml is missing."""
    if not yaml or not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def apply_overrides(
    built: list[dict[str, Any]], overrides: dict[str, Any]
) -> list[dict[str, Any]]:
    """Hand tiers win over every signal, and can resurrect pruned models.

    A key with no live catalog entry is *created* (2023-era models that vendors
    have since delisted) but only if it states a complexity — a bare aliases
    entry for a nonexistent model would emit a tier-less registry row.
    """
    by_key = {e["canonical_id"]: e for e in built}
    for key, ov in (overrides.get("models") or {}).items():
        entry = by_key.get(key)
        if entry is None:
            if "complexity" not in ov:
                continue
            entry = {
                "canonical_id": key,
                "display_name": ov.get("display_name", key),
                "aliases": [],
                "released": ov.get("released"),
                "mode": "chat",
                "complexity": int(ov["complexity"]),
                "score": float(ov.get("score", ov["complexity"])),
                "signals": {
                    k: ov.get(k) for k in OVERRIDE_SIGNAL_KEYS
                },
                "basis": "override",
                "notes": ov.get(
                    "notes", "created via override; absent from live catalogs"
                ),
                "_quality": None,
                # WART, preserved deliberately: `_raws` is an internal of the old
                # build that leaked into models.json. Scored entries had it popped
                # before writing; override-created ones did not, so exactly these
                # entries carry an always-empty `_raws`. Nothing reads it. Kept so
                # a rebuild stays byte-identical to the committed registry; drop it
                # (and `_quality`) as its own change if you want a clean schema.
                "_raws": [],
            }
            by_key[key] = entry
            built.append(entry)
        for field in ("complexity", "score", "released", "display_name", "notes"):
            if field in ov:
                entry[field] = ov[field]
        if "aliases" in ov:
            entry["aliases"] = sorted(set(entry["aliases"]) | set(ov["aliases"]))
        if "signals" in ov:
            entry["signals"].update(
                {k: v for k, v in ov["signals"].items() if v is not None}
            )
        if entry["basis"] != "override":
            entry["basis"] = "override"
    return built


def alias_collisions(built: list[dict[str, Any]]) -> list[str]:
    """Names claimed by more than one canonical model — a normalization bug."""
    alias_index: dict[str, str] = {}
    collisions: list[str] = []
    for entry in built:
        names = {entry["canonical_id"], *entry["aliases"]}
        for a in names:
            if a in alias_index and alias_index[a] != entry["canonical_id"]:
                collisions.append(f"{a}: {alias_index[a]} vs {entry['canonical_id']}")
            alias_index[a] = entry["canonical_id"]
    return collisions
