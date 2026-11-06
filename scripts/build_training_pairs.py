#!/usr/bin/env python3
"""Derive training pairs from data/router_training_dataset.jsonl.

Emits one row per (prompt, tier, outcome) sample for pairwise training:
the model scores a prompt against each complexity tier and learns whether
that tier wins, loses, or ties on the prompt. At inference the router
scores tiers 1-5 and picks the argmax.

Row shape: {"prompt", "tier", "outcome", "source"} — provenance lives in
the dataset file (pairs are derivable from it), so no ref field.

Rules:
  arena, decisive (winner a/b), different tiers -> (winner tier, 1.0) and
      (loser tier, 0.0)
  arena, tie or both_bad, different tiers       -> both tiers at 0.5
      ("no preference between these tiers on this prompt");
      --drop-ties skips those instead (ties compress the between-tier
      separation the router needs)
  same-tier battles                              -> skipped (a decisive
      battle between two equal tiers carries no complexity gradient and
      self-cancels in training; --keep-same-tier emits them anyway)
  routerbench-style rows (anything with candidate_tiers/solving_tiers:
      routerbench, complexity_router) -> every solving tier at 1.0, every
      other candidate tier at 0.0 (complexity_router emits a single gold
      tier, so its label tier wins and the other four tiers lose)
  nothing solved (solving_tiers [])              -> skipped (no ordering
      signal between tiers)

--max-pairs N caps the output with a seeded stratified sample over
(source, tier) strata, proportional allocation. Never truncate the file by
head: rows are ordered by source generation, so head-truncation would
train on a single era. Deterministic for a given --seed.

Usage:
  python scripts/build_training_pairs.py                       # full file
  python scripts/build_training_pairs.py --max-pairs 15000     # capped
  python scripts/build_training_pairs.py --keep-same-tier
  python scripts/build_training_pairs.py --min-tier-gap 2 --drop-ties
  python scripts/build_training_pairs.py --sources complexity_router

--sources a,b restricts which dataset sources contribute pairs (default:
all sources in the dataset file).

--min-tier-gap N keeps only battles whose tiers differ by >= N: adjacent-tier
human votes are near-coin-flips (the higher tier wins ~57% at gap 1 vs ~71%
at gap 3), so those pairs mostly inject label noise.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parent.parent

IN_DEFAULT = ROOT / "data" / "router_training_dataset.jsonl"
OUT_DEFAULT = ROOT / "data" / "router_training_pairs.jsonl"
REPORT_PATH = ROOT / "eval_reports" / "training_pairs_report.md"


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


def arena_pairs(row: dict[str, object], keep_same_tier: bool,
                min_tier_gap: int, drop_ties: bool,
                stats: PairStats) -> list[dict[str, object]]:
    tier_a = row["tier_a"]
    tier_b = row["tier_b"]
    winner = row["winner"]
    if not isinstance(tier_a, int) or not isinstance(tier_b, int):
        stats.no_signal_skipped += 1
        return []
    if tier_a == tier_b:
        if not keep_same_tier or winner not in ("a", "b"):
            stats.same_tier_skipped += 1
            return []
    if min_tier_gap and abs(tier_a - tier_b) < min_tier_gap:
        stats.filter_skipped += 1
        return []
    if winner in ("tie", "both_bad") and drop_ties:
        stats.ties_skipped += 1
        return []
    prompt = row["prompt"]
    source = row["source"]
    if winner == "a":
        outs = [(tier_a, 1.0), (tier_b, 0.0)]
    elif winner == "b":
        outs = [(tier_b, 1.0), (tier_a, 0.0)]
    elif winner in ("tie", "both_bad"):
        outs = [(tier_a, 0.5), (tier_b, 0.5)]
    else:
        stats.no_signal_skipped += 1
        return []
    return [
        {"prompt": prompt, "tier": t, "outcome": o, "source": source}
        for t, o in outs
    ]


def solving_tier_pairs(row: dict[str, object],
                       stats: PairStats) -> list[dict[str, object]]:
    """Pairs from min-solving-tier rows (routerbench, complexity_router):
    every solving tier at 1.0, every other candidate tier at 0.0.
    complexity_router labels a single gold tier, so the other four tiers
    all lose -- the min-cost routing signal."""
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
        {"prompt": prompt, "tier": t, "outcome": 1.0 if t in solving else 0.0,
         "source": row["source"]}
        for t in candidates
    ]


def strata_counts(infile: Path, keep_same_tier: bool, min_tier_gap: int,
                  drop_ties: bool,
                  only_sources: set[str] | None) -> Counter[tuple[str, int]]:
    """Pass 1: how many pairs fall in each (source, tier) stratum."""
    counts: Counter[tuple[str, int]] = Counter()
    for line in infile.open():
        if not line.strip():
            continue
        row = json.loads(line)
        stats = PairStats()
        source = str(row.get("source", "?"))
        if only_sources is not None and source not in only_sources:
            continue
        if isinstance(row.get("candidate_tiers"), list):
            pairs = solving_tier_pairs(row, stats)
        else:
            pairs = arena_pairs(row, keep_same_tier, min_tier_gap, drop_ties,
                                stats)
        for p in pairs:
            counts[(source, cast(int, p["tier"]))] += 1
    return counts


def allocate(counts: Counter[tuple[str, int]], cap: int) -> dict[tuple[str, int], int]:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", type=Path, default=IN_DEFAULT,
                    help="dataset JSONL from build_training_dataset.py")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--keep-same-tier", action="store_true",
                    help="also emit decisive same-tier battles as 1.0/0.0")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="cap output via stratified (source, tier) sampling")
    ap.add_argument("--min-tier-gap", type=int, default=0,
                    help="keep only arena battles whose tiers differ by >= N")
    ap.add_argument("--drop-ties", action="store_true",
                    help="skip tie / both_bad battles instead of 0.5 targets")
    ap.add_argument("--sources", default=None,
                    help="comma list restricting which sources contribute "
                         "pairs (default: all sources in the dataset file)")
    ap.add_argument("--seed", type=int, default=13,
                    help="sampling seed (deterministic builds)")
    args = ap.parse_args()

    if not args.infile.exists():
        raise SystemExit(f"missing {args.infile} - run "
                         f"scripts/build_training_dataset.py first")
    requested_sources = (
        {s.strip() for s in args.sources.split(",") if s.strip()}
        if args.sources is not None else None
    )

    counts = strata_counts(args.infile, args.keep_same_tier,
                           args.min_tier_gap, args.drop_ties, requested_sources)
    if requested_sources is not None:
        available = {key for key, _tier in counts}
        missing = requested_sources - available
        if missing:
            raise SystemExit(
                f"sources {sorted(missing)} contribute no pairs in "
                f"{args.infile}; sources with pairs: {sorted(available)}"
            )
    total_available = sum(counts.values())
    alloc = allocate(counts, args.max_pairs) if args.max_pairs else dict(counts)
    # pass 2 must draw the SAME subset: precompute chosen indices per stratum
    rng = random.Random(args.seed)
    chosen: dict[tuple[str, int], set[int]] = {
        key: set(rng.sample(range(counts[key]), alloc[key]))
        for key in counts if alloc[key] < counts[key]
    }

    stats_by_source: dict[str, PairStats] = {}
    seen_in_stratum: Counter[tuple[str, int]] = Counter()
    total = 0
    tmp = args.out.with_suffix(".jsonl.tmp")
    with args.infile.open() as fin, tmp.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("source", "?"))
            if requested_sources is not None and source not in requested_sources:
                continue
            stats = stats_by_source.setdefault(source, PairStats())
            stats.rows_in += 1
            if isinstance(row.get("candidate_tiers"), list):
                pairs = solving_tier_pairs(row, stats)
            else:
                pairs = arena_pairs(row, args.keep_same_tier,
                                    args.min_tier_gap, args.drop_ties, stats)
            for p in pairs:
                key = (source, cast(int, p["tier"]))
                idx = seen_in_stratum[key]
                seen_in_stratum[key] = idx + 1
                if key in chosen and idx not in chosen[key]:
                    continue
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")
                outcome = f'{p["outcome"]:.1f}'
                stats.outcome_hist[outcome] += 1
                assert isinstance(p["tier"], int)
                stats.tier_hist[p["tier"]] += 1
                stats.pairs += 1
                total += 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as f:
        f.write("# training pairs report\n\n")
        f.write(f"generated: {date.today().isoformat()} | "
                f"sources: {args.sources or 'all'} | "
                f"keep_same_tier: {args.keep_same_tier} | "
                f"min_tier_gap: {args.min_tier_gap} | "
                f"drop_ties: {args.drop_ties} | "
                f"max_pairs: {args.max_pairs or 'none'} | seed: {args.seed}\n\n")
        f.write("| source | dataset rows | pairs | same-tier skipped | "
                "gap/ties skipped | no-signal skipped |\n|---|---|---|---|---|---|\n")
        for source, s in stats_by_source.items():
            f.write(f"| {source} | {s.rows_in} | {s.pairs} | "
                    f"{s.same_tier_skipped} | "
                    f"{s.filter_skipped + s.ties_skipped} | "
                    f"{s.no_signal_skipped} |\n")
        f.write(f"\navailable pairs: {total_available} | emitted: {total}\n\n")
        if args.max_pairs and args.max_pairs < total_available:
            f.write(f"stratified sample over (source, tier); "
                    f"{len(counts)} strata, seed {args.seed}:\n\n")
            f.write("| source | tier | kept | available |\n|---|---|---|---|\n")
            for (source, tier) in sorted(counts):
                f.write(f"| {source} | {tier} | {alloc[(source, tier)]} | "
                        f"{counts[(source, tier)]} |\n")
            f.write("\n")
        for source, s in stats_by_source.items():
            f.write(f"## {source}\n\n")
            f.write(f"outcome histogram: {dict(sorted(s.outcome_hist.items()))}\n\n")
            f.write(f"tier histogram (rows per tier): "
                    f"{dict(sorted(s.tier_hist.items()))}\n\n")
        f.write(f"total pairs: {total}\n")
    tmp.replace(args.out)
    for source, s in stats_by_source.items():
        print(f"{source}: rows={s.rows_in} pairs={s.pairs} "
              f"same-tier-skip={s.same_tier_skipped} "
              f"gap/ties-skip={s.filter_skipped + s.ties_skipped} "
              f"no-signal={s.no_signal_skipped}")
    print(f"wrote {args.out} ({total} pairs"
          + (f", capped from {total_available}" if args.max_pairs
             and args.max_pairs < total_available else "")
          + f") and {REPORT_PATH}")


if __name__ == "__main__":
    main()
