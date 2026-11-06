#!/usr/bin/env python3
"""Derive training pairs from data/router_training_dataset.jsonl.

Emits one row per (prompt, tier, outcome) sample for pairwise training:
the model scores a prompt against each complexity tier and learns whether
that tier wins, loses, or ties on the prompt. At inference the router
scores tiers 1-5 and picks the argmax.

Row shape: {"prompt", "tier", "outcome", "source"} — provenance lives in
the dataset file (pairs are derivable from it), so no ref field.

The labeling rules and the two-row-shape dispatch live in `router.pairs`;
this file is the streaming driver, the sampling, and the report. Read the
rules there rather than re-deriving them here.

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
from collections import Counter
from datetime import date
from pathlib import Path

from router.pairs import (
    PairFilter,
    PairStats,
    allocate,
    pairs_from_row,
    strata_key,
)

ROOT = Path(__file__).resolve().parent.parent

IN_DEFAULT = ROOT / "data" / "router_training_dataset.jsonl"
OUT_DEFAULT = ROOT / "data" / "router_training_pairs.jsonl"
REPORT_PATH = ROOT / "eval_reports" / "training_pairs_report.md"


def iter_rows(path: Path):
    """Stream dataset rows without materializing them (~365 MB file)."""
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def strata_counts(
    infile: Path, flt: PairFilter, only_sources: set[str] | None
) -> Counter[tuple[str, int]]:
    """Pass 1: how many pairs fall in each (source, tier) stratum."""
    counts: Counter[tuple[str, int]] = Counter()
    for row in iter_rows(infile):
        source = str(row.get("source", "?"))
        if only_sources is not None and source not in only_sources:
            continue
        stats = PairStats()  # pass 1 only needs the pairs, not the skip counts
        for p in pairs_from_row(row, flt, stats):
            counts[strata_key(p)] += 1
    return counts


def render_report(
    args: argparse.Namespace,
    stats_by_source: dict[str, PairStats],
    counts: Counter[tuple[str, int]],
    alloc: dict[tuple[str, int], int],
    total_available: int,
    total: int,
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        f.write("# training pairs report\n\n")
        f.write(
            f"generated: {date.today().isoformat()} | "
            f"sources: {args.sources or 'all'} | "
            f"keep_same_tier: {args.keep_same_tier} | "
            f"min_tier_gap: {args.min_tier_gap} | "
            f"drop_ties: {args.drop_ties} | "
            f"max_pairs: {args.max_pairs or 'none'} | seed: {args.seed}\n\n"
        )
        f.write(
            "| source | dataset rows | pairs | same-tier skipped | "
            "gap/ties skipped | no-signal skipped |\n|---|---|---|---|---|---|\n"
        )
        for source, s in stats_by_source.items():
            f.write(
                f"| {source} | {s.rows_in} | {s.pairs} | "
                f"{s.same_tier_skipped} | "
                f"{s.filter_skipped + s.ties_skipped} | "
                f"{s.no_signal_skipped} |\n"
            )
        f.write(f"\navailable pairs: {total_available} | emitted: {total}\n\n")
        if args.max_pairs and args.max_pairs < total_available:
            f.write(
                f"stratified sample over (source, tier); "
                f"{len(counts)} strata, seed {args.seed}:\n\n"
            )
            f.write("| source | tier | kept | available |\n|---|---|---|---|\n")
            for (source, tier) in sorted(counts):
                f.write(
                    f"| {source} | {tier} | {alloc[(source, tier)]} | "
                    f"{counts[(source, tier)]} |\n"
                )
            f.write("\n")
        for source, s in stats_by_source.items():
            f.write(f"## {source}\n\n")
            f.write(f"outcome histogram: {dict(sorted(s.outcome_hist.items()))}\n\n")
            f.write(
                f"tier histogram (rows per tier): "
                f"{dict(sorted(s.tier_hist.items()))}\n\n"
            )
        f.write(f"total pairs: {total}\n")


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
    ap.add_argument("--report", type=Path, default=REPORT_PATH,
                    help="markdown report to write")
    args = ap.parse_args()

    if not args.infile.exists():
        raise SystemExit(f"missing {args.infile} - run "
                         f"scripts/build_training_dataset.py first")
    flt = PairFilter(
        keep_same_tier=args.keep_same_tier,
        min_tier_gap=args.min_tier_gap,
        drop_ties=args.drop_ties,
    )
    requested_sources = (
        {s.strip() for s in args.sources.split(",") if s.strip()}
        if args.sources is not None else None
    )

    counts = strata_counts(args.infile, flt, requested_sources)
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
    with tmp.open("w") as fout:
        for row in iter_rows(args.infile):
            source = str(row.get("source", "?"))
            if requested_sources is not None and source not in requested_sources:
                continue
            stats = stats_by_source.setdefault(source, PairStats())
            stats.rows_in += 1
            for p in pairs_from_row(row, flt, stats):
                key = strata_key(p)
                idx = seen_in_stratum[key]
                seen_in_stratum[key] = idx + 1
                if key in chosen and idx not in chosen[key]:
                    continue
                fout.write(json.dumps(p.as_json_dict(), ensure_ascii=False) + "\n")
                stats.outcome_hist[f"{p.outcome:.1f}"] += 1
                stats.tier_hist[p.tier] += 1
                stats.pairs += 1
                total += 1

    render_report(args, stats_by_source, counts, alloc, total_available,
                  total, args.report)
    tmp.replace(args.out)
    for source, s in stats_by_source.items():
        print(f"{source}: rows={s.rows_in} pairs={s.pairs} "
              f"same-tier-skip={s.same_tier_skipped} "
              f"gap/ties-skip={s.filter_skipped + s.ties_skipped} "
              f"no-signal={s.no_signal_skipped}")
    print(f"wrote {args.out} ({total} pairs"
          + (f", capped from {total_available}" if args.max_pairs
             and args.max_pairs < total_available else "")
          + f") and {args.report}")


if __name__ == "__main__":
    main()
