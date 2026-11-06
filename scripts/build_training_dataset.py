#!/usr/bin/env python3
"""Build the (prompt -> complexity) training dataset.

Thin CLI over `router.dataset` (one adapter per source family) and
`router.sources` (the catalog of what feeds the pipeline, shared with
`audit_model_coverage.py`). The labeling rules, the row shapes each family
emits, and the coverage gate all live in `router.dataset`.

Gates: model-name coverage per source must be >= 99% (add gaps to
data/model_registry/overrides.yaml, rebuild the registry, retry); duplicate
battles (same prompt + model pair) across releases are dropped. Model-free
sources (complexity_router) have no names to resolve and are exempt — that
exemption is derived from whether the source resolved any names, not from a
hardcoded list of source keys.

Usage:
  python scripts/build_training_dataset.py                     # arena only
  python scripts/build_training_dataset.py --sources arena,routerbench
  python scripts/build_training_dataset.py --sources arena,routerbench,complexity_router
  python scripts/build_training_dataset.py --limit 500         # smoke run
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import GatedRepoError

from router.contracts import COVERAGE_TARGET
from router.dataset import (
    SourceStats,
    arena_rows,
    complexity_router_rows,
    coverage_failures,
    routerbench_rows,
)
from router.model_registry import ModelRegistry, get_registry
from router.sources import (
    FAMILY_ARENA,
    FAMILY_COMPLEXITY_ROUTER,
    FAMILY_ROUTERBENCH,
    Source,
    for_families,
    parse_sources,
)

ROOT = Path(__file__).resolve().parent.parent

OUT_DEFAULT = ROOT / "data" / "router_training_dataset.jsonl"
REPORT_PATH = ROOT / "eval_reports" / "training_dataset_report.md"


def rows_for(
    src: Source,
    registry: ModelRegistry,
    stats: SourceStats,
    seen: set[str],
    epsilon: float,
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    """The one family dispatch. A new family adds a branch here plus an adapter."""
    if src.family == FAMILY_ARENA:
        return arena_rows(src.key, src.hf_id, registry, stats, seen, limit)
    if src.family == FAMILY_ROUTERBENCH:
        return routerbench_rows(registry, stats, seen, epsilon, limit)
    if src.family == FAMILY_COMPLEXITY_ROUTER:
        return complexity_router_rows(stats, seen, limit)
    raise SystemExit(f"no adapter for source family {src.family!r} ({src.key})")


def percentile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]


def render_report(
    built: list[tuple[Source, SourceStats]],
    args: argparse.Namespace,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_hist: dict[int, int] = {}
    with REPORT_PATH.open("w") as f:
        f.write("# training dataset build report\n\n")
        f.write(
            f"generated: {date.today().isoformat()} | sources: {args.sources} | "
            f"epsilon: {args.epsilon}\n\n"
        )
        f.write(
            "| source | scanned | emitted | dup skipped | unresolved dropped | "
            "model coverage | tie/both_bad |\n|---|---|---|---|---|---|---|\n"
        )
        for src, s in built:
            # "n/a" means the source resolved no model names at all (model-free),
            # not that coverage was skipped for a source that has names.
            coverage = s.coverage
            cov_text = "n/a" if coverage is None else f"{coverage:.2f}%"
            ties = s.winner_hist.get("tie", 0) + s.winner_hist.get("both_bad", 0)
            f.write(
                f"| {src.key} | {s.scanned} | {s.emitted} | {s.dup_skipped} | "
                f"{s.unresolved_dropped} | {cov_text} | {ties} |\n"
            )
        f.write("\n")
        for src, s in built:
            f.write(f"## {src.key}\n\n")
            if s.unmapped:
                f.write("unmapped model names (rows dropped):\n\n")
                f.write("| name | rows |\n|---|---|\n")
                for name, n in s.unmapped.most_common():
                    f.write(f"| `{name}` | {n} |\n")
                f.write("\n")
            if s.winner_hist:  # arena battles only; model-free sources have none
                f.write(f"winner distribution: {dict(s.winner_hist)}\n\n")
            lengths = sorted(s.prompt_chars)
            f.write(
                f"prompt chars p50/p95/p99/max: {percentile(lengths, 0.5)} / "
                f"{percentile(lengths, 0.95)} / {percentile(lengths, 0.99)} / "
                f"{lengths[-1] if lengths else 0}\n\n"
            )
            f.write(f"label_tier histogram: {dict(sorted(s.tier_hist.items()))}\n\n")
            if s.label_rule:
                f.write(f"label rule: {s.label_rule}\n\n")
            for tier, n in s.tier_hist.items():
                total_hist[tier] = total_hist.get(tier, 0) + n
        f.write(
            f"total emitted: {sum(s.emitted for _src, s in built)} | "
            f"combined label_tier histogram: {dict(sorted(total_hist.items()))}\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=FAMILY_ARENA,
                    help="comma list: arena, routerbench, complexity_router")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="routerbench: a tier 'solves' a prompt within this "
        "much of the best model's quality",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="cap rows per source (smoke runs)"
    )
    args = ap.parse_args()

    try:
        sources = for_families(parse_sources(args.sources))
    except ValueError as e:
        raise SystemExit(str(e)) from None

    registry = get_registry()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    built: list[tuple[Source, SourceStats]] = []
    seen: set[str] = set()

    tmp = args.out.with_suffix(".jsonl.tmp")
    with tmp.open("w") as out:
        for src in sources:
            print(f"building {src.key} ({src.hf_id}) ...")
            stats = SourceStats()
            try:
                for row in rows_for(src, registry, stats, seen, args.epsilon,
                                    args.limit):
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
            except (DatasetNotFoundError, GatedRepoError) as e:
                print(
                    f"  SKIP {src.key}: gated or inaccessible "
                    f"({type(e).__name__}: {str(e)[:120]})"
                )
                print(
                    "  -> if the repo is gated, accept its license on "
                    "huggingface.co and run `hf auth login` to include it"
                )
                continue
            built.append((src, stats))
            print(
                f"  scanned={stats.scanned} emitted={stats.emitted} "
                f"dup={stats.dup_skipped} unresolved={stats.unresolved_dropped}"
            )

    # coverage gate: fail loudly rather than emit a silently-gappy dataset
    if not built:
        tmp.unlink(missing_ok=True)
        raise SystemExit("no sources were built (all gated or filtered out)")
    stats_by_key = {src.key: s for src, s in built}
    render_report(built, args)
    failed = coverage_failures(stats_by_key)
    if failed:
        tmp.unlink(missing_ok=True)
        for failure in failed:
            print(
                f"FAIL {failure.key}: coverage {failure.rate:.2f}% < "
                f"{COVERAGE_TARGET}%"
            )
            for name, n in failure.unmapped.most_common(20):
                print(f"    {n:7d}  {name}")
            print(
                "    -> add these to data/model_registry/overrides.yaml, "
                "rebuild the registry, retry"
            )
        raise SystemExit(1)

    tmp.replace(args.out)
    total = sum(s.emitted for _src, s in built)
    print(f"wrote {args.out} ({total} rows) and {REPORT_PATH}")


if __name__ == "__main__":
    main()
