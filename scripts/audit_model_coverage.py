#!/usr/bin/env python3
"""Audit registry coverage over a dataset's raw model names.

Reads every model name that appears in a routing dataset (Hugging Face arena
data, RouterBench, or a local csv/parquet/jsonl file), resolves each against
the model complexity registry, and reports the resolution rate plus a
frequency-ranked list of unmapped names. Iterate on overrides.yaml until
every training source is >= 99%.

`--all` audits the same source catalog `build_training_dataset.py` builds from
(`router.sources.AUDITABLE`), so this predicts exactly what the build's
coverage gate will decide. Model-free sources have no names to resolve and are
not listed.

Usage:
  python scripts/audit_model_coverage.py                       # arena 140k
  python scripts/audit_model_coverage.py --all                 # full training suite
  python scripts/audit_model_coverage.py --dataset <hf_id>     # other HF set
  python scripts/audit_model_coverage.py --file data/foo.csv   # local file
  python scripts/audit_model_coverage.py --columns model_a,model_b
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import TextIO

from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import GatedRepoError

from router.contracts import COVERAGE_TARGET
from router.model_registry import ModelRegistry, get_registry
from router.sources import (
    AUDITABLE,
    NAME_MODE_PKL_MODEL_COLUMNS,
    ROUTERBENCH_FILE,
    ROUTERBENCH_HF,
    Source,
    by_key,
    routerbench_model_columns,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_REPORT = ROOT / "eval_reports" / "model_coverage_audit.md"

# Derived from the catalog rather than restated, so the audit's defaults cannot
# drift from the sources it is auditing.
_ARENA_140K = by_key("arena_140k")
DEFAULT_DATASET = _ARENA_140K.hf_id
DEFAULT_COLUMNS = list(_ARENA_140K.model_columns)

# (source or None, display label, resolved, unmapped, total rows)
AuditResult = tuple["Source | None", str, Counter[str], Counter[str], int]


def names_from_hf(dataset_id: str, columns: list[str]) -> Counter[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")
    missing = [c for c in columns if c not in ds.column_names]
    if missing:
        raise SystemExit(f"columns {missing} not in {dataset_id}; has: {ds.column_names}")
    counts: Counter[str] = Counter()
    cols = [ds[c] for c in columns]
    for values in zip(*cols):
        for value in values:
            name = str(value).strip()
            if name:
                counts[name] += 1
    return counts


def names_from_routerbench() -> Counter[str]:
    """The wide pkl's model names are its column headers, one row per prompt."""
    from huggingface_hub import hf_hub_download

    import pandas as pd

    path = hf_hub_download(repo_id=ROUTERBENCH_HF,
                           filename=ROUTERBENCH_FILE, repo_type="dataset")
    df = pd.read_pickle(path)
    model_cols = routerbench_model_columns(list(df.columns))
    if not model_cols:
        raise SystemExit(f"routerbench pkl has no per-model columns; has: {list(df.columns)}")
    counts: Counter[str] = Counter()
    for col in model_cols:
        n = sum(1 for v in df[col].tolist() if v is not None and v == v)  # skip NaN
        if n:
            counts[str(col).strip()] += n
    return counts


def names_from_file(path: Path, columns: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path, columns=columns)
        for row in df[columns].itertuples(index=False, name=None):
            for value in row:
                name = str(value).strip()
                if name:
                    counts[name] += 1
    else:  # csv / jsonl
        if path.suffix == ".jsonl":
            with path.open() as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
        for row in rows:
            for c in columns:
                name = row.get(c)
                if name:
                    counts[str(name).strip()] += 1
    return counts


def resolve_counts(registry: ModelRegistry,
                   counts: Counter[str]) -> tuple[Counter[str], Counter[str]]:
    """Split a name-frequency counter into resolved / unmapped, weighted by rows."""
    resolved: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    for name, n in counts.items():
        if registry.resolve(name) is None:
            unmapped[name] += n
        else:
            resolved[name] += n
    return resolved, unmapped


def load_names(source: Source) -> Counter[str]:
    if source.name_mode == NAME_MODE_PKL_MODEL_COLUMNS:
        return names_from_routerbench()
    return names_from_hf(source.hf_id, list(source.model_columns))


def _print_counts(label: str, resolved: Counter[str], unmapped: Counter[str],
                  total: int) -> None:
    n_bad = sum(unmapped.values())
    print(f"{label}: unique names: {len(resolved) + len(unmapped)} | rows: {total}")
    print(f"  resolved rows: {total - n_bad}/{total} "
          f"({100 * (total - n_bad) / max(total, 1):.2f}%) | unmapped names: {len(unmapped)}")
    if unmapped:
        print("  top unmapped:")
        for name, n in unmapped.most_common(20):
            print(f"    {n:7d}  {name}")


def _write_unmapped(f: TextIO, header: str, unmapped: Counter[str]) -> None:
    f.write(f"## unmapped names — {header} (by frequency)\n\n")
    f.write("| name | rows |\n|---|---|\n")
    for name, n in unmapped.most_common():
        f.write(f"| `{name}` | {n} |\n")
    f.write("\n")


def write_report(results: list[AuditResult], label: str, columns: list[str],
                 skipped: list[str]) -> None:
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_REPORT.open("w") as f:
        f.write("# model coverage audit\n\n")
        if skipped:
            f.write(f"skipped (gated / not accessible — accept the license on "
                    f"huggingface.co, then `hf auth login`): {', '.join(skipped)}\n\n")
        if len(results) == 1:
            _source, src_label, resolved, unmapped, total = results[0]
            n_bad = sum(unmapped.values())
            rate = 100 * (total - n_bad) / max(total, 1)
            f.write(f"source: {src_label} | columns: {columns}\n\n")
            f.write(f"unique names: {len(resolved) + len(unmapped)} | battle rows: {total}\n")
            f.write(f"resolved: {rate:.2f}% of rows | {len(unmapped)} unmapped names\n\n")
            if unmapped:
                _write_unmapped(f, src_label, unmapped)
        else:
            f.write(f"source: {label}\n\n")
            f.write("| source | rows | unique names | resolved % | unmapped names | pass? |\n")
            f.write("|---|---|---|---|---|---|\n")
            for source, _label, resolved, unmapped, total in results:
                n_bad = sum(unmapped.values())
                rate = 100 * (total - n_bad) / max(total, 1)
                ok = "yes" if rate >= COVERAGE_TARGET else "**NO**"
                key = source.key if source else "?"
                f.write(f"| {key} | {total} | {len(resolved) + len(unmapped)} | "
                        f"{rate:.2f}% | {len(unmapped)} | {ok} |\n")
            f.write(f"\ntarget: >= {COVERAGE_TARGET}% per source\n\n")
            for source, _label, _resolved, unmapped, _total in results:
                if unmapped:
                    _write_unmapped(f, source.key if source else "?", unmapped)
    print(f"wrote {OUT_REPORT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id")
    ap.add_argument("--file", type=Path, help="local csv/parquet/jsonl instead of HF dataset")
    ap.add_argument("--columns", default=",".join(DEFAULT_COLUMNS),
                    help="comma-separated columns holding model names")
    ap.add_argument("--all", action="store_true",
                    help="audit the full training suite (4 arena sets + RouterBench)")
    args = ap.parse_args()

    registry = get_registry()
    results: list[AuditResult] = []
    skipped: list[str] = []
    columns: list[str] = []
    label: str
    if args.all:
        if args.file:
            raise SystemExit("--all cannot be combined with --file")
        label = "full training suite"
        for source in AUDITABLE:
            print(f"auditing {source.key} ({source.hf_id}) ...")
            try:
                counts = load_names(source)
            except (DatasetNotFoundError, GatedRepoError) as e:
                print(f"  SKIP {source.key}: gated or inaccessible "
                      f"({type(e).__name__}: {str(e)[:120]})")
                print("  -> accept the license on huggingface.co and run "
                      "`hf auth login` to include it")
                skipped.append(source.key)
                continue
            resolved, unmapped = resolve_counts(registry, counts)
            total = sum(counts.values())
            results.append((source, source.hf_id, resolved, unmapped, total))
            _print_counts(source.key, resolved, unmapped, total)
    elif args.file:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
        counts = names_from_file(args.file, columns)
        resolved, unmapped = resolve_counts(registry, counts)
        total = sum(counts.values())
        results.append((None, str(args.file), resolved, unmapped, total))
        _print_counts(str(args.file), resolved, unmapped, total)
        label = str(args.file)
    else:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
        counts = names_from_hf(args.dataset, columns)
        resolved, unmapped = resolve_counts(registry, counts)
        total = sum(counts.values())
        results.append((None, args.dataset, resolved, unmapped, total))
        _print_counts(args.dataset, resolved, unmapped, total)
        label = args.dataset

    write_report(results, label, columns, skipped)


if __name__ == "__main__":
    main()
