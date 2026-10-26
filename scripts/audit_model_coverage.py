#!/usr/bin/env python3
"""Audit registry coverage over a dataset's raw model names.

Reads every model name that appears in a routing dataset (Hugging Face arena
data or a local csv/parquet/jsonl file), resolves each against the model
complexity registry, and reports the resolution rate plus a frequency-ranked
list of unmapped names. Iterate on overrides.yaml until coverage is >= 99%.

Usage:
  python scripts/audit_model_coverage.py                       # arena 140k
  python scripts/audit_model_coverage.py --dataset <hf_id>     # other HF set
  python scripts/audit_model_coverage.py --file data/foo.csv   # local file
  python scripts/audit_model_coverage.py --columns model_a,model_b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from router.model_registry import get_registry  # noqa: E402

DEFAULT_DATASET = "lmarena-ai/arena-human-preference-140k"
DEFAULT_COLUMNS = ["model_a", "model_b"]
OUT_REPORT = ROOT / "eval_reports" / "model_coverage_audit.md"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id")
    ap.add_argument("--file", type=Path, help="local csv/parquet/jsonl instead of HF dataset")
    ap.add_argument("--columns", default=",".join(DEFAULT_COLUMNS),
                    help="comma-separated columns holding model names")
    args = ap.parse_args()

    columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    counts = names_from_file(args.file, columns) if args.file else names_from_hf(args.dataset, columns)

    registry = get_registry()
    resolved: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    total_names = total_rows = 0
    for name, n in counts.items():
        total_names += 1
        total_rows += n
        entry = registry.resolve(name)
        if entry is None:
            unmapped[name] += n
        else:
            resolved[name] += n

    n_unmapped_rows = sum(unmapped.values())
    print(f"unique names: {total_names} | rows: {total_rows}")
    print(f"resolved rows: {total_rows - n_unmapped_rows}/{total_rows} "
          f"({100 * (total_rows - n_unmapped_rows) / max(total_rows, 1):.2f}%)")
    print(f"unmapped unique names: {len(unmapped)}")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_REPORT.open("w") as f:
        f.write("# model coverage audit\n\n")
        src = args.file or args.dataset
        f.write(f"source: {src} | columns: {columns}\n\n")
        f.write(f"unique names: {total_names} | battle rows: {total_rows}\n")
        rate = 100 * (total_rows - n_unmapped_rows) / max(total_rows, 1)
        f.write(f"resolved: {rate:.2f}% of rows | {len(unmapped)} unmapped names\n\n")
        if unmapped:
            f.write("## unmapped names (by frequency)\n\n")
            f.write("| name | rows |\n|---|---|\n")
            for name, n in unmapped.most_common():
                f.write(f"| `{name}` | {n} |\n")
    print(f"wrote {OUT_REPORT}")
    if unmapped:
        print("\ntop unmapped:")
        for name, n in unmapped.most_common(20):
            print(f"  {n:7d}  {name}")


if __name__ == "__main__":
    main()
