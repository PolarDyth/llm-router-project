#!/usr/bin/env python3
"""Build the (prompt -> complexity) training dataset.

Sources (all labeled by resolving raw model names through the complexity
registry in data/model_registry/models.json):

  arena  -- four LMArena human-preference battle sets (33k/55k/100k/140k,
            mid-2023 -> mid-2025). One row per battle: prompt, both models,
            their tiers, and the human winner. label_kind="winner_tier":
            label_tier is the tier of the model that won (null for ties /
            both_bad, which are kept for training-side use).

  routerbench -- withmartian/routerbench 0-shot pkl (~30k prompts x 11
            models, per-prompt quality + cost). One row per prompt.
            label_kind="min_solving_tier": the lowest tier whose best model
            reaches the quality threshold (best - epsilon, or exact-correct
            when scores are binary); solving_tiers lists every tier that
            cleared the bar (empty when nothing did).

Gates: model-name coverage per source must be >= 99% (add gaps to
data/model_registry/overrides.yaml, rebuild the registry, retry); duplicate
battles (same prompt + model pair) across releases are dropped.

Usage:
  python scripts/build_training_dataset.py                     # arena only
  python scripts/build_training_dataset.py --sources arena,routerbench
  python scripts/build_training_dataset.py --limit 500         # smoke run
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO, TextIO, cast

from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import GatedRepoError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from router.model_registry import ModelRegistry, get_registry  # noqa: E402

OUT_DEFAULT = ROOT / "data" / "router_training_dataset.jsonl"
REPORT_PATH = ROOT / "eval_reports" / "training_dataset_report.md"
COVERAGE_TARGET = 99.0

ARENA_SOURCES = [
    ("arena_33k", "lmsys/chatbot_arena_conversations"),
    ("arena_55k", "lmarena-ai/arena-human-preference-55k"),
    ("arena_100k", "lmarena-ai/arena-human-preference-100k"),
    ("arena_140k", "lmarena-ai/arena-human-preference-140k"),
]

ROUTERBENCH_FILE = "routerbench_0shot.pkl"


@dataclass
class SourceStats:
    scanned: int = 0
    emitted: int = 0
    dup_skipped: int = 0
    unresolved_dropped: int = 0
    tier_hist: Counter[int] = field(default_factory=Counter)
    winner_hist: Counter[str] = field(default_factory=Counter)
    unmapped: Counter[str] = field(default_factory=Counter)
    prompt_chars: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------
# prompt / winner normalization shared by all arena generations


def _content_text(content: Any) -> str:
    """Message content: plain string, or a list of parts with 'text' fields."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def first_user_text(conversation: Any) -> str:
    """First user turn from a conversation in any arena schema.

    Conversations arrive as a list of message dicts (role/content), sometimes
    as a JSON string wrapping that list. Content is a string (openai format)
    or a list of parts (140k format).
    """
    if isinstance(conversation, str):
        try:
            conversation = json.loads(conversation)
        except json.JSONDecodeError:
            try:
                conversation = ast.literal_eval(conversation)  # python-repr lists
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                return conversation.strip()
    if not isinstance(conversation, list):
        return ""
    if any(isinstance(t, str) for t in conversation):
        # 55k style: a JSON list of turn texts, not message dicts
        return "\n".join(
            t for t in conversation if isinstance(t, str) and t.strip()
        ).strip()
    for msg in conversation:  # prefer a turn actually marked as user/human
        if isinstance(msg, dict) and str(msg.get("role", "")).lower() in (
            "user",
            "human",
            "prompter",
            "none",
            "",
        ):
            text = _content_text(msg.get("content")).strip()
            if text:
                return text
    for msg in conversation:  # fallback: first non-empty content
        if isinstance(msg, dict):
            text = _content_text(msg.get("content")).strip()
            if text:
                return text
    return ""


def norm_winner(raw: str) -> str:
    s = str(raw).strip().lower()
    if s == "model_a":
        return "a"
    if s == "model_b":
        return "b"
    if "bothbad" in s or "both_bad" in s:
        return "both_bad"
    if "tie" in s:
        return "tie"
    return "unknown"


def dedup_key(prompt: str, models: list[str]) -> str:
    norm = " ".join(prompt.split())
    pair = "\x00".join(sorted(m.strip().lower() for m in models))
    return hashlib.sha1(f"{pair}\x00{norm}".encode()).hexdigest()


# --------------------------------------------------------------------------
# arena adapter


def iter_arena_rows(key: str, hf_id: str) -> Any:
    """Yield normalized dicts {prompt, model_a, model_b, winner, extra}."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, split="train")
    cols = set(ds.column_names)
    flat_prompt = "prompt" in cols  # 55k generation: flat columns, one-hot winner

    wanted = ["model_a", "model_b"]
    if flat_prompt:
        wanted += ["prompt", "winner_model_a", "winner_model_b", "winner_tie"]
    else:
        wanted += ["conversation_a", "winner"]
        for opt in ("language", "is_code", "category_tag"):
            if opt in cols:
                wanted.append(opt)
    ds = ds.select_columns(wanted)

    for raw in ds:
        row = cast(dict[str, Any], raw)
        if flat_prompt:
            prompt = first_user_text(row["prompt"])
            if row.get("winner_model_a"):
                winner = "a"
            elif row.get("winner_model_b"):
                winner = "b"
            elif row.get("winner_tie"):
                winner = "tie"
            else:
                winner = "unknown"
            extra: dict[str, Any] = {}
        else:
            prompt = first_user_text(row["conversation_a"])
            winner = norm_winner(row["winner"])
            extra = {
                opt: row[opt]
                for opt in ("language", "is_code", "category_tag")
                if opt in row and row[opt] is not None
            }
            if isinstance(extra.get("category_tag"), (dict, list)):
                extra["category_tag"] = json.dumps(
                    extra["category_tag"], separators=(",", ":")
                )
        yield {
            "prompt": prompt,
            "model_a": str(row["model_a"]).strip(),
            "model_b": str(row["model_b"]).strip(),
            "winner": winner,
            "extra": extra,
        }


def build_arena(
    key: str,
    hf_id: str,
    registry: ModelRegistry,
    out: TextIO,
    stats: SourceStats,
    seen: set[str],
    limit: int | None,
) -> None:
    for item in iter_arena_rows(key, hf_id):
        if limit is not None and stats.scanned >= limit:
            break
        stats.scanned += 1
        prompt = item["prompt"]
        if not prompt:
            stats.unresolved_dropped += 1
            continue
        tier_a = registry.complexity(item["model_a"])
        tier_b = registry.complexity(item["model_b"])
        if tier_a is None:
            stats.unmapped[item["model_a"]] += 1
        if tier_b is None:
            stats.unmapped[item["model_b"]] += 1
        if tier_a is None or tier_b is None:
            stats.unresolved_dropped += 1
            continue
        winner = item["winner"]
        stats.winner_hist[winner] += 1
        dk = dedup_key(prompt, [item["model_a"], item["model_b"]])
        if dk in seen:
            stats.dup_skipped += 1
            continue
        seen.add(dk)
        label_tier = tier_a if winner == "a" else tier_b if winner == "b" else None
        if label_tier is not None:
            stats.tier_hist[label_tier] += 1
        stats.prompt_chars.append(len(prompt))
        out.write(
            json.dumps(
                {
                    "id": f"{key}:{stats.emitted:07d}",
                    "source": key,
                    "label_kind": "winner_tier",
                    "prompt": prompt,
                    "model_a": item["model_a"],
                    "model_b": item["model_b"],
                    "tier_a": tier_a,
                    "tier_b": tier_b,
                    "winner": winner,
                    "label_tier": label_tier,
                    **item["extra"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        stats.emitted += 1


# --------------------------------------------------------------------------
# routerbench adapter

ROUTERBENCH_META_COLS = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}


def routerbench_model_columns(columns: list[str]) -> list[str]:
    """Wide pkl: quality lives in bare '<model>' columns; '<model>|suffix' is not."""
    return [
        c for c in columns if "|" not in str(c) and str(c) not in ROUTERBENCH_META_COLS
    ]


def build_routerbench(
    registry: ModelRegistry,
    out: TextIO,
    stats: SourceStats,
    seen: set[str],
    epsilon: float,
    limit: int | None,
) -> str:
    """Label each prompt with the cheapest tier whose model solves it.

    The 0-shot pkl is wide: one row per prompt, one bare '<model>' quality
    column per model (plus '<model>|model_response' and '<model>|total_cost'
    detail columns). Returns the quality rule applied, for the report.
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(
        repo_id="withmartian/routerbench",
        filename=ROUTERBENCH_FILE,
        repo_type="dataset",
    )
    df = pd.read_pickle(path)
    if "prompt" not in df.columns:
        raise SystemExit(
            f"routerbench pkl has no 'prompt' column; has: {list(df.columns)}"
        )
    model_cols = routerbench_model_columns(list(df.columns))
    if not model_cols:
        raise SystemExit(
            f"routerbench pkl has no per-model columns; has: {list(df.columns)}"
        )

    prompts = [str(p) for p in df["prompt"].tolist()]
    evals = (
        [str(v) for v in df["eval_name"].tolist()]
        if "eval_name" in df.columns
        else None
    )
    oracles = (
        [str(v) for v in df["oracle_model_to_route_to"].tolist()]
        if "oracle_model_to_route_to" in df.columns
        else None
    )
    # quality per model column; NaN/None -> None
    qualities: dict[str, list[float | None]] = {}
    for col in model_cols:
        vals: list[float | None] = []
        for v in df[col].tolist():
            try:
                f = float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                f = None
            vals.append(f if f is not None and f == f else None)
        qualities[col] = vals

    tier_by_col: dict[str, int | None] = {c: registry.complexity(c) for c in model_cols}
    all_q = [q for col in model_cols for q in qualities[col] if q is not None]
    binary = bool(all_q) and set(all_q) <= {0.0, 1.0}
    thr_desc = (
        "quality == 1.0 (scores are binary correct/incorrect)"
        if binary
        else f"quality >= per-prompt best - {epsilon}; prompts where the "
        f"best quality is 0 (nothing solved) get a null label"
    )
    print(
        f"routerbench quality: min={min(all_q):.3f} max={max(all_q):.3f} "
        f"binary={binary} | label rule: {thr_desc}"
    )

    for i in range(len(prompts)):
        if limit is not None and stats.scanned >= limit:
            break
        stats.scanned += 1
        prompt = first_user_text(prompts[i]).strip()
        if not prompt:
            stats.unresolved_dropped += 1
            continue
        dk = dedup_key(prompt, ["routerbench"])
        if dk in seen:
            stats.dup_skipped += 1
            continue
        pairs: list[tuple[int, float]] = []
        for col in model_cols:
            q = qualities[col][i]
            if q is None:
                continue
            tier = tier_by_col[col]
            if tier is None:
                stats.unmapped[col] += 1
                continue
            pairs.append((tier, q))
        if not pairs:
            stats.unresolved_dropped += 1
            continue
        best = max(q for _t, q in pairs)
        thr = 1.0 if binary else best - epsilon
        tiers = {t for t, _q in pairs}
        solving = {t for t, q in pairs if q >= thr}
        seen.add(dk)
        # best quality 0 = the oracle itself says no model solved the prompt;
        # an "everyone solves it" label would be noise, so null it like a tie
        label_tier = min(solving) if solving and best > 0.0 else None
        if label_tier is not None:
            stats.tier_hist[label_tier] += 1
        stats.prompt_chars.append(len(prompt))
        row: dict[str, Any] = {
            "id": f"routerbench:{stats.emitted:07d}",
            "source": "routerbench",
            "label_kind": "min_solving_tier",
            "prompt": prompt,
            "candidate_tiers": sorted(tiers),
            "solving_tiers": sorted(solving) if best > 0.0 else [],
            "label_tier": label_tier,
            "best_quality": round(best, 4),
        }
        if evals is not None:
            row["dataset"] = evals[i]
        if oracles is not None:
            row["oracle_model"] = oracles[i]
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats.emitted += 1
    return thr_desc


# --------------------------------------------------------------------------
# reporting


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.2f}%" if d else "n/a"


def percentile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]


def model_slots(key: str, s: SourceStats) -> int:
    """Model-name slots the source contributes (2 per arena battle, 1 per rb row)."""
    rows = s.emitted + s.unresolved_dropped
    return 2 * rows if key != "routerbench" else rows


def write_report(
    sources: list[tuple[str, str, SourceStats]],
    args: argparse.Namespace,
    thr_desc: str | None,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_hist: Counter[int] = Counter()
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
        for key, _hf_id, s in sources:
            slots = model_slots(key, s)
            coverage = pct(slots - sum(s.unmapped.values()), slots)
            ties = s.winner_hist.get("tie", 0) + s.winner_hist.get("both_bad", 0)
            f.write(
                f"| {key} | {s.scanned} | {s.emitted} | {s.dup_skipped} | "
                f"{s.unresolved_dropped} | {coverage} | {ties} |\n"
            )
        f.write("\n")
        for key, _hf_id, s in sources:
            f.write(f"## {key}\n\n")
            if s.unmapped:
                f.write("unmapped model names (rows dropped):\n\n")
                f.write("| name | rows |\n|---|---|\n")
                for name, n in s.unmapped.most_common():
                    f.write(f"| `{name}` | {n} |\n")
                f.write("\n")
            if key != "routerbench":
                f.write(f"winner distribution: {dict(s.winner_hist)}\n\n")
            lengths = sorted(s.prompt_chars)
            f.write(
                f"prompt chars p50/p95/p99/max: {percentile(lengths, 0.5)} / "
                f"{percentile(lengths, 0.95)} / {percentile(lengths, 0.99)} / "
                f"{lengths[-1] if lengths else 0}\n\n"
            )
            f.write(f"label_tier histogram: {dict(sorted(s.tier_hist.items()))}\n\n")
            total_hist.update(s.tier_hist)
        if thr_desc:
            f.write(f"routerbench label rule: {thr_desc}\n\n")
        f.write(
            f"total emitted: {sum(s.emitted for _k, _i, s in sources)} | "
            f"combined label_tier histogram: {dict(sorted(total_hist.items()))}\n"
        )


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="arena", help="comma list: arena, routerbench")
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

    requested = {s.strip() for s in args.sources.split(",") if s.strip()}
    unknown = requested - {"arena", "routerbench"}
    if unknown:
        raise SystemExit(
            f"unknown sources: {sorted(unknown)}; valid: arena, routerbench"
        )

    registry = get_registry()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sources: list[tuple[str, str, SourceStats]] = []
    seen: set[str] = set()
    thr_desc: str | None = None

    tmp = args.out.with_suffix(".jsonl.tmp")
    with tmp.open("w") as out:
        if "arena" in requested:
            for key, hf_id in ARENA_SOURCES:
                print(f"building {key} ({hf_id}) ...")
                stats = SourceStats()
                try:
                    build_arena(key, hf_id, registry, out, stats, seen, args.limit)
                except (DatasetNotFoundError, GatedRepoError) as e:
                    print(
                        f"  SKIP {key}: gated or inaccessible "
                        f"({type(e).__name__}: {str(e)[:120]})"
                    )
                    print(
                        "  -> accept the license on huggingface.co and run "
                        "`hf auth login` to include it"
                    )
                    continue
                sources.append((key, hf_id, stats))
                print(
                    f"  scanned={stats.scanned} emitted={stats.emitted} "
                    f"dup={stats.dup_skipped} unresolved={stats.unresolved_dropped}"
                )
        if "routerbench" in requested:
            print("building routerbench (withmartian/routerbench, 0-shot) ...")
            stats = SourceStats()
            try:
                thr_desc = build_routerbench(
                    registry, out, stats, seen, args.epsilon, args.limit
                )
            except (DatasetNotFoundError, GatedRepoError) as e:
                print(
                    f"  SKIP routerbench: gated or inaccessible "
                    f"({type(e).__name__}: {str(e)[:120]})"
                )
            else:
                sources.append(("routerbench", "withmartian/routerbench", stats))
                print(
                    f"  scanned={stats.scanned} emitted={stats.emitted} "
                    f"dup={stats.dup_skipped} unresolved={stats.unresolved_dropped}"
                )

    # coverage gate: fail loudly rather than emit a silently-gappy dataset
    if not sources:
        tmp.unlink(missing_ok=True)
        raise SystemExit("no sources were built (all gated or filtered out)")
    failed = []
    for key, _hf_id, s in sources:
        slots = model_slots(key, s)
        dropped = sum(s.unmapped.values())
        rate = 100 * (slots - dropped) / slots if slots else 100.0
        if rate < COVERAGE_TARGET:
            failed.append((key, rate, s.unmapped))
    write_report(sources, args, thr_desc)
    if failed:
        tmp.unlink(missing_ok=True)
        for key, rate, unmapped in failed:
            print(f"FAIL {key}: coverage {rate:.2f}% < {COVERAGE_TARGET}%")
            for name, n in unmapped.most_common(20):
                print(f"    {n:7d}  {name}")
            print(
                "    -> add these to data/model_registry/overrides.yaml, "
                "rebuild the registry, retry"
            )
        raise SystemExit(1)

    tmp.replace(args.out)
    total = sum(s.emitted for _k, _i, s in sources)
    print(f"wrote {args.out} ({total} rows) and {REPORT_PATH}")


if __name__ == "__main__":
    main()
