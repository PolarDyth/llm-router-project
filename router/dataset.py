"""Training-dataset stage: raw source rows -> labeled (prompt, tier) rows.

Three source families, each a generator that yields dataset rows and accumulates
into a `SourceStats` it is handed. Generators rather than writers because the
dataset file is ~365 MB: nothing here may materialize a source's rows, and the
caller stays a single `for row in ...: out.write(...)`.

What each family produces, and why the label is trustworthy:

  arena              one row per human battle. `winner_tier`: the tier of the
                     model a human preferred. Ties/both_bad keep both tiers and
                     a null label, so the pairs stage can still use them.
  routerbench        one row per prompt, wide pkl of per-model quality.
                     `min_solving_tier`: the cheapest tier that clears
                     best - epsilon. A prompt nobody solves is null-labeled
                     rather than "everyone solves it", which would be noise.
  complexity_router  one row per prompt, complexity levels annotated directly by
                     a model (no router involved), mapped linearly onto tiers.

The coverage gate is this stage's contract: a source that resolves model names
must resolve >= 99% of them, or the build hard-fails. Silent name-dropping would
skew the tier histogram toward whatever the registry happened to know.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from router.contracts import COVERAGE_TARGET
from router.sources import (
    COMPLEXITY_ROUTER_HF,
    ROUTERBENCH_FILE,
    ROUTERBENCH_HF,
    by_key,
    routerbench_model_columns,
)

# ComplexityRouter's 4-level annotation scale onto the router's 5-tier scale.
# Linear: level 3 lands on tier 4, so tier 5 stays arena/routerbench-only
# signal and the synthetic rewrites ("frontier needed") stop short of it.
LEVEL_TO_TIER = {0: 1, 1: 2, 2: 3, 3: 4}
COMPLEXITY_ROUTER_SYNTHETIC_ID = "Level0_to_Level3"  # id marker of rewrites


class NameResolver(Protocol):
    """What the adapters need of a registry: raw model name -> complexity tier.

    Declaring the narrow capability instead of taking a concrete `ModelRegistry`
    is what makes these adapters testable with a dict-backed fake -- no catalog
    file, no network, no class mocking. `router.model_registry.ModelRegistry`
    satisfies it structurally.
    """

    def complexity(self, name: str) -> int | None: ...


@dataclass
class SourceStats:
    scanned: int = 0
    emitted: int = 0
    dup_skipped: int = 0
    unresolved_dropped: int = 0
    #: Model-name slots this source actually tried to resolve. The coverage gate
    #: is applied iff this is > 0, which is what keeps a model-free source exempt
    #: without hardcoding which sources qualify.
    name_slots: int = 0
    tier_hist: Counter[int] = field(default_factory=Counter)
    winner_hist: Counter[str] = field(default_factory=Counter)
    unmapped: Counter[str] = field(default_factory=Counter)
    prompt_chars: list[int] = field(default_factory=list)
    label_rule: str | None = None

    @property
    def coverage(self) -> float | None:
        """Resolved share of model-name slots, or None when there were none."""
        if not self.name_slots:
            return None
        return 100 * (self.name_slots - sum(self.unmapped.values())) / self.name_slots



@dataclass
class CoverageFailure:
    key: str
    rate: float
    unmapped: Counter[str]


def coverage_failures(
    stats: dict[str, SourceStats], target: float = COVERAGE_TARGET
) -> list[CoverageFailure]:
    """Sources whose resolved-name rate is below the gate.

    A source with no model-name slots is not a failure: there was nothing to
    resolve (complexity_router). Sources that *do* name models but under-resolve
    them are the ones that would silently skew the tier mix.
    """
    failed = []
    for key, s in stats.items():
        rate = s.coverage
        if rate is not None and rate < target:
            failed.append(CoverageFailure(key, rate, s.unmapped))
    return failed


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


def iter_arena_rows(key: str, hf_id: str) -> Iterator[dict[str, Any]]:
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


def arena_rows(
    key: str,
    hf_id: str,
    registry: NameResolver,
    stats: SourceStats,
    seen: set[str],
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    """Labeled arena dataset rows: both tiers plus the human winner's tier."""
    label_kind = by_key(key).label_kind
    for item in iter_arena_rows(key, hf_id):
        if limit is not None and stats.scanned >= limit:
            break
        stats.scanned += 1
        prompt = item["prompt"]
        if not prompt:
            stats.unresolved_dropped += 1
            continue
        stats.name_slots += 2  # a battle names exactly two models
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
        # Counted *before* the yield, not after: these are generators, so a
        # consumer that breaks early would otherwise under-report rows it has
        # already taken. The id keeps the pre-increment value.
        row_id = f"{key}:{stats.emitted:07d}"
        stats.emitted += 1
        yield {
            "id": row_id,
            "source": key,
            "label_kind": label_kind,
            "prompt": prompt,
            "model_a": item["model_a"],
            "model_b": item["model_b"],
            "tier_a": tier_a,
            "tier_b": tier_b,
            "winner": winner,
            "label_tier": label_tier,
            **item["extra"],
        }


# --------------------------------------------------------------------------
# routerbench adapter


def routerbench_rows(
    registry: NameResolver,
    stats: SourceStats,
    seen: set[str],
    epsilon: float,
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    """Label each prompt with the cheapest tier whose model solves it.

    The 0-shot pkl is wide: one row per prompt, one bare '<model>' quality
    column per model (plus '<model>|model_response' and '<model>|total_cost'
    detail columns). The rule that was applied is recorded on `stats.label_rule`
    for the report, since whether scores are binary changes it.
    """
    from huggingface_hub import hf_hub_download

    import pandas as pd

    path = hf_hub_download(
        repo_id=ROUTERBENCH_HF,
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
                f = float(v)
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
    stats.label_rule = thr_desc
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
                continue  # this model was never run on this prompt: no slot
            stats.name_slots += 1
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
        row_id = f"routerbench:{stats.emitted:07d}"
        stats.emitted += 1  # before yield, as in arena_rows
        row: dict[str, Any] = {
            "id": row_id,
            "source": "routerbench",
            "label_kind": by_key("routerbench").label_kind,
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
        yield row


# --------------------------------------------------------------------------
# complexity_router adapter


def complexity_router_rows(
    stats: SourceStats,
    seen: set[str],
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    """Ingest RowRed/ComplexityRouter (model-free, direct complexity labels).

    Train split only -- the 400-row test split stays untouched as an external
    benchmark. Dedup runs through the shared `seen` set so prompts already
    labeled by arena/routerbench are not double-labeled under a second,
    possibly conflicting label (the notebook's val split groups by prompt).

    Emits the routerbench row shape with `solving_tiers=[label_tier]`, so the
    pairs stage needs no third branch: the gold tier wins, the other four lose.
    """
    from datasets import load_dataset

    label_kind = by_key("complexity_router").label_kind
    ds = load_dataset(COMPLEXITY_ROUTER_HF, split="train")
    for raw in ds:
        if limit is not None and stats.scanned >= limit:
            break
        stats.scanned += 1
        row = cast(dict[str, Any], raw)
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            stats.unresolved_dropped += 1
            continue
        try:
            level = int(row["level"])
        except (KeyError, TypeError, ValueError):
            stats.unresolved_dropped += 1
            continue
        tier = LEVEL_TO_TIER.get(level)
        if tier is None:
            stats.unresolved_dropped += 1
            continue
        dk = dedup_key(prompt, ["complexity_router"])
        if dk in seen:
            stats.dup_skipped += 1
            continue
        seen.add(dk)
        stats.tier_hist[tier] += 1
        stats.prompt_chars.append(len(prompt))
        row_id = f"complexity_router:{stats.emitted:07d}"
        stats.emitted += 1  # before yield, as in arena_rows
        yield {
            "id": row_id,
            "source": "complexity_router",
            "label_kind": label_kind,
            "prompt": prompt,
            "level": level,
            "category": str(row.get("category", "")),
            "reason": str(row.get("reason", "")),
            "synthetic": str(row.get("original_message_id", ""))
            == COMPLEXITY_ROUTER_SYNTHETIC_ID,
            "candidate_tiers": [1, 2, 3, 4, 5],
            "solving_tiers": [tier],
            "label_tier": tier,
        }
