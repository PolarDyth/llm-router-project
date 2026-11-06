#!/usr/bin/env python3
"""Evaluate a saved complexity-router checkpoint locally (inference only).

Loads a checkpoint (weights + tokenizer + tier_texts.json), reproduces the
notebook's prompt-hash validation split over a pairs file, and reports the
same numbers the notebook prints per epoch: the majority-tier baseline,
pairwise accuracy (winner tier scored above loser tier on held-out prompts)
and routing accuracy (argmax over the five tier scores vs the single gold
tier). Then routes a few demo prompts taken straight from the training data,
with their gold tiers shown.

This never trains: batched forward passes only, safe on the Mac
(MPS/CPU) against a Colab-trained checkpoint. The split, max length and
tier texts must match training — they are read from the checkpoint's
tier_texts.json and reproduce the notebook's constants by default.

Usage:
  python scripts/eval_checkpoint.py                          # defaults below
  python scripts/eval_checkpoint.py --checkpoint models/complexity_router/final
  python scripts/eval_checkpoint.py --skip-train --device cpu
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import torch

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DEFAULT = ROOT / "models" / "complexity_router" / "final"
PAIRS_DEFAULT = ROOT / "data" / "router_training_pairs.jsonl"
MAX_LEN = 256        # must match training
VAL_PROMPT_MOD = 20  # notebook split: sha1(prompt) % 20 == 0 -> validation

# One demo prompt per labeled level, all pulled from the pairs source data
# (gold = the labeled tier, shown next to the routing decision).
DEMO_PROMPTS = [
    ("Is playing bass easier than playing guitar?", 1),
    ("What's a version control system and how do I get started using one?", 2),
    ("How does an induction stove work? Is it any better than gas or electric?", 3),
    ("Design an adaptive, self-healing parity-checking algorithm for distributed "
     "quantum computing nodes with fault-tolerant error correction", 4),
    ("Derive the sum of 1 and 2 by modeling them as abstract algebraic generators "
     "within a non-commutative ring theory framework while simultaneously "
     "computing the resulting cohomology classes", 4),
]


def pick_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    try:
        torch_directml = importlib.import_module("torch_directml")  # never a hard import
        device_fn = getattr(torch_directml, "device", None)
        if callable(device_fn):
            return cast(torch.device, device_fn())
    except (ImportError, OSError):
        pass
    return torch.device("cpu")


def load_pairs(path: Path) -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append((r["prompt"], int(r["tier"]), float(r["outcome"])))
    return rows


def notebook_val_split(rows: list[tuple[str, int, float]],
                       val_mod: int) -> list[tuple[str, int, float]]:
    return [
        triplet for triplet in rows
        if int(hashlib.sha1(triplet[0].encode()).hexdigest(), 16) % val_mod == 0
    ]


@torch.no_grad()
def score_matrix(model: Any, tokenizer: Any, device: torch.device,
                 tier_texts: dict[int, str], prompts: list[str],
                 batch_size: int) -> dict[str, dict[int, float]]:
    """Score every (prompt, tier) combo; batched forward passes."""
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    keys: list[tuple[str, int]] = []
    for prompt in prompts:
        for tier, text in tier_texts.items():
            enc = tokenizer(prompt, text, truncation=True, max_length=MAX_LEN)
            input_ids.append(enc["input_ids"])
            attention_mask.append(enc["attention_mask"])
            keys.append((prompt, tier))

    scores: dict[str, dict[int, float]] = {p: {} for p in prompts}
    for start in range(0, len(keys), batch_size):
        chunk_ids = input_ids[start:start + batch_size]
        chunk_mask = attention_mask[start:start + batch_size]
        padded = tokenizer.pad({"input_ids": chunk_ids,
                                "attention_mask": chunk_mask},
                               return_tensors="pt").to(device)
        logits = model(**padded).logits.squeeze(-1).cpu().tolist()
        for (prompt, tier), value in zip(keys[start:start + batch_size], logits):
            scores[prompt][tier] = float(value)
    return scores


def metric_cohort(by_prompt: dict[str, dict[int, float]]
                  ) -> tuple[list[str], int]:
    """Decisive prompts (exactly one 1.0 outcome) + their eval-pair count."""
    decisive = [
        prompt for prompt, tiers in by_prompt.items()
        if list(tiers.values()).count(1.0) == 1
    ]
    pairs = 0
    for prompt in decisive:
        gold = next(t for t, o in by_prompt[prompt].items() if o == 1.0)
        pairs += sum(1 for o in by_prompt[prompt].values() if o == 0.0)
        _ = gold
    return decisive, pairs


def pairwise_accuracy(by_prompt: dict[str, dict[int, float]],
                      scores: dict[str, dict[int, float]]) -> float:
    correct = total = 0
    for prompt, tiers in by_prompt.items():
        wins = [t for t, o in tiers.items() if o == 1.0]
        losses = [t for t, o in tiers.items() if o == 0.0]
        for w in wins:
            for l in losses:
                total += 1
                correct += scores[prompt][w] > scores[prompt][l]
    return correct / max(total, 1)


def routing_accuracy(by_prompt: dict[str, dict[int, float]],
                     scores: dict[str, dict[int, float]]) -> float:
    correct = total = 0
    for prompt, tiers in by_prompt.items():
        gold = [t for t, o in tiers.items() if o == 1.0]
        if len(gold) != 1:
            continue
        total += 1
        correct += max(scores[prompt], key=lambda t: scores[prompt][t]) == gold[0]
    return correct / max(total, 1)


def majority_baseline(by_prompt: dict[str, dict[int, float]]) -> tuple[float, int] | None:
    golds = [
        next(t for t, o in tiers.items() if o == 1.0)
        for tiers in by_prompt.values()
        if list(tiers.values()).count(1.0) == 1
    ]
    if not golds:
        return None
    top, count = Counter(golds).most_common(1)[0]
    return count / len(golds), top


def route_demo(model: Any, tokenizer: Any, device: torch.device,
               tier_texts: dict[int, str]) -> None:
    print("\ndemo routes (gold tier from the labeled data):")
    scores = score_matrix(model, tokenizer, device, tier_texts,
                          [p for p, _g in DEMO_PROMPTS], batch_size=8)
    for prompt, gold in DEMO_PROMPTS:
        logits = scores[prompt]
        ranked = sorted(logits.items(), key=lambda kv: -kv[1])
        picks = " ".join(f"t{t}={v:+.2f}" for t, v in ranked)
        marker = "ok" if ranked[0][0] == gold else "MISS"
        print(f"  [{marker}] gold {gold} | {picks}")
        print(f"        > {prompt[:90]}")


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    ap.add_argument("--pairs", type=Path, default=PAIRS_DEFAULT)
    ap.add_argument("--device", default=None, help="cpu / mps / cuda override")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-train", action="store_true",
                    help="skip the training-split diagnostic")
    args = ap.parse_args()

    tier_texts = {int(k): v for k, v in json.loads(
        (args.checkpoint / "tier_texts.json").read_text()).items()}
    rows = load_pairs(args.pairs)

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.checkpoint), num_labels=1).to(device).eval()
    print(f"checkpoint: {args.checkpoint} | device: {device} | "
          f"pairs: {len(rows)} from {args.pairs.name}")

    val = notebook_val_split(rows, VAL_PROMPT_MOD)
    val_prompts = sorted({p for p, _t, _o in val})
    by_prompt: dict[str, dict[int, float]] = {}
    for prompt, tier, outcome in val:
        by_prompt.setdefault(prompt, {})[tier] = outcome
    val_scores = score_matrix(model, tokenizer, device, tier_texts,
                              val_prompts, args.batch_size)
    decisive, eval_pairs = metric_cohort(by_prompt)
    print(f"\nval split: {len(val)} rows / {len(val_prompts)} prompts "
          f"({decisive and len(decisive)} decisive, {eval_pairs} eval pairs)")
    baseline = majority_baseline(by_prompt)
    if baseline:
        print(f"majority-tier baseline: {baseline[0]:.3f} "
              f"(always tier {baseline[1]})")
    print(f"pairwise accuracy:  {pairwise_accuracy(by_prompt, val_scores):.3f}")
    print(f"routing accuracy:   {routing_accuracy(by_prompt, val_scores):.3f}")

    if not args.skip_train:
        train = [triplet for triplet in rows if triplet not in set(val)]
        train_prompts = sorted({p for p, _t, _o in train})
        train_by: dict[str, dict[int, float]] = {}
        for prompt, tier, outcome in train:
            train_by.setdefault(prompt, {})[tier] = outcome
        train_scores = score_matrix(model, tokenizer, device, tier_texts,
                                    train_prompts, args.batch_size)
        print(f"train split diagnostic: {len(train)} rows / "
              f"{len(train_prompts)} prompts")
        print(f"  pairwise accuracy:  "
              f"{pairwise_accuracy(train_by, train_scores):.3f}")
        print(f"  routing accuracy:   "
              f"{routing_accuracy(train_by, train_scores):.3f}")

    route_demo(model, tokenizer, device, tier_texts)


if __name__ == "__main__":
    main()
