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
(MPS/CPU) against a Colab-trained checkpoint.

The split, max length and metric definitions come from `router.contracts` and
`router.metrics` — the same modules the notebook's inline copies are tested
against — so the numbers printed here are the notebook's numbers by
construction rather than by discipline. `tier_texts.json` is still read from
the checkpoint, because the checkpoint defines what its own scores mean; the
constants only describe the split and the truncation.

Usage:
  python scripts/eval_checkpoint.py                          # defaults below
  python scripts/eval_checkpoint.py --checkpoint models/complexity_router/final
  python scripts/eval_checkpoint.py --skip-train --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from router.contracts import MAX_LEN, VAL_PROMPT_MOD
from router.devices import VALID_DEVICES, select_device
from router.metrics import (
    Triplet,
    group_by_prompt,
    majority_baseline,
    metric_cohort,
    pairwise_accuracy,
    routing_accuracy,
    train_split,
    val_split,
)

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DEFAULT = ROOT / "models" / "complexity_router" / "final"
PAIRS_DEFAULT = ROOT / "data" / "router_training_pairs.jsonl"

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


def load_pairs(path: Path) -> list[Triplet]:
    """The pairs file reduced to the three fields the router trains on."""
    rows: list[Triplet] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append((r["prompt"], int(r["tier"]), float(r["outcome"])))
    return rows


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
    ap.add_argument("--device", default=None,
                    help=f"force one of: {', '.join(VALID_DEVICES[1:])} (default: auto)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-train", action="store_true",
                    help="skip the training-split diagnostic")
    args = ap.parse_args()

    tier_texts = {int(k): v for k, v in json.loads(
        (args.checkpoint / "tier_texts.json").read_text()).items()}
    rows = load_pairs(args.pairs)

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))
    # num_labels=1 must be restated: the head is a scalar regressor over the
    # tier text, and the default would rebuild a 2-class head that cannot load.
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.checkpoint), num_labels=1).to(device).eval()
    print(f"checkpoint: {args.checkpoint} | device: {device} | "
          f"pairs: {len(rows)} from {args.pairs.name}")

    val = val_split(rows, VAL_PROMPT_MOD)
    by_prompt = group_by_prompt(val)
    val_scores = score_matrix(model, tokenizer, device, tier_texts,
                              sorted(by_prompt), args.batch_size)
    decisive, eval_pairs = metric_cohort(by_prompt)
    print(f"\nval split: {len(val)} rows / {len(by_prompt)} prompts "
          f"({len(decisive)} decisive, {eval_pairs} eval pairs)")
    baseline = majority_baseline(by_prompt)
    if baseline:
        print(f"majority-tier baseline: {baseline[0]:.3f} "
              f"(always tier {baseline[1]})")
    print(f"pairwise accuracy:  {pairwise_accuracy(by_prompt, val_scores):.3f}")
    print(f"routing accuracy:   {routing_accuracy(by_prompt, val_scores):.3f}")

    if not args.skip_train:
        train_by = group_by_prompt(train_split(rows, VAL_PROMPT_MOD))
        train_scores = score_matrix(model, tokenizer, device, tier_texts,
                                    sorted(train_by), args.batch_size)
        print(f"train split diagnostic: {len(rows) - len(val)} rows / "
              f"{len(train_by)} prompts")
        print(f"  pairwise accuracy:  "
              f"{pairwise_accuracy(train_by, train_scores):.3f}")
        print(f"  routing accuracy:   "
              f"{routing_accuracy(train_by, train_scores):.3f}")

    route_demo(model, tokenizer, device, tier_texts)


if __name__ == "__main__":
    main()
