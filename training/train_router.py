#!/usr/bin/env python3
"""Train the LLM router cross-encoder.

Works on NVIDIA GPUs, AMD GPUs (ROCm on Linux/Windows, DirectML on Windows),
Apple Silicon (MPS), and CPU. Run with --help for options.
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import batch_to_device
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from device_utils import VALID_DEVICES, select_device

# The build scripts emit these, most preferred first.
DATASET_CANDIDATES = [
    "router_training_pairs_scrubbed.csv",
    "router_training_pairs_v2.csv",
    "router_training_pairs.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Training CSV with columns sentence1,sentence2,label. "
        "Defaults to the first of these found under data/: "
        + ", ".join(DATASET_CANDIDATES),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "dynamic_router" / "final",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--model-name",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Base cross-encoder model to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--device",
        default="auto",
        choices=VALID_DEVICES,
        help="Accelerator to train on (default: auto-detect).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_dataset(arg: Path | None) -> Path:
    if arg is not None:
        for candidate in (arg, PROJECT_ROOT / arg):
            if candidate.is_file():
                return candidate
        raise SystemExit(f"[data] Dataset not found: {arg}")
    for name in DATASET_CANDIDATES:
        candidate = PROJECT_ROOT / "data" / name
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "[data] No training CSV found under data/ (looked for "
        + ", ".join(DATASET_CANDIDATES)
        + "). Generate one first, e.g.: python scripts/build_scrubbed_arena.py"
    )


def load_pairs(csv_path: Path) -> tuple[list[list[str]], list[float]]:
    df = pd.read_csv(csv_path, dtype={"sentence1": str, "sentence2": str})
    missing = {"sentence1", "sentence2", "label"} - set(df.columns)
    if missing:
        raise SystemExit(
            f"[data] {csv_path} is missing column(s): {', '.join(sorted(missing))}"
        )
    df = df.dropna(subset=["sentence1", "sentence2", "label"])
    if df.empty:
        raise SystemExit(f"[data] {csv_path} contains no usable rows")

    pairs: list[list[str]] = []
    labels: list[float] = []
    for sentence1, sentence2, label in zip(df["sentence1"], df["sentence2"], df["label"]):
        try:
            value = float(label)
        except (TypeError, ValueError):
            continue
        if math.isnan(value):
            continue
        pairs.append([str(sentence1), str(sentence2)])
        labels.append(value)
    dropped = len(df) - len(pairs)
    if dropped:
        print(f"[data] Dropped {dropped} row(s) with missing or non-numeric values")
    return pairs, labels


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    dataset_path = resolve_dataset(args.dataset)
    pairs, labels = load_pairs(dataset_path)
    num_batches = math.ceil(len(pairs) / args.batch_size)
    print(
        f"[data] {len(pairs)} pairs from {dataset_path.name} "
        f"({num_batches} batches/epoch, {args.epochs} epochs)"
    )

    # Identity activation so forward() yields raw logits for BCEWithLogitsLoss;
    # the Sigmoid default is restored before saving.
    model = CrossEncoder(
        args.model_name,
        num_labels=1,
        max_length=512,
        # sentence-transformers accepts torch.device at runtime (its predict()
        # signature allows str | torch.device); the __init__ annotation is
        # just narrower than reality.
        device=device,  # pyright: ignore[reportArgumentType]
        activation_fn=torch.nn.Identity(),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_batches * args.epochs,
    )

    print(f"[train] Starting training on {device}...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        start = time.perf_counter()
        order = list(range(len(pairs)))
        random.shuffle(order)
        for step, start_idx in enumerate(range(0, len(pairs), args.batch_size), 1):
            batch_idx = order[start_idx : start_idx + args.batch_size]
            batch_pairs = [pairs[i] for i in batch_idx]
            batch_labels = torch.tensor(
                [labels[i] for i in batch_idx], dtype=torch.float32, device=device
            )
            features = model.preprocess(batch_pairs)
            features = batch_to_device(features, device)

            optimizer.zero_grad()
            # CrossEncoder.forward returns raw logits under "scores" (the
            # Identity activation guarantees no sigmoid is applied here).
            logits = model(features)["scores"]
            loss = loss_fn(logits.squeeze(-1).float(), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            if step % 100 == 0 or step == num_batches:
                print(
                    f"[train] epoch {epoch} step {step}/{num_batches} "
                    f"loss {loss.item():.4f}"
                )
        print(
            f"[train] epoch {epoch} done: avg loss "
            f"{epoch_loss / num_batches:.4f} in {time.perf_counter() - start:.1f}s"
        )

    # Restore the standard sentence-transformers behaviour for num_labels=1:
    # predict() returns sigmoid probabilities.
    model.activation_fn = torch.nn.Sigmoid()
    model.save_pretrained(str(args.output_dir))
    print(f"[train] Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
