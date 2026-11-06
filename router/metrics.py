"""Router eval math: the split, the cohorts, and the three numbers.

`scripts/eval_checkpoint.py` imports these; the Colab notebook re-declares the
same math inline because it has no repo checkout. `tests/test_contracts.py`
holds the two to the same prompt-hash split and the same majority baseline, so
the local numbers stay the notebook's numbers.

What each metric actually means here -- and why the shapes look fussy:

  routing_accuracy   argmax over the five tier scores == the gold tier. Only
                     defined for prompts with exactly one 1.0, and it caps near
                     the human label-agreement rate (~0.42), so beating it is a
                     different claim than the raw number suggests.
  pairwise_accuracy  gold tier scored above loser tier, over all winner/loser
                     pairs. This is the one that tracks training progress
                     (0.66-0.68 is the current target band), because it credits
                     ordering the tiers even when the argmax lands elsewhere.
  majority_baseline  always answering the most common gold tier. A router that
                     does not beat this has learned to ignore the prompt.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Iterable

from router.contracts import VAL_PROMPT_MOD

#: One training pair as it arrives from the pairs file: (prompt, tier, outcome).
Triplet = tuple[str, int, float]
#: Rows grouped by prompt: prompt -> {tier: outcome}.
ByPrompt = dict[str, dict[int, float]]
#: Model output: prompt -> {tier: logit}.
Scores = dict[str, dict[int, float]]


def is_validation(triplet: Triplet, mod: int = VAL_PROMPT_MOD) -> bool:
    return int(hashlib.sha1(triplet[0].encode()).hexdigest(), 16) % mod == 0


def val_split(
    rows: Iterable[Triplet], mod: int = VAL_PROMPT_MOD
) -> list[Triplet]:
    """The notebook's held-out prompts: sha1(prompt) % mod == 0.

    Prompt-grouped, so a prompt's winner and loser tiers never straddle the
    split -- which is why routing accuracy on "val" is honest.
    """
    return [t for t in rows if is_validation(t, mod)]


def train_split(
    rows: Iterable[Triplet], mod: int = VAL_PROMPT_MOD
) -> list[Triplet]:
    return [t for t in rows if not is_validation(t, mod)]


def group_by_prompt(rows: Iterable[Triplet]) -> ByPrompt:
    by_prompt: ByPrompt = {}
    for prompt, tier, outcome in rows:
        by_prompt.setdefault(prompt, {})[tier] = outcome
    return by_prompt


def gold_tiers(by_prompt: ByPrompt) -> list[int]:
    """The single gold tier of every decisive prompt.

    "Decisive" is load-bearing: a prompt with two 1.0s (an arena tie kept as
    0.5/0.5 never produces one, but a hand-edited file could) has no argmax
    target, and one with no 1.0 has no winner to score against.
    """
    return [
        next(t for t, o in tiers.items() if o == 1.0)
        for tiers in by_prompt.values()
        if list(tiers.values()).count(1.0) == 1
    ]


def metric_cohort(by_prompt: ByPrompt) -> tuple[list[str], int]:
    """Decisive prompts + how many winner/loser eval pairs they contribute."""
    decisive = [
        prompt
        for prompt, tiers in by_prompt.items()
        if list(tiers.values()).count(1.0) == 1
    ]
    pairs = sum(
        sum(1 for o in by_prompt[prompt].values() if o == 0.0) for prompt in decisive
    )
    return decisive, pairs


def pairwise_accuracy(by_prompt: ByPrompt, scores: Scores) -> float:
    """Fraction of (gold tier, losing tier) pairs the model ordered correctly."""
    correct = total = 0
    for prompt, tiers in by_prompt.items():
        wins = [t for t, o in tiers.items() if o == 1.0]
        losses = [t for t, o in tiers.items() if o == 0.0]
        for w in wins:
            for l in losses:
                total += 1
                correct += scores[prompt][w] > scores[prompt][l]
    return correct / max(total, 1)


def routing_accuracy(by_prompt: ByPrompt, scores: Scores) -> float:
    """Fraction of decisive prompts whose argmax tier is the gold tier."""
    correct = total = 0
    for prompt, tiers in by_prompt.items():
        gold = [t for t, o in tiers.items() if o == 1.0]
        if len(gold) != 1:
            continue
        total += 1
        correct += max(scores[prompt], key=lambda t: scores[prompt][t]) == gold[0]
    return correct / max(total, 1)


def majority_baseline(by_prompt: ByPrompt) -> tuple[float, int] | None:
    """(accuracy, tier) of always answering the most common gold tier."""
    golds = gold_tiers(by_prompt)
    if not golds:
        return None
    top, count = Counter(golds).most_common(1)[0]
    return count / len(golds), top
