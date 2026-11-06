"""The (prompt, tier) interface: the constants training and inference must agree on.

Routing is argmax over five tier scores, so the tier strings are part of the
model contract, not presentation: change ``TIER_TEXTS`` and every existing
checkpoint is scoring a different question than it was trained on.

The Colab notebook cannot import this module (it runs with only the pairs file
uploaded, no repo checkout), so it re-declares these values inline. That copy
is forced; the drift is not accepted silently -- `tests/test_contracts.py`
parses the notebook JSON and fails if any constant here stops matching the
notebook cell that uses it. ``scripts/eval_checkpoint.py`` imports this module
directly and has no reason to carry a third copy.
"""

from __future__ import annotations

import hashlib

TIERS: tuple[int, ...] = (1, 2, 3, 4, 5)

#: Scored against the prompt, in order; argmax of these five is the routing decision.
TIER_TEXTS: dict[int, str] = {
    1: "complexity tier 1 - a tiny flash-tier model for simple tasks",
    2: "complexity tier 2 - a light mini-tier model",
    3: "complexity tier 3 - a mid-tier pro model",
    4: "complexity tier 4 - a high-capability frontier model",
    5: "complexity tier 5 - a top frontier model for the hardest tasks",
}

#: Prompts are truncated at this length. 256 left 11.1% of labeled prompts
#: tokenizing ``(prompt, tier_text)`` past the cut -- and the over-budget rows
#: were disproportionately the long, hard ones -- so the budget is the backbone's
#: ceiling rather than a speed knob.
#:
#: 512 IS that ceiling: DeBERTa-v3's ``max_position_embeddings`` is exactly 512,
#: which is what ``tests/test_contracts.py`` asserts a checkpoint satisfies. Going
#: above 512 is not a constant edit -- it needs the position embeddings resized,
#: which invalidates the pretrained weights that make the fine-tune work at all.
#: The tier text eats ~15 tokens of the budget, so the prompt itself gets ~497.
MAX_LEN = 512

#: Prompt-grouped validation split: sha1(prompt) % VAL_PROMPT_MOD == 0 -> val.
#: Splitting on the prompt (not the row) is what keeps a prompt's winning and
#: losing tiers out of opposite sides of the split. ~5% at 20.
VAL_PROMPT_MOD = 20

#: A source that resolves model names must resolve at least this share of them
#: or `build_training_dataset.py` hard-fails and writes no file. Two scripts
#: enforce/predict this number (the build gate and the coverage audit), which is
#: why it is a contract rather than a constant in either of them: if they drift,
#: the audit reports a pass for a build that then refuses to run.
COVERAGE_TARGET = 99.0

#: Registry-side meaning of the same 1-5 scale, written into models.json.
COMPLEXITY_SCALE: dict[str, str] = {
    "1": "tiny/flash class (sub-3B, flash/mini/nano tiers)",
    "2": "light class (7-14B open, mini/haiku hosted, legacy 2022 frontier)",
    "3": "mid class (70B open, pro hosted, 2023 frontier)",
    "4": "pro class (gpt-4o/sonnet/2024 frontier)",
    "5": "frontier class (o1+/opus/gpt-5 era)",
}


def is_validation_prompt(prompt: str, mod: int = VAL_PROMPT_MOD) -> bool:
    """The split predicate, shared by training and eval so neither re-derives it."""
    return int(hashlib.sha1(prompt.encode()).hexdigest(), 16) % mod == 0
