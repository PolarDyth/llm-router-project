"""Build training pairs from LMArena battles with scrubbed model descriptions.

Maps each battle's model names to OpenRouter profiles (data/model_profiles.json),
scrubs model-identifying strings out of the descriptions, and writes
winner/loser pairs (label 1.0 / 0.0) to data/router_training_pairs_scrubbed.csv.
"""

import argparse
import csv
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset

# Trailing tokens that mark a variant of the same base model rather than a
# different model (e.g. "claude-opus-4-20250514-thinking-16k" -> claude-opus-4).
VARIANT_TOKENS = {
    "thinking", "no-thinking", "non-thinking", "high", "low", "latest",
    "preview", "beta", "experimental", "exp", "ondemand",
}
# Words that may trail the matched base name on the OpenRouter side without
# counting as a different model ("mini", "lite", "image", "pro" are NOT here:
# those are different models).
SAFE_SUFFIX_WORDS = {
    "instruct", "thinking", "chat", "turbo", "latest", "ondemand",
    "preview", "beta", "exp", "it",
}

# 1. Locate files
project_root = Path(__file__).resolve().parents[1]
json_path = project_root / "data" / "model_profiles.json"
output_csv = project_root / "data" / "router_training_pairs_scrubbed.csv"


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    """Load model profiles, accepting both the curated dict format and a raw
    OpenRouter /api/v1/models response."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        entries = cast("list[dict[str, Any]]", payload["data"])
        payload = {m["id"]: m for m in entries if isinstance(m, dict) and "id" in m}
    elif isinstance(payload, list):
        entries = cast("list[dict[str, Any]]", payload)
        payload = {m["id"]: m for m in entries if isinstance(m, dict) and "id" in m}
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(
            f"[profiles] {path} is empty or not a recognized format. Expected "
            ' {"<model-id>": {"name": ..., "description": ...}}  or a raw '
            "OpenRouter /api/v1/models response."
        )
    usable = sum(1 for p in payload.values() if str(p.get("description") or "").strip())
    print(f"[profiles] Loaded {len(payload)} models ({usable} with descriptions)")
    if usable == 0:
        raise SystemExit("[profiles] No model has a description - cannot build pairs.")
    return cast("dict[str, dict[str, Any]]", payload)


# 2. Load the real OpenRouter profiles
print("Loading OpenRouter profiles...")
profiles = load_profiles(json_path)


@lru_cache(maxsize=None)
def _normalize(name: str) -> tuple[str, frozenset[str]]:
    """Canonical form for comparing arena names with OpenRouter ids.

    Returns (canonical form, date/version tokens that were stripped). The
    canonical form lowercases and drops trailing variant tokens (thinking,
    no-thinking, 16k, beta, ...) and date fragments (2025-04-16, 0528), so
    "claude-3.7-sonnet" and "claude-3-7-sonnet-20250219" both become
    "claude37sonnet". Single digits are kept: they are version numbers
    (grok-4), and the stripped date tokens are returned so callers can tell
    two differently-dated snapshots (mistral-small-2506 vs -2603) apart.
    """
    tokens = [t for t in re.split(r"[-_.\s:]+", name.lower().strip()) if t]
    dates: set[str] = set()
    while len(tokens) > 1:
        if "-".join(tokens[-2:]) in VARIANT_TOKENS:  # e.g. "no-thinking"
            tokens.pop()
            tokens.pop()
        elif tokens[-1] in VARIANT_TOKENS or re.fullmatch(r"\d+k", tokens[-1]):
            tokens.pop()
        elif re.fullmatch(r"\d{2,}", tokens[-1]):
            dates.add(tokens.pop())
        else:
            break
    kept: list[str] = []
    for token in tokens:
        stripped = re.sub(r"\d{4,}", "", token)
        if stripped != token:
            dates.update(re.findall(r"\d{4,}", token))
        if stripped:
            kept.append(stripped)
    return "".join(kept), frozenset(dates)


def _safe_tail(remainder: str) -> bool:
    """True if the unmatched trailing text is a harmless variant suffix
    (word like "instruct", or a size spec like "17b-16e"), not a version bump
    ("claude-opus-4" must not match "claude-opus-4.5", "grok-3" must not
    match "grok-4")."""
    if not remainder or any(remainder.startswith(w) for w in SAFE_SUFFIX_WORDS):
        return True
    # A size spec like "17b-16e", "70b", "128e" (at most three digits);
    # longer digit runs are version numbers ("3224b..." from 3.2-24b).
    return re.match(r"\d{1,3}[bex]", remainder) is not None


def _match_score(arena_norm: str, profile_norm: str) -> int:
    if arena_norm == profile_norm:
        return 3
    if arena_norm.startswith(profile_norm) and _safe_tail(arena_norm[len(profile_norm):]):
        return 2
    if profile_norm.startswith(arena_norm) and _safe_tail(profile_norm[len(arena_norm):]):
        return 1
    return 0


def find_matching_profile(arena_model_name: str) -> tuple[str, dict[str, Any]] | None:
    """Best (or_id, profile) match for an arena model name, or None."""
    arena_norm, arena_dates = _normalize(arena_model_name)
    if not arena_norm:
        return None
    best: tuple[int, int, str] | None = None  # (score, name length, profile id)
    for or_id in profiles:
        short_id = or_id.split(":")[0].split("/")[-1]
        profile_norm, profile_dates = _normalize(short_id)
        # If both names carry dates and they differ, these are two different
        # snapshots (mistral-small-2506 vs mistral-small-2603), not variants.
        if arena_dates and profile_dates and arena_dates != profile_dates:
            continue
        score = _match_score(arena_norm, profile_norm)
        if score == 0:
            continue
        rank = (score, len(profile_norm), or_id)
        if best is None or rank > best:
            best = rank
    if best is None:
        return None
    return best[2], profiles[best[2]]


@lru_cache(maxsize=None)
def get_aggressively_scrubbed_description(arena_model_name: str) -> str | None:
    match = find_matching_profile(arena_model_name)
    if match is None:
        return None
    _, profile = match

    desc = str(profile.get("description") or "").strip()
    if not desc:
        return None
    # OpenRouter truncates long descriptions with a trailing "...". Keep the
    # text that is there; rejecting it used to discard ~90% of the catalog.
    if desc.endswith("..."):
        desc = desc[:-3].rstrip()
    if len(desc) < 30:
        return None

    # Greedy Regex: Swallows attached numbers, hyphens, and dots (e.g., Qwen3-30B)
    model_roots = [
        "gpt", "chatgpt", "o1", "o3", "o4", "claude", "sonnet", "opus", "haiku",
        "llama", "qwen", "mistral", "mixtral", "magistral", "gemma", "gemini",
        "phi", "grok", "yi", "command", "nemotron", "deepseek", "kimi",
        "minimax", "nova", "hunyuan", "ernie", "glm",
    ]
    roots_pattern = r'(?i)\b(?:' + '|'.join(model_roots) + r')[a-zA-Z0-9\-\.]*'
    desc = re.sub(roots_pattern, "this AI model", desc)

    # Scrub provider companies
    providers = [
        "OpenAI", "Anthropic", "Google", "Meta", "Mistral", "Cohere", "Qwen",
        "Microsoft", "Alibaba", "xAI", "Tencent", "DeepSeek", "MoonshotAI",
        "Moonshot", "MiniMax", "Amazon", "Nvidia", "Salesforce", "AI21",
    ]
    provider_pattern = r'(?i)\b(?:' + '|'.join(providers) + r')\b'
    desc = re.sub(provider_pattern, "the developer", desc)

    # Scrub exact name strings that might have slipped through
    name = str(profile.get("name") or "")
    if len(name) > 3:
        desc = re.sub(re.escape(name), "this AI model", desc, flags=re.IGNORECASE)

    return desc


# 3. Parse arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--max-pairs", type=int, default=15000,
                    help="Stop after writing this many pairs (default 15000).")
args = parser.parse_args()
max_pairs = args.max_pairs

# 4. Load the Hugging Face arena dataset
print("Downloading/Loading 140k dataset from Hugging Face...")
dataset = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")

# 5. Process and generate pairs
print("Processing pairs and aggressively scrubbing names...")
training_data: list[list[Any]] = []
unmatched_names: Counter[str] = Counter()
battles = 0
used_battles = 0

for row in dataset:
    # Hugging Face datasets rows are plain dicts at runtime, but the
    # Dataset iteration type does not express that to type checkers.
    battle = cast("dict[str, Any]", row)
    battles += 1

    if len(training_data) + 2 > max_pairs:
        break

    try:
        prompt = battle["conversation_a"][0]["content"][0]["text"]
    except (IndexError, KeyError, TypeError):
        continue

    winner = battle["winner"]
    if winner not in ["model_a", "model_b"]:
        continue

    desc_a = get_aggressively_scrubbed_description(battle["model_a"])
    desc_b = get_aggressively_scrubbed_description(battle["model_b"])

    # Only keep battles where both descriptions resolved to something usable
    if desc_a is None or desc_b is None:
        for name, desc in ((battle["model_a"], desc_a), (battle["model_b"], desc_b)):
            if desc is None:
                unmatched_names[name] += 1
        continue

    used_battles += 1
    if winner == "model_a":
        training_data.append([prompt, desc_a, 1.0])
        training_data.append([prompt, desc_b, 0.0])
    else:
        training_data.append([prompt, desc_b, 1.0])
        training_data.append([prompt, desc_a, 0.0])

    if battles % 500 == 0:
        print(f"[arena] {battles} battles seen, {len(training_data)} pairs written")

# 6. Save the new dataset
with output_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sentence1", "sentence2", "label"])
    writer.writerows(training_data)

print(f"\nSuccess! Generated {len(training_data)} scrubbed training pairs "
      f"from {used_battles} of {battles} battles.")
print(f"Saved to: {output_csv}")
if unmatched_names:
    print(f"\nSkipped battles because these arena models had no matching "
          f"profile with a description (often delisted from OpenRouter):")
    for name, count in unmatched_names.most_common(15):
        print(f"  {count:5d}  {name}")
