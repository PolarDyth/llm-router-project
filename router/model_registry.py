"""Model complexity registry: loader and name resolver.

The registry (data/model_registry/models.json) maps canonical chat LLMs
(2022-2026) to a complexity tier 1-5. Dataset pipelines call
``resolve(name)`` to swap raw model names for complexity values.

Tier scale:
  1 tiny/flash   2 light/mini   3 mid/pro   4 pro/2024-frontier   5 frontier
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = ROOT / "data" / "model_registry" / "models.json"

# hand aliases for names whose raw form loses tier information when stripped.
# Keep in sync with scripts/build_model_registry.py (imported from here).
HAND_ALIASES = {
    "gpt-4-1106-preview": "openai/gpt-4-turbo",
    "gpt-4-0125-preview": "openai/gpt-4-turbo",
    "gpt-4-turbo-preview": "openai/gpt-4-turbo",
    "gpt-4-vision-preview": "openai/gpt-4-turbo",
    "gpt-4-32k": "openai/gpt-4",
    "gpt-4-32k-0314": "openai/gpt-4",
    "gpt-4-32k-0613": "openai/gpt-4",
    "chatgpt-4o-latest": "openai/gpt-4o",
    "gemini-pro": "google/gemini-1.0-pro",
    "gemini-pro-vision": "google/gemini-1.0-pro-vision",
    "gemini-flash-1.5": "google/gemini-1.5-flash",
    "gemini-pro-1.5": "google/gemini-1.5-pro",
    "bard": "google/gemini-1.0-pro",
    "deepseek": "deepseek/deepseek-chat",
    "o1-preview": "openai/o1",
    "o1-2024-12-17": "openai/o1",
    "claude-instant": "anthropic/claude-instant-1.2",
    # release-number spellings: normalization strips "-v2", which would
    # otherwise collapse onto whatever entry claims bare "claude", so these
    # must be matched on the raw name (see resolve())
    "claude-v1": "anthropic/claude-1",
    "claude-v2": "anthropic/claude-2",
    # arena's "amazon-nova-experimental-chat" reduces to bare "nova"
    "nova": "amazon/nova-pro",
}

# vendor families whose -chat/-instruct/-turbo suffixes are serving noise
OPEN_STRIP = re.compile(
    r"^(?:llama|vicuna|qwen|qwq|mistral|mixtral|ministral|gemma|codellama|deepseek|"
    r"yi-|glm|chatglm|phi-|dbrx|falcon|olmo|nemotron|exaone|jamba|command|solar|"
    r"hermes|openchat|starling|tulu|tinyllama|smol|stablelm|baichuan|internlm|"
    r"mpt|wizardlm|wizardcoder|gpt4all|dolly|alpaca|guanaco|koala|rwkv|stablecode)"
)


def normalize_name(raw: str) -> str:
    """Reduce any model name variant to its canonical registry name."""
    s = raw.strip().lower()
    if s.startswith("~"):  # OpenRouter pointer aliases (e.g. ~unbiased/pareto-latest)
        s = s.lstrip("~")
    if "/" in s:
        s = s.rpartition("/")[2]  # hosting/org prefixes are never tier-bearing
    s = re.sub(r"^(?:us|eu|apac|au|global|us-gov|us-gov-1)\.", "", s)
    changed = True
    while changed:
        changed = False
        if re.match(r"^[a-z]{2}\.", s):  # bedrock regions ca. jp. kr. (us-gov. handled above)
            s = re.sub(r"^[a-z]{2}\.", "", s, count=1)
            changed = True
        for ns in ("anthropic.", "amazon.", "meta.", "ai21.", "cohere.", "mistral.",
                   "deepseek.", "writer.", "luma.", "stability.", "twelvelabs.",
                   "nvidia.", "openai.", "google.", "qwen.", "lg.", "moonshot."):
            if s.startswith(ns):
                s = s[len(ns):]
                changed = True
    s = re.sub(r"^[a-z0-9_.]+?--", "", s)             # gateway double prefixes
    s = re.sub(r":[\w.-]+$", "", s)                   # :0, :free, :extended, :beta
    s = re.sub(r"@\w[\w.-]*$", "", s)                 # vertex @20250219, edenai @us
    # trailing qualifiers: strip until stable (thinking variants, context
    # windows, serving dates, effort tags are all same-tier noise)
    while True:
        prev = s
        s = re.sub(r"-v\d+(?::[\w.]+)?$", "", s)      # bedrock -v1:0
        s = re.sub(r"-20\d{6}", "", s)                # -20240229
        s = re.sub(r"-20\d{2}-\d{2}-\d{2}", "", s)    # -2024-08-06
        s = re.sub(r"-\d{6}$", "", s)                 # -0613, -1106
        if re.match(r"^(?:gpt|qwen|mistral|magistral|ministral|deepseek|gemini)", s):
            s = re.sub(r"-\d{4}$", "", s)             # -0428, -2407, -0324
        if s.startswith("gemini"):
            s = re.sub(r"-\d{3}$", "", s)             # -001 serving snapshots
        s = re.sub(r"-\d+k$", "", s)                  # -16k, -32k context variants
        s = re.sub(r"-\d{2}-\d{2,4}$", "", s)         # -02-24, -05-14, -12-2024
        s = re.sub(
            r"-(?:preview|beta|latest|exp|experimental|online|hf|"
            r"thinking|no-thinking|high|low|none)$", "", s)
        if s.startswith(("meta-", "amazon-")) or OPEN_STRIP.match(s):
            s = re.sub(r"-(?:chat|instruct|turbo|math|moe|base|it)$", "", s)
            s = s.replace("v3p", "3.").replace("v2p", "2.")     # llama-v3p1-70b
            s = s.replace("llama-v", "llama-").replace("v3-", "3-").replace("v2-", "2-")
            s = re.sub(r"^(?:meta|amazon)-", "", s)
        if s == prev:
            break
    if s in HAND_ALIASES:
        return HAND_ALIASES[s]
    if re.match(r"^(?:gpt-3\.5|gpt-4|davinci)", s) and s.endswith("-16k"):
        s = s[: -len("-16k")]
    return s.strip("-")


class ModelRegistry:
    """Alias index over the generated registry."""

    def __init__(self, data: dict[str, Any]):
        self.generated_at: str = data.get("generated_at", "")
        self.models: list[dict[str, Any]] = data["models"]
        self._by_key: dict[str, dict[str, Any]] = {m["canonical_id"]: m for m in self.models}
        self._alias: dict[str, str] = {}
        for m in self.models:
            names = [m["canonical_id"], m["canonical_id"].split("/", 1)[1], *m["aliases"]]
            for a in names:
                self._alias.setdefault(a, m["canonical_id"])

    def entry(self, canonical_id: str) -> dict[str, Any] | None:
        return self._by_key.get(canonical_id)

    def resolve(self, name: str) -> dict[str, Any] | None:
        """Resolve any raw model name to its registry entry, or None."""
        s = name.strip().lower()
        # historical meaning wins for ambiguous marketing names ("gemini-pro"):
        # our training data is 2022-2025, where these names meant the originals
        hand = HAND_ALIASES.get(s) or HAND_ALIASES.get(normalize_name(s))
        if hand is not None:
            entry = self._by_key.get(hand)
            if entry is not None:
                return entry
        for candidate in self._candidates(s):
            key = self._alias.get(candidate)
            if key is not None:
                return self._by_key[key]
        return None

    def complexity(self, name: str) -> int | None:
        entry = self.resolve(name)
        return entry["complexity"] if entry else None

    @staticmethod
    def _candidates(s: str) -> list[str]:
        outs = [s, re.sub(r"[\s_]+", "-", s)]  # "Claude Opus 5.5" -> claude-opus-5.5
        norm = normalize_name(outs[1])
        outs.append(norm)
        # a normalized name that is itself "vendor/model" may hit either form
        if "/" in norm:
            outs.append(norm.rpartition("/")[2])
        # date-stamped snapshots of unlisted models: strip trailing stamps
        stripped = re.sub(r"-\d{6,8}$", "", norm)
        if stripped != norm:
            outs.append(stripped)
            if "/" in stripped:
                outs.append(stripped.rpartition("/")[2])
        return outs


@lru_cache(maxsize=1)
def get_registry(path: str | None = None) -> ModelRegistry:
    file = Path(path) if path else DEFAULT_REGISTRY_PATH
    with file.open() as f:
        return ModelRegistry(json.load(f))
