"""Vendor catalog acquisition: four public catalogs -> merged canonical groups.

This module turns downloaded catalog files into `Canonical` groups (one per real
model, with every raw alias and the best-known price / release date / quality
signal attached). It deliberately does **not** assign tiers -- that is
`router.scoring`'s job, and keeping the two apart is what lets the scoring
invariants be tested against hand-built groups instead of a network download.

The four catalogs each have a different shape and different reliability, which
is what the merge rules in `merge_records` are for:

`VENDOR_CATALOGS` (the four downloadable catalogs) is distinct from
`router.sources.SOURCES` (the dataset sources that feed training); the old shared
name for both made every import ambiguous in practice.

  LiteLLM      price-only, per-token floats, no release dates
  OpenRouter   price + artificial-analysis intelligence index + design-arena ELO,
               release date is a *listing* timestamp
  models.dev   price + a real release_date, keyed by provider
  Epoch AI     parameter counts + publication dates, keyed by a mangled name
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

from router.model_registry import normalize_name  # shared with the runtime resolver

VENDOR_CATALOGS = {
    "litellm.json": "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
    "openrouter.json": "https://openrouter.ai/api/v1/models",
    "modelsdev.json": "https://models.dev/api.json",
    "epoch_ai_models.csv": "https://datahub.io/ai/epoch-data-on-ai-models/r/0.csv",
}

# vendors whose models have public parameter counts and commodity hosting,
# where price reflects the hoster not the capability
OPEN_PARAMS_VENDORS = {
    "meta",
    "mistral",
    "alibaba",
    "google",
    "microsoft",
    "deepseek",
    "tii",
    "ai2",
    "mosaic",
    "nvidia",
    "huggingface",
    "nous",
    "01-ai",
    "shanghai-ai-lab",
    "lg-ai",
    "databricks",
    "upstage",
    "laion",
    "berkeley",
    "01-ai",
}

VENDOR_PATTERNS = [
    (r"claude", "anthropic"),
    (r"\bgpt|davinci|babbage|chatgpt|\bo[134](?:-|$)|gpt-oss", "openai"),
    (r"gemini|gemma|palm|bard|maddison", "google"),
    (r"llama|vicuna|guanaco|alpaca", "meta"),
    (r"mistral|mixtral|ministral|codestral|magistral|pixtral|devstral", "mistral"),
    (r"qwen|qwq", "alibaba"),
    (r"deepseek", "deepseek"),
    (r"command\b|command-", "cohere"),
    (r"glm|chatglm", "zhipu"),
    (r"kimi|moonshot", "moonshot"),
    (r"ernie", "baidu"),
    (r"falcon", "tii"),
    (r"\bmpt\b", "mosaic"),
    (r"\bphi-\d", "microsoft"),
    (r"\byi-\d", "01-ai"),
    (r"internlm", "shanghai-ai-lab"),
    (r"\bnova\b|titan\b", "amazon"),
    (r"jamba|\bj2\b", "ai21"),
    (r"dbrx|dolly", "databricks"),
    (r"grok", "xai"),
    (r"olmo", "ai2"),
    (r"nemotron", "nvidia"),
    (r"hermes|openhermes|nous-", "nous"),
    (r"zephyr|starcoder|starchat|santacoder", "huggingface"),
    (r"\bsolar\b", "upstage"),
    (r"exaone", "lg-ai"),
    (r"reka", "reka"),
    (r"oasst|pythia", "laion"),
    (r"koala", "berkeley"),
    (r"wizardlm|wizardcoder", "microsoft"),
    (r"\bairoboros", "jondurbin"),
    (r"\btoppy\b", "jebcarter"),
    (r"\batlas\b", "intellignosis"),
    (r"rwkv", "rwkv"),
    (r"fastchat", "lmsys"),
    (r"gpt4all", "nomic-ai"),
    (r"stablelm|stable-code", "stability-ai"),
]


class Rec:
    __slots__ = (
        "raw",
        "source",
        "name",
        "vendor_hint",
        "pin",
        "pout",
        "released",
        "ii",
        "elo",
        "params_b",
        "context",
    )

    def __init__(
        self,
        raw: str,
        source: str,
        name: str,
        vendor_hint: str | None = None,
        pin: float | None = None,
        pout: float | None = None,
        released: str | None = None,
        ii: float | None = None,
        elo: float | None = None,
        params_b: float | None = None,
        context: int | None = None,
    ):
        self.raw, self.source, self.name, self.vendor_hint = (
            raw,
            source,
            name,
            vendor_hint,
        )
        self.pin, self.pout, self.released = pin, pout, released
        self.ii, self.elo, self.params_b, self.context = ii, elo, params_b, context


class Canonical:
    """One real model: every raw name it appeared under, plus merged signals."""

    __slots__ = ("key", "aliases", "raws", "fields")

    def __init__(self, key: str):
        self.key = key
        self.aliases: set[str] = set()
        self.raws: list[str] = []
        self.fields: dict[str, Any] = {}


def infer_vendor(name_norm: str, hint: str | None = None) -> str:
    for pat, vendor in VENDOR_PATTERNS:
        if re.search(pat, name_norm):
            return vendor
    return hint or "unknown"


def parse_params(token: str | None) -> float | None:
    """'116830000000.0' / '7B (estimated)' -> billions."""
    if not token:
        return None
    t = token.strip()
    m = re.match(r"^([\d.]+)\s*(?:([kmb])\b)", t, re.I)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "k":
            return val / 1e6
        if unit == "m":
            return val / 1e3
        return val
    try:
        raw = float(t)
    except ValueError:
        return None
    return raw / 1e9 if raw > 1e6 else raw


def params_from_name(name_norm: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)b\b", name_norm)  # MoE: 8x7b -> 56
    if m:
        return float(m.group(1)) * float(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*b(?:illion)?\b", name_norm)
    if m:
        return float(m.group(1))
    return None


def epoch_key(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9.+-]", "", s)
    return re.sub(r"-{2,}", "-", s).strip("-")


def per_m(per_token: float | str | None) -> float | None:
    if per_token in (None, ""):
        return None
    try:
        v = float(per_token)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return v * 1e6


class CatalogCache:
    """The four downloaded catalogs, read from a directory.

    Holding the directory on the object (rather than a module-level `CACHE`
    constant) is what makes the loaders runnable against a fixture dir in
    tests, and against a different mirror, without editing the module.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, name: str) -> Path:
        return self.root / name

    def refresh(self, urls: dict[str, str] = VENDOR_CATALOGS, force: bool = False) -> None:
        """Download any catalog not already cached (or all of them with force)."""
        for name, url in urls.items():
            path = self.path(name)
            if force or not path.exists():
                self.root.mkdir(parents=True, exist_ok=True)
                print(f"fetching {name} ...", file=sys.stderr)
                req = urllib.request.Request(
                    url, headers={"User-Agent": "llm-router-registry/1.0"}
                )
                with urllib.request.urlopen(req, timeout=120) as r:
                    path.write_bytes(r.read())

    def records(self) -> list[Rec]:
        """Price/name records from the three priced catalogs."""
        return (
            self._litellm() + self._openrouter() + self._modelsdev()
        )

    def epoch(self) -> dict[str, dict[str, Any]]:
        """Epoch AI params + publication dates, keyed by `epoch_key`."""
        with self.path("epoch_ai_models.csv").open(
            newline="", encoding="utf-8"
        ) as f:
            rows = list(csv.DictReader(f))
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            if "Language" not in (r.get("Domain") or ""):
                continue
            date = (r.get("Publication date") or "").strip()
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
                continue
            key = epoch_key(r["Model"])
            params = parse_params(r.get("Parameters"))
            cur = out.get(key)
            if cur is None or (date and date > cur["released"]):
                out[key] = {"released": date, "params_b": params}
        return out

    def _litellm(self) -> list[Rec]:
        data = json.loads(self.path("litellm.json").read_text())
        data.pop("sample_spec", None)
        keep = {"chat", "responses", "completion"}
        recs = []
        for key, v in data.items():
            if v.get("mode") not in keep:
                continue
            if key.endswith("/"):  # entry-point placeholders
                continue
            recs.append(
                Rec(
                    raw=key,
                    source="litellm",
                    name=normalize_name(key),
                    pin=per_m(v.get("input_cost_per_token")),
                    pout=per_m(v.get("output_cost_per_token")),
                    context=v.get("max_input_tokens") or v.get("max_tokens"),
                )
            )
        return recs

    def _openrouter(self) -> list[Rec]:
        data = json.loads(self.path("openrouter.json").read_text())["data"]
        recs = []
        for v in data:
            pricing = v.get("pricing") or {}
            bench = v.get("benchmarks") or {}
            aa = bench.get("artificial_analysis") or {}
            da = bench.get("design_arena") or []
            elos = [d["elo"] for d in da if isinstance(d, dict) and d.get("elo")]
            created = v.get("created")
            released = (
                dt.datetime.fromtimestamp(created, dt.UTC).date().isoformat()
                if created
                else None
            )
            recs.append(
                Rec(
                    raw=v["id"],
                    source="openrouter",
                    name=normalize_name(v["id"]),
                    pin=per_m(pricing.get("prompt")),
                    pout=per_m(pricing.get("completion")),
                    released=released,
                    ii=aa.get("intelligence_index"),
                    elo=sum(elos) / len(elos) if elos else None,
                    context=v.get("context_length"),
                )
            )
        return recs

    def _modelsdev(self) -> list[Rec]:
        data = json.loads(self.path("modelsdev.json").read_text())
        recs = []
        for provider, pv in data.items():
            for key, m in (pv.get("models") or {}).items():
                modalities = m.get("modalities") or {}
                if modalities.get("output") and "text" not in modalities["output"]:
                    continue  # image/audio/embedding models are not routed chat models
                cost = m.get("cost") or {}
                pin = (
                    cost.get("input")
                    if isinstance(cost.get("input"), (int, float))
                    else None
                )
                pout = (
                    cost.get("output")
                    if isinstance(cost.get("output"), (int, float))
                    else None
                )
                recs.append(
                    Rec(
                        raw=f"{provider}/{key}",
                        source="models.dev",
                        name=normalize_name(key),
                        vendor_hint=provider,
                        pin=pin,
                        pout=pout,
                        released=m.get("release_date"),
                    )
                )
        return recs


def merge_records(
    recs: list[Rec], epoch: dict[str, dict[str, Any]]
) -> dict[str, Canonical]:
    """Group raw records by canonical model and take the best signal per field.

    Release dates prefer models.dev (a real publication date) over OpenRouter (a
    listing timestamp); prices accumulate every hoster because `scoring` treats
    the priciest listing as the capability signal; params come from Epoch keyed
    on a mangled name, else from the size suffix in the model name.
    """
    groups: dict[str, Canonical] = {}
    for r in recs:
        if not r.name or len(r.name) < 2:
            continue
        # hand aliases and vendor-prefixed names are already canonical keys
        key = (
            r.name
            if "/" in r.name
            else f"{infer_vendor(r.name, r.vendor_hint)}/{r.name}"
        )
        g = groups.setdefault(key, Canonical(key))
        g.aliases.add(r.raw.lower())
        g.aliases.add(r.name)
        g.raws.append(r.raw)
        f = g.fields
        if r.pin is not None or r.pout is not None:
            f["prices"] = f.get("prices", []) + [(r.source, r.pin, r.pout)]
        for field in ("released", "ii", "elo", "context"):
            val = getattr(r, field)
            if val is not None:
                cur = f.get(field)
                # models.dev release dates beat openrouter listing dates
                if field == "released":
                    if cur is None or (
                        r.source == "models.dev" and cur[0] != "models.dev"
                    ):
                        f[field] = (r.source, val)
                elif cur is None:
                    f[field] = val
        if f.get("params_b") is None:
            ep = epoch.get(epoch_key(r.raw)) or epoch.get(epoch_key(r.name))
            if ep:
                f["params_b"] = ep["params_b"]
                if f.get("released") is None and ep["released"]:
                    f["released"] = ("epoch", ep["released"])
            else:
                pfn = params_from_name(r.name)
                if pfn:
                    f["params_b"] = pfn
    return groups
