#!/usr/bin/env python3
"""Build the model complexity registry.

Merges four catalogs into one canonical registry mapping every chat LLM name
(2022-2026) to a complexity tier:

  1 = tiny/flash class          4 = pro/legacy-frontier class
  2 = mini/light class          5 = frontier class
  3 = mid/pro class

Scoring is hybrid (per docs in README):
  * price prior: log-interpolated from blended $/1M tokens (0.75*in + 0.25*out)
  * quality check: Artificial-Analysis intelligence index / design-arena ELO
    pulls the tier one step toward quality when they disagree
  * fallback for unpriced models: parameter count + name keywords
  * era caps: a model cannot outrank the frontier of its release window
  * overrides.yaml wins over everything

Usage:
  python scripts/build_model_registry.py [--refresh]

Outputs:
  data/model_registry/models.json       the registry (checked in)
  data/model_registry/build_report.md   disagreements + coverage stats (local)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.model_registry import normalize_name  # shared with the runtime resolver

try:
    import yaml
except ImportError:  # overrides are optional without pyyaml
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "model_registry" / "cache"
OUT_JSON = ROOT / "data" / "model_registry" / "models.json"
OUT_OVERRIDES = ROOT / "data" / "model_registry" / "overrides.yaml"
OUT_REPORT = ROOT / "data" / "model_registry" / "build_report.md"

SOURCES = {
    "litellm.json": "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
    "openrouter.json": "https://openrouter.ai/api/v1/models",
    "modelsdev.json": "https://models.dev/api.json",
    "epoch_ai_models.csv": "https://datahub.io/ai/epoch-data-on-ai-models/r/0.csv",
}

# hand aliases are normalized names of models whose name loses tier info when
# stripped; values are canonical registry keys (defined in router/model_registry.py)

PRICE_EDGES = [
    (0.05, 1.0),
    (0.15, 1.7),
    (0.70, 2.7),
    (4.0, 3.9),
    (15.0, 4.9),
    (30.0, 5.0),
]
II_CUTS = [(45.0, 5), (28.0, 4), (14.0, 3), (7.0, 2), (0.0, 1)]
ELO_FRONTIER = 1300.0  # design-arena ELO only used to lift, never to downgrade
PARAMS_CUTS_MODERN = [(300.0, 5), (67.0, 4), (13.0, 3), (3.0, 2), (0.0, 1)]
PARAMS_CUTS_LEGACY = [(300.0, 4), (70.0, 3), (7.0, 2), (0.0, 1)]

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

# absolute-capability era windows (see build_report for calibration notes)
ERA_PRE_FRONTIER = dt.date(2023, 3, 15)  # before gpt-4: everything ~ tier 2
ERA_CAP_4_END = dt.date(2024, 9, 1)  # before o1: nothing is tier 5
ERA_MODERN = dt.date(2025, 1, 1)  # before this, today's intelligence
# index is not a fair downgrade signal

TINY_WORDS = re.compile(
    r"\b(?:tiny|mini|micro|nano|slim|small|lite|flash|instant|1b|2b|3b|4b|5b|6b|7b|8b|9b)\b|-(?:1|2|3)b\b"
)
HUGE_WORDS = re.compile(r"\b(?:opus|ultra|max|large|405b|671b|200b| frontier)\b")

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


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def fetch(name: str, url: str, refresh: bool) -> Path:
    path = CACHE / name
    if refresh or not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        log(f"fetching {name} ...")
        req = urllib.request.Request(
            url, headers={"User-Agent": "llm-router-registry/1.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    return path


# ---------------------------------------------------------------- normalization
# normalize_name lives in router/model_registry.py and is shared verbatim


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


# ---------------------------------------------------------------- record loading


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


def load_litellm() -> list[Rec]:
    data = json.loads((CACHE / "litellm.json").read_text())
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


def load_openrouter() -> list[Rec]:
    data = json.loads((CACHE / "openrouter.json").read_text())["data"]
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


def load_modelsdev() -> list[Rec]:
    data = json.loads((CACHE / "modelsdev.json").read_text())
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


def epoch_key(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9.+-]", "", s)
    return re.sub(r"-{2,}", "-", s).strip("-")


def load_epoch() -> dict[str, dict[str, Any]]:
    with (CACHE / "epoch_ai_models.csv").open(newline="", encoding="utf-8") as f:
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


# ---------------------------------------------------------------- canonicalization


class Canonical:
    __slots__ = ("key", "aliases", "raws", "fields")

    def __init__(self, key: str):
        self.key = key
        self.aliases: set[str] = set()
        self.raws: list[str] = []
        self.fields: dict[str, Any] = {}


def merge_records(
    recs: list[Rec], epoch: dict[str, dict[str, Any]]
) -> dict[str, Canonical]:
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


# ---------------------------------------------------------------- scoring


def interp_price(pin: float | None, pout: float | None) -> float | None:
    if pin is None and pout is None:
        return None
    pin = pin or 0.0
    pout = pout or 0.0
    if pin == 0.0 and pout == 0.0:
        return 1.0  # free tier
    blended = 0.75 * pin + 0.25 * pout
    if blended <= PRICE_EDGES[0][0]:
        return 1.0
    if blended >= PRICE_EDGES[-1][0]:
        return 5.0
    for (p0, s0), (p1, s1) in zip(PRICE_EDGES, PRICE_EDGES[1:]):
        if p0 < blended <= p1:
            frac = (blended - p0) / (p1 - p0)
            return s0 + frac * (s1 - s0)
    return None


def cut_tier(value: float | None, cuts) -> int | None:
    if value is None:
        return None
    for threshold, tier in cuts:
        if value >= threshold:
            return tier
    return cuts[-1][1]


def era_of(released: str | None) -> dt.date | None:
    if not released:
        return None
    try:
        return dt.date.fromisoformat(released[:10])
    except ValueError:
        return None


def score_group(g: Canonical) -> dict[str, Any] | None:
    f = g.fields
    prices = f.get("prices") or []
    # list price is the capability signal: use the priciest hoster, ignore free mirrors
    best: tuple[float, float] | None = None
    best_blended = -1.0
    for _, p_in, p_out in prices:
        if not (p_in or p_out):
            continue
        blended = 0.75 * (p_in or 0.0) + 0.25 * (p_out or 0.0)
        if blended > best_blended:
            best_blended = blended
            best = (p_in or 0.0, p_out or 0.0)
    pin, pout = best if best else (None, None)

    price_cont = interp_price(pin, pout)
    released = (f.get("released") or (None, None))[1]
    era = era_of(released)

    quality_ii = cut_tier(f.get("ii"), II_CUTS)
    quality = quality_ii
    if quality is None and f.get("elo") and f["elo"] >= ELO_FRONTIER:
        quality = 5  # design-arena ELO only lifts frontier-grade models

    params_b = f.get("params_b")
    legacy = era is not None and era < dt.date(2024, 9, 1)
    params_tier = cut_tier(
        params_b, PARAMS_CUTS_LEGACY if legacy else PARAMS_CUTS_MODERN
    )

    nm = g.key.split("/", 1)[1]
    vendor = g.key.split("/", 1)[0]
    tiny = bool(TINY_WORDS.search(nm))
    huge = bool(HUGE_WORDS.search(nm))

    # --- base tier selection
    basis: str
    base_tier: int
    pre_frontier = era is not None and era < ERA_PRE_FRONTIER
    legacy_open = (
        legacy
        and params_tier is not None
        and vendor in OPEN_PARAMS_VENDORS
        and not pre_frontier
    )
    if pre_frontier and not nm.startswith("gpt-4"):
        # pre-gpt-4 models were priced for scarcity, not capability
        base_tier, basis = 2, "era(pre-2023-03)"
    elif legacy_open and params_tier is not None:
        # commodity hosting prices legacy open models for scarcity, not capability
        base_tier, basis = params_tier, f"params({params_b:g}B, legacy open)"
    elif price_cont is not None:
        base_tier = max(1, min(5, int(price_cont + 0.5)))  # half-up
        basis = (
            f"price({pin:.2f}/{pout:.2f}$ per M)" if (pin or pout) else "price(free)"
        )
    elif params_tier is not None:
        base_tier, basis = params_tier, f"params({params_b:g}B)"
    elif tiny:
        base_tier, basis = 1, "keyword(tiny)"
    elif huge:
        base_tier, basis = 4, "keyword(huge)"
    else:
        base_tier, basis = 2, "guess"

    # --- quality adjustment: +1 toward quality; downgrades only for modern
    # models, where today's intelligence index is a fair absolute signal
    adj = 0
    if quality is not None:
        diff = quality - base_tier
        if diff >= 2 or (diff == 1 and (quality == 5 or base_tier <= 2)):
            adj = 1
        elif (
            diff <= -2
            and quality_ii is not None
            and era is not None
            and era >= ERA_MODERN
        ):
            adj = -1
    tier = base_tier + adj

    # --- era caps
    if era is not None:
        if era < ERA_PRE_FRONTIER:
            tier = min(tier, 3)
        elif era < ERA_CAP_4_END:
            tier = min(tier, 4)
    elif tier == 5:
        tier = 5  # allowed, but flagged below: no release date to cap it

    tier = max(1, min(5, int(tier)))
    score_cont = float(tier)
    if price_cont is not None and basis.startswith("price"):
        score_cont = price_cont
    if adj:
        score_cont = min(
            max(score_cont + adj * 0.5, float(tier) - 0.5), float(tier) + 0.5
        )
    score_cont = min(max(score_cont, 1.0), 5.0)

    notes = []
    if basis == "guess":
        notes.append("unpriced, no params found - review")
    if adj:
        notes.append(f"quality adj {adj:+d} (ii={f.get('ii')}, elo={f.get('elo')})")
    if released is None and tier == 5:
        notes.append("tier 5 with no release date - review")
    if (
        price_cont is not None
        and quality is not None
        and abs(quality - round(price_cont)) >= 2
    ):
        notes.append("price/quality disagree >=2 tiers")

    return {
        "canonical_id": g.key,
        "display_name": None,  # filled from best raw name below
        "aliases": sorted(a for a in g.aliases if a and a != g.key.split("/", 1)[1]),
        "released": released,
        "mode": "chat",
        "complexity": tier,
        "score": round(score_cont, 2),
        "signals": {
            "price_in_per_m": round(pin, 4) if pin is not None else None,
            "price_out_per_m": round(pout, 4) if pout is not None else None,
            "elo": round(f["elo"]) if f.get("elo") else None,
            "intelligence_index": f.get("ii"),
            "params_b": params_b,
        },
        "basis": basis,
        "notes": "; ".join(notes),
        "_quality": quality,
        "_raws": g.raws,
    }


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download catalogs")
    args = ap.parse_args()

    for name, url in SOURCES.items():
        fetch(name, url, args.refresh)

    recs = load_litellm() + load_openrouter() + load_modelsdev()
    log(f"records: litellm+openrouter+models.dev = {len(recs)}")
    epoch = load_epoch()
    log(f"epoch language models with dates: {len(epoch)}")

    overrides: dict[str, Any] = {}
    if yaml and OUT_OVERRIDES.exists():
        overrides = yaml.safe_load(OUT_OVERRIDES.read_text()) or {}

    groups = merge_records(recs, epoch)
    log(f"canonical groups: {len(groups)}")

    excluded = set(overrides.get("excludes") or []) if yaml else set()
    built = []
    for g in groups.values():
        if g.key in excluded:
            continue  # junk merge group; real model recreated via overrides
        entry = score_group(g)
        if entry is None:
            continue
        # display name: longest alias, capitalized by the source that had it
        raws = [r for r in entry.pop("_raws") if "/" not in r or r.count("/") == 1]
        entry["display_name"] = max((r for r in raws), key=len) if raws else g.key
        built.append(entry)

    # apply overrides
    by_key = {e["canonical_id"]: e for e in built}
    for key, ov in (overrides.get("models") or {}).items():
        entry = by_key.get(key)
        if entry is None:
            # create entries the live catalogs have pruned (2023-era models etc.)
            if "complexity" not in ov:
                continue
            entry = {
                "canonical_id": key,
                "display_name": ov.get("display_name", key),
                "aliases": [],
                "released": ov.get("released"),
                "mode": "chat",
                "complexity": int(ov["complexity"]),
                "score": float(ov.get("score", ov["complexity"])),
                "signals": {
                    k: ov.get(k)
                    for k in (
                        "price_in_per_m",
                        "price_out_per_m",
                        "elo",
                        "intelligence_index",
                        "params_b",
                    )
                },
                "basis": "override",
                "notes": ov.get(
                    "notes", "created via override; absent from live catalogs"
                ),
                "_quality": None,
                "_raws": [],
            }
            by_key[key] = entry
            built.append(entry)
        for field in ("complexity", "score", "released", "display_name", "notes"):
            if field in ov:
                entry[field] = ov[field]
        if "aliases" in ov:
            entry["aliases"] = sorted(set(entry["aliases"]) | set(ov["aliases"]))
        if "signals" in ov:
            entry["signals"].update(
                {k: v for k, v in ov["signals"].items() if v is not None}
            )
        if entry["basis"] != "override":
            entry["basis"] = "override"

    # alias → canonical index + collision check
    alias_index: dict[str, str] = {}
    collisions: list[str] = []
    for entry in built:
        names = {entry["canonical_id"], *entry["aliases"]}
        for a in names:
            if a in alias_index and alias_index[a] != entry["canonical_id"]:
                collisions.append(f"{a}: {alias_index[a]} vs {entry['canonical_id']}")
            alias_index[a] = entry["canonical_id"]

    tier_hist = defaultdict(int)
    for e in built:
        tier_hist[e["complexity"]] += 1

    registry = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "description": "Model complexity registry: canonical chat LLMs (2022-2026) -> complexity 1-5. Generated by scripts/build_model_registry.py; hand fixes live in overrides.yaml.",
        "complexity_scale": {
            "1": "tiny/flash class (sub-3B, flash/mini/nano tiers)",
            "2": "light class (7-14B open, mini/haiku hosted, legacy 2022 frontier)",
            "3": "mid class (70B open, pro hosted, 2023 frontier)",
            "4": "pro class (gpt-4o/sonnet/2024 frontier)",
            "5": "frontier class (o1+/opus/gpt-5 era)",
        },
        "models": sorted(built, key=lambda e: e["canonical_id"]),
    }
    OUT_JSON.write_text(json.dumps(registry, indent=1, ensure_ascii=False) + "\n")
    log(f"wrote {OUT_JSON} ({len(built)} models)")

    # ---- report
    flagged = [e for e in built if e["notes"] or e["basis"] == "guess"]
    flagged.sort(key=lambda e: e["canonical_id"])
    with OUT_REPORT.open("w") as f:
        f.write("# registry build report\n\n")
        f.write(f"generated: {registry['generated_at']}\n\n")
        f.write(
            f"models: {len(built)} | tier histogram: {dict(sorted(tier_hist.items()))}\n\n"
        )
        f.write(f"alias collisions: {len(collisions)}\n")
        for c in collisions[:50]:
            f.write(f"- {c}\n")
        f.write(
            "\n## flagged for review (quality-adjusted, guessed, or era-capped)\n\n"
        )
        for e in flagged:
            f.write(
                f"- **{e['complexity']}** {e['canonical_id']} | {e['basis']} | {e['notes']}\n"
            )
        f.write("\n## sample by tier (spot check)\n\n")
        seen: dict[int, list[str]] = defaultdict(list)
        for e in sorted(
            built, key=lambda x: -(x["signals"]["intelligence_index"] or 0)
        ):
            if len(seen[e["complexity"]]) < 12:
                seen[e["complexity"]].append(e["canonical_id"])
        for tier in sorted(seen):
            f.write(f"### tier {tier}\n")
            for mid in seen[tier]:
                f.write(f"- {mid}\n")
    log(f"wrote {OUT_REPORT}")
    log(f"tier histogram: {dict(sorted(tier_hist.items()))}")
    if collisions:
        log(f"WARNING: {len(collisions)} alias collisions - see report")


if __name__ == "__main__":
    main()
