#!/usr/bin/env python3
"""Select the balanced capability-floor dataset from the arena battle rows.

A stage between the dataset build and the pairs build:

    dataset.jsonl --(floor rules, era-balanced quota)--> floor.jsonl --pairs-->

It reads the ALREADY-BUILT `data/router_training_dataset.jsonl` -- no Hub
downloads, no gated re-ingest, no change to `build_training_dataset.py` -- and
writes one row per selected prompt in the solving shape that
`router/pairs.py:solving_tier_pairs` already understands. That seam is what keeps
the pairs stage, the metric layer, the notebook's `pos_weight=4.0` and the seven
byte-compared goldens untouched.

Every rule lives in `router/floor.py`; this file parses arguments, orders the
passes, and renders the report. The report is the deliverable as much as the
JSONL is: it states the per-tier supply, the shortfalls that a stricter balance
could not reach, the era spread that the 2023-only tier-1/2 supply makes
unavoidable, and a sample per tier for a human to eyeball before anyone spends a
Colab session.

Usage:
  python scripts/select_floor_dataset.py                       # defaults below
  python scripts/select_floor_dataset.py --per-tier 1500 --seed 13
  python scripts/select_floor_dataset.py --skip-semantic-dedup --limit 2000   # fast
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from router.contracts import TIERS
from router.floor import (
    PREVIOUS_MAX_LEN,
    RULE_BOTH_BAD,
    RULE_UPSET,
    DEFAULT_RULES,
    DedupResult,
    FloorRules,
    FloorStats,
    PromptAgg,
    Selection,
    SelectionQuota,
    collect_intervals,
    floor_rows,
    near_duplicate_representatives,
    norm_prompt,
    select_balanced,
)
from router.model_registry import get_registry
from router.sources import FAMILY_ARENA, SOURCES

ROOT = Path(__file__).resolve().parent.parent

IN_DEFAULT = ROOT / "data" / "router_training_dataset.jsonl"
OUT_DEFAULT = ROOT / "data" / "router_training_floor.jsonl"
QUARANTINE_DEFAULT = ROOT / "data" / "router_floor_quarantine.jsonl"
HOLDOUT_DEFAULT = ROOT / "data" / "router_floor_holdout.jsonl"
REPORT_DEFAULT = ROOT / "eval_reports" / "floor_selection_report.md"
EMBED_CACHE = ROOT / "data" / "floor_embeddings.npz"

#: Arena releases, in catalog order (which is chronological). Derived from the
#: source catalog rather than restated here -- `router/sources.py` owns that list.
ARENA_KEYS: tuple[str, ...] = tuple(s.key for s in SOURCES if s.family == FAMILY_ARENA)
GATED_KEYS: frozenset[str] = frozenset(s.key for s in SOURCES if s.gated)

#: Default embedding model for the near-duplicate pass. Cached locally, small,
#: and English-only is fine: the report prints the non-English share instead of
#: silently clustering across languages.
DEDUP_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
DEDUP_THRESHOLD_DEFAULT = 0.92

#: How many prompts per tier go into the human review table.
REVIEW_ROWS = 40


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Stream the dataset file; it is ~365 MB, so never materialize it."""
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def registry_stamp(path: Path = ROOT / "data" / "model_registry" / "models.json") -> str:
    """First 12 hex of the registry the gates were applied against.

    The floor stage reads `score` from the live registry while trusting the
    `tier_a`/`tier_b` frozen in the dataset file, so the pairing has to be
    recorded: this is what tells a later reader whether a floor file and a
    models.json belong to the same world.
    """
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def check_registry_consistency(rows: Iterator[dict[str, Any]], limit: int | None,
                               tolerance: float) -> tuple[int, int]:
    """Tripwire for the invalidation cascade, not a coverage gate.

    Re-resolving model names and comparing to the dataset's frozen tiers cannot
    protect label *quality* -- it only detects that the dataset file predates a
    registry rebuild. A coverage number here would be vacuous: every name in the
    file was resolved when it was written, so it could only ever report 100%.
    """
    registry = get_registry()
    checked = mismatched = 0
    for row in rows:
        if not str(row.get("source", "")).startswith("arena"):
            continue
        if limit is not None and checked >= limit:
            break
        checked += 1
        for name, key in (("model_a", "tier_a"), ("model_b", "tier_b")):
            want, got = row.get(key), registry.complexity(str(row.get(name, "")))
            if got is not None and want is not None and int(want) != got:
                mismatched += 1
                break
    rate = mismatched / max(checked, 1)
    if rate > tolerance:
        raise SystemExit(
            f"{mismatched}/{checked} battle rows disagree with the current "
            f"registry tiers ({rate:.1%} > {tolerance:.1%}): "
            f"{IN_DEFAULT} predates a registry rebuild. Run "
            f"scripts/build_model_registry.py then "
            f"scripts/build_training_dataset.py before selecting a floor set."
        )
    return checked, mismatched


class TokenLength:
    """Prompt length in tokens, with a stated fallback so the estimate is honest.

    Selection reserves a share of prompts past the OLD 256-token budget, which is
    a claim about tokens -- so which measurer ran has to appear in the report
    rather than be assumed.
    """

    def __init__(self, tokenizer_dir: Path | None) -> None:
        self.backend = "chars/4 estimate"
        self._tokenizer: Any = None
        if tokenizer_dir is not None and (tokenizer_dir / "tokenizer.json").exists():
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
                self.backend = f"tokenizer at {tokenizer_dir.name}"
            except Exception as exc:                       # pragma: no cover
                print(f"[warn] tokenizer unavailable ({exc}); using chars/4")
        self._cache: dict[str, int] = {}

    def __call__(self, prompt: str) -> int:
        key = norm_prompt(prompt)
        if key not in self._cache:
            if self._tokenizer is None:
                self._cache[key] = len(prompt) // 4
            else:
                self._cache[key] = len(
                    self._tokenizer(prompt, add_special_tokens=False)["input_ids"])
        return self._cache[key]


class SbEmbedder:
    """Adapts a sentence-transformers model to floor.py's `Embedder` protocol.

    The import happens here, in a script, not in `router/floor.py`: the
    clustering *rule* stays testable and import-cheap while the 90 MB model stays
    optional. CPU on purpose -- MPS float jitter would move prompts across the
    similarity threshold between runs and make the core non-reproducible.
    """

    def __init__(self, model_id: str, batch_size: int) -> None:
        from sentence_transformers import SentenceTransformer

        self._model: Any = SentenceTransformer(model_id, device="cpu")
        self.batch_size = batch_size

    def encode(self, texts: Any) -> Any:
        return self._model.encode(list(texts), batch_size=self.batch_size,
                                  normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False)


def supply_table(aggs: dict[str, PromptAgg]) -> dict[int, Counter[str]]:
    """Per-tier era supply of point-labelled prompts -- the balance ceiling."""
    out: dict[int, Counter[str]] = {t: Counter() for t in TIERS}
    for item in aggs.values():
        tier = item.tier
        if tier is None:
            continue
        for era in item.eras:
            out[tier][era] += 1
    return out


def review_sample(aggs: dict[str, PromptAgg], chosen: set[str], per_tier: int,
                  seed: int) -> dict[int, list[PromptAgg]]:
    """A spread of labelled prompts per tier, biased toward the risky provenance.

    Tier 5 is mostly `both_bad` (a satisfaction verdict, not a capability one) and
    tiers 1-2 are mostly upsets, so those are the rows worth reading -- and the
    ones the core excludes, since the core is what a reader has already implicitly
    accepted. Deterministic: sha1 of seed+prompt, never file order.
    """
    by_tier: dict[int, list[PromptAgg]] = {t: [] for t in TIERS}
    for item in aggs.values():
        tier = item.tier
        if tier is None or item.prompt in chosen:
            continue
        by_tier[tier].append(item)

    def score_of_urgency(tier: int | None, item: PromptAgg) -> int:
        if tier == 5 and item.provenance == RULE_BOTH_BAD:
            return 0          # the claim "needs tier 5" from a dissatisfied voter
        if tier is not None and tier <= 2 and item.provenance == RULE_UPSET:
            return 0          # the claim "a light model sufficed" from one ballot
        return 1

    out: dict[int, list[PromptAgg]] = {}
    for tier, pool in by_tier.items():
        ordered = sorted(pool, key=lambda i, t=tier: (
            score_of_urgency(t, i),
            hashlib.sha1(f"{seed}:{i.prompt}".encode()).hexdigest()))
        out[tier] = ordered[:per_tier]
    return out


def render_report(args: argparse.Namespace, stats: FloorStats, rules: FloorRules,
                  aggs: dict[str, PromptAgg], selection: Selection,
                  dedup: DedupResult | None, holdout: list[PromptAgg],
                  length_of: TokenLength, stamp: str, core_sha: str,
                  checked: int, mismatched: int, review: dict[int, list[PromptAgg]],
                  collect_keys: frozenset[str], core_keys: tuple[str, ...],
                  holdout_era: str | None, path: Path) -> None:
    lines: list[str] = ["# capability-floor selection report", ""]
    lines.append(f"input: `{args.infile.name}` | per_tier {args.per_tier} | "
                 f"review seed {args.seed}")
    lines.append(f"releases read: {', '.join(sorted(collect_keys))} | core pool: "
                 f"{', '.join(core_keys)}"
                 + (f" | holdout era: `{holdout_era}`" if holdout_era else ""))
    lines.append(f"registry: `{stamp}` | tiers frozen in file re-resolved against it: "
                 f"{mismatched} mismatches in {checked} rows (tolerance "
                 f"{args.tier_tolerance:.3%})")
    lines.append(f"length backend: {length_of.backend} | long = >{PREVIOUS_MAX_LEN} "
                 f"tokens (the budget the rules were designed against)")
    if dedup is None:
        lines.append("near-duplicate collapse: SKIPPED (--skip-semantic-dedup) -- "
                     "paraphrases may straddle the prompt-hash validation split")
    else:
        lines.append(f"near-duplicate collapse: {args.dedup_model} @ "
                     f"{args.dedup_threshold} -> {dedup.n_clusters} clusters from "
                     f"{dedup.n_prompts} prompts ({dedup.dropped} dropped)")
    lines.append(f"core sha256: `{core_sha}`")
    lines.append("")
    lines.append("## yield")
    lines.append("")
    lines.append("| stage | count |")
    lines.append("|---|---|")
    lines.append(f"| dataset rows scanned | {stats.rows_scanned} |")
    lines.append(f"| arena battle rows | {stats.battle_rows} |")
    lines.append(f"| battles that implied a constraint | {stats.constraints} |")
    lines.append(f"| battles that implied nothing (same-tier, gated) | "
                 f"{stats.unconstrained} |")
    lines.append(f"| prompts with >=1 constraint | {stats.prompts} |")
    lines.append(f"| collapsed to a point (labelable) | {stats.collapsed} |")
    lines.append(f"| consistent but still an interval | {stats.intervals} |")
    lines.append(f"| self-contradictory (quarantined) | {stats.contradictory} "
                 f"({stats.contradiction_rate:.1%}) |")
    lines.append("")
    lines.append("Contradictory prompts are the corpus's own disagreement about one "
                 "prompt's floor -- a noise floor no single-label scheme can see, "
                 "because it needs two battles to disagree.")
    lines.append("")

    lines.append("## rules in force")
    lines.append("")
    lines.append("| gate | value | why |")
    lines.append("|---|---|---|")
    lines.append(f"| min `|delta score|` | {rules.min_delta_score} | a decisive win "
                 "needs genuinely different models, not adjacent tier buckets |")
    lines.append(f"| both_bad min tier | {rules.both_bad_min_tier} | two frontier "
                 "models failing is difficulty; two mini models failing is normal |")
    lines.append(f"| max upset gap | {rules.max_upset_gap} | a huge-gap upset is more "
                 "likely a bad ballot or a wrong registry tier |")
    lines.append(f"| max tie gap | {rules.max_tie_gap} | ties are weaker than wins |")
    lines.append("")

    lines.append("## era supply of labelable prompts (the balance ceiling)")
    lines.append("")
    supply = supply_table(aggs)
    lines.append("| tier | " + " | ".join(ARENA_KEYS) + " | total |")
    lines.append("|---|" + "---|" * (len(ARENA_KEYS) + 1))
    for tier in TIERS:
        row = supply[tier]
        lines.append(f"| {tier} | " + " | ".join(str(row.get(k, 0)) for k in ARENA_KEYS)
                     + f" | {sum(row.values())} |")
    lines.append("")
    lines.append("Tiers 1-2 have no meaningful supply in the 2025 release because "
                 "weak models were not fielded there -- an upset needs a weak model "
                 "to beat a strong one. That is why exact per-tier balance and "
                 "era-cleanliness cannot both be satisfied from arena data, and why "
                 "the selection maximizes the smaller side instead of pretending.")
    lines.append("")

    lines.append("## selection")
    lines.append("")
    lines.append("| tier | labelable pool | chosen | shortfall | era spread | "
                 "provenance | corroborated | >old MAX_LEN | val prompts |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    pool_sizes: dict[int, int] = {t: 0 for t in TIERS}
    for item in aggs.values():
        tier = item.tier
        if tier is not None and (holdout_era is None or holdout_era not in item.eras):
            pool_sizes[tier] += 1
    for tier in TIERS:
        eras = selection.per_tier_era.get(tier, Counter())
        prov = selection.per_tier_provenance.get(tier, Counter())
        lines.append(
            f"| {tier} | {pool_sizes[tier]} | {selection.per_tier[tier]} | "
            f"{selection.shortfalls.get(tier, 0)} | "
            + ", ".join(f"{k.replace('arena_', '')}:{v}" for k, v in
                        sorted(eras.items(), key=lambda x: -x[1])) + " | "
            + ", ".join(f"{k}:{v}" for k, v in sorted(prov.items(),
                                                      key=lambda x: -x[1])) + " | "
            f"{selection.per_tier_corroborated[tier]} | "
            f"{selection.per_tier_long[tier]} | {selection.per_tier_val[tier]} |")
    lines.append("")
    lines.append(f"chosen prompts: {len(selection.chosen)} -> "
                 f"{len(selection.chosen) * len(TIERS)} expected pair rows")
    if selection.shortfalls:
        lines.append("")
        lines.append("Two different reasons a tier can fall short, and they want "
                     "opposite responses:")
        for tier, short in sorted(selection.shortfalls.items()):
            pool_n, chosen_n = pool_sizes[tier], selection.per_tier[tier]
            if pool_n >= args.per_tier:
                lines.append(f"- **tier {tier}: {pool_n} labelable, {chosen_n} chosen.** "
                             f"The pool is big enough, so this is the `both_bad` "
                             f"provenance cap ({args.both_bad_share_cap:.0%} of a tier) "
                             f"doing its job -- most tier-{tier} points come from that "
                             f"one rule. Raising `--per-tier` will not help; raising the "
                             f"cap would let a satisfaction verdict define the tier.")
            else:
                lines.append(f"- **tier {tier}: {pool_n} labelable < {args.per_tier} "
                             f"asked.** Genuine scarcity -- the battles that release "
                             f"fielded cannot evidence this tier. Lower `--per-tier` to "
                             f"the smallest pool, or accept the imbalance.")
    lines.append("")
    lines.append("`corroborated` is prompts whose label rests on >=2 battles. At "
                 "91.5% of arena prompts having exactly one battle, the core is "
                 "mostly single-vote evidence -- which is the property the "
                 "out-of-era probe exists to price.")
    lines.append("")

    if holdout:
        lines.append("## temporal holdout (out-of-era evaluation set)")
        lines.append("")
        hist = Counter(i.tier for i in holdout if i.tier is not None)
        lines.append("| tier | prompts |")
        lines.append("|---|---|")
        for tier in TIERS:
            lines.append(f"| {tier} | {hist.get(tier, 0)} |")
        lines.append("")
        lines.append(f"Every holdout prompt's evidence comes only from "
                     f"`{args.holdout_era}`, and none of them are in the core. "
                     f"Tiers with no supply in that release are absent by "
                     f"construction -- the out-of-era number measures the top of the "
                     f"scale only, and any report that averages over all five tiers "
                     f"here is wrong.")
        lines.append("")

    lines.append("## review sample (read this before training)")
    lines.append("")
    for tier in TIERS:
        rows = review.get(tier, [])
        if not rows:
            continue
        lines.append(f"### tier {tier} ({len(rows)} of "
                     f"{sum(supply[tier].values())} labelable)")
        lines.append("")
        lines.append("| prompt | rules | n | eras |")
        lines.append("|---|---|---|---|")
        for item in rows:
            text = item.prompt.replace("|", "\\|").replace("\n", " ")[:110]
            lines.append(f"| {text} | {item.provenance} | {item.n_constraints} | "
                         + ",".join(e.replace("arena_", "") for e in item.eras) + " |")
        lines.append("")
    lines.append("Tier 5 rows are `both_bad`: two capable models produced answers the "
                 "voter rejected. That is difficulty *unless* the failure was a "
                 "refusal, a safety trip, or a picky voter -- which is exactly what "
                 "this table is for.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", type=Path, default=IN_DEFAULT,
                    help="dataset JSONL from build_training_dataset.py")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--quarantine-out", type=Path, default=QUARANTINE_DEFAULT,
                    help="self-contradictory prompts, for inspection")
    ap.add_argument("--holdout-out", type=Path, default=HOLDOUT_DEFAULT)
    ap.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    ap.add_argument("--per-tier", type=int, default=1500)
    ap.add_argument("--val-floor-per-tier", type=int, default=60,
                    help="minimum held-out prompts per tier, met by same-tier swaps")
    ap.add_argument("--long-min-per-tier", type=int, default=None,
                    help="prompts past the old 256-token budget kept per tier "
                         "(default: per_tier // 10)")
    ap.add_argument("--both-bad-share-cap", type=float, default=0.6,
                    help="max share of a tier whose label comes only from both_bad")
    ap.add_argument("--delta-score", type=float, default=DEFAULT_RULES.min_delta_score)
    ap.add_argument("--both-bad-min-tier", type=int,
                    default=DEFAULT_RULES.both_bad_min_tier)
    ap.add_argument("--upset-gap", type=int, default=DEFAULT_RULES.max_upset_gap)
    ap.add_argument("--tie-gap", type=int, default=DEFAULT_RULES.max_tie_gap)
    ap.add_argument("--exclude-source", default=None,
                    help="comma list of arena releases to drop (e.g. arena_33k for "
                         "its CC-BY-NC licence, or arena_140k to build a train set "
                         "with a temporal holdout)")
    ap.add_argument("--holdout-era", default=None,
                    help="arena release to reserve entirely as an out-of-era "
                         "evaluation set")
    ap.add_argument("--holdout-per-tier", type=int, default=300)
    ap.add_argument("--seed", type=int, default=13,
                    help="review-sample seed only: selection itself is deterministic "
                         "via prompt-hash tiebreaks, not seeded random")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke: scan at most N dataset rows")
    ap.add_argument("--dedup-model", default=DEDUP_MODEL_DEFAULT)
    ap.add_argument("--dedup-threshold", type=float, default=DEDUP_THRESHOLD_DEFAULT)
    ap.add_argument("--dedup-batch", type=int, default=256)
    ap.add_argument("--skip-semantic-dedup", action="store_true",
                    help="skip the near-duplicate pass (fast, offline, byte-stable)")
    ap.add_argument("--tokenizer-dir", type=Path,
                    default=ROOT / "models" / "complexity_router" / "final",
                    help="used only to measure token length for the long-prompt quota")
    ap.add_argument("--tier-tolerance", type=float, default=0.001,
                    help="max share of rows whose frozen tier disagrees with the "
                         "registry before the build refuses to run")
    ap.add_argument("--check-rows", type=int, default=20000,
                    help="how many battle rows to re-resolve for the consistency check")
    args = ap.parse_args()

    if not args.infile.exists():
        raise SystemExit(f"missing {args.infile} - run "
                         f"scripts/build_training_dataset.py first")

    rules = FloorRules(min_delta_score=args.delta_score,
                      both_bad_min_tier=args.both_bad_min_tier,
                      max_upset_gap=args.upset_gap, max_tie_gap=args.tie_gap)
    excluded = {s.strip() for s in (args.exclude_source or "").split(",") if s.strip()}
    unknown = excluded - set(ARENA_KEYS)
    if unknown:
        raise SystemExit(f"--exclude-source names non-arena sources: {sorted(unknown)}; "
                         f"arena releases are {list(ARENA_KEYS)}")
    collect_keys = frozenset(k for k in ARENA_KEYS if k not in excluded)
    if not collect_keys:
        raise SystemExit("every arena release is excluded -- nothing would label")
    holdout_era = args.holdout_era
    if holdout_era is not None and holdout_era not in collect_keys:
        raise SystemExit(f"--holdout-era {holdout_era} is not among the releases being "
                         f"read ({sorted(collect_keys)}); it cannot be excluded and "
                         f"held out at once")
    # The holdout release is still *collected* -- dropping it from the scan would
    # leave nothing to hold out. It is removed from the core's pool instead, and
    # only era-exclusive prompts become holdout rows: a prompt with one 2024 and
    # one 2025 battle is partly in-distribution, so it measures nothing.
    core_keys = tuple(k for k in ARENA_KEYS
                      if k in collect_keys and k != holdout_era)
    if not core_keys:
        raise SystemExit("no releases left for the core once the holdout era is "
                         "removed -- drop --holdout-era or --exclude-source")

    stamp = registry_stamp()
    checked, mismatched = check_registry_consistency(
        iter_rows(args.infile), args.check_rows, args.tier_tolerance)

    def stream() -> Iterator[dict[str, Any]]:
        for i, row in enumerate(iter_rows(args.infile)):
            if args.limit is not None and i >= args.limit:
                break
            yield row

    stats = FloorStats()
    aggs = collect_intervals(stream(), get_registry(), rules, stats, collect_keys)
    print(f"battles={stats.battle_rows} constraints={stats.constraints} "
          f"prompts={stats.prompts} collapsed={stats.collapsed} "
          f"contradictory={stats.contradictory} "
          f"({stats.contradiction_rate:.1%})")

    length_of = TokenLength(args.tokenizer_dir)
    quota = SelectionQuota(per_tier=args.per_tier,
                           val_floor_per_tier=args.val_floor_per_tier,
                           both_bad_share_cap=args.both_bad_share_cap,
                           long_min_per_tier=args.long_min_per_tier)

    def points_only(pool: dict[str, PromptAgg],
                    era_filter: set[str] | None = None,
                    skip: frozenset[str] = frozenset()) -> dict[str, PromptAgg]:
        out: dict[str, PromptAgg] = {}
        for key, item in pool.items():
            if item.tier is None or key in skip:
                continue
            if era_filter is not None and set(item.eras) != era_filter:
                continue
            out[key] = item
        return out

    # The holdout is chosen first and then removed from the core's pool, so a
    # prompt cannot be both a training example and a temporal test case.
    holdout: list[PromptAgg] = []
    holdout_keys: frozenset[str] = frozenset()
    if holdout_era is not None:
        h_pool = points_only(aggs, era_filter={holdout_era})
        h_sel = select_balanced(h_pool, (holdout_era,), length_of,
                                SelectionQuota(per_tier=args.holdout_per_tier,
                                               val_floor_per_tier=0,
                                               both_bad_share_cap=1.0,
                                               long_min_per_tier=0))
        holdout = h_sel.chosen
        holdout_keys = frozenset(norm_prompt(i.prompt) for i in holdout)
        print(f"holdout ({holdout_era} only): {len(holdout)} prompts across tiers "
              f"{dict(sorted(Counter(i.tier for i in holdout if i.tier).items()))}")

    dedup: DedupResult | None = None
    keep: frozenset[str] | None = None
    if not args.skip_semantic_dedup:
        candidates = [item.prompt for key, item in sorted(aggs.items())
                      if item.tier is not None and key not in holdout_keys]
        embedder = SbEmbedder(args.dedup_model, args.dedup_batch)
        dedup = near_duplicate_representatives(candidates, embedder,
                                               args.dedup_threshold)
        keep = dedup.keep
        print(f"near-duplicates: {dedup.n_prompts} -> {dedup.n_clusters} clusters "
              f"({dedup.dropped} paraphrases dropped)")
    else:
        print("near-duplicates: skipped (--skip-semantic-dedup)")

    # Any prompt with holdout-era evidence leaves the core, not just the ones
    # chosen into the holdout: a 2025 battle in the training side of a temporal
    # split makes the split measure nothing.
    core_pool = {k: v for k, v in aggs.items()
                 if holdout_era is None or holdout_era not in v.eras}
    selection = select_balanced(core_pool, core_keys, length_of, quota, keep=keep)
    if selection.shortfalls:
        print("SHORTFALLS (supply-limited, not a knob): "
              + ", ".join(f"tier {t} short by {n}" for t, n in
                          sorted(selection.shortfalls.items())))

    tmp = args.out.with_suffix(".jsonl.tmp")
    digest = hashlib.sha256()
    written = 0
    with tmp.open("w") as fout:
        for row in floor_rows(selection.chosen):
            line = json.dumps(row, ensure_ascii=False) + "\n"
            fout.write(line)
            digest.update(line.encode())
            written += 1
    tmp.replace(args.out)

    qtmp = args.quarantine_out.with_suffix(".jsonl.tmp")
    with qtmp.open("w") as qout:
        for key, item in sorted(aggs.items()):
            if not item.contradictory:
                continue
            qout.write(json.dumps({"prompt": item.prompt,
                                   "n_constraints": item.n_constraints,
                                   "rules": dict(item.rules_seen),
                                   "supporting_sources": dict(item.supporting_sources),
                                   "last_interval": [item.lo, item.hi]},
                                  ensure_ascii=False) + "\n")
    qtmp.replace(args.quarantine_out)

    holdout_tmp = args.holdout_out.with_suffix(".jsonl.tmp")
    with holdout_tmp.open("w") as hout:
        for row in floor_rows(holdout):
            hout.write(json.dumps(row, ensure_ascii=False) + "\n")
    holdout_tmp.replace(args.holdout_out)

    chosen_prompts = {i.prompt for i in selection.chosen}
    review = review_sample(aggs, chosen_prompts, REVIEW_ROWS, args.seed)
    render_report(args, stats, rules, aggs, selection, dedup, holdout, length_of,
                  stamp, digest.hexdigest()[:12], checked, mismatched, review,
                  collect_keys, core_keys, holdout_era, args.report)

    print(f"wrote {args.out} ({written} floor rows -> {written * len(TIERS)} "
          f"expected pairs), {args.quarantine_out} ({stats.contradictory}), "
          f"{args.holdout_out} ({len(holdout)}), {args.report}")
    print("next: scripts/build_training_pairs.py --in "
          f"{args.out} --sources capability_floor   (never pass --max-pairs: "
          "it samples pair rows and would sever the 5-tier groups)")


if __name__ == "__main__":
    main()
