"""Arena battles read as capability-floor *intervals*, not as winner tiers.

Why this module exists. ``winner_tier`` -- the registry tier of whichever model
won the human vote -- is not a property of the prompt. It is bounded by the
models that happened to be fielded: arena_33k can only ever emit tiers 1, 2 and
4 (``router/scoring.py`` era-caps everything pre-2024-09 at tier 4), arena_140k
skews to 4s and 5s, and 32-44% of battles have ``tier_a == tier_b`` so the label
is set by the matchup alone. Prompts that appear in more than one release
disagree on ``winner_tier`` 31% of the time.

A battle does not say what a prompt's tier *is*. It says what **sufficed** and
what **did not**, which is an interval on the lowest tier that would do:

  stronger tier won      ``[loser_tier + 1, winner_tier]``  the weaker one was
                                                             not enough
  upset (weaker won)     ``[1, winner_tier]``               a light model was
                                                             good enough
  tie across a gap       ``[1, lower_tier]``                ditto, weaker evidence
  both_bad               ``[max_tier + 1, 5]``              nothing at that level
                                                             solved it
  same-tier decisive,
  same-tier tie          no constraint                      says nothing about
                                                             tiers

Each rule is a statement about absolute tiers, so era censoring disappears by
construction, and tier 5 becomes reachable -- it is what "two frontier-class
models both failed" means. Constraints from every battle over a prompt are
intersected; a prompt whose intersection is a single tier gets a label, and one
whose intersection is **empty is quarantined**: self-contradiction is detectable
vote noise, and discarding those prompts is part of the point.

Selection then has to make the set balanced. Era supply is the binding
constraint -- weak models were only fielded in 2023, so tiers 1-2 have no
point-labelled prompts at all in arena_140k. "Balanced" and "era-clean" are not
jointly achievable, so the quota equalizes eras *as far as supply allows* and
reports the shortfall instead of hiding it behind a loosened rule.

The rows emitted here reuse ``router/pairs.py:solving_tier_pairs``' shape exactly
(``candidate_tiers`` + ``solving_tiers``), which is what lets the pairs stage,
the metric layer, the notebook's ``pos_weight=4.0`` and the byte-golden
fingerprints all stay untouched.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

from router.contracts import TIERS, is_validation_prompt

#: Source key written into every floor row and matched by
#: `build_training_pairs.py --sources`. Deliberately does NOT start with
#: "arena_": the fixture guard in `tests/test_registry_rebuild.py` routes rows
#: whose source starts with "arena" and whose label is null through a
#: `row["winner"]` lookup that floor rows cannot satisfy.
FLOOR_SOURCE_KEY = "capability_floor"

#: Provenance for humans and the notebook. Never a dispatch key -- the pairs
#: stage branches on the row shape (`router/pairs.pairs_from_row`).
LABEL_KIND_CAPABILITY_FLOOR = "capability_floor"

#: The budget the rules were designed against. Prompts longer than this in
#: tokens are the ones the old 256-token model never actually read, so selection
#: keeps a share of them on purpose rather than inheriting a one-liner corpus.
PREVIOUS_MAX_LEN = 256

# ----------------------------------------------------------------- rules of doubt

#: A "decisive" win needs genuinely different models. `tier` is a 1-5 bucket, so
#: two models can share a tier while being far apart, or sit one tier apart and
#: be peers; the registry's continuous `score` is what says which. Without this
#: gate, a gpt-3.5-turbo (tier 4 by price) beating claude-v1 (tier 2) reads as a
#: two-tier capability statement when they are near-peers.
MIN_DELTA_SCORE = 1.0

#: both_bad is only a difficulty claim if the pair that failed was capable. Two
#: mini models failing is normal; two frontier models failing is not.
BOTH_BAD_MIN_TIER = 4

#: A tier-1 model beating a tier-4 model is evidence about the ballot, not the
#: prompt -- mismatched formatting, a lazy vote, or a wrong tier in the registry.
MAX_UPSET_GAP = 2

#: Ties are weaker than wins, so they get the same gap ceiling.
MAX_TIE_GAP = 2


@dataclass(frozen=True)
class FloorRules:
    """The four gates as data: pinnable by name and value in a test, variable by
    CLI flag, with the defaults stated exactly once (here)."""

    min_delta_score: float = MIN_DELTA_SCORE
    both_bad_min_tier: int = BOTH_BAD_MIN_TIER
    max_upset_gap: int = MAX_UPSET_GAP
    max_tie_gap: int = MAX_TIE_GAP


DEFAULT_RULES = FloorRules()

#: Rule names, as they appear in `PromptAgg.rules_seen`. These drive both the
#: human review table and the tier-5 provenance cap.
RULE_FAVORED = "stronger_won"
RULE_UPSET = "upset"
RULE_TIE = "tie_lower_sufficed"
RULE_BOTH_BAD = "both_bad"

#: `arena_55k` stores every turn joined into `prompt` while the other releases
#: store only the first turn (`router/dataset.py:147-151`), so "prompt" does not
#: mean the same thing in every row. Recorded rather than silently averaged over.
MULTI_TURN_MARKER = "\n"

#: The battle fields this stage reads. Checked together so a malformed row is
#: skipped rather than half-trusted.
BATTLE_FIELDS = ("tier_a", "tier_b", "winner", "model_a", "model_b")

#: Extras carried through for stratification, present only in some releases.
PROVENANCE_FIELDS = ("language", "is_code", "category_tag")


def norm_prompt(prompt: str) -> str:
    """The pooling key: whitespace-normalized, unhashed, human-readable.

    Same normalization as `router.dataset.dedup_key` (:184-187) but without the
    hash, because this stage must both group a prompt's battles and show the text
    in a review table. `dedup_key` cannot be reused: it hashes, and it folds in
    the model pair -- it exists to *allow* different matchups of one prompt to
    coexist, which is the opposite of what pooling wants.
    """
    return " ".join(prompt.split())


def _delta(score_a: float | None, score_b: float | None) -> float | None:
    """Capability distance, or None when it cannot be established.

    Fails closed. Every registry entry carries a `score` today; a future catalog
    gap must not silently upgrade a peer battle into a capability statement.
    """
    if score_a is None or score_b is None:
        return None
    return abs(score_a - score_b)


def _favored(win: int, lose: int, delta: float | None,
             rules: FloorRules) -> tuple[int, int] | None:
    """The stronger model won: the weaker tier was insufficient, this one served.

    A 1-tier gap makes the interval a single tier, which is where most of the
    labelled pool comes from.
    """
    if delta is None or delta < rules.min_delta_score:
        return None
    return (lose + 1, win)


def _upset(win: int, lose: int, delta: float | None,
          rules: FloorRules) -> tuple[int, int] | None:
    """The weaker model won: a light tier was good enough.

    The only route to tiers 1-2 that does not need a same-tier battle, and
    therefore the scarcest evidence in the corpus -- upsets are rare by nature,
    which is exactly why tiers 1-2 are the short side of this dataset.
    """
    if delta is None or delta < rules.min_delta_score:
        return None
    if lose - win > rules.max_upset_gap:
        return None
    return (1, win)


def _tie(lo: int, hi: int, delta: float | None,
         rules: FloorRules) -> tuple[int, int] | None:
    """A tie across a tier gap says the lesser model matched the greater one."""
    if hi == lo:  # same-tier tie: no tier information at all
        return None
    if delta is None or delta < rules.min_delta_score:
        return None
    if hi - lo > rules.max_tie_gap:
        return None
    return (1, lo)


def _both_bad(lo: int, hi: int, rules: FloorRules) -> tuple[int, int] | None:
    """Both answers were rejected: the prompt outstrips that tier.

    Clamped at the top of the scale -- if the field's best two models failed and
    one was already tier 5, the claim is "tier 5", not "tier 6".
    """
    if lo < rules.both_bad_min_tier:
        return None
    return (min(hi + 1, max(TIERS)), max(TIERS))


def battle_constraint(tier_a: int, tier_b: int, winner: str,
                      delta_score: float | None,
                      rules: FloorRules = DEFAULT_RULES) -> tuple[int, int] | None:
    """The interval this one battle implies for the prompt's capability floor.

    `delta_score` is |score_a - score_b| from the registry, None if either side
    is unknown. Returns None for "no constraint"; it never returns an inverted
    interval -- inversion only arises when two constraints meet.
    """
    lo_tier, hi_tier = min(tier_a, tier_b), max(tier_a, tier_b)
    if winner in ("a", "b"):
        win = tier_a if winner == "a" else tier_b
        lose = tier_b if winner == "a" else tier_a
        if win > lose:
            return _favored(win, lose, delta_score, rules)
        if win < lose:
            return _upset(win, lose, delta_score, rules)
        return None  # same-tier decisive: the matchup decided the label
    if winner == "tie":
        return _tie(lo_tier, hi_tier, delta_score, rules)
    if winner == "both_bad":
        return _both_bad(lo_tier, hi_tier, rules)
    return None  # "unknown", or anything the normalizer could not classify


def constraint_rule(tier_a: int, tier_b: int, winner: str) -> str:
    """Which rule a battle answers to, for provenance.

    Names the branch `battle_constraint` took, without re-checking the gates:
    only called for rows that already produced a constraint, so an empty return
    here would mean the two functions disagree about the branch structure.
    """
    lo_tier, hi_tier = min(tier_a, tier_b), max(tier_a, tier_b)
    if winner in ("a", "b"):
        win = tier_a if winner == "a" else tier_b
        lose = tier_b if winner == "a" else tier_a
        return RULE_FAVORED if win > lose else RULE_UPSET if win < lose else ""
    if winner == "tie":
        return RULE_TIE if hi_tier > lo_tier else ""
    if winner == "both_bad":
        return RULE_BOTH_BAD
    return ""


def intersect(current: tuple[int, int] | None,
              nxt: tuple[int, int]) -> tuple[int, int] | None:
    """Narrow an interval by a new constraint; None once they cannot coexist.

    `current` is None for a prompt's first constraint. Order-independent, so a
    prompt's battles arrive in any sequence and yield the same interval.
    """
    if current is None:
        return nxt
    lo, hi = max(current[0], nxt[0]), min(current[1], nxt[1])
    return (lo, hi) if lo <= hi else None


def collapse(intervals: Iterable[tuple[int, int]]) -> tuple[int, int] | None:
    """Fold a prompt's constraints into one interval; None if contradictory."""
    acc: tuple[int, int] | None = None
    for nxt in intervals:
        acc = intersect(acc, nxt)
        if acc is None:
            return None
    return acc


# ----------------------------------------------------------------------- pooling

@dataclass
class PromptAgg:
    """Everything the floor stage learns about one prompt."""

    prompt: str                        # first-seen original text
    lo: int = 1
    hi: int = max(TIERS)
    n_constraints: int = 0
    contradictory: bool = False
    rules_seen: Counter[str] = field(default_factory=Counter)
    supporting_sources: Counter[str] = field(default_factory=Counter)
    multi_turn: bool = False           # a contributing battle joined all turns
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def tier(self) -> int | None:
        """The label: defined only when the intersection is a single tier."""
        if self.contradictory or self.lo != self.hi:
            return None
        return self.lo

    @property
    def eras(self) -> tuple[str, ...]:
        return tuple(sorted(self.supporting_sources))

    @property
    def corroborated(self) -> bool:
        """More than one battle contributed, so the label is not a single vote."""
        return self.n_constraints >= 2

    @property
    def provenance(self) -> str:
        """Short rule mix, e.g. `stronger_won+upset` -- what the label rests on."""
        return "+".join(sorted(self.rules_seen))


class ScoreResolver(Protocol):
    """The slice of `ModelRegistry` this stage needs.

    Mirrors `router.dataset.NameResolver` (which exposes only `.complexity`) so
    tests can drive this stage with a dict-backed fake and never read
    `data/model_registry/models.json`. `score` rather than `tier` is the point:
    the gates need sub-tier resolution.
    """

    def resolve(self, name: str) -> Mapping[str, Any] | None: ...


@dataclass
class FloorStats:
    """Counters for the selection report: what was thrown away, and why."""

    rows_scanned: int = 0
    battle_rows: int = 0
    constraints: int = 0
    unconstrained: int = 0        # battles that implied nothing (same-tier)
    prompts: int = 0              # prompts with at least one constraint
    collapsed: int = 0
    intervals: int = 0            # consistent but still wide: no point label
    contradictory: int = 0
    tier_hist: Counter[int] = field(default_factory=Counter)

    @property
    def contradiction_rate(self) -> float:
        """Share of constrained prompts whose own battles disagree.

        The dataset's honest noise floor: these are prompts where two human
        ballots placed the capability floor on opposite sides of an interval.
        """
        return self.contradictory / max(self.prompts, 1)


def _score_of(resolver: ScoreResolver, name: str,
              cache: dict[str, float | None]) -> float | None:
    """Registry `score` for a model name, memoized (a release has ~50 names)."""
    if name not in cache:
        entry = resolver.resolve(name)
        raw = entry.get("score") if entry is not None else None
        cache[name] = float(raw) if isinstance(raw, (int, float)) else None
    return cache[name]


def collect_intervals(rows: Iterable[Mapping[str, Any]],
                      resolver: ScoreResolver,
                      rules: FloorRules = DEFAULT_RULES,
                      stats: FloorStats | None = None,
                      battle_keys: frozenset[str] = frozenset(),
                      ) -> dict[str, PromptAgg]:
    """One streaming pass over dataset rows -> per-prompt interval aggregates.

    `rows` is the built dataset file: arena battle rows (carrying `tier_a`,
    `tier_b`, `winner`) are consumed and everything else is skipped, so the
    caller can point this straight at `data/router_training_dataset.jsonl`
    without rebuilding anything. `battle_keys` restricts which releases count --
    used to drop the gated `arena_33k`, or to hold a release out for temporal
    testing.

    Memory shape: aggregates, never rows -- ~286k small records for 327k
    battles. Only prompts that produced at least one constraint are kept, so
    `FloorStats.prompts` stays a meaningful denominator.
    """
    out: dict[str, PromptAgg] = {}
    scores: dict[str, float | None] = {}
    stats = stats if stats is not None else FloorStats()

    for row in rows:
        stats.rows_scanned += 1
        source = str(row.get("source", ""))
        if battle_keys and source not in battle_keys:
            continue
        if not all(k in row for k in BATTLE_FIELDS):
            continue
        stats.battle_rows += 1
        prompt_raw = str(row.get("prompt", ""))
        if not prompt_raw.strip():
            continue
        tier_a, tier_b = int(row["tier_a"]), int(row["tier_b"])  # type: ignore[arg-type]
        winner = str(row["winner"])
        delta = _delta(_score_of(resolver, str(row["model_a"]), scores),
                       _score_of(resolver, str(row["model_b"]), scores))
        nxt = battle_constraint(tier_a, tier_b, winner, delta, rules)
        if nxt is None:
            stats.unconstrained += 1
            continue
        stats.constraints += 1

        key = norm_prompt(prompt_raw)
        item = out.get(key)
        if item is None:
            item = PromptAgg(prompt=prompt_raw.strip())
            out[key] = item
        item.supporting_sources[source] += 1
        item.rules_seen[constraint_rule(tier_a, tier_b, winner)] += 1
        if MULTI_TURN_MARKER in prompt_raw:
            item.multi_turn = True
        for k in PROVENANCE_FIELDS:
            if row.get(k) is not None:
                item.extras.setdefault(k, row[k])
        if item.contradictory:
            continue

        merged = intersect((item.lo, item.hi), nxt) if item.n_constraints else nxt
        item.n_constraints += 1
        if merged is None:
            item.contradictory = True
            continue
        item.lo, item.hi = merged

    for item in out.values():
        stats.prompts += 1
        tier = item.tier
        if item.contradictory:
            stats.contradictory += 1
        elif tier is not None:
            stats.collapsed += 1
            stats.tier_hist[tier] += 1
        else:
            stats.intervals += 1
    return out


# --------------------------------------------------------------------- selection

@dataclass(frozen=True)
class SelectionQuota:
    """What "balanced" means, with the places it cannot be honoured made explicit."""

    per_tier: int = 1500
    #: Held-out prompts per tier. The split itself stays `VAL_PROMPT_MOD`-hashed
    #: (a tested contract); selection only guarantees the hash lands on enough
    #: val prompts per tier, so a 5% split of a small core never leaves a tier
    #: with three evaluation prompts.
    val_floor_per_tier: int = 60
    #: Tier 5 is mostly both_bad rows and both_bad is a *satisfaction* verdict,
    #: not a capability one. Capping the share stops one noisy rule from
    #: defining the top of the scale.
    both_bad_share_cap: float = 0.6
    #: Prompts past the old 256-token budget kept per tier, so the raised
    #: MAX_LEN is exercised instead of inheriting a one-liner corpus.
    long_min_per_tier: int | None = None

    def __post_init__(self) -> None:
        if self.long_min_per_tier is None:
            object.__setattr__(self, "long_min_per_tier", self.per_tier // 10)

    def both_bad_allowance(self) -> int:
        """How many of a tier's picks may rest on `both_bad` alone.

        `int(per_tier * cap)` floors to 0 for a small tier -- at per_tier=1 a 0.6
        cap would empty tier 5 entirely, since nearly all of its points come from
        that one rule. A positive cap therefore admits at least one prompt; only
        `both_bad_share_cap=0.0` forbids the rule outright.
        """
        if self.both_bad_share_cap >= 1.0:
            return self.per_tier
        if self.both_bad_share_cap <= 0.0:
            return 0
        return max(1, int(self.per_tier * self.both_bad_share_cap))


def _rank(item: PromptAgg) -> tuple[int, int, str]:
    """Deterministic preference: corroboration, then rule mix, then prompt hash.

    Never dict order, never `PYTHONHASHSEED` -- the same input file must select
    the same core byte for byte, because the pairs goldens are byte-compared.
    """
    return (0 if item.corroborated else 1,
            0 if len(item.rules_seen) > 1 else 1,
            hashlib.sha1(item.prompt.encode()).hexdigest())


def waterfill(target: int, supply: Mapping[str, int],
              order: Sequence[str]) -> dict[str, int]:
    """Spread `target` picks over eras as evenly as supply allows.

    Equal shares first, then the remainder to whichever eras still have supply,
    in `order`. Returns less than `target` when the pool is short; the shortfall
    is reported rather than papered over by loosening a rule mid-build.
    """
    take = {era: 0 for era in order}
    pool = {era: supply.get(era, 0) for era in order}
    left = min(target, sum(p for p in pool.values() if p > 0))
    live = [era for era in order if pool[era] > 0]
    while left > 0 and live:
        share = max(1, left // len(live))
        for era in list(live):
            add = min(share, pool[era] - take[era], left)
            if add > 0:
                take[era] += add
                left -= add
            if take[era] >= pool[era]:
                live.remove(era)
            if left <= 0:
                break
    return take


def _era_fill(candidates: Sequence[PromptAgg], want: int, eras: Sequence[str],
              quota: SelectionQuota, taken: set[str],
              chosen: list[PromptAgg]) -> None:
    """Append up to `want` era-balanced, provenance-capped picks to `chosen`."""
    if want <= 0:
        return
    supply = {era: sum(1 for i in candidates if era in i.eras) for era in eras}
    by_era: dict[str, list[PromptAgg]] = {era: [] for era in eras}
    for item in candidates:
        for era in item.eras:
            by_era[era].append(item)

    for era, allowance in waterfill(want, supply, eras).items():
        got = 0
        for item in sorted(by_era[era], key=_rank):
            if got >= allowance or item.prompt in taken:
                continue
            # Cap is on the tier as a whole, and `chosen` is this tier's list.
            if item.provenance == RULE_BOTH_BAD:
                bb = sum(1 for c in chosen if c.provenance == RULE_BOTH_BAD)
                if bb >= quota.both_bad_allowance():
                    continue
            chosen.append(item)
            taken.add(item.prompt)
            got += 1


def _pick_tier(pool: Sequence[PromptAgg], want: int, eras: Sequence[str],
               quota: SelectionQuota, long_of: Callable[[str], int]
               ) -> list[PromptAgg]:
    """`want` prompts for one tier: length floor first, then era-balanced fill."""
    chosen: list[PromptAgg] = []
    taken: set[str] = set()
    long_cands = [i for i in pool if long_of(i.prompt) > PREVIOUS_MAX_LEN]
    # Long prompts are the scarcest and the most valuable to keep, so they are
    # picked before the general fill rather than hoping the ordering surfaces
    # some: without this the core would silently be all one-liners again.
    _era_fill(sorted(long_cands, key=_rank), min(want, quota.long_min_per_tier or 0),
              eras, quota, taken, chosen)

    # Mop up in rounds. One pass is not enough: a prompt that fought in two
    # releases is counted in both eras' supply, so the first pass hands out
    # allowances that overlap and leaves several era cells short. Without the
    # refill a tier with 4,600 labelable prompts ships 1,490 and the shortfall
    # looks like scarcity when it is arithmetic. Each round either adds a prompt
    # or the pool is genuinely exhausted (provenance cap, or fewer than `want`
    # collapsed points) and the caller reports the real number.
    while len(chosen) < want:
        before = len(chosen)
        _era_fill([i for i in pool if i.prompt not in taken],
                  want - before, eras, quota, taken, chosen)
        if len(chosen) == before:
            break
    return chosen[:want]


@dataclass
class Selection:
    """The chosen prompts, plus what a reviewer needs in order to distrust them."""

    chosen: list[PromptAgg] = field(default_factory=list)
    shortfalls: dict[int, int] = field(default_factory=dict)
    per_tier: Counter[int] = field(default_factory=Counter)
    per_tier_era: dict[int, Counter[str]] = field(default_factory=dict)
    per_tier_provenance: dict[int, Counter[str]] = field(default_factory=dict)
    per_tier_val: Counter[int] = field(default_factory=Counter)
    per_tier_long: Counter[int] = field(default_factory=Counter)
    per_tier_corroborated: Counter[int] = field(default_factory=Counter)


def select_balanced(aggs: Mapping[str, PromptAgg],
                    eras: Sequence[str],
                    long_of: Callable[[str], int],
                    quota: SelectionQuota = SelectionQuota(),
                    keep: frozenset[str] | None = None) -> Selection:
    """Exactly `per_tier` collapsed prompts per tier, era-balanced where possible.

    `keep` is the near-duplicate representative set: when supplied only those
    prompts are eligible, so a paraphrase cluster cannot straddle the
    prompt-hash split (which keys on exact text and would otherwise put "how do I
    reverse a list" in train and its paraphrase in val).

    The split is never chosen here -- it is `sha1(prompt) % 20 == 0`, a tested
    contract -- but at 1,500/tier that hash lands ~75 val prompts per tier with
    an 8-point spread, so a thin tier is corrected by swapping in same-tier val
    prompts for same-tier train ones. Same-tier keeps the balance exact, and
    because the predicate is label-independent the swap cannot bias what val
    measures.
    """
    out = Selection()
    by_tier: dict[int, list[PromptAgg]] = {t: [] for t in TIERS}
    for item in aggs.values():
        tier = item.tier
        if tier is None:
            continue
        if keep is not None and norm_prompt(item.prompt) not in keep:
            continue
        by_tier[tier].append(item)

    for tier in TIERS:
        picks = _pick_tier(by_tier[tier], quota.per_tier, eras, quota, long_of)
        taken = {p.prompt for p in picks}

        need = quota.val_floor_per_tier - sum(1 for p in picks
                                             if is_validation_prompt(p.prompt))
        if need > 0:
            unchosen = sorted([p for p in by_tier[tier]
                               if p.prompt not in taken
                               and is_validation_prompt(p.prompt)], key=_rank)
            train_side = sorted([p for p in picks if not is_validation_prompt(p.prompt)],
                                key=_rank, reverse=True)
            for swap_in in unchosen[:need]:
                if not train_side:
                    break
                swap_out = train_side.pop()
                picks.remove(swap_out)
                taken.discard(swap_out.prompt)
                picks.append(swap_in)
                taken.add(swap_in.prompt)

        if len(picks) < quota.per_tier:
            out.shortfalls[tier] = quota.per_tier - len(picks)
        out.chosen.extend(picks)
        out.per_tier[tier] = len(picks)
        out.per_tier_era[tier] = Counter(e for p in picks for e in p.eras)
        out.per_tier_provenance[tier] = Counter(p.provenance for p in picks)
        out.per_tier_val[tier] = sum(1 for p in picks
                                     if is_validation_prompt(p.prompt))
        out.per_tier_long[tier] = sum(1 for p in picks
                                      if long_of(p.prompt) > PREVIOUS_MAX_LEN)
        out.per_tier_corroborated[tier] = sum(1 for p in picks if p.corroborated)
    return out


@dataclass
class DedupResult:
    """What the near-duplicate pass decided, so the report can state its blast."""

    keep: frozenset[str]                 # normalized prompts allowed to be selected
    n_prompts: int = 0
    n_clusters: int = 0

    @property
    def dropped(self) -> int:
        return self.n_prompts - self.n_clusters


class Embedder(Protocol):
    """The slice of a sentence-transformers model this stage needs.

    Injected rather than imported: the clustering rule below is domain logic and
    must stay testable (and pyright-clean) without a 90 MB model on disk.
    """

    def encode(self, texts: Sequence[str]) -> Any: ...


def near_duplicate_representatives(prompts: Sequence[str], embedder: Embedder,
                                   threshold: float,
                                   block: int = 512) -> DedupResult:
    """One representative per semantic cluster, by transitive closure of
    ``cosine >= threshold``.

    Why this exists: the validation split hashes the **exact** prompt text, so
    "how do i reverse a list in python" and "reverse a list python" can land on
    opposite sides and val stops measuring generalization. Selection only ever
    draws from `keep`, which collapses each cluster to one member.

    Order-independent by construction (edges then union-find, no greedy
    first-seed), and the representative is the lexicographically smallest
    normalized prompt -- deterministic without leaning on input order.

    Cost is O(n^2 / 2) dot products but run in `block`-row slices, so a 40k
    candidate pool peaks at ~80 MB of similarity matrix. Called on the collapsed
    pool rather than every constrained prompt: only selectable prompts can leak.
    """
    import numpy as np

    result = DedupResult(keep=frozenset(), n_prompts=len(prompts))
    if not prompts:
        return result

    norm = np.asarray(embedder.encode(list(prompts)), dtype=np.float32)
    norm = norm / np.clip(np.linalg.norm(norm, axis=1, keepdims=True), 1e-9, None)

    parent = list(range(len(prompts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for start in range(0, len(prompts), block):
        stop = min(start + block, len(prompts))
        sim = norm[start:stop] @ norm.T                    # (rows, all prompts)
        rows, cols = np.where(sim >= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if start + r < c:                              # each edge once
                union(start + r, c)
        del sim

    clusters: dict[int, list[int]] = {}
    for i in range(len(prompts)):
        clusters.setdefault(find(i), []).append(i)
    keys = {min(norm_prompt(prompts[i]) for i in group) for group in clusters.values()}
    result.keep = frozenset(keys)
    result.n_clusters = len(clusters)
    return result


def floor_rows(selected: Iterable[PromptAgg]) -> Iterator[dict[str, Any]]:
    """Selected prompts as dataset rows, in the shape `solving_tier_pairs` eats.

    Field-for-field the `complexity_router` shape (`router/dataset.py:490-498`):
    five candidate tiers, exactly one solving. That is what preserves the pairs
    stage, the "exactly one gold per prompt" metric cohort, the notebook's 4:1
    `pos_weight`, and the seven byte-compared goldens -- nothing downstream learns
    a new row shape here.

    `supporting_sources` is not decoration. `arena_33k` is CC-BY-NC-4.0 and its
    rows may never be committed; the existing fixture guard checks only a row's
    own `source` key, so a *derived* row quoting a gated battle would sail
    through. Anything from this generator that needs to reach a tracked file must
    be filtered on this field first.
    """
    n = 0
    for item in selected:
        tier = item.tier
        if tier is None:
            # Hard invariant: a floor row is always a collapsed point. A null
            # label would send the pairs stage down the arena branch of the
            # dispatch and emit unusable 0.5/0.5 rows.
            continue
        n += 1  # counted before the yield: this is a generator, and a consumer
               # that breaks early must still report the truth (dataset.py:285)
        yield {
            "id": f"{FLOOR_SOURCE_KEY}:{n - 1:07d}",
            "source": FLOOR_SOURCE_KEY,
            "label_kind": LABEL_KIND_CAPABILITY_FLOOR,
            "prompt": item.prompt,
            "candidate_tiers": list(TIERS),
            "solving_tiers": [tier],
            "label_tier": tier,
            "n_constraints": item.n_constraints,
            "rules": dict(item.rules_seen),
            "supporting_sources": dict(item.supporting_sources),
            "multi_turn": item.multi_turn,
            # Original arena names, as `arena_rows` splats them (dataset.py:301),
            # so a floor row's provenance reads like the battle rows it came from.
            **item.extras,
        }
