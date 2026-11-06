"""The capability-floor rules: each gate pinned individually, then selection.

`winner_tier` said "whichever model won is the prompt's tier". These tests pin the
replacement: a battle is a statement about what *sufficed*, and a prompt's label
only exists where its battles agree on one tier. The rules that decide that are
the whole dataset's semantics, so each one gets its own test with its boundary
value -- a silent change to `MIN_DELTA_SCORE` or `BOTH_BAD_MIN_TIER` is a change
to what the labels mean.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from router.contracts import TIERS, is_validation_prompt
from router.floor import (
    BOTH_BAD_MIN_TIER,
    FLOOR_SOURCE_KEY,
    MAX_TIE_GAP,
    MAX_UPSET_GAP,
    MIN_DELTA_SCORE,
    PREVIOUS_MAX_LEN,
    RULE_BOTH_BAD,
    RULE_FAVORED,
    RULE_TIE,
    RULE_UPSET,
    FloorRules,
    FloorStats,
    PromptAgg,
    SelectionQuota,
    battle_constraint,
    collapse,
    collect_intervals,
    constraint_rule,
    floor_rows,
    intersect,
    norm_prompt,
    select_balanced,
    waterfill,
)


class ScoreFake:
    """Registry-shaped `score` lookup, so no test reads data/model_registry."""

    def __init__(self, scores: dict[str, float | None],
                 tiers: dict[str, int] | None = None) -> None:
        self.scores = scores
        self.tiers = tiers or {}

    def resolve(self, name: str) -> dict[str, Any] | None:
        if name not in self.scores:
            return None
        return {"canonical_id": name, "score": self.scores[name],
                "complexity": self.tiers.get(name)}


BOTH = ScoreFake({"weak": 2.0, "strong": 4.0, "mid": 3.0, "top": 5.0,
                  "peer_a": 2.4, "peer_b": 2.6, "unknown_a": None})


def battle(tier_a: int, tier_b: int, winner: str, ma: str = "weak",
           mb: str = "strong", **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"source": "arena_100k", "prompt": f"p {tier_a}{tier_b}",
                           "tier_a": tier_a, "tier_b": tier_b, "winner": winner,
                           "model_a": ma, "model_b": mb}
    row.update(extra)
    return row


class TestNormPrompt:
    def test_collapses_whitespace_runs(self) -> None:
        assert norm_prompt("  a\t\n b  ") == "a b"

    def test_is_not_hashed(self) -> None:
        # Reports and the review table must show text, and `dedup_key` hashes.
        assert norm_prompt("hello world") == "hello world"


class TestFavoredRule:
    """stronger won -> [loser+1, winner]; a 1-tier gap is already a point."""

    def test_gap_one_is_a_point(self) -> None:
        assert battle_constraint(3, 4, "b", 2.0) == (4, 4)

    def test_gap_three_is_an_interval(self) -> None:
        # tier 4 ("b") beats tier 1 ("a"): the weaker side was insufficient, so
        # the floor is somewhere in [2, 4] and the prompt gets no point label.
        assert battle_constraint(1, 4, "b", 3.0) == (2, 4)

    def test_loser_tier_plus_one_is_the_lower_bound(self) -> None:
        # "tier 2 was not enough" is the claim; it survives even when the winner
        # is far above, so the interval never starts at 1.
        assert battle_constraint(2, 5, "b", 3.0) == (3, 5)

    def test_delta_below_the_gate_is_no_constraint(self) -> None:
        assert battle_constraint(2, 4, "b", MIN_DELTA_SCORE - 0.01) is None

    def test_delta_at_the_gate_counts(self) -> None:
        assert battle_constraint(2, 4, "b", MIN_DELTA_SCORE) == (3, 4)

    def test_missing_score_fails_closed(self) -> None:
        assert battle_constraint(2, 4, "b", None) is None

    def test_same_tier_decisive_says_nothing(self) -> None:
        # 32-44% of battles: the label would be pure matchup. This is the defect
        # the whole module exists to fix.
        assert battle_constraint(3, 3, "a", 2.0) is None


class TestUpsetRule:
    """weaker won -> [1, winner]: the only era-free route to tiers 1-2."""

    def test_upset_bounds_the_floor_above_at_the_winner(self) -> None:
        assert battle_constraint(4, 2, "b", 2.0) == (1, 2)

    def test_gap_at_the_cap_still_counts(self) -> None:
        # tier_a=3, tier_b=1, "b" won: the lighter model was good enough.
        assert battle_constraint(1 + MAX_UPSET_GAP, 1, "b", 2.0) == (1, 1)

    def test_wide_upset_is_distrusted(self) -> None:
        # A tier-1 beating a tier-4 is more likely a bad ballot or a wrong
        # registry tier than a light model solving the prompt.
        assert battle_constraint(1 + MAX_UPSET_GAP + 1, 1, "b", 3.0) is None

    def test_delta_gate_applies_to_upsets_too(self) -> None:
        assert battle_constraint(4, 2, "b", MIN_DELTA_SCORE - 0.5) is None


class TestTieRule:
    def test_tie_across_a_gap_says_the_lesser_sufficed(self) -> None:
        assert battle_constraint(2, 3, "tie", 2.0) == (1, 2)

    def test_same_tier_tie_is_silent(self) -> None:
        assert battle_constraint(3, 3, "tie", 2.0) is None

    def test_wide_tie_is_distrusted(self) -> None:
        assert battle_constraint(1, 1 + MAX_TIE_GAP + 1, "tie", 3.0) is None

    def test_delta_gate_applies(self) -> None:
        assert battle_constraint(2, 3, "tie", None) is None


class TestBothBadRule:
    """both_bad is how tier 5 becomes reachable at all."""

    def test_frontier_pair_failing_means_tier_five(self) -> None:
        assert battle_constraint(4, 4, "both_bad", None) == (5, 5)

    def test_clamped_at_the_top_of_the_scale(self) -> None:
        # Two tier-5 models failing is "tier 5", not "tier 6" -- and it must not
        # register as a self-contradiction.
        assert battle_constraint(5, 5, "both_bad", None) == (5, 5)

    def test_capable_pair_with_a_gap(self) -> None:
        assert battle_constraint(4, 5, "both_bad", None) == (5, 5)

    def test_weak_pair_failing_proves_little(self) -> None:
        assert battle_constraint(2, 2, "both_bad", None) is None

    def test_gate_needs_both_models_capable(self) -> None:
        # min(tier) is the gate, not max: a tier-2 side makes the verdict about
        # the tier-2 model's answer.
        lo = BOTH_BAD_MIN_TIER - 1
        assert battle_constraint(lo, 5, "both_bad", None) is None

    def test_at_the_gate_boundary(self) -> None:
        assert battle_constraint(BOTH_BAD_MIN_TIER, 4, "both_bad", None) == (5, 5)

    def test_rules_are_data_not_hardcoded_in_the_branch(self) -> None:
        strict = FloorRules(both_bad_min_tier=5)
        assert battle_constraint(4, 5, "both_bad", None, strict) is None


class TestUnknownWinner:
    @pytest.mark.parametrize("winner", ["unknown", "", "both_good"])
    def test_unclassifiable_ballots_are_dropped(self, winner: str) -> None:
        assert battle_constraint(2, 4, winner, 3.0) is None


class TestConstraintRule:
    def test_names_the_branch(self) -> None:
        assert constraint_rule(2, 4, "b") == RULE_FAVORED
        assert constraint_rule(4, 2, "b") == RULE_UPSET
        assert constraint_rule(2, 4, "tie") == RULE_TIE
        assert constraint_rule(4, 4, "both_bad") == RULE_BOTH_BAD

    def test_silent_cases_name_no_rule(self) -> None:
        assert constraint_rule(3, 3, "a") == ""
        assert constraint_rule(3, 3, "tie") == ""

    def test_agrees_with_the_constraint_functions_rule(self) -> None:
        # `collect_intervals` records provenance for every row that produced a
        # constraint, so a rule name of "" there means the two disagree.
        for tier_a, tier_b, winner in [(2, 4, "b"), (4, 2, "b"), (2, 4, "tie"),
                                       (4, 4, "both_bad"), (5, 5, "both_bad")]:
            if battle_constraint(tier_a, tier_b, winner, 3.0) is not None:
                assert constraint_rule(tier_a, tier_b, winner) != ""


class TestIntersectAndCollapse:
    def test_first_constraint_passes_through(self) -> None:
        assert intersect(None, (2, 4)) == (2, 4)

    def test_narrowing(self) -> None:
        assert intersect((1, 4), (3, 5)) == (3, 4)

    def test_disjoint_is_none(self) -> None:
        assert intersect((1, 2), (4, 5)) is None

    def test_touching_edges_still_a_point(self) -> None:
        assert intersect((1, 3), (3, 5)) == (3, 3)

    def test_collapse_is_order_independent(self) -> None:
        sets = [(1, 3), (2, 5), (2, 3)]
        forward = collapse(sets)
        for _ in range(8):
            shuffled = sets[:]
            random.shuffle(shuffled)
            assert collapse(shuffled) == forward

    def test_collapse_of_a_contradiction_is_none_in_every_order(self) -> None:
        for perm in ([(1, 2), (3, 4)], [(3, 4), (1, 2)]):
            assert collapse(perm) is None

    def test_empty_input_is_none(self) -> None:
        assert collapse([]) is None


class TestCollectIntervals:
    def test_pools_two_battles_of_one_prompt(self) -> None:
        # A gap-1 win says [3,3] and an upset says [1,2]: contradictory, so the
        # prompt is quarantined rather than averaged.
        rows = [battle(2, 3, "b"), battle(4, 2, "b")]
        rows[1]["prompt"] = rows[0]["prompt"]
        aggs = collect_intervals(rows, BOTH)
        item = next(iter(aggs.values()))
        assert item.contradictory is True
        assert item.tier is None
        assert item.n_constraints == 2

    def test_intersecting_constraints_give_a_point(self) -> None:
        rows = [battle(2, 4, "a"), battle(1, 3, "b")]  # [1,2] and [2,3] -> [2,2]
        rows[1]["prompt"] = rows[0]["prompt"]
        aggs = collect_intervals(rows, BOTH)
        item = next(iter(aggs.values()))
        assert item.contradictory is False
        assert item.tier == 2
        assert item.corroborated is True

    def test_unconstrained_battles_never_create_a_prompt(self) -> None:
        # Otherwise `FloorStats.prompts` inflates and contradiction_rate, which is
        # the dataset's noise floor, reads lower than the truth.
        stats = FloorStats()
        aggs = collect_intervals([battle(3, 3, "a")], BOTH, stats=stats)
        assert aggs == {}
        assert stats.battle_rows == 1 and stats.unconstrained == 1
        assert stats.prompts == 0

    def test_non_battle_rows_are_skipped(self) -> None:
        rows = [{"source": "complexity_router", "prompt": "x",
                 "candidate_tiers": [1, 2, 3, 4, 5], "solving_tiers": [3]}]
        stats = FloorStats()
        assert collect_intervals(rows, BOTH, stats=stats) == {}
        assert stats.rows_scanned == 1 and stats.battle_rows == 0

    def test_battle_keys_filter_to_the_named_releases(self) -> None:
        rows = [battle(2, 3, "b"), battle(2, 4, "b")]
        rows[1]["source"] = "arena_33k"
        stats = FloorStats()
        aggs = collect_intervals(rows, BOTH, stats=stats,
                                 battle_keys=frozenset({"arena_100k"}))
        assert sum(sum(a.supporting_sources.values()) for a in aggs.values()) == 1
        assert stats.battle_rows == 1  # the filtered row never became a battle
        assert all("arena_33k" not in a.supporting_sources for a in aggs.values())

    def test_records_era_provenance_per_prompt(self) -> None:
        # The license guard needs it (arena_33k is CC-BY-NC) and so does the
        # era-balance report.
        rows = [battle(2, 3, "b"), battle(2, 4, "b")]
        rows[1]["prompt"] = rows[0]["prompt"]
        rows[1]["source"] = "arena_33k"
        item = next(iter(collect_intervals(rows, BOTH).values()))
        assert dict(item.supporting_sources) == {"arena_100k": 1, "arena_33k": 1}
        assert item.eras == ("arena_100k", "arena_33k")

    def test_records_rule_provenance(self) -> None:
        rows = [battle(2, 4, "b"), battle(4, 2, "b")]
        rows[1]["prompt"] = rows[0]["prompt"]
        item = next(iter(collect_intervals(rows, BOTH).values()))
        assert item.rules_seen[RULE_FAVORED] == 1
        assert item.rules_seen[RULE_UPSET] == 1
        assert item.provenance == f"{RULE_FAVORED}+{RULE_UPSET}"

    def test_multi_turn_prompts_are_flagged_not_merged_silently(self) -> None:
        # arena_55k joins every turn into `prompt`; the others keep turn 1 only.
        rows = [{"source": "arena_55k", "prompt": "line one\nline two",
                 "tier_a": 2, "tier_b": 4, "winner": "b",
                 "model_a": "weak", "model_b": "strong"}]
        item = next(iter(collect_intervals(rows, BOTH).values()))
        assert item.multi_turn is True

    def test_extras_are_carried_once(self) -> None:
        rows = [battle(2, 4, "b", language="English", is_code=True,
                       category_tag='["writing"]')]
        item = next(iter(collect_intervals(rows, BOTH).values()))
        assert item.extras["language"] == "English"
        assert item.extras["is_code"] is True

    def test_empty_prompt_is_skipped(self) -> None:
        rows = [battle(2, 4, "b")]
        rows[0]["prompt"] = "   "
        assert collect_intervals(rows, BOTH) == {}

    def test_stats_tier_histogram_counts_points_only(self) -> None:
        rows = [battle(2, 3, "b"),      # gap 1 -> [3,3]: a point
                battle(1, 4, "b")]      # gap 3 -> [2,4]: consistent but wide
        stats = FloorStats()
        collect_intervals(rows, BOTH, stats=stats)
        assert stats.collapsed == 1 and stats.intervals == 1
        assert stats.prompts == 2
        assert dict(stats.tier_hist) == {3: 1}

    def test_peer_scores_are_not_read_as_a_capability_gap(self) -> None:
        # The registry's continuous score is the point: tier 2 vs tier 4 can be
        # a 0.2-score gap, and that vote must not become a [3,4] claim.
        rows = [{"source": "arena_100k", "prompt": "peer battle", "tier_a": 2,
                 "tier_b": 4, "winner": "b", "model_a": "peer_a",
                 "model_b": "peer_b"}]
        stats = FloorStats()
        assert collect_intervals(rows, BOTH, stats=stats) == {}
        assert stats.unconstrained == 1


class TestWaterfill:
    def test_equal_shares_when_supply_allows(self) -> None:
        eras = ("a", "b", "c")
        assert waterfill(1500, {"a": 900, "b": 900, "c": 900}, eras) == {
            "a": 500, "b": 500, "c": 500}

    def test_remainder_goes_to_eras_with_supply(self) -> None:
        # Mirrors the real tier-1 shape: 140k has almost none, 100k runs out
        # early, and the two 2023 releases absorb the rest within one pick of
        # each other -- which is as era-balanced as that supply allows.
        eras = ("arena_33k", "arena_55k", "arena_100k", "arena_140k")
        supply = {"arena_33k": 2663, "arena_55k": 1671, "arena_100k": 404,
                  "arena_140k": 9}
        got = waterfill(1500, supply, eras)
        assert sum(got.values()) == 1500
        assert got["arena_100k"] == 404 and got["arena_140k"] == 9  # capped
        assert abs(got["arena_33k"] - got["arena_55k"]) <= 1
        assert got["arena_33k"] > got["arena_100k"]

    def test_shortfall_is_reported_not_invented(self) -> None:
        got = waterfill(1500, {"a": 10, "b": 20}, ("a", "b"))
        assert sum(got.values()) == 30

    def test_zero_supply_era_is_never_picked(self) -> None:
        assert waterfill(100, {"a": 0, "b": 500}, ("a", "b")) == {"a": 0, "b": 100}

    def test_deterministic_and_floored_to_shared_remainder(self) -> None:
        # 'a' runs out, so b and c split the rest; the last pick goes to whichever
        # comes first in `order`, which is what makes the core byte-reproducible.
        supply = {"a": 7, "b": 900, "c": 900}
        expected = {"a": 7, "b": 747, "c": 746}
        for _ in range(5):
            assert waterfill(1500, supply, ("a", "b", "c")) == expected
        assert waterfill(1500, supply, ("a", "c", "b")) == {"a": 7, "c": 747,
                                                            "b": 746}


def agg(prompt: str, tier: int, eras: tuple[str, ...], n_constraints: int = 1,
        rules: tuple[str, ...] = (RULE_FAVORED,), multi_turn: bool = False
        ) -> PromptAgg:
    item = PromptAgg(prompt=prompt, lo=tier, hi=tier, n_constraints=n_constraints)
    for era in eras:
        item.supporting_sources[era] = 1
    for rule in rules:
        item.rules_seen[rule] = 1
    item.multi_turn = multi_turn
    return item


def synthetic_pool(tier: int, n: int, eras: tuple[str, ...],
                   start: int = 0, rules: tuple[str, ...] = (RULE_FAVORED,),
                   n_constraints: int = 1) -> list[PromptAgg]:
    # Prompts long enough to be "over the old budget" are marked by padding, so
    # `long_of=len` in these tests sees the difference.
    pad = "x " * (PREVIOUS_MAX_LEN + 5)
    return [agg(f"{pad}prompt {tier} {i + start}" if (i + start) % 3 == 0
                else f"prompt {tier} {i + start}", tier, eras, n_constraints, rules)
            for i in range(n)]


ERAS = ("arena_33k", "arena_55k", "arena_100k", "arena_140k")


class TestSelectBalanced:
    def pool(self) -> dict[str, PromptAgg]:
        pool: dict[str, PromptAgg] = {}
        # tiers 1-2 deliberately short of per_tier to exercise the shortfall path
        for tier, counts in {1: {"arena_33k": 40, "arena_55k": 30},
                             2: {"arena_33k": 50},
                             3: {e: 40 for e in ERAS},
                             4: {e: 40 for e in ERAS},
                             5: {"arena_100k": 60, "arena_140k": 60}}.items():
            items: list[PromptAgg] = []
            for era, n in counts.items():
                items += synthetic_pool(tier, n, (era,), start=len(items))
            for item in items:
                pool[norm_prompt(item.prompt)] = item
        return pool

    def test_exact_balance_where_supply_allows(self) -> None:
        quota = SelectionQuota(per_tier=100, val_floor_per_tier=0,
                               both_bad_share_cap=1.0, long_min_per_tier=0)
        sel = select_balanced(self.pool(), ERAS, len, quota)
        assert sel.per_tier[3] == 100 and sel.per_tier[4] == 100
        assert sel.per_tier[5] == 100
        assert 3 not in sel.shortfalls and 5 not in sel.shortfalls

    def test_short_tiers_ship_short_and_say_so(self) -> None:
        # Never fill a quota by loosening a rule: the report is where the
        # scarcity shows up.
        quota = SelectionQuota(per_tier=100, val_floor_per_tier=0,
                               both_bad_share_cap=1.0, long_min_per_tier=0)
        sel = select_balanced(self.pool(), ERAS, len, quota)
        assert sel.per_tier[1] == 70 and sel.shortfalls[1] == 30
        assert sel.per_tier[2] == 50 and sel.shortfalls[2] == 50

    def test_eras_are_spread_within_a_tier(self) -> None:
        quota = SelectionQuota(per_tier=80, val_floor_per_tier=0,
                               both_bad_share_cap=1.0, long_min_per_tier=0)
        sel = select_balanced(self.pool(), ERAS, len, quota)
        assert dict(sel.per_tier_era[3]) == {e: 20 for e in ERAS}

    def test_a_positive_share_cap_never_empties_a_small_tier(self) -> None:
        # int(1 * 0.6) == 0 would empty tier 5 entirely, since almost all tier-5
        # points come from both_bad alone -- a smoke run would report a
        # supply shortage that was really arithmetic.
        assert SelectionQuota(per_tier=1, both_bad_share_cap=0.6).both_bad_allowance() == 1
        assert SelectionQuota(per_tier=1500, both_bad_share_cap=0.6).both_bad_allowance() == 900
        assert SelectionQuota(per_tier=1500, both_bad_share_cap=0.0).both_bad_allowance() == 0
        assert SelectionQuota(per_tier=1500, both_bad_share_cap=1.0).both_bad_allowance() == 1500
        pool = {norm_prompt(p.prompt): p for p in
                synthetic_pool(5, 5, ("arena_140k",), rules=(RULE_BOTH_BAD,))}
        quota = SelectionQuota(per_tier=1, val_floor_per_tier=0,
                               both_bad_share_cap=0.6, long_min_per_tier=0)
        assert select_balanced(pool, ERAS, len, quota).per_tier[5] == 1

    def test_selection_is_deterministic_across_input_orders(self) -> None:
        # The pairs goldens are byte-compared, so a dict-order or PYTHONHASHSEED
        # dependency would break rebuilds while looking like a data change.
        quota = SelectionQuota(per_tier=60, val_floor_per_tier=0,
                               both_bad_share_cap=1.0, long_min_per_tier=0)
        pool = self.pool()
        keys = list(pool)
        random.Random(1).shuffle(keys)
        shuffled = {k: pool[k] for k in keys}
        a = [p.prompt for p in select_balanced(pool, ERAS, len, quota).chosen]
        b = [p.prompt for p in select_balanced(shuffled, ERAS, len, quota).chosen]
        assert sorted(a) == sorted(b)

    def test_corroborated_prompts_precede_single_vote_ones(self) -> None:
        pool = {norm_prompt(p.prompt): p for p in
                synthetic_pool(3, 30, ("arena_100k",)) +
                [agg("corroborated three", 3, ("arena_100k",), n_constraints=2),
                 agg("corroborated four", 3, ("arena_100k",), n_constraints=4)]}
        quota = SelectionQuota(per_tier=2, val_floor_per_tier=0,
                               both_bad_share_cap=1.0, long_min_per_tier=0)
        sel = select_balanced(pool, ERAS, len, quota)
        assert sel.per_tier_corroborated[3] == 2

    def test_both_bad_share_is_capped(self) -> None:
        pool = {norm_prompt(p.prompt): p for p in synthetic_pool(
            5, 200, ("arena_140k",), rules=(RULE_BOTH_BAD,))}
        quota = SelectionQuota(per_tier=100, val_floor_per_tier=0,
                               both_bad_share_cap=0.5, long_min_per_tier=0)
        sel = select_balanced(pool, ERAS, len, quota)
        # Every candidate is both_bad-only, so the cap limits the tier: a cap that
        # silently filled to per_tier would be a cap that does not exist.
        assert sel.per_tier[5] == 50
        assert sel.shortfalls[5] == 50
        assert sel.per_tier_provenance[5][RULE_BOTH_BAD] == 50

    def test_long_prompts_are_kept_so_max_len_is_exercised(self) -> None:
        pool = {norm_prompt(p.prompt): p for p in
                synthetic_pool(4, 120, ("arena_140k",))}
        quota = SelectionQuota(per_tier=60, val_floor_per_tier=0,
                               both_bad_share_cap=1.0, long_min_per_tier=20)
        sel = select_balanced(pool, ERAS, len, quota)
        assert sel.per_tier[4] == 60
        assert sel.per_tier_long[4] >= 20

    def test_val_floor_is_enforced_per_tier(self) -> None:
        pool = {norm_prompt(p.prompt): p for p in
                synthetic_pool(3, 200, ("arena_100k",))}
        quota = SelectionQuota(per_tier=40, val_floor_per_tier=12,
                               both_bad_share_cap=1.0, long_min_per_tier=0)
        sel = select_balanced(pool, ERAS, len, quota)
        assert sel.per_tier[3] == 40
        assert sel.per_tier_val[3] >= 12
        # The swap must not re-derive the split: whatever it counted has to agree
        # with the contract's predicate, or val is not the cohort eval will use.
        chosen = [p for p in sel.chosen if p.tier == 3]
        counted = sum(1 for p in chosen if is_validation_prompt(p.prompt))
        assert counted == sel.per_tier_val[3]

    def test_keep_set_restricts_to_cluster_representatives(self) -> None:
        items = synthetic_pool(3, 60, ("arena_100k",))
        keep = frozenset({norm_prompt(items[0].prompt)})
        quota = SelectionQuota(per_tier=50, val_floor_per_tier=0,
                               both_bad_share_cap=1.0, long_min_per_tier=0)
        sel = select_balanced({norm_prompt(i.prompt): i for i in items},
                              ERAS, len, quota, keep=keep)
        assert sel.per_tier[3] == 1
        assert sel.shortfalls[3] == 49

    def test_no_prompt_is_selected_twice(self) -> None:
        quota = SelectionQuota(per_tier=100, val_floor_per_tier=5,
                               both_bad_share_cap=1.0, long_min_per_tier=10)
        sel = select_balanced(self.pool(), ERAS, len, quota)
        prompts = [p.prompt for p in sel.chosen]
        assert len(prompts) == len(set(prompts))

    def test_overlapping_era_membership_does_not_short_the_tier(self) -> None:
        # Every prompt fought in two releases, so era supply double-counts and a
        # single allowance pass hands out more picks than exist. The tier has
        # 4x the needed supply and must still fill: a shortfall reported here
        # would read as label scarcity when it is arithmetic.
        pool = {}
        for i in range(400):
            item = agg(f"multi era prompt {i}", 4, ("arena_100k", "arena_55k"))
            pool[norm_prompt(item.prompt)] = item
        quota = SelectionQuota(per_tier=150, val_floor_per_tier=0,
                               both_bad_share_cap=1.0, long_min_per_tier=0)
        sel = select_balanced(pool, ("arena_55k", "arena_100k"), len, quota)
        assert sel.per_tier[4] == 150
        assert 4 not in sel.shortfalls

    def test_shortfall_is_reported_when_a_cap_binds_not_only_when_supply_runs_out(self) -> None:
        # 3,000 labelable tier-5 prompts, but a 0.3 both_bad cap and no other
        # provenance means the honest answer is 900 chosen + a shortfall.
        pool = {norm_prompt(p.prompt): p for p in
                synthetic_pool(5, 300, ("arena_100k",), rules=(RULE_BOTH_BAD,))}
        quota = SelectionQuota(per_tier=150, val_floor_per_tier=0,
                               both_bad_share_cap=0.3, long_min_per_tier=0)
        sel = select_balanced(pool, ERAS, len, quota)
        assert sel.per_tier[5] == 45
        assert sel.shortfalls[5] == 105
        assert sel.per_tier_provenance[5][RULE_BOTH_BAD] == 45

    def test_wide_intervals_are_not_labelled(self) -> None:
        item = PromptAgg(prompt="wide", lo=2, hi=4, n_constraints=1)
        pool = {"wide": item, **{norm_prompt(p.prompt): p for p in
                                 synthetic_pool(2, 40, ("arena_33k",))}}
        quota = SelectionQuota(per_tier=10, val_floor_per_tier=0,
                               both_bad_share_cap=1.0, long_min_per_tier=0)
        sel = select_balanced(pool, ERAS, len, quota)
        assert all(p.tier is not None for p in sel.chosen)


class TestFloorRows:
    def one(self) -> list[dict[str, Any]]:
        item = agg("reverse a linked list", 3, ("arena_100k",), n_constraints=2,
                   rules=(RULE_FAVORED, RULE_UPSET))
        item.extras["language"] = "English"
        return list(floor_rows([item]))

    def test_shape_is_what_solving_tier_pairs_expects(self) -> None:
        row = self.one()[0]
        assert row["candidate_tiers"] == list(TIERS)
        assert row["solving_tiers"] == [3]
        assert row["label_tier"] == 3
        # The pairs stage dispatches on this field being a list; without it the
        # row would take the arena branch and look for `winner`.
        assert isinstance(row["candidate_tiers"], list)

    def test_source_is_not_prefixed_arena(self) -> None:
        # test_registry_rebuild routes arena-prefixed null-label rows through a
        # row["winner"] lookup these rows cannot satisfy.
        row = self.one()[0]
        assert not row["source"].startswith("arena")
        assert row["source"] == FLOOR_SOURCE_KEY

    def test_provenance_fields_are_present(self) -> None:
        row = self.one()[0]
        assert row["n_constraints"] == 2
        assert row["rules"] == {RULE_FAVORED: 1, RULE_UPSET: 1}
        assert row["supporting_sources"] == {"arena_100k": 1}
        assert row["label_kind"] == "capability_floor"

    def test_ids_are_sequential_from_zero(self) -> None:
        items = [agg(f"prompt {i}", 3, ("arena_100k",)) for i in range(3)]
        ids = [r["id"] for r in floor_rows(items)]
        assert ids == [f"{FLOOR_SOURCE_KEY}:{i:07d}" for i in range(3)]

    def test_uncollapsed_prompts_emit_nothing(self) -> None:
        # A null label would produce 0.5/0.5 pairs the loss cannot use, so the
        # stage's invariant is "never emit one" rather than "emit and let a
        # downstream filter catch it".
        assert list(floor_rows([PromptAgg(prompt="wide", lo=2, hi=4,
                                          n_constraints=1)])) == []
        # Contradictory prompts are quarantined upstream, and a point interval is
        # no label once the flags say so: lo == hi alone must not be enough.
        quarantined = PromptAgg(prompt="x", lo=3, hi=3, n_constraints=2,
                                contradictory=True)
        assert quarantined.tier is None
        assert list(floor_rows([quarantined])) == []

    def test_rows_are_json_serializable(self) -> None:
        row = self.one()[0]
        assert json.loads(json.dumps(row))["label_tier"] == 3

    def test_survives_the_real_pairs_dispatch(self) -> None:
        # The contract that lets this stage exist without touching pairs.py:
        # one gold, four losers, complete 5-tier group.
        from router.metrics import group_by_prompt, metric_cohort
        from router.pairs import PairFilter, PairStats, pairs_from_row

        stats = PairStats()
        pairs = [p for row in floor_rows([agg(f"prompt {i}", (i % 5) + 1,
                                              ("arena_100k",))
                                          for i in range(5)])
                 for p in pairs_from_row(row, PairFilter(), stats)]
        assert len(pairs) == 25
        assert stats.no_signal_skipped == 0
        by_prompt = group_by_prompt([(p.prompt, p.tier, p.outcome) for p in pairs])
        decisive, eval_pairs = metric_cohort(by_prompt)
        assert len(decisive) == 5 and eval_pairs == 20
