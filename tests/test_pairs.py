"""Pairs derivation: the row-shape dispatch, the filtering rules, the sampler.

The pairs file is what training actually sees, so the load-bearing property here
is the one `BCEWithLogitsLoss(pos_weight=4.0)` and `routing_accuracy` both rely
on: a decisive row contributes exactly one 1.0 and four 0.0s (or, for arena
battles, one 1.0 and one 0.0).
"""

from __future__ import annotations

from typing import Any

import pytest

from router.pairs import (
    Pair,
    PairFilter,
    PairStats,
    allocate,
    arena_pairs,
    pairs_from_row,
    solving_tier_pairs,
    strata_key,
)


def arena_row(tier_a: Any = 4, tier_b: Any = 2, winner: Any = "a",
              prompt: str = "p", source: str = "arena_100k") -> dict[str, Any]:
    """Tiers are Any on purpose: several tests feed malformed rows to check that
    unresolvable labels are counted as no-signal rather than crashing."""
    return {"prompt": prompt, "tier_a": tier_a, "tier_b": tier_b,
            "winner": winner, "source": source}


def solving_row(candidates=(1, 2, 3, 4, 5), solving=(3,), prompt="p",
                source="complexity_router"):
    return {"prompt": prompt, "source": source,
            "candidate_tiers": list(candidates), "solving_tiers": list(solving),
            "label_tier": min(solving) if solving else None}


class TestArenaRules:
    def test_decisive_winner_gets_one_1_0(self):
        s = PairStats()
        pairs = arena_pairs(arena_row(4, 2, "a"), PairFilter(), s)
        assert [(p.tier, p.outcome) for p in pairs] == [(4, 1.0), (2, 0.0)]
        assert s.no_signal_skipped == 0

    def test_winner_b_is_reversed_not_relabelled(self):
        pairs = arena_pairs(arena_row(4, 2, "b"), PairFilter(), PairStats())
        assert [(p.tier, p.outcome) for p in pairs] == [(2, 1.0), (4, 0.0)]

    def test_ties_are_half_targets_not_dropped_by_default(self):
        for winner in ("tie", "both_bad"):
            pairs = arena_pairs(arena_row(4, 2, winner), PairFilter(), PairStats())
            assert [p.outcome for p in pairs] == [0.5, 0.5], winner

    def test_drop_ties_skips_both_tie_flavours(self):
        for winner in ("tie", "both_bad"):
            s = PairStats()
            assert arena_pairs(arena_row(4, 2, winner),
                               PairFilter(drop_ties=True), s) == []
            assert s.ties_skipped == 1

    def test_same_tier_decisive_battles_are_skipped(self):
        s = PairStats()
        assert arena_pairs(arena_row(3, 3, "a"), PairFilter(), s) == []
        assert s.same_tier_skipped == 1

    def test_keep_same_tier_emits_the_self_cancelling_pair(self):
        pairs = arena_pairs(arena_row(3, 3, "a"), PairFilter(keep_same_tier=True),
                            PairStats())
        assert [(p.tier, p.outcome) for p in pairs] == [(3, 1.0), (3, 0.0)]

    def test_keep_same_tier_still_skips_same_tier_ties(self):
        s = PairStats()
        assert arena_pairs(arena_row(3, 3, "tie"), PairFilter(keep_same_tier=True),
                           s) == []
        assert s.same_tier_skipped == 1

    def test_min_tier_gap_drops_adjacent_tiers(self):
        """Adjacent-tier human votes are near coin-flips, so gap 1 is noise."""
        s = PairStats()
        assert arena_pairs(arena_row(3, 2, "a"), PairFilter(min_tier_gap=2), s) == []
        assert s.filter_skipped == 1
        assert len(arena_pairs(arena_row(4, 2, "a"), PairFilter(min_tier_gap=2),
                               PairStats())) == 2

    def test_zero_gap_is_falsy_and_filters_nothing(self):
        assert len(arena_pairs(arena_row(3, 2, "a"), PairFilter(min_tier_gap=0),
                               PairStats())) == 2

    def test_unresolvable_tiers_are_no_signal(self):
        for bad in (None, "4", 4.0):
            s = PairStats()
            assert arena_pairs(arena_row(4, bad, "a"), PairFilter(), s) == []
            assert s.no_signal_skipped == 1

    def test_unknown_winner_is_no_signal(self):
        s = PairStats()
        assert arena_pairs(arena_row(4, 2, "tiebroken_by_coin"), PairFilter(), s) == []
        assert s.no_signal_skipped == 1

    def test_prompt_and_source_are_carried_through(self):
        pairs = arena_pairs(arena_row(4, 2, "a", prompt="hi", source="arena_33k"),
                            PairFilter(), PairStats())
        assert all(p.prompt == "hi" and p.source == "arena_33k" for p in pairs)


class TestSolvingTierRules:
    def test_gold_tier_wins_and_the_other_four_lose(self):
        pairs = solving_tier_pairs(solving_row(solving=(3,)), PairStats())
        assert sorted(p.tier for p in pairs) == [1, 2, 3, 4, 5]
        assert {p.tier: p.outcome for p in pairs}[3] == 1.0
        assert [p.outcome for p in pairs if p.tier != 3] == [0.0, 0.0, 0.0, 0.0]

    def test_every_solving_tier_wins(self):
        pairs = solving_tier_pairs(solving_row(solving=(3, 4, 5)), PairStats())
        assert {p.tier: p.outcome for p in pairs} == {1: 0.0, 2: 0.0, 3: 1.0,
                                                     4: 1.0, 5: 1.0}

    def test_exactly_one_gold_per_prompt_is_what_training_assumes(self):
        """`routing_accuracy` is defined on this; a row with two golds is the
        arena-tie case and must not reach the argmax metric."""
        for solving in ((1,), (2,), (3,), (4,), (5,)):
            golds = [p for p in solving_tier_pairs(solving_row(solving=solving),
                                                  PairStats())
                     if p.outcome == 1.0]
            assert len(golds) == 1

    def test_nothing_solved_is_no_signal(self):
        """A prompt no model solved carries no ordering information."""
        s = PairStats()
        assert solving_tier_pairs(solving_row(solving=()), s) == []
        assert s.no_signal_skipped == 1

    def test_empty_candidate_list_is_no_signal(self):
        s = PairStats()
        assert solving_tier_pairs(solving_row(candidates=(), solving=(3,)), s) == []
        assert s.no_signal_skipped == 1

    def test_missing_candidates_is_no_signal(self):
        s = PairStats()
        row = solving_row()
        del row["candidate_tiers"]
        assert solving_tier_pairs(row, s) == []
        assert s.no_signal_skipped == 1

    def test_old_dataset_files_fall_back_to_label_tier(self):
        """Datasets built before solving_tiers existed carry only label_tier."""
        row = {"prompt": "p", "source": "routerbench", "candidate_tiers": [1, 2, 3],
               "label_tier": 2}
        pairs = solving_tier_pairs(row, PairStats())
        assert {p.tier: p.outcome for p in pairs} == {1: 0.0, 2: 1.0, 3: 0.0}

    def test_null_label_tier_is_no_signal(self):
        s = PairStats()
        row = {"prompt": "p", "source": "routerbench", "candidate_tiers": [1, 2],
               "label_tier": None}
        assert solving_tier_pairs(row, s) == []
        assert s.no_signal_skipped == 1


class TestRowShapeDispatch:
    def test_candidate_tiers_list_selects_the_solving_shape(self):
        """Dispatch is on field presence, not on `source` -- complexity_router
        deliberately reuses the routerbench shape."""
        row = solving_row(solving=(4,))
        row["tier_a"] = 4
        row["tier_b"] = 2
        row["winner"] = "a"
        pairs = pairs_from_row(row, PairFilter(), PairStats())
        assert len(pairs) == 5  # the solving shape, not the 2-pair arena shape

    def test_arena_shape_is_used_when_candidate_tiers_absent(self):
        pairs = pairs_from_row(arena_row(4, 2, "a"), PairFilter(), PairStats())
        assert len(pairs) == 2

    def test_callers_cannot_disagree_about_the_shape(self):
        """Both streaming passes of the pairs build call this one function; the
        strata count and the emitted subset must agree row by row."""
        for row in (arena_row(4, 2, "a"), solving_row(solving=(3,))):
            a = pairs_from_row(row, PairFilter(), PairStats())
            b = pairs_from_row(row, PairFilter(), PairStats())
            assert [(p.tier, p.outcome) for p in a] == [(p.tier, p.outcome) for p in b]


class TestSerializationShape:
    def test_key_order_is_the_on_disk_contract(self):
        """The pairs file's key order is byte-compared by the golden manifest,
        so field order in Pair is part of the artifact, not an accident."""
        p = Pair(prompt="hi", tier=3, outcome=1.0, source="routerbench")
        assert list(p.as_json_dict()) == ["prompt", "tier", "outcome", "source"]


class TestStrata:
    def test_stratifies_on_source_and_tier(self):
        p = Pair(prompt="x", tier=4, outcome=1.0, source="arena_33k")
        assert strata_key(p) == ("arena_33k", 4)

    def test_source_is_part_of_the_key_so_a_cap_cannot_collapse_the_mix(self):
        counts = {("arena_33k", 4): 100, ("complexity_router", 4): 100}
        assert allocate(counts, 40) == {("arena_33k", 4): 20,
                                        ("complexity_router", 4): 20}


class TestAllocate:
    def test_cap_above_total_is_a_no_op(self):
        counts = {("a", 1): 5, ("a", 2): 7}
        assert allocate(counts, 999) == counts

    def test_allocation_sums_exactly_to_the_cap(self):
        counts = {("a", 1): 7, ("a", 2): 13, ("b", 3): 11}
        for cap in (1, 5, 17, 30, 31):
            assert sum(allocate(counts, cap).values()) == cap, cap

    def test_allocation_is_proportonal_within_rounding(self):
        counts = {("a", 1): 100, ("a", 2): 300}
        alloc = allocate(counts, 40)
        assert alloc == {("a", 1): 10, ("a", 2): 30}

    def test_no_stratum_is_over_allocated(self):
        """Allocation never exceeds what the stratum actually has."""
        counts = {("a", 1): 1000, ("a", 5): 1}
        for cap in (10, 50, 1001):
            alloc = allocate(counts, cap)
            assert all(alloc[k] <= counts[k] for k in counts), (cap, alloc)

    def test_a_stratum_that_is_millionths_of_the_file_rounds_to_zero(self):
        """Honest limit of proportional sampling: at cap 10 a single rare row
        loses its remainder race and is dropped. Widening --max-pairs is the
        fix, not a different allocator."""
        counts = {("a", 1): 1000, ("a", 5): 1}
        assert allocate(counts, 10)[("a", 5)] == 0
        assert allocate(counts, 1000)[("a", 5)] >= 1

    def test_cap_of_zero_emits_nothing(self):
        assert sum(allocate({("a", 1): 3}, 0).values()) == 0

    def test_is_deterministic(self):
        counts = {("a", 1): 7, ("b", 2): 13, ("c", 3): 5}
        assert allocate(counts, 12) == allocate(counts, 12)


class TestStatsCountersAreNotLost:
    @pytest.mark.parametrize("row,attr", [
        (arena_row(3, 3, "a"), "same_tier_skipped"),
        (arena_row(4, 2, "tie"), "ties_skipped"),
        (arena_row(None, 2, "a"), "no_signal_skipped"),
    ])
    def test_every_skip_reason_is_counted_for_the_report(self, row, attr):
        s = PairStats()
        flt = PairFilter(drop_ties=True)
        arena_pairs(row, flt, s)
        assert getattr(s, attr) == 1
