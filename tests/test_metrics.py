"""Eval metrics and the prompt-grouped split.

`pairwise_accuracy` is the number the project's standing protocol is measured
against (0.66-0.68 target band), and `routing_accuracy` is capped near the human
label-agreement rate. Both are only meaningful with the cohort filters below, so
these tests pin the filters as hard as the arithmetic.
"""

from __future__ import annotations

import hashlib

from router.contracts import TIERS, VAL_PROMPT_MOD
from router.metrics import (
    gold_tiers,
    group_by_prompt,
    majority_baseline,
    metric_cohort,
    pairwise_accuracy,
    routing_accuracy,
    train_split,
    val_split,
)


def scores_for(order: dict[str, list[int]]) -> dict[str, dict[int, float]]:
    """Build {prompt: {tier: logit}} from a preference order (best first)."""
    return {
        prompt: {tier: float(len(order[prompt]) - i) for i, tier in enumerate(tiers)}
        for prompt, tiers in order.items()
    }


class TestSplit:
    def test_split_partitions_without_overlap_or_loss(self):
        rows = [(f"prompt number {i}", 1 + i % 5, 1.0) for i in range(400)]
        val, train = val_split(rows), train_split(rows)
        assert len(val) + len(train) == len(rows)
        assert not set(val) & set(train)

    def test_split_is_by_prompt_not_by_row(self):
        """The whole point: a prompt's winning and losing tiers must land on the
        same side, or "held out" would be leaking the answer."""
        rows = [("shared prompt", t, 1.0 if t == 3 else 0.0) for t in TIERS]
        rows += [("other prompt", t, 1.0 if t == 2 else 0.0) for t in TIERS]
        val_prompts = {p for p, _t, _o in val_split(rows)}
        train_prompts = {p for p, _t, _o in train_split(rows)}
        assert not val_prompts & train_prompts

    def test_predicate_matches_the_notebook_formula(self):
        for prompt in ("hello", "a" * 300, "unicode: 你好 🚀"):
            expected = int(hashlib.sha1(prompt.encode()).hexdigest(), 16) % 20 == 0
            row = (prompt, 1, 1.0)
            assert (row in val_split([row])) is expected

    def test_split_is_roughly_five_percent(self):
        rows = [(f"p{i}", 1, 1.0) for i in range(4000)]
        rate = len(val_split(rows)) / len(rows)
        assert 0.03 < rate < 0.07

    def test_modulus_controls_the_split_size(self):
        rows = [(f"p{i}", 1, 1.0) for i in range(2000)]
        assert len(val_split(rows, mod=10)) > len(val_split(rows, mod=40))

    def test_default_modulus_is_the_contract_value(self):
        rows = [(f"p{i}", 1, 1.0) for i in range(200)]
        assert len(val_split(rows)) == len(val_split(rows, VAL_PROMPT_MOD))


class TestGrouping:
    def test_groups_rows_by_prompt(self):
        grouped = group_by_prompt([("a", 1, 0.0), ("a", 3, 1.0), ("b", 2, 1.0)])
        assert grouped == {"a": {1: 0.0, 3: 1.0}, "b": {2: 1.0}}

    def test_five_tier_group_is_a_complete_prompt_cohort(self):
        grouped = group_by_prompt([("x", t, 1.0 if t == 4 else 0.0) for t in TIERS])
        assert sorted(grouped["x"]) == [1, 2, 3, 4, 5]


class TestCohort:
    def test_decisive_requires_exactly_one_gold(self):
        grouped = {
            "one-gold": {1: 0.0, 3: 1.0},
            "two-gold": {1: 1.0, 3: 1.0},
            "no-gold": {1: 0.0, 3: 0.0},
            "half": {1: 0.5, 3: 0.5},
        }
        decisive, pairs = metric_cohort(grouped)
        assert decisive == ["one-gold"]
        assert pairs == 1

    def test_pair_count_is_gold_times_losers(self):
        grouped = {"x": {t: (1.0 if t == 3 else 0.0) for t in TIERS}}
        assert metric_cohort(grouped) == (["x"], 4)

    def test_gold_tiers_ignores_ambiguous_prompts(self):
        grouped = {"a": {1: 1.0, 2: 1.0}, "b": {4: 1.0, 5: 0.0}}
        assert gold_tiers(grouped) == [4]


class TestPairwiseAccuracy:
    def test_perfect_ordering_scores_one(self):
        grouped = {"x": {1: 0.0, 4: 1.0}}
        s = scores_for({"x": [4, 1]})
        assert pairwise_accuracy(grouped, s) == 1.0

    def test_reversed_ordering_scores_zero(self):
        grouped = {"x": {1: 0.0, 4: 1.0}}
        s = scores_for({"x": [1, 4]})
        assert pairwise_accuracy(grouped, s) == 0.0

    def test_every_gold_loser_pair_is_counted(self):
        grouped = {"x": {t: (1.0 if t == 3 else 0.0) for t in TIERS}}
        s = scores_for({"x": [3, 4, 2, 5, 1]})
        # gold 3 beats 4, 2 and 5 but loses to... 3 is first, so it beats all four
        assert pairwise_accuracy(grouped, s) == 1.0

    def test_ties_at_zero_5_contribute_no_pairs(self):
        """A 0.5/0.5 row has no winner to be above and no loser to be below."""
        grouped = {"x": {1: 0.5, 3: 0.5}}
        s = scores_for({"x": [3, 1]})
        assert pairwise_accuracy(grouped, s) == 0.0

    def test_empty_cohort_does_not_divide_by_zero(self):
        assert pairwise_accuracy({}, {}) == 0.0


class TestRoutingAccuracy:
    def test_argmax_is_the_decision(self):
        grouped = {"x": {t: (1.0 if t == 2 else 0.0) for t in TIERS}}
        s = scores_for({"x": [2, 1, 3, 4, 5]})
        assert routing_accuracy(grouped, s) == 1.0

    def test_wrong_argmax_scores_zero_even_when_ordering_is_mostly_right(self):
        grouped = {"x": {t: (1.0 if t == 2 else 0.0) for t in TIERS}}
        s = scores_for({"x": [3, 2, 1, 4, 5]})
        assert routing_accuracy(grouped, s) == 0.0

    def test_ambiguous_prompts_are_excluded_from_the_denominator(self):
        grouped = {"skipped": {1: 1.0, 2: 1.0}, "counted": {4: 1.0, 5: 0.0}}
        s = {"skipped": {1: 9.0, 2: 1.0}, "counted": {4: 9.0, 5: 1.0}}
        assert routing_accuracy(grouped, s) == 1.0

    def test_is_at_most_one(self):
        grouped = group_by_prompt([("a", 1, 1.0), ("a", 2, 0.0), ("b", 3, 1.0)])
        s = {"a": {1: 5.0, 2: 1.0}, "b": {3: 2.0}}
        assert 0.0 <= routing_accuracy(grouped, s) <= 1.0


class TestMajorityBaseline:
    def test_is_the_share_of_the_most_common_gold(self):
        grouped = {f"p{i}": {4: 1.0, 1: 0.0} for i in range(6)}
        grouped.update({f"q{i}": {2: 1.0, 1: 0.0} for i in range(4)})
        baseline = majority_baseline(grouped)
        assert baseline is not None
        rate, top = baseline
        assert top == 4
        assert rate == 0.6

    def test_none_when_there_is_nothing_to_be_majority_of(self):
        assert majority_baseline({}) is None
        assert majority_baseline({"a": {1: 0.0, 2: 0.0}}) is None

    def test_beating_the_baseline_is_the_real_bar(self):
        """A router that ignores the prompt lands here, and the notebook prints
        this number next to routing accuracy for exactly that comparison."""
        grouped = {"a": {4: 1.0, 1: 0.0}, "b": {4: 1.0, 2: 0.0}}
        baseline = majority_baseline(grouped)
        assert baseline is not None
        baseline_rate, top = baseline
        s = {"a": {top: 9.0, 1: 1.0}, "b": {top: 9.0, 2: 1.0}}
        assert routing_accuracy(grouped, s) == baseline_rate


class TestRealPairsShape:
    def test_model_free_rows_form_complete_decisive_cohorts(self, dataset_sample):
        """complexity_router emits a complete 5-tier group with exactly one gold
        per prompt, so every such prompt is decisive and contributes 4 eval
        pairs -- and routing accuracy is then defined on all of them."""
        import json

        from router.pairs import PairFilter, PairStats, pairs_from_row

        rows = [
            (p.prompt, p.tier, p.outcome)
            for line in dataset_sample.read_text().splitlines() if line.strip()
            for row in [json.loads(line)] if row["source"] == "complexity_router"
            for p in pairs_from_row(row, PairFilter(), PairStats())
        ]
        grouped = group_by_prompt(rows)
        assert grouped, "fixture has no complexity_router rows"
        assert all(sorted(t) == [1, 2, 3, 4, 5] for t in grouped.values())
        decisive, pairs = metric_cohort(grouped)
        assert sorted(decisive) == sorted(grouped)
        assert pairs == 4 * len(grouped)
        assert len(gold_tiers(grouped)) == len(grouped)
        assert routing_accuracy(grouped, {
            p: {t: (9.0 if o == 1.0 else 1.0) for t, o in tiers.items()}
            for p, tiers in grouped.items()
        }) == 1.0

    def test_the_two_cohort_helpers_agree_on_real_data(self, dataset_sample):
        """metric_cohort and gold_tiers filter the same way independently; if one
        is edited the two must not silently disagree about the denominator."""
        import json

        from router.pairs import PairFilter, PairStats, pairs_from_row

        rows = [
            (p.prompt, p.tier, p.outcome)
            for line in dataset_sample.read_text().splitlines() if line.strip()
            for p in pairs_from_row(json.loads(line), PairFilter(), PairStats())
        ]
        grouped = group_by_prompt(rows)
        decisive, _pairs = metric_cohort(grouped)
        assert len(decisive) == len(gold_tiers(grouped))
