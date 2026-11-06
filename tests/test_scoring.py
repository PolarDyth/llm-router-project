"""Tier-scoring invariants.

These are the rules AGENTS.md labels "don't break silently". Each one is a
deliberate response to a signal that lies in a specific way, so the tests pin
the *reason* as much as the number: if you change an era boundary or let ELO
downgrade a model, a test should fail and make you say why.
"""

from __future__ import annotations

import datetime as dt

import pytest

from router.catalogs import Canonical
from router.contracts import COMPLEXITY_SCALE, TIERS
from router.scoring import (
    ELO_FRONTIER,
    II_CUTS,
    PARAMS_CUTS_LEGACY,
    PARAMS_CUTS_MODERN,
    alias_collisions,
    apply_era_caps,
    apply_overrides,
    best_price,
    cut_tier,
    era_of,
    interp_price,
    quality_adjust,
    quality_tier,
    score_all,
    score_group,
)


def group(key: str, *, prices=(), released=None, ii=None, elo=None,
          params_b=None, raws=(), aliases=()) -> Canonical:
    """A canonical group built by hand, so scoring is tested without catalogs."""
    g = Canonical(key)
    g.raws = list(raws)
    g.aliases = set(aliases) | {key}
    if prices:
        g.fields["prices"] = list(prices)
    if released is not None:
        g.fields["released"] = released
    if ii is not None:
        g.fields["ii"] = ii
    if elo is not None:
        g.fields["elo"] = elo
    if params_b is not None:
        g.fields["params_b"] = params_b
    return g


def scored(g: Canonical) -> dict:
    """`score_group` returns None for an unscoreable group. Every group built in
    these tests is scoreable, so unwrap once with a failure naming the model
    rather than 15 `assert entry is not None` lines."""
    entry = score_group(g)
    assert entry is not None, g.key
    return entry


class TestPricePrior:
    def test_no_price_information_is_not_zero_price(self):
        assert interp_price(None, None) is None

    def test_free_is_tiny(self):
        assert interp_price(0.0, 0.0) == 1.0

    def test_below_lowest_edge_is_clamped_to_floor(self):
        assert interp_price(0.01, 0.01) == 1.0

    def test_above_top_edge_is_clamped_to_ceiling(self):
        assert interp_price(30.0, 30.0) == 5.0

    def test_monotonic_in_blended_price(self):
        scores: list[float] = []
        for p in (0.05, 0.2, 1.0, 5.0, 15.0, 40.0):
            v = interp_price(p, p)
            assert v is not None, p
            scores.append(v)
        assert scores == sorted(scores)
        assert scores[0] < scores[-1]

    def test_input_price_dominates_output_price(self):
        # blended = 0.75*in + 0.25*out, so a doubling of input price moves the
        # prior further than doubling output price
        in_heavy, out_heavy = interp_price(2.0, 1.0), interp_price(1.0, 2.0)
        assert in_heavy is not None and out_heavy is not None
        assert in_heavy > out_heavy

    def test_best_price_takes_the_priciest_hoster(self):
        prices = [("litellm", 0.5, 1.0), ("openrouter", 30.0, 60.0)]
        assert best_price(prices) == (30.0, 60.0)

    def test_best_price_ignores_free_mirrors(self):
        prices = [("openrouter", 0.0, 0.0), ("litellm", 0.5, 1.0)]
        assert best_price(prices) == (0.5, 1.0)

    def test_best_price_of_nothing_gives_no_price(self):
        assert best_price([]) == (None, None)
        assert best_price([("litellm", None, None)]) == (None, None)


class TestQualitySignals:
    def test_intelligence_index_cuts(self):
        assert cut_tier(50.0, II_CUTS) == 5
        assert cut_tier(45.0, II_CUTS) == 5
        assert cut_tier(44.9, II_CUTS) == 4
        assert cut_tier(0.0, II_CUTS) == 1
        assert cut_tier(None, II_CUTS) is None

    def test_elo_lifts_only_when_the_index_is_absent(self):
        """The invariant: ELO promotes to frontier, it never demotes."""
        assert quality_tier(None, 1500.0) == (None, 5)
        # a low index is not overruled by a high ELO
        assert quality_tier(10.0, 1500.0) == (2, 2)

    def test_elo_below_frontier_does_nothing(self):
        assert quality_tier(None, ELO_FRONTIER - 1) == (None, None)
        assert quality_tier(None, None) == (None, None)

    def test_zero_elo_is_not_a_signal(self):
        assert quality_tier(None, 0.0) == (None, None)


class TestQualityAdjust:
    def test_big_disagreement_lifts(self):
        assert quality_adjust(2, 5, 5, dt.date(2023, 1, 1)) == 1

    def test_one_step_lifts_at_the_extremes_only(self):
        assert quality_adjust(4, 5, 5, None) == 1  # toward frontier
        assert quality_adjust(2, 3, 3, None) == 1  # from the cheap end
        assert quality_adjust(3, 4, 4, None) == 0  # middle: not enough evidence

    def test_downgrade_requires_a_modern_release(self):
        """Era-biased index: gpt-4o scores 8.4 on the 2026 index, which says
        nothing about how good it was in 2024."""
        modern = dt.date(2026, 1, 1)
        legacy = dt.date(2023, 6, 1)
        assert quality_adjust(5, 3, 3, modern) == -1
        assert quality_adjust(5, 3, 3, legacy) == 0
        assert quality_adjust(5, 3, 3, None) == 0

    def test_downgrade_requires_the_index_not_just_elo(self):
        assert quality_adjust(5, 3, None, dt.date(2026, 1, 1)) == 0

    def test_no_quality_signal_no_adjustment(self):
        assert quality_adjust(3, None, None, dt.date(2026, 1, 1)) == 0


class TestEraCaps:
    def test_pre_gpt4_models_cannot_be_frontier(self):
        assert apply_era_caps(5, dt.date(2023, 3, 14)) == 3

    def test_pre_o1_models_cannot_be_tier5(self):
        assert apply_era_caps(5, dt.date(2024, 8, 31)) == 4
        assert apply_era_caps(4, dt.date(2024, 8, 31)) == 4

    def test_modern_models_are_uncapped(self):
        assert apply_era_caps(5, dt.date(2025, 6, 1)) == 5

    def test_unknown_era_is_uncapped_but_flagged_by_score_group(self):
        assert apply_era_caps(5, None) == 5

    def test_caps_never_raise_a_tier(self):
        assert apply_era_caps(2, dt.date(2023, 3, 14)) == 2

    def test_era_of_tolerates_bad_dates(self):
        assert era_of(None) is None
        assert era_of("not-a-date") is None
        assert era_of("2024-08-06T00:00:00Z") == dt.date(2024, 8, 6)


class TestBaseTierSelection:
    def test_legacy_open_model_scores_by_params_not_hosting_price(self):
        """A 2023 Llama-2-70B listing at commodity prices is not a mid-tier
        model; params say otherwise and params win for open weights."""
        g = group(
            "meta/llama-2-70b",
            prices=[("litellm", 0.9, 0.9)],
            released=("models.dev", "2023-07-18"),
            params_b=70.0,
        )
        entry = scored(g)
        assert entry is not None
        assert entry["basis"].startswith("params")
        assert "legacy open" in entry["basis"]
        assert entry["complexity"] == 3

    def test_pre_frontier_models_get_the_era_floor(self):
        g = group(
            "openai/text-babbage-001",
            prices=[("litellm", 0.1, 0.2)],
            released=("litellm", "2022-01-11"),
        )
        entry = scored(g)
        assert entry["basis"] == "era(pre-2023-03)"
        assert entry["complexity"] == 2

    def test_gpt4_is_exempt_from_the_era_floor_but_not_the_era_cap(self):
        """ERA_PRE_FRONTIER is gpt-4's own release date: the exemption stops the
        floor from applying to it, but it still cannot outrank its own era."""
        g = group(
            "openai/gpt-4-0314",
            prices=[("litellm", 20.0, 60.0)],
            released=("litellm", "2023-03-14"),
        )
        entry = scored(g)
        assert entry["basis"].startswith("price")
        assert entry["complexity"] == 3  # priced as 5, capped to its era

    def test_unpriced_modern_model_falls_back_to_params(self):
        """Post-o1, so the legacy-open branch does not apply and params are the
        plain fallback for having no price at all."""
        g = group("deepseek/deepseek-v2", released=("openrouter", "2025-06-06"),
                  params_b=236.0)
        entry = scored(g)
        assert entry["basis"] == "params(236B)"

    def test_legacy_open_branch_is_labelled_as_such(self):
        g = group("deepseek/deepseek-v2", released=("openrouter", "2024-05-06"),
                  params_b=236.0)
        assert scored(g)["basis"] == "params(236B, legacy open)"

    def test_keyword_fallbacks(self):
        tiny = scored(group("alibaba/qwen-7b-chat",
                                 released=("openrouter", "2024-06-01")))
        assert tiny["basis"] == "keyword(tiny)"
        assert tiny["complexity"] == 1
        huge = scored(group("xai/grok-max", released=("openrouter", "2025-06-01")))
        assert huge["basis"] == "keyword(huge)"

    def test_nothing_known_is_flagged_for_review(self):
        entry = scored(group("unknownorg/mystery-model"))
        assert entry["basis"] == "guess"
        assert "unpriced, no params found - review" in entry["notes"]

    def test_tier5_without_a_release_date_is_flagged(self):
        entry = scored(group("somevendor/ultra-max", ii=50.0))
        assert entry["complexity"] == 5
        assert "tier 5 with no release date - review" in entry["notes"]

    def test_price_quality_disagreement_is_flagged(self):
        entry = scored(
            group("openai/cheap-but-brilliant",
                  prices=[("litellm", 0.05, 0.05)],
                  released=("openrouter", "2025-06-01"),
                  ii=50.0)
        )
        assert "price/quality disagree >=2 tiers" in entry["notes"]

    def test_every_tier_stays_in_1_to_5(self):
        for params in (
            dict(prices=[("litellm", 500.0, 900.0)], released=("x", "2026-01-01"), ii=99.0),
            dict(prices=[("litellm", 0.0, 0.0)]),
            dict(released=("x", "2020-01-01")),
        ):
            entry = scored(group("vendor/model-name", **params))
            assert entry["complexity"] in TIERS
            assert 1.0 <= entry["score"] <= 5.0


class TestParamsCuts:
    def test_legacy_scale_is_lower_than_modern(self):
        """405B was 2024's frontier; the same count means less in the o1 era."""
        legacy, modern = (cut_tier(405.0, PARAMS_CUTS_LEGACY),
                          cut_tier(405.0, PARAMS_CUTS_MODERN))
        assert legacy is not None and modern is not None
        assert legacy < modern
        assert cut_tier(70.0, PARAMS_CUTS_LEGACY) == 3
        assert cut_tier(70.0, PARAMS_CUTS_MODERN) == 4


class TestOverrides:
    def test_override_beats_scored_tier(self):
        built = [scored(group("openai/model-x",
                                   prices=[("litellm", 1.0, 1.0)],
                                   released=("x", "2025-01-01")))]
        before = built[0]["complexity"]
        out = apply_overrides(built, {"models": {"openai/model-x": {"complexity": 5}}})
        entry = next(e for e in out if e["canonical_id"] == "openai/model-x")
        assert entry["complexity"] == 5
        assert entry["complexity"] != before
        assert entry["basis"] == "override"

    def test_override_aliases_are_merged_not_replaced(self):
        entry = scored(group("openai/model-x", aliases={"openai/model-x", "old-name"},
                                  prices=[("litellm", 1.0, 1.0)]))
        out = apply_overrides([entry], {
            "models": {"openai/model-x": {"aliases": ["brand-new-name"]}}
        })
        assert {"old-name", "brand-new-name"} <= set(out[0]["aliases"])

    def test_null_override_signals_do_not_erase_real_ones(self):
        entry = scored(group("openai/model-x", prices=[("litellm", 2.0, 4.0)],
                                  released=("x", "2025-01-01")))
        assert entry["signals"]["price_in_per_m"] == 2.0
        out = apply_overrides([entry], {
            "models": {"openai/model-x": {"signals": {"price_in_per_m": None,
                                                     "elo": 1400}}}
        })
        assert out[0]["signals"]["price_in_per_m"] == 2.0  # not clobbered
        assert out[0]["signals"]["elo"] == 1400

    def test_absent_model_is_created_only_when_a_tier_is_given(self):
        out = apply_overrides([], {"models": {
            "anthropic/claude-0": {"complexity": 2, "display_name": "Claude 0"},
            "vendor/aliases-only": {"aliases": ["whatever"]},
        }})
        keys = {e["canonical_id"] for e in out}
        assert "anthropic/claude-0" in keys
        assert "vendor/aliases-only" not in keys

    def test_excluded_group_is_dropped(self):
        groups = {
            "junk/merge-group": group("junk/merge-group"),
            "openai/real-model": group("openai/real-model",
                                       prices=[("litellm", 1.0, 1.0)]),
        }
        built = score_all(groups, excludes={"junk/merge-group"})
        assert [e["canonical_id"] for e in built] == ["openai/real-model"]

    def test_created_override_entry_gets_full_signal_shape(self):
        out = apply_overrides([], {"models": {
            "a/b": {"complexity": 3, "intelligence_index": 20.0}}})
        assert out[0]["signals"]["intelligence_index"] == 20.0
        assert set(out[0]["signals"]) == {
            "price_in_per_m", "price_out_per_m", "elo",
            "intelligence_index", "params_b"}


class TestAliasCollisions:
    def test_shared_alias_between_two_models_is_reported(self):
        built = [
            {"canonical_id": "a/one", "aliases": ["dupe"]},
            {"canonical_id": "a/two", "aliases": ["dupe"]},
        ]
        collisions = alias_collisions(built)
        assert len(collisions) == 1
        assert "dupe" in collisions[0]

    def test_a_name_claiming_itself_is_not_a_collision(self):
        built = [{"canonical_id": "a/one", "aliases": ["a/one", "one"]}]
        assert alias_collisions(built) == []


class TestRegistryScaleContract:
    def test_scale_covers_every_tier(self):
        assert set(COMPLEXITY_SCALE) == {str(t) for t in TIERS}

    def test_scale_descriptions_match_the_registry_artifact(self, committed_registry):
        """models.json carries the scale to humans; contracts.py is what scripts
        read. If these diverge, the doc in the artifact is lying."""
        assert committed_registry["complexity_scale"] == COMPLEXITY_SCALE
