"""The coverage gate and the arena schema normalization it depends on.

The gate is the reason a registry gap cannot quietly become a training-set bias:
a source that names models must resolve nearly all of them or the build refuses
to write a file. These tests pin both the arithmetic and the exemption rule.
"""

from __future__ import annotations

import pytest

from router.contracts import COVERAGE_TARGET
from router.dataset import (
    LEVEL_TO_TIER,
    SourceStats,
    coverage_failures,
    dedup_key,
    first_user_text,
    norm_winner,
)


class TestCoverageArithmetic:
    def test_no_names_resolved_means_no_coverage_claim(self):
        assert SourceStats().coverage is None

    def test_all_slots_resolved_is_100(self):
        s = SourceStats(name_slots=100)
        assert s.coverage == 100.0

    def test_unmapped_names_lower_the_rate_weighted_by_rows(self):
        s = SourceStats(name_slots=100)
        s.unmapped["missing-model"] = 5
        assert s.coverage == pytest.approx(95.0)

    def test_a_name_appearing_in_many_rows_matters_more(self):
        s = SourceStats(name_slots=1000)
        s.unmapped["rare-gap"] = 1
        s.unmapped["common-gap"] = 200
        assert s.coverage == pytest.approx(79.9)

    def test_slots_zero_is_not_a_division_error(self):
        s = SourceStats(name_slots=0)
        s.unmapped["x"] = 3  # cannot happen in practice, must not crash
        assert s.coverage is None


class TestGateDecisions:
    def test_a_model_free_source_never_fails_the_gate(self):
        """complexity_router resolves nothing, so it has no rate to gate."""
        s = SourceStats(scanned=100, emitted=100)
        assert not coverage_failures({"complexity_router": s})

    def test_below_target_fails(self):
        s = SourceStats(name_slots=100)
        s.unmapped["gap"] = 2  # 98%
        failed = coverage_failures({"arena_33k": s})
        assert [f.key for f in failed] == ["arena_33k"]
        assert failed[0].rate == pytest.approx(98.0)

    def test_exactly_at_target_passes(self):
        s = SourceStats(name_slots=100)
        s.unmapped["gap"] = int(100 * (100 - COVERAGE_TARGET) / 100)  # 1 -> 99%
        assert not coverage_failures({"arena_33k": s})

    def test_the_gate_carries_the_names_for_the_fix_itself(self):
        failed = coverage_failures({
            "routerbench": _stats(100, {"never-seen-model": 30})
        })
        assert failed[0].unmapped.most_common(1)[0] == ("never-seen-model", 30)

    def test_every_failing_source_is_reported_not_just_the_first(self):
        stats = {"arena_33k": _stats(100, {"gap": 20}),
                 "arena_140k": _stats(100, {"gap": 30}),
                 "arena_100k": _stats(100, {})}
        assert {f.key for f in coverage_failures(stats)} == {"arena_33k", "arena_140k"}

    def test_target_can_be_loosened_only_explicitly(self):
        stats = {"arena_33k": _stats(100, {"gap": 5})}
        assert coverage_failures(stats, target=90.0) == []
        assert len(coverage_failures(stats, target=99.0)) == 1


def _stats(slots: int, unmapped: dict[str, int]) -> SourceStats:
    s = SourceStats(name_slots=slots)
    for name, n in unmapped.items():
        s.unmapped[name] = n
    return s


class TestGateIsStructuralNotANameList:
    """The old gate kept a hardcoded list of exempt source keys and got one
    wrong (routerbench). Exemption now follows from whether names were resolved."""

    def test_slots_are_what_switch_the_gate_on(self):
        named = _stats(50, {})
        free = SourceStats(scanned=50, emitted=50)
        assert named.carries_model_names and not free.carries_model_names
        assert coverage_failures({"a": named}) == []
        assert coverage_failures({"b": free}) == []

    def test_a_new_model_name_source_is_gated_without_edits_anywhere(self):
        """A hypothetical 5th source that resolved names is failed by the same
        code path, with no family list to remember to update."""
        assert coverage_failures({"brand_new_source": _stats(100, {"gap": 40})})


class TestArenaSchemaNormalization:
    """Four arena generations, four different conversation encodings."""

    def test_plain_message_list(self):
        conv = [{"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "hi"}]
        assert first_user_text(conv) == "hello there"

    def test_json_string_wrapping_the_list(self):
        conv = '[{"role": "user", "content": "from json"}]'
        assert first_user_text(conv) == "from json"

    def test_python_repr_wrapping_the_list(self):
        conv = "[{'role': 'user', 'content': 'from repr'}]"
        assert first_user_text(conv) == "from repr"

    def test_multimodal_content_parts_are_joined(self):
        conv = [{"role": "user", "content": [
            {"type": "text", "text": "line one"},
            {"type": "image_url", "url": "…"},
            {"type": "text", "text": "line two"},
        ]}]
        assert first_user_text(conv) == "line one\nline two"

    def test_flat_turn_list_generation(self):
        """The 55k release stores a JSON list of bare turn texts."""
        assert first_user_text('["a turn", "b turn"]') == "a turn\nb turn"

    def test_non_user_roles_are_accepted_when_nothing_better_exists(self):
        for role in ("human", "prompter", "none", ""):
            assert first_user_text([{"role": role, "content": "asked"}]) == "asked"

    def test_falls_back_to_first_non_empty_content(self):
        conv = [{"role": "assistant", "content": ""},
                {"role": "assistant", "content": "only content"}]
        assert first_user_text(conv) == "only content"

    def test_nonConversation_shapes_give_empty_prompt(self):
        assert first_user_text(42) == ""
        assert first_user_text(None) == ""
        assert first_user_text([]) == ""

    def test_unparseable_string_is_returned_stripped(self):
        assert first_user_text("  just a prompt  ") == "just a prompt"


class TestWinnerNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("model_a", "a"), ("MODEL_B", "b"), (" model_a ", "a"),
        ("tie", "tie"), ("TIE", "tie"),
        ("both_bad", "both_bad"), ("bothbad", "both_bad"),
        ("something_else", "unknown"), (None, "unknown"),
    ])
    def test_maps_arena_verdicts(self, raw, expected):
        assert norm_winner(raw) == expected


class TestDedup:
    def test_same_prompt_and_pair_is_one_key(self):
        assert dedup_key("hello world", ["a-model", "b-model"]) == \
               dedup_key("hello  world", ["b-model", "a-model"])

    def test_whitespace_is_normalized(self):
        assert dedup_key("a b", ["m"]) == dedup_key("  a   b ", ["m"])

    def test_model_name_case_is_ignored(self):
        assert dedup_key("p", ["GPT-4o"]) == dedup_key("p", ["gpt-4O"])

    def test_different_prompt_or_pair_diverge(self):
        assert dedup_key("p", ["m"]) != dedup_key("q", ["m"])
        assert dedup_key("p", ["m"]) != dedup_key("p", ["n"])

    def test_model_free_sources_use_their_own_namespace(self):
        """routerbench and complexity_router dedup on a synthetic key, so a
        prompt seen in both is only labeled once -- first source wins."""
        assert dedup_key("p", ["routerbench"]) != dedup_key("p", ["complexity_router"])


class TestComplexityLevelMapping:
    def test_levels_map_onto_tiers_one_to_four(self):
        assert LEVEL_TO_TIER == {0: 1, 1: 2, 2: 3, 3: 4}

    def test_tier_five_is_not_reachable_from_annotations(self):
        """Keeps frontier-only signal coming from arena and routerbench."""
        assert 5 not in LEVEL_TO_TIER.values()

    def test_every_mapped_tier_is_a_real_tier(self):
        from router.contracts import TIERS
        assert set(LEVEL_TO_TIER.values()) <= set(TIERS)
