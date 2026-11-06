"""The shared source catalog.

Its whole purpose is that the coverage audit and the dataset build cannot drift
apart, so these tests are mostly about the *relationship* between the two
consumers rather than the data itself.
"""

from __future__ import annotations

import pytest

from router.sources import (
    AUDITABLE,
    FAMILIES,
    FAMILY_ARENA,
    FAMILY_COMPLEXITY_ROUTER,
    FAMILY_ROUTERBENCH,
    LABEL_KIND_ANNOTATED_LEVEL,
    LABEL_KIND_MIN_SOLVING_TIER,
    LABEL_KIND_WINNER_TIER,
    NAME_MODE_COLUMNS,
    NAME_MODE_NONE,
    NAME_MODE_PKL_MODEL_COLUMNS,
    ROUTERBENCH_META_COLS,
    SOURCES,
    by_key,
    for_families,
    parse_sources,
    routerbench_model_columns,
)


class TestCatalogShape:
    def test_keys_are_unique(self):
        keys = [s.key for s in SOURCES]
        assert len(keys) == len(set(keys))

    def test_every_source_has_a_known_family_and_name_mode(self):
        for s in SOURCES:
            assert s.family in FAMILIES, s.key
            assert s.name_mode in (
                NAME_MODE_COLUMNS, NAME_MODE_PKL_MODEL_COLUMNS, NAME_MODE_NONE,
            ), s.key

    def test_catalog_order_is_the_build_order(self):
        """Arena generations first, then the model-free sources: the shared
        dedup set means whichever source sees a prompt first owns its label."""
        assert [s.key for s in SOURCES] == [
            "arena_33k", "arena_55k", "arena_100k", "arena_140k",
            "routerbench", "complexity_router",
        ]

    def test_four_arena_generations_are_one_cli_family(self):
        arena = [s for s in SOURCES if s.family == FAMILY_ARENA]
        assert len(arena) == 4
        assert all(s.name_mode == NAME_MODE_COLUMNS for s in arena)
        assert all(s.model_columns == ("model_a", "model_b") for s in arena)

    def test_label_kinds_match_the_families(self):
        assert by_key("arena_100k").label_kind == LABEL_KIND_WINNER_TIER
        assert by_key("routerbench").label_kind == LABEL_KIND_MIN_SOLVING_TIER
        assert by_key("complexity_router").label_kind == LABEL_KIND_ANNOTATED_LEVEL

    def test_unknown_key_raises_with_the_valid_list(self):
        with pytest.raises(KeyError) as exc:
            by_key("arena_999k")
        assert "arena_33k" in str(exc.value)


class TestCoverageGateEligibility:
    """The bug this catalog exists to prevent.

    RouterBench resolves ~401k model names through the registry, so it must be
    audited *and* gated. It previously appeared in the audit's list but was
    treated as model-free by the build, leaving its names ungateable.
    """

    def test_routerbench_carries_model_names(self):
        assert by_key("routerbench").carries_model_names is True

    def test_routerbench_is_in_the_audited_set(self):
        assert by_key("routerbench") in AUDITABLE

    def test_complexity_router_is_the_only_model_free_source(self):
        free = [s.key for s in SOURCES if not s.carries_model_names]
        assert free == ["complexity_router"]

    def test_audited_set_is_exactly_the_name_carrying_sources(self):
        assert [s.key for s in AUDITABLE] == [
            s.key for s in SOURCES if s.name_mode != NAME_MODE_NONE
        ]

    def test_a_model_free_source_has_no_columns_to_audit(self):
        assert by_key("complexity_router").model_columns == ()
        assert by_key("routerbench").model_columns == ()  # names come from headers

    def test_name_mode_decides_both_consumers_without_a_hardcoded_list(self):
        for s in SOURCES:
            assert s.carries_model_names == (s.name_mode != NAME_MODE_NONE)


class TestFamilySelection:
    def test_arena_expands_to_four_sources(self):
        assert [s.key for s in for_families(["arena"])] == [
            "arena_33k", "arena_55k", "arena_100k", "arena_140k",
        ]

    def test_selection_is_returned_in_catalog_order_not_request_order(self):
        assert [s.key for s in for_families(["complexity_router", "arena"])] == [
            "arena_33k", "arena_55k", "arena_100k", "arena_140k",
            "complexity_router",
        ]

    def test_all_three_families_cover_the_whole_catalog(self):
        assert len(for_families(list(FAMILIES))) == len(SOURCES)

    def test_unknown_family_is_rejected_before_any_download(self):
        with pytest.raises(ValueError) as exc:
            for_families(["arena", "imaginary"])
        assert "imaginary" in str(exc.value)


class TestParseSources:
    def test_default_is_arena_only(self):
        assert parse_sources(None) == [FAMILY_ARENA]
        assert parse_sources("") == [FAMILY_ARENA]

    def test_splits_trims_and_validates(self):
        assert parse_sources(" arena , routerbench ") == [FAMILY_ARENA, "routerbench"]

    def test_deduplicates_preserving_first_position(self):
        assert parse_sources("routerbench,arena,routerbench") == [
            "routerbench", FAMILY_ARENA,
        ]

    def test_bad_name_lists_the_valid_ones(self):
        with pytest.raises(ValueError) as exc:
            parse_sources("arenia")
        assert "arena" in str(exc.value)

    def test_default_argument_can_be_overridden(self):
        assert parse_sources(None, default=FAMILY_COMPLEXITY_ROUTER) == [
            FAMILY_COMPLEXITY_ROUTER,
        ]


class TestRouterbenchColumnLayout:
    def test_bare_columns_are_model_names(self):
        assert routerbench_model_columns(["gpt-4o", "gemini-1.5-pro"]) == [
            "gpt-4o", "gemini-1.5-pro",
        ]

    def test_suffixed_detail_columns_are_not(self):
        cols = ["gpt-4o", "gpt-4o|model_response", "gpt-4o|total_cost"]
        assert routerbench_model_columns(cols) == ["gpt-4o"]

    def test_metadata_columns_are_excluded(self):
        cols = ["sample_id", "prompt", "eval_name", "oracle_model_to_route_to", "gpt-4o"]
        assert routerbench_model_columns(cols) == ["gpt-4o"]
        assert {"prompt", "sample_id", "eval_name",
                "oracle_model_to_route_to"} == set(ROUTERBENCH_META_COLS)

    def test_non_string_headers_are_tolerated(self):
        assert routerbench_model_columns([1, "gpt-4o"]) == [1, "gpt-4o"]

    def test_defined_once_and_shared_by_both_consumers(self):
        """The auditor and the builder must agree on which pkl columns are models,
        or they count different denominators and disagree about coverage. These
        two files each carried a verbatim copy before `router.sources` existed."""
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        for path in (root / "scripts/audit_model_coverage.py",
                     root / "scripts/build_training_dataset.py",
                     root / "router/dataset.py"):
            text = path.read_text()
            assert re.search(r"^\s*ROUTERBENCH_META_COLS\s*=", text, re.M) is None, path
            assert re.search(r"^def routerbench_model_columns", text, re.M) is None, path
        owners = [p.name for p in (root / "router").glob("*.py")
                  if re.search(r"^def routerbench_model_columns", p.read_text(), re.M)]
        assert owners == ["sources.py"]
