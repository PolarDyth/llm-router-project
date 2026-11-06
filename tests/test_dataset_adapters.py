"""Dataset-stage adapters, driven with fake catalogs instead of the Hub.

This stage is the one producer whose output the notebook trains on, and its
rules are not arithmetic you can eyeball: which prompts get a null label, how
many model-name slots a source claims, and what `id` a row carries. The 99%
coverage gate reads `name_slots`, so a wrong denominator here is a wrong gate
downstream -- and it would stay green, because the gate's own tests only feed it
hand-built stats.

Everything here monkeypatches `load_dataset` / `hf_hub_download` / `read_pickle`,
so the rules are testable without network, gated licenses, or a 365 MB file.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
from typing import Any

import pytest

from router.dataset import (
    COVERAGE_TARGET,
    LEVEL_TO_TIER,
    SourceStats,
    arena_rows,
    complexity_router_rows,
    coverage_failures,
    routerbench_rows,
)
from router.sources import by_key

ROOT = pathlib.Path(__file__).resolve().parent.parent


class FakeRegistry:
    """Tier lookup from a dict, so resolution failures are easy to stage."""

    def __init__(self, tiers: dict[str, int | None]) -> None:
        self.tiers = tiers

    def complexity(self, name: str) -> int | None:
        return self.tiers.get(name)

    def resolve(self, name: str) -> dict[str, Any] | None:
        tier = self.tiers.get(name)
        return None if tier is None else {"canonical_id": name, "complexity": tier}


class FakeDataset:
    """The slice of the HF Dataset API the adapters use."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.column_names = list(rows[0]) if rows else []

    def select_columns(self, names: list[str]) -> "FakeDataset":
        return FakeDataset([{k: r.get(k) for k in names} for r in self._rows])

    def __iter__(self):
        return iter(self._rows)


class FakeFrame:
    """The slice of pandas the routerbench adapter uses: columns + tolist()."""

    def __init__(self, columns: list[str], data: dict[str, list[Any]]) -> None:
        self.columns = columns
        self._data = data

    def __getitem__(self, col: str) -> Any:
        return type("_Col", (), {"tolist": lambda self=None, v=self._data[col]: v})()


@pytest.fixture
def arena(monkeypatch):
    """Load `arena_rows` against a fake `datasets.load_dataset`."""

    def install(rows: list[dict[str, Any]]):
        import datasets

        monkeypatch.setattr(datasets, "load_dataset",
                            lambda hf_id, split="train": FakeDataset(rows),
                            raising=True)

    return install


@pytest.fixture
def routerbench(monkeypatch):
    """Load `routerbench_rows` against a fake pkl download."""
    import pandas as pd

    def install(columns: list[str], data: dict[str, list[Any]]):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                            lambda **kw: pathlib.Path("/fake/routerbench.pkl"),
                            raising=True)
        monkeypatch.setattr(pd, "read_pickle",
                            lambda path: FakeFrame(columns, data), raising=True)

    return install


@pytest.fixture
def complexity_router(monkeypatch):
    def install(rows: list[dict[str, Any]]):
        import datasets

        monkeypatch.setattr(datasets, "load_dataset",
                            lambda hf_id, split="train": FakeDataset(rows),
                            raising=True)

    return install


# --------------------------------------------------------------------------
# arena


NESTED = [
    {"model_a": "gpt-4o", "model_b": "gemini-1.5-flash",
     "conversation_a": [{"role": "user", "content": "Which is better for prose?"}],
     "winner": "model_a", "language": "en", "is_code": False,
     "category_tag": {"topic": "writing"}},
    {"model_a": "gpt-4o", "model_b": "llama-3.1-70b-instruct",
     "conversation_a": [{"role": "user", "content": "Summarize this"}],
     "winner": "model_b", "language": None, "is_code": None, "category_tag": None},
    {"model_a": "gpt-4o", "model_b": "gemini-1.5-flash",
     "conversation_a": [{"role": "user", "content": "Tied question, distinct prompt"}],
     "winner": "tie"},
    {"model_a": "mystery-model", "model_b": "gpt-4o",
     "conversation_a": [{"role": "user", "content": "unknown peer"}],
     "winner": "model_a"},
    {"model_a": "gpt-4o", "model_b": "gemini-1.5-flash",
     "conversation_a": [{"role": "user", "content": ""}], "winner": "model_a"},
]
REG = FakeRegistry({"gpt-4o": 4, "gemini-1.5-flash": 2, "llama-3.1-70b-instruct": 3,
                    "mystery-model": None})


class TestArenaAdapter:
    def test_rows_are_labeled_by_the_winning_model_tier(self, arena):
        arena(NESTED)
        stats, seen = SourceStats(), set()
        rows = list(arena_rows("arena_100k", "hf/x", REG, stats, seen, None))
        decisive = [r for r in rows if r["label_tier"] is not None]
        assert [r["label_tier"] for r in decisive[:2]] == [4, 3]
        assert decisive[0]["tier_a"] == 4 and decisive[0]["tier_b"] == 2

    def test_ties_keep_both_tiers_and_a_null_label(self, arena):
        arena(NESTED)
        rows = list(arena_rows("arena_100k", "hf/x", REG, SourceStats(), set(), None))
        tie = next(r for r in rows if r["winner"] == "tie")
        assert tie["label_tier"] is None
        assert (tie["tier_a"], tie["tier_b"]) == (4, 2)

    def test_ids_are_sequential_per_source_from_zero(self, arena):
        arena(NESTED)
        rows = list(arena_rows("arena_100k", "hf/x", REG, SourceStats(), set(), None))
        assert [r["id"] for r in rows] == [f"arena_100k:{i:07d}" for i in range(len(rows))]

    def test_each_battle_claims_exactly_two_name_slots(self, arena):
        """This number IS the coverage gate's denominator; halving it would
        double every reported arena coverage and still pass a green suite."""
        arena(NESTED)
        stats = SourceStats()
        list(arena_rows("arena_100k", "hf/x", REG, stats, set(), None))
        # four rows reached resolution (the empty-prompt row never got there)
        assert stats.name_slots == 8
        assert stats.unmapped["mystery-model"] == 1
        assert stats.coverage == pytest.approx(100 * 7 / 8)

    def test_empty_prompt_consumes_no_slots_and_is_not_gateable(self, arena):
        arena([{"model_a": "gpt-4o", "model_b": "gemini-1.5-flash",
                "conversation_a": [{"role": "user", "content": "   "}],
                "winner": "model_a"}])
        stats = SourceStats()
        rows = list(arena_rows("arena_100k", "hf/x", REG, stats, set(), None))
        assert rows == []
        assert (stats.scanned, stats.unresolved_dropped, stats.name_slots) == (1, 1, 0)
        assert stats.coverage is None

    def test_dupe_battles_across_generations_are_dropped_once_counted(self, arena):
        arena([NESTED[0], NESTED[0]])
        stats, seen = SourceStats(), set()
        rows = list(arena_rows("arena_100k", "hf/x", REG, stats, seen, None))
        assert len(rows) == 1
        assert stats.dup_skipped == 1
        # the dupe still consumed its two slots: names were resolved before dedup
        assert stats.name_slots == 4

    def test_limit_caps_scanned_rows_not_emitted_rows(self, arena):
        arena(NESTED)
        stats = SourceStats()
        list(arena_rows("arena_100k", "hf/x", REG, stats, set(), 2))
        assert stats.scanned == 2

    def test_extra_columns_pass_through_and_tags_are_compact_json(self, arena):
        arena(NESTED)
        rows = list(arena_rows("arena_100k", "hf/x", REG, SourceStats(), set(), None))
        first = rows[0]
        assert first["language"] == "en" and first["is_code"] is False
        assert first["category_tag"] == '{"topic":"writing"}'
        # None-valued optionals are omitted rather than written as null
        assert "language" not in rows[1]

    def test_label_kind_comes_from_the_source_catalog(self, arena):
        arena(NESTED)
        rows = list(arena_rows("arena_100k", "hf/x", REG, SourceStats(), set(), None))
        assert {r["label_kind"] for r in rows} == {by_key("arena_100k").label_kind}

    def test_flat_prompt_generation_with_one_hot_winner(self, arena):
        """The 55k release: flat columns, winner as three booleans."""
        arena([{"model_a": "gpt-4o", "model_b": "gemini-1.5-flash",
                "prompt": "flat prompt", "winner_model_a": 0,
                "winner_model_b": 1, "winner_tie": 0}])
        stats = SourceStats()
        rows = list(arena_rows("arena_55k", "hf/x", REG, stats, set(), None))
        assert rows[0]["prompt"] == "flat prompt"
        assert rows[0]["winner"] == "b" and rows[0]["label_tier"] == 2

    def test_row_shape_is_the_arena_one_for_the_pairs_dispatch(self, arena):
        """No candidate_tiers on arena rows: that is what routes them through
        `arena_pairs` in the pairs stage."""
        arena(NESTED)
        rows = list(arena_rows("arena_100k", "hf/x", REG, SourceStats(), set(), None))
        assert all("candidate_tiers" not in r for r in rows)
        assert all("tier_a" in r and "tier_b" in r for r in rows)

    def test_stats_are_updated_before_each_yield(self, arena):
        """Generators are lazy: a consumer that breaks early must still have an
        accurate `emitted` for the rows it already took."""
        arena(NESTED)
        stats = SourceStats()
        gen = arena_rows("arena_100k", "hf/x", REG, stats, set(), None)
        for _i, _row in enumerate(gen, start=1):
            if _i == 2:
                break
        assert stats.emitted == 2


# --------------------------------------------------------------------------
# routerbench

RB_COLS = ["sample_id", "prompt", "eval_name", "oracle_model_to_route_to",
           "gpt-4o", "gemini-1.5-flash", "o1"]


def rb_data(qualities: list[dict[str, float | None]]) -> dict[str, list[Any]]:
    data: dict[str, list[Any]] = {
        "sample_id": list(range(len(qualities))),
        "prompt": [f"prompt {i}" for i in range(len(qualities))],
        "eval_name": ["mmlu"] * len(qualities),
        "oracle_model_to_route_to": ["gpt-4o"] * len(qualities),
    }
    for col in ("gpt-4o", "gemini-1.5-flash", "o1"):
        data[col] = [q.get(col) for q in qualities]
    return data


RB_REG = FakeRegistry({"gpt-4o": 4, "gemini-1.5-flash": 2, "o1": 5})


class TestRouterbenchAdapter:
    def test_min_solving_tier_is_the_cheapest_tier_clearing_the_bar(self, routerbench):
        # best 0.95, epsilon 0.05 -> bar at 0.90; flash at 0.90 clears it exactly
        routerbench(RB_COLS, rb_data([{"gpt-4o": 0.9, "gemini-1.5-flash": 0.90,
                                      "o1": 0.95}]))
        stats = SourceStats()
        rows = list(routerbench_rows(RB_REG, stats, set(), 0.05, None))
        assert rows[0]["solving_tiers"] == [2, 4, 5]
        assert rows[0]["label_tier"] == 2
        assert rows[0]["candidate_tiers"] == [2, 4, 5]

    def test_a_cheap_tier_just_below_the_bar_does_not_solve_it(self, routerbench):
        """The epsilon boundary is the whole labeling rule: 0.87 vs a 0.95 best
        is a miss, so the prompt's min-solving tier moves up to 4."""
        routerbench(RB_COLS, rb_data([{"gpt-4o": 0.9, "gemini-1.5-flash": 0.87,
                                      "o1": 0.95}]))
        rows = list(routerbench_rows(RB_REG, SourceStats(), set(), 0.05, None))
        assert rows[0]["solving_tiers"] == [4, 5]
        assert rows[0]["label_tier"] == 4

    def test_epsilon_widens_or_narrows_the_solving_set(self, routerbench):
        routerbench(RB_COLS, rb_data([{"gpt-4o": 0.9, "gemini-1.5-flash": 0.87,
                                      "o1": 0.95}]))
        loose = list(routerbench_rows(RB_REG, SourceStats(), set(), 0.10, None))
        assert loose[0]["solving_tiers"] == [2, 4, 5]

    def test_a_prompt_nobody_solves_gets_a_null_label_not_an_everyone_label(
            self, routerbench):
        """All-zero quality: the oracle says no model solved it. Labeling the
        cheapest tier here would teach the router that trivial prompts need a
        model at all -- 1,307 rows of the real dataset are this case."""
        routerbench(RB_COLS, rb_data([{"gpt-4o": 0.0, "gemini-1.5-flash": 0.0,
                                      "o1": 0.0}]))
        stats = SourceStats()
        rows = list(routerbench_rows(RB_REG, stats, set(), 0.05, None))
        assert len(rows) == 1
        assert rows[0]["label_tier"] is None
        assert rows[0]["solving_tiers"] == []

    def test_a_zero_quality_prompt_is_null_labeled_even_under_a_negative_bar(
            self, routerbench):
        """`best > 0.0` is the only thing standing between a prompt nobody solved
        and an "everyone solved it" label. Once any fractional score exists the
        rule becomes best - epsilon, so a best of 0.0 has a bar of -0.05 that
        every zero clears -- the guard, not the threshold, is what nulls it."""
        routerbench(RB_COLS, rb_data([
            {"gpt-4o": 0.5, "gemini-1.5-flash": 0.4, "o1": 0.6},  # non-binary scores
            {"gpt-4o": 0.0, "gemini-1.5-flash": 0.0, "o1": 0.0},  # nobody solved this
        ]))
        rows = list(routerbench_rows(RB_REG, SourceStats(), set(), 0.05, None))
        assert rows[0]["solving_tiers"] == [5] and rows[0]["label_tier"] == 5
        assert rows[1]["best_quality"] == 0.0
        assert rows[1]["solving_tiers"] == []
        assert rows[1]["label_tier"] is None

    def test_binary_scores_switch_to_the_exact_correct_rule(self, routerbench):
        routerbench(RB_COLS, rb_data([{"gpt-4o": 1.0, "gemini-1.5-flash": 0.0,
                                      "o1": 1.0}]))
        stats = SourceStats()
        rows = list(routerbench_rows(RB_REG, stats, set(), 0.05, None))
        assert rows[0]["solving_tiers"] == [4, 5]
        assert rows[0]["label_tier"] == 4
        assert stats.label_rule is not None and "binary" in stats.label_rule

    def test_missing_quality_cells_are_not_name_slots(self, routerbench):
        """A model never run on a prompt must not dilute the coverage
        denominator -- only cells with a value were candidates."""
        routerbench(RB_COLS, rb_data([{"gpt-4o": 0.5, "gemini-1.5-flash": None,
                                      "o1": float("nan")}]))
        stats = SourceStats()
        list(routerbench_rows(RB_REG, stats, set(), 0.05, None))
        assert stats.name_slots == 1

    def test_unresolved_model_columns_count_as_slots_and_gaps(self, routerbench):
        routerbench(RB_COLS, rb_data([{"gpt-4o": 0.5, "gemini-1.5-flash": 0.4,
                                      "o1": None}]))
        stats = SourceStats()
        rows = list(routerbench_rows(FakeRegistry({"gpt-4o": 4}), stats, set(),
                                     0.05, None))
        assert stats.name_slots == 2
        assert stats.unmapped["gemini-1.5-flash"] == 1
        assert stats.coverage == pytest.approx(50.0)
        assert rows[0]["candidate_tiers"] == [4]  # only the resolved tier survives

    def test_routerbench_is_coverage_gated_not_exempt(self, routerbench):
        """The bug this refactor fixed: it used to be reported as n/a and never
        gated, despite resolving ~401k names."""
        routerbench(RB_COLS, rb_data([{"gpt-4o": 0.5, "gemini-1.5-flash": 0.4,
                                      "o1": 0.6}]))
        stats = SourceStats()
        list(routerbench_rows(FakeRegistry({"gpt-4o": 4}), stats, set(), 0.05, None))
        failed = coverage_failures({"routerbench": stats})
        assert [f.key for f in failed] == ["routerbench"]
        assert failed[0].rate < COVERAGE_TARGET

    def test_prompts_with_no_resolved_models_are_dropped(self, routerbench):
        routerbench(RB_COLS, rb_data([{"gpt-4o": 0.5, "gemini-1.5-flash": 0.4,
                                      "o1": 0.6}]))
        stats = SourceStats()
        rows = list(routerbench_rows(FakeRegistry({}), stats, set(), 0.05, None))
        assert rows == []
        assert stats.unresolved_dropped == 1

    def test_best_quality_is_rounded_for_the_row(self, routerbench):
        routerbench(RB_COLS, rb_data([{"gpt-4o": 0.91234567, "gemini-1.5-flash": 0.1,
                                      "o1": 0.2}]))
        rows = list(routerbench_rows(RB_REG, SourceStats(), set(), 0.05, None))
        assert rows[0]["best_quality"] == 0.9123

    def test_oracle_and_eval_name_are_carried(self, routerbench):
        routerbench(RB_COLS, rb_data([{"gpt-4o": 0.9, "gemini-1.5-flash": 0.1,
                                      "o1": 0.2}]))
        rows = list(routerbench_rows(RB_REG, SourceStats(), set(), 0.05, None))
        assert rows[0]["dataset"] == "mmlu"
        assert rows[0]["oracle_model"] == "gpt-4o"

    def test_detail_columns_are_not_treated_as_models(self, routerbench):
        cols = RB_COLS + ["gpt-4o|model_response", "gpt-4o|total_cost"]
        data = rb_data([{"gpt-4o": 0.9, "gemini-1.5-flash": 0.1, "o1": 0.2}])
        data.update({"gpt-4o|model_response": ["text"], "gpt-4o|total_cost": [0.01]})
        routerbench(cols, data)
        stats = SourceStats()
        list(routerbench_rows(RB_REG, stats, set(), 0.05, None))
        assert stats.name_slots == 3  # the two detail columns contributed none

    def test_limit_caps_scanned_prompts(self, routerbench):
        routerbench(RB_COLS, rb_data([{"gpt-4o": 0.9, "gemini-1.5-flash": 0.5,
                                      "o1": 0.9}] * 5))
        stats = SourceStats()
        list(routerbench_rows(RB_REG, stats, set(), 0.05, 2))
        assert stats.scanned == 2

    def test_missing_prompt_column_is_a_clear_failure(self, routerbench):
        routerbench(["sample_id", "o1"], {"sample_id": [0], "o1": [0.5]})
        with pytest.raises(SystemExit, match="no 'prompt' column"):
            list(routerbench_rows(RB_REG, SourceStats(), set(), 0.05, None))

    def test_no_model_columns_at_all_is_a_clear_failure(self, routerbench):
        routerbench(["sample_id", "prompt"],
                    {"sample_id": [0], "prompt": ["p"]})
        with pytest.raises(SystemExit, match="no per-model columns"):
            list(routerbench_rows(RB_REG, SourceStats(), set(), 0.05, None))

    def test_stats_are_updated_before_each_yield(self, routerbench):
        routerbench(RB_COLS, rb_data([{"gpt-4o": 0.9, "gemini-1.5-flash": 0.5,
                                      "o1": 0.9}] * 4))
        stats = SourceStats()
        for i, _row in enumerate(routerbench_rows(RB_REG, stats, set(), 0.05, None), 1):
            if i == 3:
                break
        assert stats.emitted == 3


# --------------------------------------------------------------------------
# complexity_router

CR_ROWS = [
    {"prompt": "simple thing", "level": 0, "category": "Guidance",
     "reason": "trivial", "original_message_id": "m1"},
    {"prompt": "harder thing", "level": "2", "category": "Coding",
     "reason": "multi-step", "original_message_id": "m2"},
    {"prompt": "frontier rewrite", "level": 3, "category": "Math",
     "reason": "abstract", "original_message_id": "Level0_to_Level3"},
    {"prompt": "level out of range", "level": 7, "category": "X",
     "reason": "?", "original_message_id": "m4"},
    {"prompt": "", "level": 1, "category": "X", "reason": "?",
     "original_message_id": "m5"},
    {"prompt": "unparseable level", "level": "abc", "category": "X",
     "reason": "?", "original_message_id": "m6"},
    {"prompt": "missing level", "category": "X", "original_message_id": "m7"},
]


class TestComplexityRouterAdapter:
    def test_levels_map_linearly_onto_tiers_one_to_four(self, complexity_router):
        complexity_router(CR_ROWS)
        rows = list(complexity_router_rows(SourceStats(), set(), None))
        assert [r["label_tier"] for r in rows] == [1, 3, 4]

    def test_reuses_the_solving_tier_row_shape_on_purpose(self, complexity_router):
        """Gold tier wins, the other four lose: this is why the pairs stage has
        no third branch."""
        complexity_router(CR_ROWS)
        rows = list(complexity_router_rows(SourceStats(), set(), None))
        for r in rows:
            assert r["candidate_tiers"] == [1, 2, 3, 4, 5]
            assert r["solving_tiers"] == [r["label_tier"]]
            assert r["label_kind"] == by_key("complexity_router").label_kind

    def test_tier_five_is_unreachable_from_annotations(self, complexity_router):
        complexity_router(CR_ROWS)
        rows = list(complexity_router_rows(SourceStats(), set(), None))
        assert 5 not in {r["label_tier"] for r in rows}
        assert set(LEVEL_TO_TIER) == {0, 1, 2, 3}

    def test_out_of_range_level_is_dropped_not_silently_binned(self, complexity_router):
        """`.get(level)` must stay default-less: a `.get(level, 3)` would
        mislabel junk levels mid-tier and still emit a full 5-tier group."""
        complexity_router([{"prompt": "p", "level": 7, "category": "c",
                            "reason": "r", "original_message_id": "m"}])
        stats = SourceStats()
        assert list(complexity_router_rows(stats, set(), None)) == []
        assert stats.unresolved_dropped == 1
        assert stats.emitted == 0

    def test_numeric_string_levels_are_coerced(self, complexity_router):
        complexity_router([{"prompt": "p", "level": "1", "category": "c",
                            "reason": "r", "original_message_id": "m"}])
        rows = list(complexity_router_rows(SourceStats(), set(), None))
        assert rows[0]["label_tier"] == 2

    def test_unparseable_or_missing_level_is_dropped(self, complexity_router):
        complexity_router(CR_ROWS)
        stats = SourceStats()
        rows = list(complexity_router_rows(stats, set(), None))
        # level 7, "abc", missing key, and the empty prompt: four drops, three rows
        assert stats.unresolved_dropped == 4
        assert stats.scanned == len(CR_ROWS)
        assert len(rows) == stats.emitted == 3

    def test_synthetic_flag_marks_only_the_rewrites(self, complexity_router):
        complexity_router(CR_ROWS)
        rows = list(complexity_router_rows(SourceStats(), set(), None))
        assert [r["synthetic"] for r in rows] == [False, False, True]

    def test_model_free_source_resolves_no_names_so_no_coverage_denominator(
            self, complexity_router):
        complexity_router(CR_ROWS)
        stats = SourceStats()
        list(complexity_router_rows(stats, set(), None))
        assert stats.name_slots == 0
        assert stats.coverage is None
        assert coverage_failures({"complexity_router": stats}) == []

    def test_dedup_shares_one_namespace_across_sources(self, complexity_router):
        """A prompt already labeled by arena must not be relabeled here: `seen`
        is passed in, not created inside."""
        complexity_router(CR_ROWS)
        shared = {"routerbench-or-arena-key"}
        from router.dataset import dedup_key

        first_prompt = CR_ROWS[0]["prompt"]
        shared.add(dedup_key(first_prompt, ["complexity_router"]))
        stats = SourceStats()
        rows = list(complexity_router_rows(stats, shared, None))
        assert first_prompt not in {r["prompt"] for r in rows}
        assert stats.dup_skipped == 1

    def test_ids_are_sequential_per_source(self, complexity_router):
        complexity_router(CR_ROWS)
        rows = list(complexity_router_rows(SourceStats(), set(), None))
        assert [r["id"] for r in rows] == [f"complexity_router:{i:07d}"
                                           for i in range(len(rows))]

    def test_limit_caps_scanned_rows(self, complexity_router):
        """The check is `scanned >= limit` before processing, so exactly `limit`
        rows are read no matter how many of them survive labeling."""
        complexity_router(CR_ROWS)
        stats = SourceStats()
        rows = list(complexity_router_rows(stats, set(), 1))
        assert stats.scanned == 1 and len(rows) == 1

        complexity_router(CR_ROWS)
        stats2 = SourceStats()
        rows2 = list(complexity_router_rows(stats2, set(), 2))
        assert stats2.scanned == 2 and len(rows2) == 2

        complexity_router(CR_ROWS)
        stats3 = SourceStats()
        rows3 = list(complexity_router_rows(stats3, set(), 5))
        # rows 4-5 are the empty prompt and an unparseable level: read, not emitted
        assert stats3.scanned == 5 and len(rows3) == 3

    def test_stats_are_updated_before_each_yield(self, complexity_router):
        complexity_router(CR_ROWS)
        stats = SourceStats()
        gen = complexity_router_rows(stats, set(), None)
        for i, _row in enumerate(gen, start=1):
            if i == 1:
                break
        assert stats.emitted == 1


# --------------------------------------------------------------------------
# the CLI's family dispatch


def load_build_cli():
    spec = importlib.util.spec_from_file_location(
        "_btd", ROOT / "scripts" / "build_training_dataset.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFamilyDispatch:
    @pytest.mark.parametrize("key,fixture,seed", [
        ("arena_100k", "arena", NESTED),
        ("routerbench", "routerbench", (RB_COLS, rb_data(
            [{"gpt-4o": 0.9, "gemini-1.5-flash": 0.5, "o1": 0.95}]))),
        ("complexity_router", "complexity_router", CR_ROWS),
    ])
    def test_every_family_routes_to_the_right_adapter(self, key, fixture, seed,
                                                     request):
        """One fake installed per case: all three patch `load_dataset`, so
        installing them together would silently feed every adapter the last
        fixture's rows and still look green."""
        install = request.getfixturevalue(fixture)
        if fixture == "routerbench":
            install(*seed)
        else:
            install(seed)
        cli = load_build_cli()
        rows = list(cli.rows_for(by_key(key), RB_REG, SourceStats(), set(), 0.05, 1))
        assert rows, key
        assert rows[0]["source"] == key
        assert rows[0]["label_kind"] == by_key(key).label_kind

    def test_an_unfamilied_source_fails_loudly(self):
        """A catalog entry with no adapter must not silently produce zero rows."""
        from router.sources import Source

        cli = load_build_cli()
        orphan = Source("brand_new", "brand_new_family", "hf/x", "winner_tier",
                        "columns")
        with pytest.raises(SystemExit, match="no adapter"):
            list(cli.rows_for(orphan, FakeRegistry({}), SourceStats(), set(),
                              0.05, None))

    def test_rows_for_is_the_only_family_branch(self):
        """One dispatch point, three families. A new family means one branch here
        plus one adapter -- and this assertion fails until both are added."""
        text = (ROOT / "scripts" / "build_training_dataset.py").read_text()
        assert text.count("src.family ==") == 3
        body = text.split("def rows_for")[1].split("\ndef ")[0]
        assert body.count("return ") == 3
