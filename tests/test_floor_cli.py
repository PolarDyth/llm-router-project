"""The floor selection CLI: its failure modes, its seams, and its report.

These run offline against tiny tmp files (never the 365 MB dataset), because what
needs pinning here is not arithmetic -- `router/floor.py` covers that. What the
suite must hold is the *contract between stages*: rows that come out of this
script have to be consumable by `build_training_pairs.py` with no changes to it,
the registry tripwire has to actually stop a stale build, and the temporal holdout
has to be exclusive, because a leaky holdout is worse than no holdout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "select_floor_dataset.py"
PAIRS_SCRIPT = ROOT / "scripts" / "build_training_pairs.py"

# No tokenizer: the chars/4 fallback is deterministic and keeps these tests from
# needing models/ (which is gitignored). The report prints which backend ran.
NO_TOKENIZER = ROOT / "tests" / "fixtures" / "definitely-not-a-tokenizer"


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=ROOT)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


#: Registry names used to build synthetic battles. They are deliberately real
#: names rather than fakes: the CLI re-resolves every row's tiers against the
#: live registry and refuses to run on a stale dataset file, so a battle has to
#: state the tier its models actually carry. `TestBattleFixture` below pins that
#: correspondence, so a registry edit breaks a named assertion instead of
#: silently turning every case in this file into a tripwire hit.
TINY = "chatglm-6b"               # tier 1, score 1.0
WEAK = "mistral-7b-instruct"      # tier 2, score 2.0
MID = "llama-2-70b-chat"         # tier 3, score 3.0
STRONG = "gpt-4-0613"             # tier 4, score 5.0
TOP = "o3-2025-04-16"             # tier 5, score 4.91

#: A battle's models must carry the tiers the row states, so the fixture picks
#: them from the tier rather than fixing them and lying about the tier. Getting
#: this wrong is exactly what the CLI's registry tripwire is for -- it aborted
#: these tests until the pairing was made honest.
MODEL_BY_TIER: dict[int, str] = {1: TINY, 2: WEAK, 3: MID, 4: STRONG, 5: TOP}


class TestBattleFixture:
    def test_synthetic_models_carry_the_tiers_the_cases_assume(self) -> None:
        from router.model_registry import get_registry

        reg = get_registry()
        assert {t: reg.complexity(n) for t, n in MODEL_BY_TIER.items()} \
            == {t: t for t in MODEL_BY_TIER}

    def test_adjacent_tiers_clear_the_delta_gate(self) -> None:
        # |score_a - score_b| >= 1.0 is what makes "decisive" mean something. If
        # an adjacent pair ever becomes near-peers, the happy-path cases stop
        # producing a constraint at all and quietly assert nothing.
        from router.model_registry import get_registry

        reg = get_registry()
        for lower, upper in ((2, 3), (3, 4), (1, 2)):
            entry_a = reg.resolve(MODEL_BY_TIER[lower])
            entry_b = reg.resolve(MODEL_BY_TIER[upper])
            assert entry_a is not None and entry_b is not None
            gap = abs(float(entry_a["score"]) - float(entry_b["score"]))
            assert gap >= 1.0, f"tiers {lower}->{upper} score gap {gap} < 1.0"


def battle(prompt: str, tier_a: int, tier_b: int, winner: str,
           source: str = "arena_100k", model_a: str | None = None,
           model_b: str | None = None) -> dict[str, Any]:
    """An arena-shaped battle, registry-consistent by construction."""
    ma = model_a or MODEL_BY_TIER[tier_a]
    mb = model_b or MODEL_BY_TIER[tier_b]
    return {"id": f"{source}:0", "source": source, "label_kind": "winner_tier",
            "prompt": prompt, "model_a": ma, "model_b": mb,
            "tier_a": tier_a, "tier_b": tier_b, "winner": winner,
            "label_tier": tier_a if winner == "a" else tier_b if winner == "b" else None}


@pytest.fixture
def core_inputs(tmp_path: Path) -> dict[str, Path]:
    return {"in": tmp_path / "dataset.jsonl", "out": tmp_path / "floor.jsonl",
            "quarantine": tmp_path / "quarantine.jsonl",
            "holdout": tmp_path / "holdout.jsonl",
            "report": tmp_path / "report.md"}


def base_args(paths: dict[str, Path], **overrides: str) -> list[str]:
    args = ["--in", str(paths["in"]), "--out", str(paths["out"]),
            "--quarantine-out", str(paths["quarantine"]),
            "--holdout-out", str(paths["holdout"]),
            "--report", str(paths["report"]),
            "--skip-semantic-dedup", "--tokenizer-dir", str(NO_TOKENIZER),
            "--per-tier", overrides.get("per_tier", "1"),
            "--val-floor-per-tier", overrides.get("val_floor", "0"),
            "--long-min-per-tier", "0"]
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestHappyPath:
    def test_selects_and_writes_the_solving_shape(self, core_inputs: dict[str, Path]
                                                  ) -> None:
        # tier 4 beats tier 3 -> [4,4]: a point label from one decisive battle.
        write_rows(core_inputs["in"], [battle("alpha prompt", 3, 4, "b"),
                                       battle("beta prompt", 2, 3, "b")])
        result = run(base_args(core_inputs))
        assert result.returncode == 0, result.stderr
        rows = read_jsonl(core_inputs["out"])
        assert len(rows) == 2
        for row in rows:
            assert row["candidate_tiers"] == [1, 2, 3, 4, 5]
            assert len(row["solving_tiers"]) == 1
            assert row["label_tier"] == row["solving_tiers"][0]
            assert row["source"] == "capability_floor"
            assert row["supporting_sources"]  # licence provenance is never empty

    def test_output_flows_through_the_pairs_build_unchanged(self, core_inputs: dict[str, Path]
                                                           ) -> None:
        # The load-bearing seam: pairs.py was not touched, so these rows must
        # produce complete 5-tier groups with exactly one gold there.
        write_rows(core_inputs["in"], [battle("alpha prompt", 3, 4, "b"),
                                       battle("beta prompt", 2, 3, "b")])
        assert run(base_args(core_inputs)).returncode == 0
        pairs_out = core_inputs["out"].with_name("pairs.jsonl")
        result = subprocess.run(
            [sys.executable, str(PAIRS_SCRIPT), "--in", str(core_inputs["out"]),
             "--out", str(pairs_out), "--sources", "capability_floor",
             "--report", str(core_inputs["report"].with_name("pairs.md"))],
            capture_output=True, text=True, cwd=ROOT)
        assert result.returncode == 0, result.stderr
        pairs = read_jsonl(pairs_out)
        assert len(pairs) == 10                     # 2 prompts x 5 tiers
        assert list(pairs[0].keys()) == ["prompt", "tier", "outcome", "source"]
        by_prompt: dict[str, dict[int, float]] = {}
        for p in pairs:
            by_prompt.setdefault(p["prompt"], {})[p["tier"]] = p["outcome"]
        assert all(sorted(g) == [1, 2, 3, 4, 5] for g in by_prompt.values())
        assert all(list(g.values()).count(1.0) == 1 for g in by_prompt.values())
        assert "no-signal=0" in result.stdout

    def test_report_states_supply_shortfalls_and_review(self, core_inputs: dict[str, Path]
                                                       ) -> None:
        write_rows(core_inputs["in"], [battle(f"prompt number {i}", 3, 4, "b")
                                       for i in range(12)])
        assert run(base_args(core_inputs, per_tier="3")).returncode == 0
        text = core_inputs["report"].read_text()
        for heading in ("## yield", "## rules in force", "## selection",
                        "## review sample", "era supply"):
            assert heading in text
        # The length backend is a claim about tokens, so which measurer ran is
        # part of the record rather than an assumption.
        assert "chars/4 estimate" in text
        assert "near-duplicate collapse: SKIPPED" in text

    def test_quarantine_file_holds_the_contradictory_prompts(self, core_inputs: dict[str, Path]
                                                            ) -> None:
        # [3,4] from the win and [1,2] from the upset: no tier satisfies both.
        write_rows(core_inputs["in"], [battle("contested prompt", 2, 4, "a"),
                                       battle("contested prompt", 3, 4, "b")])
        result = run(base_args(core_inputs))
        assert result.returncode == 0, result.stderr
        quarantined = read_jsonl(core_inputs["quarantine"])
        assert len(quarantined) == 1
        assert quarantined[0]["prompt"] == "contested prompt"
        assert quarantined[0]["n_constraints"] == 2
        assert "contradictory=1" in result.stdout
        assert read_jsonl(core_inputs["out"]) == []


class TestRegistryTripwire:
    def test_stale_tiers_stop_the_build(self, core_inputs: dict[str, Path]) -> None:
        # `gpt-4-0613` is tier 4 in the current registry; a file that says the
        # battle it was in was a tier-1-vs-tier-2 matchup predates a registry
        # rebuild, and its labels would be computed against a scale that no
        # longer exists. This row is inconsistent on purpose: model_a carries a
        # tier-4 name while the row states tier 1.
        write_rows(core_inputs["in"], [battle("stale row", 1, 2, "b",
                                             model_a=STRONG)])
        result = run(base_args(core_inputs))
        assert result.returncode != 0
        assert "predates a registry rebuild" in result.stderr

    def test_tolerance_is_stated_not_implied(self, core_inputs: dict[str, Path]) -> None:
        write_rows(core_inputs["in"], [battle("fine row", 3, 4, "b")])
        result = run(base_args(core_inputs) + ["--tier-tolerance", "0.5"])
        assert result.returncode == 0, result.stderr


class TestHoldoutExclusivity:
    def test_holdout_prompts_never_appear_in_the_core(self, core_inputs: dict[str, Path]
                                                     ) -> None:
        rows = [battle(f"old era prompt {i}", 3, 4, "b", source="arena_100k")
                for i in range(6)]
        rows += [battle(f"new era prompt {i}", 3, 4, "b", source="arena_140k")
                 for i in range(6)]
        # one prompt fought in BOTH releases: it must leave the core entirely,
        # or "out-of-era" is evaluated on partly in-distribution data.
        rows.append(battle("shared prompt", 3, 4, "b", source="arena_100k"))
        rows.append(battle("shared prompt", 3, 4, "b", source="arena_140k"))
        write_rows(core_inputs["in"], rows)
        result = run(base_args(core_inputs, per_tier="2")
                     + ["--holdout-era", "arena_140k", "--holdout-per-tier", "2"])
        assert result.returncode == 0, result.stderr
        core = {r["prompt"] for r in read_jsonl(core_inputs["out"])}
        hold = {r["prompt"] for r in read_jsonl(core_inputs["holdout"])}
        assert core and hold
        assert not (core & hold)
        # era-exclusive evidence only, and nothing with 140k backing at all
        assert all(set(r["supporting_sources"]) == {"arena_140k"} for r in
                   read_jsonl(core_inputs["holdout"]))
        assert all("arena_140k" not in r["supporting_sources"] for r in
                   read_jsonl(core_inputs["out"]))

    def test_excluding_and_holding_out_the_same_release_is_refused(
            self, core_inputs: dict[str, Path]) -> None:
        write_rows(core_inputs["in"], [battle("p", 3, 4, "b")])
        result = run(base_args(core_inputs) + ["--exclude-source", "arena_140k",
                                               "--holdout-era", "arena_140k"])
        assert result.returncode != 0
        assert "cannot be excluded and held out" in result.stderr


class TestArgumentValidation:
    def test_missing_input_names_the_build_to_run_first(self, tmp_path: Path) -> None:
        result = run(["--in", str(tmp_path / "nope.jsonl")])
        assert result.returncode != 0
        assert "build_training_dataset.py" in result.stderr

    def test_non_arena_exclude_is_rejected(self, core_inputs: dict[str, Path]) -> None:
        write_rows(core_inputs["in"], [battle("p", 3, 4, "b")])
        result = run(base_args(core_inputs) + ["--exclude-source", "routerbench"])
        assert result.returncode != 0
        assert "non-arena sources" in result.stderr

    def test_excluding_every_release_is_rejected(self, core_inputs: dict[str, Path]) -> None:
        write_rows(core_inputs["in"], [battle("p", 3, 4, "b")])
        result = run(base_args(core_inputs)
                     + ["--exclude-source", ",".join(
                         ["arena_33k", "arena_55k", "arena_100k", "arena_140k"])])
        assert result.returncode != 0
        assert "nothing would label" in result.stderr

    def test_rule_flags_reach_the_rules_not_just_the_report(self, core_inputs: dict[str, Path]
                                                           ) -> None:
        # both_bad with min tier 4 gives tier 5 under default gates; raising the
        # gate to 5 must reject it. A flag that only renames a report column would
        # still pass the first assertion above and fail this one.
        rows = [battle("hard both bad", 4, 4, "both_bad", model_a=STRONG,
                       model_b=STRONG)]
        write_rows(core_inputs["in"], rows)
        default_run = run(base_args(core_inputs))
        assert default_run.returncode == 0, default_run.stderr
        assert len(read_jsonl(core_inputs["out"])) == 1
        strict = run(base_args(core_inputs) + ["--both-bad-min-tier", "5"])
        assert strict.returncode == 0, strict.stderr
        assert read_jsonl(core_inputs["out"]) == []


class TestOnTheCommittedFixture:
    def test_smoke_over_the_600_row_sample(self, tmp_path: Path, dataset_sample: Path
                                          ) -> None:
        # Real arena rows, no network, no gated data: the fixture carries 55k /
        # 100k / 140k only, so this exercises the pipeline end to end at small
        # scale without ever writing to a tracked file.
        out = tmp_path / "floor.jsonl"
        result = run(["--in", str(dataset_sample), "--out", str(out),
                      "--quarantine-out", str(tmp_path / "q.jsonl"),
                      "--holdout-out", str(tmp_path / "h.jsonl"),
                      "--report", str(tmp_path / "r.md"), "--per-tier", "1",
                      "--skip-semantic-dedup", "--tokenizer-dir", str(NO_TOKENIZER)])
        assert result.returncode == 0, result.stderr
        rows = read_jsonl(out)
        assert rows
        assert all(r["label_tier"] in (1, 2, 3, 4, 5) for r in rows)
        assert all(len(r["candidate_tiers"]) == 5 for r in rows)
