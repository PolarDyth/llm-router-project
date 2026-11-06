"""End-to-end pin on the pairs build.

The manifest in `tests/golden/` holds the byte-fingerprints of seven real builds
over a 720-row sample of the production dataset, captured before the package
restructure. Every configuration exercises a different rule path: both row
shapes, the same-tier and gap filters, tie dropping, and two seeds of the
stratified sampler.

These are the tests that say the restructure changed no behavior -- the unit
tests say the rules are right, this says the rules are still the ones that were
applied to real data.
"""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_training_pairs.py"


def sub(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_pairs_build(argv: list[str], infile: pathlib.Path,
                    outdir: pathlib.Path) -> bytes:
    out, report = outdir / "pairs.jsonl", outdir / "report.md"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--in", str(infile), "--out", str(out),
         "--report", str(report), *argv],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, f"build failed:\n{proc.stdout}\n{proc.stderr}"
    assert out.exists(), "the build reported success but wrote no pairs file"
    assert report.exists(), "the build wrote no report"
    return out.read_bytes()


def rows(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode().splitlines()]


def fingerprint(data: bytes) -> dict:
    tiers: dict[str, int] = collections.Counter()
    outcomes: dict[str, int] = collections.Counter()
    for row in rows(data):
        tiers[str(row["tier"])] += 1
        outcomes[str(row["outcome"])] += 1
    # keys are stringified to match the JSON manifest, which cannot hold ints
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "pairs": sum(tiers.values()),
        "tier_hist": dict(sorted(tiers.items(), key=lambda kv: int(kv[0]))),
        "outcome_hist": dict(sorted(outcomes.items(), key=lambda kv: float(kv[0]))),
    }


class TestGoldenBuilds:
    @pytest.mark.parametrize("config", [
        "default", "cr_only", "keep_same", "gap2_noties",
        "cap200", "cap200_seed99", "routerbench_only",
    ])
    def test_matches_the_pre_restructure_fingerprint(self, config, golden_manifest,
                                                     dataset_sample, tmp_path):
        expected = golden_manifest[config]
        got = fingerprint(run_pairs_build(expected["argv"], dataset_sample,
                                          sub(tmp_path, config)))
        # `argv` is how the build is invoked, not part of its fingerprint
        wanted = {k: v for k, v in expected.items() if k != "argv"}
        assert got == wanted, (
            f"pairs build '{config}' changed output.\n"
            f"  expected: {wanted}\n  got:      {got}"
        )


class TestCapIsASubset:
    def test_capping_reduces_counts_without_reordering_or_reshaping(
            self, dataset_sample, tmp_path):
        default = rows(run_pairs_build([], dataset_sample, sub(tmp_path, "full")))
        capped = rows(run_pairs_build(["--max-pairs", "300"], dataset_sample,
                                      sub(tmp_path, "capped")))
        assert len(capped) == 300 < len(default)
        assert set(capped[0]) == set(default[0])
        as_tuples = {(r["prompt"], r["tier"], r["outcome"], r["source"])
                     for r in default}
        assert all((r["prompt"], r["tier"], r["outcome"], r["source"]) in as_tuples
                   for r in capped), "every kept pair must also appear uncapped"


class TestBuildIsDeterministic:
    def test_same_seed_same_bytes(self, dataset_sample, tmp_path):
        a = run_pairs_build(["--max-pairs", "150"], dataset_sample, sub(tmp_path, "a"))
        b = run_pairs_build(["--max-pairs", "150"], dataset_sample, sub(tmp_path, "b"))
        assert a == b

    def test_different_seed_selects_a_different_subset(self, dataset_sample, tmp_path):
        a = run_pairs_build(["--max-pairs", "150", "--seed", "13"], dataset_sample,
                            sub(tmp_path, "a"))
        b = run_pairs_build(["--max-pairs", "150", "--seed", "77"], dataset_sample,
                            sub(tmp_path, "b"))
        assert hashlib.sha256(a).hexdigest() != hashlib.sha256(b).hexdigest()


class TestCapKeepsTheMix:
    """The point of stratified sampling over head-truncation."""

    def test_several_sources_survive_a_tight_cap(self, dataset_sample, tmp_path):
        sources = {r["source"] for r in rows(
            run_pairs_build(["--max-pairs", "120"], dataset_sample, sub(tmp_path, "x")))}
        assert len(sources) > 1, f"cap collapsed the mix onto {sources}"

    def test_every_tier_survives_a_tight_cap(self, dataset_sample, tmp_path):
        tiers = {r["tier"] for r in rows(
            run_pairs_build(["--max-pairs", "250"], dataset_sample, sub(tmp_path, "y")))}
        assert tiers == {1, 2, 3, 4, 5}

    def test_single_source_restriction_works(self, dataset_sample, tmp_path):
        data = rows(run_pairs_build(["--sources", "complexity_router"], dataset_sample,
                                    sub(tmp_path, "z")))
        assert data and {r["source"] for r in data} == {"complexity_router"}
        # complexity_router emits complete 5-tier groups, one gold per prompt
        by_prompt: dict[str, list[dict]] = collections.defaultdict(list)
        for r in data:
            by_prompt[r["prompt"]].append(r)
        assert all(len(v) == 5 for v in by_prompt.values())
        assert all(sum(x["outcome"] == 1.0 for x in v) == 1 for v in by_prompt.values())


class TestFailureModes:
    def test_missing_input_names_the_fix(self, tmp_path):
        proc = subprocess.run([sys.executable, str(SCRIPT),
                               "--in", str(tmp_path / "nope.jsonl")],
                              capture_output=True, text=True, cwd=ROOT)
        assert proc.returncode != 0
        assert "run scripts/build_training_dataset.py first" in proc.stderr + proc.stdout

    def test_source_with_no_pairs_is_an_error_not_an_empty_file(self, dataset_sample,
                                                                tmp_path):
        out = tmp_path / "o.jsonl"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--in", str(dataset_sample), "--out", str(out),
             "--report", str(tmp_path / "r.md"), "--sources", "arena_999k"],
            capture_output=True, text=True, cwd=ROOT)
        assert proc.returncode != 0
        assert "contribute no pairs" in proc.stderr + proc.stdout
        assert not out.exists()
