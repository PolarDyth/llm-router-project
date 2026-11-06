"""The committed registry must be regenerable, and the fixture's names must resolve.

`data/model_registry/models.json` is generated but checked in, so the only way to
know it is honest is to rebuild it from the cached vendor catalogs and diff.
That cache is gitignored, so the rebuild test skips itself on a fresh clone
rather than failing or reaching for the network.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import subprocess
import sys

import pytest

from router.contracts import COMPLEXITY_SCALE

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "model_registry" / "cache"
CATALOG_FILES = ("litellm.json", "openrouter.json", "modelsdev.json",
                 "epoch_ai_models.csv")

needs_cache = pytest.mark.skipif(
    not all((CACHE / n).exists() for n in CATALOG_FILES),
    reason="vendor catalog cache is gitignored; run "
           "scripts/build_model_registry.py --refresh to populate it",
)


@needs_cache
def test_rebuild_from_cache_reproduces_the_committed_registry(tmp_path,
                                                             committed_registry):
    out = tmp_path / "models.json"
    report = tmp_path / "report.md"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_model_registry.py"),
         "--out", str(out), "--report", str(report)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

    # Byte compare, not object compare: dict equality is blind to key ORDER,
    # which is part of the committed artifact (and part of any diff review).
    def stripped(text: str) -> str:
        return re.sub(r'"generated_at": "[^"]*"', "", text)

    committed_text = (ROOT / "data" / "model_registry" / "models.json").read_text()
    rebuilt_text = out.read_text()
    assert stripped(rebuilt_text) == stripped(committed_text), (
        "the committed registry is not what the current code + cached catalogs "
        "produce -- a scoring or normalization rule changed without a rebuild"
    )
    if stripped(rebuilt_text) != stripped(committed_text):
        # unreachable on success; gives a named diff instead of a 368 MB blob mismatch
        r = {m["canonical_id"]: m for m in json.loads(rebuilt_text)["models"]}
        c = {m["canonical_id"]: m for m in json.loads(committed_text)["models"]}
        only_rebuilt = sorted(set(r) - set(c))[:5]
        only_committed = sorted(set(c) - set(r))[:5]
        changed = [k for k in set(r) & set(c) if r[k] != c[k]][:5]
        raise AssertionError(
            f"registry drifted: only-rebuilt={only_rebuilt} "
            f"only-committed={only_committed} changed={changed}"
        )
    assert json.loads(rebuilt_text)["complexity_scale"] == COMPLEXITY_SCALE


@needs_cache
def test_rebuild_reports_the_same_tier_histogram(tmp_path, capfd):
    out = tmp_path / "models.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_model_registry.py"),
         "--out", str(out), "--report", str(tmp_path / "r.md")],
        capture_output=True, text=True, cwd=ROOT, check=True)
    rebuilt = collections.Counter(
        m["complexity"] for m in json.loads(out.read_text())["models"])
    committed = collections.Counter(
        m["complexity"] for m in
        json.loads((ROOT / "data" / "model_registry" / "models.json").read_text())["models"])
    assert dict(rebuilt) == dict(committed)


class TestCommittedRegistrySanity:
    """Cheap early warning for a mass re-tiering, with no catalogs needed."""

    def test_histogram_has_all_five_tiers_populated(self, committed_registry):
        hist = collections.Counter(m["complexity"] for m in committed_registry["models"])
        assert set(hist) == {1, 2, 3, 4, 5}

    def test_no_tier_has_become_dominated(self, committed_registry):
        """One tier swallowing the registry means a scoring rule collapsed."""
        hist = collections.Counter(m["complexity"] for m in committed_registry["models"])
        total = sum(hist.values())
        for tier, n in hist.items():
            assert n / total < 0.6, f"tier {tier} holds {n}/{total} models"

    def test_frontier_tier_stays_scarce(self, committed_registry):
        hist = collections.Counter(m["complexity"] for m in committed_registry["models"])
        total = sum(hist.values())
        assert hist[5] / total < 0.1

    def test_every_entry_has_the_required_fields(self, committed_registry):
        required = {"canonical_id", "display_name", "aliases", "released", "mode",
                    "complexity", "score", "signals", "basis", "notes"}
        for m in committed_registry["models"]:
            assert required <= set(m), m["canonical_id"]

    def test_canonical_ids_are_unique_and_vendor_scoped(self, committed_registry):
        ids = [m["canonical_id"] for m in committed_registry["models"]]
        assert len(ids) == len(set(ids))
        assert all("/" in i for i in ids)

    def test_scored_tiers_agree_with_their_continuous_score(self, committed_registry):
        """Held for every catalog-scored model. Hand overrides are exempt: an
        override that sets only `complexity` leaves the pre-override price-based
        `score` in place, which is why two entries disagree today."""
        for m in committed_registry["models"]:
            if m["basis"] == "override":
                continue
            assert m["complexity"] - 1 <= m["score"] <= m["complexity"] + 1, \
                m["canonical_id"]

    def test_override_score_disagreements_are_the_known_two(self, committed_registry):
        """Pinned so the list cannot grow silently. `score` is unread downstream,
        but a tier and its score disagreeing is a trap for whoever reads it next."""
        stale = [m["canonical_id"] for m in committed_registry["models"]
                 if m["basis"] == "override"
                 and not m["complexity"] - 1 <= m["score"] <= m["complexity"] + 1]
        assert sorted(stale) == ["anthropic/claude-instant-1.2",
                                 "google/gemini-1.0-pro"]

    def test_schema_version_and_description_are_stable(self, committed_registry):
        assert committed_registry["schema_version"] == 1
        assert "complexity 1-5" in committed_registry["description"]


class TestFixtureNamesResolve:
    """The 99% gate in miniature, offline, over real arena names."""

    def test_every_model_name_in_the_sample_resolves(self, dataset_sample, registry):
        unresolved = collections.Counter()
        for line in dataset_sample.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("model_a", "model_b"):
                if key in row and registry.resolve(row[key]) is None:
                    unresolved[row[key]] += 1
        assert unresolved == {}, f"registry gaps: {unresolved.most_common(10)}"

    def test_sample_covers_both_row_shapes(self, dataset_sample):
        shapes = collections.Counter()
        for line in dataset_sample.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            shapes["solving" if isinstance(row.get("candidate_tiers"), list)
                  else "arena"] += 1
        assert shapes["solving"] and shapes["arena"]

    def test_sample_covers_the_redistributable_sources(self, dataset_sample):
        """Every source except the gated CC-BY-NC one -- both row shapes and all
        three label kinds are still represented."""
        seen = set()
        kinds = set()
        for line in dataset_sample.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                seen.add(row["source"])
                kinds.add(row["label_kind"])
        assert seen == {"arena_55k", "arena_100k", "arena_140k", "routerbench",
                        "complexity_router"}
        assert kinds == {"winner_tier", "min_solving_tier", "annotated_level"}

    def test_fixture_contains_no_rows_from_gated_sources(self, dataset_sample):
        """The fixture is committed, and this repo has a remote. Any source
        flagged `gated` in the catalog -- access by accepting terms on the Hub,
        which is not a redistribution right -- must stay out of it. The guard
        reads the catalog rather than a hardcoded dataset id, so a newly added
        gated source is covered the moment it is flagged."""
        from router.sources import GATED

        assert "arena_33k" in GATED, "the catalog no longer flags the gated arena set"
        seen = set()
        for line in dataset_sample.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)["source"])
        assert not (seen & GATED), f"fixture embeds rows from gated datasets: {seen & GATED}"

    def test_fixture_rows_are_all_reproducible_from_the_dataset_file(self, dataset_sample):
        """The fixture is a slice of the real build, not hand-written data: ids
        are sequential per source and every row carries its label kind."""
        per: dict[str, list[int]] = {}
        for line in dataset_sample.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["label_kind"], row["id"]
            per.setdefault(row["source"], []).append(int(row["id"].rsplit(":", 1)[1]))
        for source, ids in per.items():
            assert ids == list(range(len(ids))), f"{source} ids are not sequential"

    def test_every_sampled_row_has_a_label_or_a_reason_not_to(self, dataset_sample):
        """A null label is only honest for an arena tie/both_bad, or a routerbench
        prompt that no model solved -- never for a decisive battle."""
        for line in dataset_sample.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["label_tier"] is not None:
                continue
            reason = (row["source"].startswith("arena")
                      and row["winner"] in ("tie", "both_bad", "unknown")) or \
                     (row["source"] == "routerbench" and not row["solving_tiers"])
            assert reason, row
