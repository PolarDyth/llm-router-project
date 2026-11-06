"""Shared fixtures. The registry is loaded once per session: `get_registry`
is lru_cached on models.json, so a fresh process is what picks up a rebuild."""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = ROOT / "tests" / "golden"
REGISTRY_JSON = ROOT / "data" / "model_registry" / "models.json"
CATALOG_CACHE = ROOT / "data" / "model_registry" / "cache"


@pytest.fixture(scope="session")
def registry():
    from router.model_registry import get_registry

    return get_registry()


@pytest.fixture(scope="session")
def committed_registry() -> dict:
    return json.loads(REGISTRY_JSON.read_text())


@pytest.fixture(scope="session")
def dataset_sample() -> pathlib.Path:
    """600 real dataset rows: 120 from each of the five redistributable sources,
    covering both row shapes and all three label kinds. The gated CC-BY-NC arena
    set is deliberately excluded because this file is committed."""
    return FIXTURES / "dataset_sample.jsonl"


@pytest.fixture(scope="session")
def golden_manifest() -> dict:
    """Pre-restructure output fingerprints of 7 pairs-build configurations."""
    return json.loads((GOLDEN / "pairs_manifest.json").read_text())
