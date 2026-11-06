"""The restructure's own guard rails.

Every rule here is a duplication that existed before and would silently come
back: two source catalogs, two copies of the pkl column rule, three copies of
device selection, two row-shape dispatches, and `sys.path` patching in place of
a package. A test is cheaper than rediscovering any of them.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


SKIP_DIRS = {".venv", ".git", ".vscode", ".zcode", ".pyright-tmp", "__pycache__",
             "llm_router.egg-info"}


_PY_FILES: list[pathlib.Path] | None = None


def py_files() -> list[pathlib.Path]:
    """Every tracked Python file: package, CLIs, tests, and any future top-level
    module. Scoping this to router/+scripts/ would let a copy reappear in tests/
    or a new root file and still report clean. Walked once, then cached -- the
    36 parametrized cases below would otherwise re-scan the tree each time."""
    global _PY_FILES
    if _PY_FILES is None:
        _PY_FILES = sorted(
            p for p in ROOT.rglob("*.py") if not SKIP_DIRS & set(p.parts)
        )
    return _PY_FILES


def scannable_files() -> list[pathlib.Path]:
    """py_files() minus this file: it necessarily contains the literal patterns
    it forbids elsewhere, so scanning it would only ever report itself."""
    return [p for p in py_files() if p.name != "test_no_duplication.py"]


def production_files() -> list[pathlib.Path]:
    return sorted([*sorted(ROOT.glob("router/*.py")),
                   *sorted(ROOT.glob("scripts/*.py"))])


def defined_names(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
    return found


def module_level_assignments(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


#: name -> the only file allowed to define it
SOLE_DEFINITION = {
    "normalize_name": "router/model_registry.py",
    "ModelRegistry": "router/model_registry.py",
    "get_registry": "router/model_registry.py",
    "merge_records": "router/catalogs.py",
    "CatalogCache": "router/catalogs.py",
    "Canonical": "router/catalogs.py",
    "infer_vendor": "router/catalogs.py",
    "score_group": "router/scoring.py",
    "apply_overrides": "router/scoring.py",
    "alias_collisions": "router/scoring.py",
    "interp_price": "router/scoring.py",
    "cut_tier": "router/scoring.py",
    "apply_era_caps": "router/scoring.py",
    "quality_adjust": "router/scoring.py",
    "routerbench_model_columns": "router/sources.py",
    "pairs_from_row": "router/pairs.py",
    "arena_pairs": "router/pairs.py",
    "solving_tier_pairs": "router/pairs.py",
    "allocate": "router/pairs.py",
    "Pair": "router/pairs.py",
    "first_user_text": "router/dataset.py",
    "norm_winner": "router/dataset.py",
    "dedup_key": "router/dataset.py",
    "coverage_failures": "router/dataset.py",
    "SourceStats": "router/dataset.py",
    "val_split": "router/metrics.py",
    "group_by_prompt": "router/metrics.py",
    "metric_cohort": "router/metrics.py",
    "pairwise_accuracy": "router/metrics.py",
    "routing_accuracy": "router/metrics.py",
    "majority_baseline": "router/metrics.py",
    "is_validation": "router/metrics.py",
    "is_validation_prompt": "router/contracts.py",
    "select_device": "router/devices.py",
    # The capability-floor rules. These are the dataset's semantics now, so a
    # second copy anywhere (a script re-deriving an interval, a notebook
    # restating the gates) would let the labels and the training data disagree
    # about what a tier claim means -- the same failure the pairs goldens catch
    # for the row rules.
    "battle_constraint": "router/floor.py",
    "constraint_rule": "router/floor.py",
    "intersect": "router/floor.py",
    "collapse": "router/floor.py",
    "collect_intervals": "router/floor.py",
    "norm_prompt": "router/floor.py",
    "waterfill": "router/floor.py",
    "select_balanced": "router/floor.py",
    "floor_rows": "router/floor.py",
}


@pytest.mark.parametrize("name,owner", sorted(SOLE_DEFINITION.items()))
def test_defined_exactly_once(name, owner):
    definers = [str(p.relative_to(ROOT)) for p in py_files() if name in defined_names(p)]
    assert definers == [owner], (
        f"{name} is defined in {definers}; it should live only in {owner}"
    )


class TestSharedConstantsHaveOneHome:
    @pytest.mark.parametrize("name", [
        "MAX_LEN", "VAL_PROMPT_MOD", "TIER_TEXTS", "TIERS", "COMPLEXITY_SCALE",
        "COVERAGE_TARGET",
    ])
    def test_contract_constant_defined_once(self, name):
        definers = [str(p.relative_to(ROOT)) for p in py_files()
                    if name in module_level_assignments(p)]
        assert definers == ["router/contracts.py"], (
            f"{name} is assigned in {definers}; only router/contracts.py may own it"
        )

    @pytest.mark.parametrize("name", [
        "HAND_ALIASES", "PRICE_EDGES", "II_CUTS", "ELO_FRONTIER",
        "ERA_PRE_FRONTIER", "ERA_CAP_4_END", "ERA_MODERN",
        "ROUTERBENCH_META_COLS", "SOURCES",
    ])
    def test_tuning_constant_defined_once(self, name):
        definers = [str(p.relative_to(ROOT)) for p in py_files()
                    if name in module_level_assignments(p)]
        assert len(definers) == 1, f"{name} assigned in {definers}"

    def test_tuning_constants_are_in_the_expected_modules(self):
        assert "HAND_ALIASES" in module_level_assignments(ROOT / "router/model_registry.py")
        scoring = module_level_assignments(ROOT / "router/scoring.py")
        assert {"PRICE_EDGES", "II_CUTS", "ELO_FRONTIER",
                "ERA_PRE_FRONTIER", "ERA_CAP_4_END", "ERA_MODERN"} <= scoring
        assert "ROUTERBENCH_META_COLS" in module_level_assignments(
            ROOT / "router/sources.py")

    def test_coverage_target_is_imported_not_restated(self):
        for rel in ("scripts/audit_model_coverage.py", "scripts/build_training_dataset.py"):
            text = (ROOT / rel).read_text()
            assert "from router.contracts import" in text, rel
            assert "COVERAGE_TARGET = " not in text, rel


class TestNoPathHacks:
    def test_no_sys_path_patching(self):
        offenders = [str(p.relative_to(ROOT)) for p in scannable_files()
                     if "sys.path.insert(" in p.read_text()
                     or "sys.path.append(" in p.read_text()]
        assert offenders == [], (
            f"these files patch sys.path instead of relying on the installed "
            f"package: {offenders}"
        )

    def test_package_is_importable_without_the_repo_on_path(self):
        """The editable install (or pyproject + `pip install -e .`) is what lets
        scripts import `router` plainly."""
        import router

        assert pathlib.Path(router.__file__).resolve().parent == (ROOT / "router").resolve()


class TestDeviceSelection:
    def test_only_the_package_module_selects_a_device(self):
        assert not any("pick_device" in defined_names(p) for p in py_files()), (
            "a script re-declared device selection; import router.devices instead"
        )

    def test_directml_is_never_a_static_import(self):
        """AGENTS.md hard rule: torch-directml cannot be installed on every
        platform, so it must be reached through importlib."""
        offenders = [str(p.relative_to(ROOT)) for p in scannable_files()
                     if re.search(r"^\s*(import torch_directml|from torch_directml)",
                                          p.read_text(), re.M)]
        assert offenders == []

    def test_directml_is_loaded_optional_via_importlib(self):
        text = (ROOT / "router/devices.py").read_text()
        assert "importlib" in text
        assert 'import_module("torch_directml")' in text

    def test_no_pin_memory_anywhere(self):
        """AGENTS.md: the manual loop exists for DirectML; pin_memory breaks it."""
        offenders = [str(p.relative_to(ROOT)) for p in scannable_files()
                     if "pin_memory" in p.read_text()]
        assert offenders == []


class TestRowShapeDispatchIsSole:
    def test_scripts_do_not_branch_on_the_row_shape(self):
        """`candidate_tiers` presence decides the pair rule. If a script branches
        on it too, the two streaming passes can disagree about what was sampled."""
        text = (ROOT / "scripts" / "build_training_pairs.py").read_text()
        assert "candidate_tiers" not in text

    def test_dispatch_lives_in_pairs(self):
        text = (ROOT / "router" / "pairs.py").read_text()
        assert text.count('isinstance(row.get("candidate_tiers"), list)') == 1


class TestNotebookIsTheOnlyDuplicateByNecessity:
    def test_no_production_module_restates_the_tier_texts(self):
        """Anything repo-local must import; only the Colab notebook may restate."""
        offenders = [
            str(p.relative_to(ROOT)) for p in production_files()
            if "complexity tier 1 -" in p.read_text()
            and str(p.relative_to(ROOT)) != "router/contracts.py"
        ]
        assert offenders == []

    def test_the_split_predicate_is_computed_in_one_place(self):
        """`metrics.is_validation` must delegate to `contracts.is_validation_prompt`.
        Two copies of the hash are how a held-out prompt leaks into "train"."""
        offenders = [
            str(p.relative_to(ROOT)) for p in production_files()
            if "hexdigest(), 16) %" in p.read_text()
            and str(p.relative_to(ROOT)) != "router/contracts.py"
        ]
        assert offenders == []
        metrics = (ROOT / "router/metrics.py").read_text()
        assert "is_validation_prompt(triplet[0]" in metrics

    def test_no_notebook_cell_imports_the_repo(self):
        """The notebook must stay self-contained: Colab gets only the pairs file
        uploaded, so an `import router` there would fail at runtime."""
        nb = (ROOT / "notebooks" / "train_router_colab.ipynb").read_text()
        assert "from router" not in nb and "import router" not in nb

    def test_the_notebook_still_declares_them(self):
        """If this fails, the notebook was changed to import from the repo --
        which cannot work in Colab and would break the upload-only workflow."""
        nb = (ROOT / "notebooks" / "train_router_colab.ipynb").read_text()
        assert "TIER_TEXTS = {" in nb
        assert "MAX_LEN = " in nb
