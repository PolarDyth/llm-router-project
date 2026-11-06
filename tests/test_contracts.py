"""The notebook <-> repo contract guard.

The Colab notebook cannot import this repo (it runs with only the pairs file
uploaded), so it re-declares the (prompt, tier) constants inline. That copy is
unavoidable; drifting silently is not. These tests parse the notebook JSON and
fail if the two disagree -- which is the check that protects every existing
checkpoint, since routing is argmax over the five tier strings.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from router import contracts

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "train_router_colab.ipynb"


def notebook_code_assignments(fragment: str, names: set[str]) -> dict:
    """literal_eval the named assignments in the first code cell containing
    `fragment`, so the notebook's own values are read rather than paraphrased."""
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if fragment not in src:
            continue
        found: dict = {}
        for node in ast.parse(src).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
        missing = names - set(found)
        assert not missing, (
            f"notebook cell defining {fragment!r} no longer assigns {sorted(missing)}"
        )
        return found
    raise AssertionError(
        f"no code cell in the notebook contains {fragment!r} -- was the notebook "
        f"restructured? tests/test_contracts.py must be updated to match."
    )


@pytest.fixture(scope="module")
def notebook_constants() -> dict:
    return notebook_code_assignments(
        "TIER_TEXTS = {", {"TIERS", "TIER_TEXTS", "MAX_LEN", "VAL_PROMPT_MOD"}
    )


class TestNotebookMatchesContracts:
    def test_tier_texts_are_identical(self, notebook_constants):
        assert notebook_constants["TIER_TEXTS"] == contracts.TIER_TEXTS

    def test_tier_list_is_identical(self, notebook_constants):
        assert tuple(notebook_constants["TIERS"]) == contracts.TIERS

    def test_max_len_is_identical(self, notebook_constants):
        assert notebook_constants["MAX_LEN"] == contracts.MAX_LEN

    def test_val_modulus_is_identical(self, notebook_constants):
        assert notebook_constants["VAL_PROMPT_MOD"] == contracts.VAL_PROMPT_MOD

    def test_notebook_split_formula_is_the_same_expression(self):
        """Not just the constant: the notebook must hash the prompt, not the row,
        and must use == 0 on the modulus."""
        nb = json.loads(NOTEBOOK.read_text())
        srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        matching = [s for s in srcs if "sha1" in s]
        assert matching, "the notebook no longer hashes prompts to split data"
        assert any(
            "int(hashlib.sha1(prompt.encode()).hexdigest(), 16) % VAL_PROMPT_MOD == 0"
            in s for s in matching
        ), "the notebook's split expression changed shape"


class TestContractIsWellFormed:
    def test_tier_texts_cover_every_tier(self):
        assert set(contracts.TIER_TEXTS) == set(contracts.TIERS)

    def test_tier_texts_are_self_describing(self):
        """The strings are model input, so each must name its own tier or the
        model can learn to key on the ordinal alone."""
        for tier, text in contracts.TIER_TEXTS.items():
            assert text.startswith(f"complexity tier {tier} -")

    def test_tier_texts_are_all_distinct(self):
        assert len(set(contracts.TIER_TEXTS.values())) == len(contracts.TIER_TEXTS)

    def test_max_len_is_positive(self):
        assert contracts.MAX_LEN > 0

    def test_val_modulus_is_at_least_two(self):
        assert contracts.VAL_PROMPT_MOD >= 2

    def test_is_validation_agrees_with_the_notebook_formula(self):
        import hashlib

        for prompt in ("a", "hello world", " x " * 100):
            expected = int(hashlib.sha1(prompt.encode()).hexdigest(), 16) % 20 == 0
            assert contracts.is_validation_prompt(prompt) is expected

    def test_is_validation_honours_an_explicit_modulus(self):
        assert contracts.is_validation_prompt("p", mod=1) is True


@pytest.fixture(scope="module")
def tier_texts():
    """A checkpoint is only interpretable if its tier texts match the contract."""
    path = ROOT / "models" / "complexity_router" / "final" / "tier_texts.json"
    if not path.exists():
        pytest.skip("no local checkpoint (models/ is gitignored)")
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


class TestSavedCheckpointAgrees:
    def test_checkpoint_tier_texts_match_contracts(self, tier_texts):
        assert tier_texts == contracts.TIER_TEXTS

    def test_checkpoint_config_sequence_length_is_the_contract(self):
        """DeBERTa was trained with truncation at MAX_LEN; a checkpoint that
        recorded a different limit would be evaluated at the wrong length."""
        config = json.loads(
            (ROOT / "models" / "complexity_router" / "final" / "config.json").read_text()
        )
        assert isinstance(config["max_position_embeddings"], int)
        assert config["max_position_embeddings"] >= contracts.MAX_LEN

    def test_num_labels_is_not_saved_so_it_must_be_restated(self):
        """The scalar head is a load-time argument, not a saved attribute: this is
        why eval_checkpoint.py has to pass num_labels=1 again or the head it
        builds cannot accept these weights."""
        config = json.loads(
            (ROOT / "models" / "complexity_router" / "final" / "config.json").read_text()
        )
        assert "num_labels" not in config


class TestEvalScriptDoesNotRedeclareConstants:
    """`eval_checkpoint.py` runs repo-local: unlike the notebook it has no excuse
    for a second copy, so importing the contract is required, not optional."""

    def test_no_local_definitions(self):
        text = (ROOT / "scripts" / "eval_checkpoint.py").read_text()
        for name in ("MAX_LEN", "VAL_PROMPT_MOD", "TIER_TEXTS", "COVERAGE_TARGET"):
            assert f"{name} = " not in text, f"eval_checkpoint.py redefines {name}"

    def test_imports_the_contract_modules(self):
        text = (ROOT / "scripts" / "eval_checkpoint.py").read_text()
        assert "from router.contracts import" in text
        assert "from router.metrics import" in text

    def test_restates_num_labels_on_load(self):
        """AutoModelForSequenceClassification would rebuild a 2-class head that
        cannot load these weights; the setting is not saved as an attribute."""
        text = (ROOT / "scripts" / "eval_checkpoint.py").read_text()
        assert "num_labels=1" in text
