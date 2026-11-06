"""Name normalization and alias resolution against the committed registry.

`normalize_name` is the single source of truth shared by the registry builder
and the runtime resolver, so these cases double as the contract for both: the
tier of already-labeled training rows depends on these mappings staying put.
"""

from __future__ import annotations

import pytest

from router.model_registry import HAND_ALIASES, normalize_name


class TestNormalizeName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            # hosting prefixes are never tier-bearing
            ("meta-llama/Llama-3.1-70B-Instruct", "llama-3.1-70b"),
            ("openai/gpt-4o-2024-08-06", "gpt-4o"),
            # bedrock/region prefixes and serving suffixes
            ("anthropic.claude-3-5-sonnet-20241022-v2:0", "claude-3-5-sonnet"),
            ("us.meta.llama3-2-11b-instruct-v1:0", "llama3-2-11b"),
            ("amazon.nova-pro-v1:0", "nova-pro"),
            # date / snapshot / context-window qualifiers
            ("gpt-4-1106-preview", "gpt-4"),
            ("gemini-1.5-flash-001", "gemini-1.5-flash"),
            ("qwen2.5-72b-instruct", "qwen2.5-72b"),
            ("gpt-4-32k-0613", "gpt-4"),
            # case is folded, whitespace is NOT (that is resolve()'s job)
            ("Claude 3.5 Sonnet", "claude 3.5 sonnet"),
        ],
    )
    def test_reduces_to_canonical_form(self, raw, expected):
        assert normalize_name(raw) == expected

    def test_is_idempotent_on_already_canonical_names(self):
        for name in ("gpt-4o", "llama-3.1-70b", "gemini-1.5-pro", "claude-3-5-sonnet"):
            assert normalize_name(name) == name

    def test_vendor_prefixed_hand_aliases_survive(self):
        """These normalize to "vendor/model" and are used as registry keys."""
        assert normalize_name("deepseek") == "deepseek/deepseek-chat"
        assert normalize_name("chatgpt-4o-latest") == "chatgpt-4o"

    def test_gateway_double_prefix_and_at_version_are_stripped(self):
        assert normalize_name("groq/llama-3.3-70b-specdec") == "llama-3.3-70b-specdec"
        assert normalize_name("google/gemini-2.0-flash-001") == "gemini-2.0-flash"


class TestResolve:
    def test_documented_example_from_the_project_docs(self, registry):
        """CLAUDE.md advertises exactly this number as a smoke check."""
        assert registry.complexity("gpt-4-1106-preview") == 4

    def test_hand_aliases_win_over_the_alias_index(self, registry):
        """Ambiguous marketing names resolve to the historical original: the
        training data is 2022-2025, where "gemini-pro" meant 1.0, not 1.5."""
        assert registry.resolve("gemini-pro")["canonical_id"] == "google/gemini-1.0-pro"
        assert registry.complexity("gemini-1.5-pro") == 4

    def test_release_number_spellings_are_matched_on_the_raw_name(self, registry):
        """normalize_name strips "-v2", which would collapse these onto bare
        "claude"; resolve() therefore checks the raw string first."""
        assert registry.resolve("claude-v1")["canonical_id"] == "anthropic/claude-1"
        assert registry.resolve("claude-v2")["canonical_id"] == "anthropic/claude-2"
        assert registry.complexity("claude-v2") == 3

    def test_tier_spread_across_the_scale(self, registry):
        assert registry.complexity("gemini-1.5-flash") == 2
        assert registry.complexity("llama-3.1-70b-instruct") == 3
        assert registry.complexity("gpt-4o") == 4
        assert registry.complexity("o1") == 5

    def test_space_and_underscore_variants_agree(self, registry):
        a = registry.resolve("Claude 3.5 Sonnet")
        b = registry.resolve("claude-3.5-sonnet")
        assert a is not None and a == b

    def test_unknown_name_resolves_to_none_not_an_error(self, registry):
        assert registry.resolve("definitely-not-a-model-xyz") is None
        assert registry.complexity("definitely-not-a-model-xyz") is None

    def test_entry_lookup_is_by_canonical_id_only(self, registry):
        assert registry.entry("openai/gpt-4o") is not None
        assert registry.entry("gpt-4o") is None

    def test_every_hand_alias_target_exists_in_the_registry(self, registry):
        """A hand alias pointing at a pruned canonical id silently resolves to
        nothing; the alias table is only correct if every target is present."""
        missing = [a for a, target in HAND_ALIASES.items()
                   if registry.entry(target) is None]
        assert missing == [], f"hand aliases point at absent models: {missing}"

    def test_generated_at_is_present_and_models_are_nonempty(self, registry):
        assert registry.generated_at
        assert len(registry.models) > 1000

    def test_registry_tiers_are_all_in_range(self, registry):
        assert {m["complexity"] for m in registry.models} <= {1, 2, 3, 4, 5}
