"""Tests for PromptBuilder — template-based extraction prompt generation."""

from unittest.mock import MagicMock

from knowledge_service.clients.prompt_builder import PromptBuilder, _MAX_TEXT_CHARS
from knowledge_service.ontology.registry import PredicateInfo


def _mock_registry():
    reg = MagicMock()
    reg.get_prompt.return_value = None
    return reg


class TestBuildCombinedPrompt:
    def test_combined_prompt_includes_entity_and_relation_instructions(self):
        reg = _mock_registry()
        reg.all_domain_names.return_value = ["base"]
        reg.get_predicates.return_value = [
            PredicateInfo(uri="http://x/causes", label="causes", domain="base"),
        ]
        builder = PromptBuilder(reg)
        prompt = builder.build_combined_prompt("Some text", title="Test", source_type="article")
        assert "Entity" in prompt
        assert "Event" in prompt
        assert "Claim" in prompt
        assert "Relationship" in prompt
        assert "snake_case" in prompt
        assert "Some text" in prompt

    def test_combined_prompt_includes_nlp_hints(self):
        reg = _mock_registry()
        reg.all_domain_names.return_value = ["base"]
        reg.get_predicates.return_value = []
        builder = PromptBuilder(reg)
        hints = [{"text": "dopamine", "label": "CHEMICAL", "wikidata_id": "Q80635"}]
        prompt = builder.build_combined_prompt("text", entity_hints=hints)
        assert "dopamine" in prompt
        assert "CHEMICAL" in prompt

    def test_combined_prompt_includes_predicates(self):
        reg = _mock_registry()
        reg.all_domain_names.return_value = ["base"]
        reg.get_predicates.return_value = [
            PredicateInfo(uri="http://x/causes", label="causes", domain="base"),
            PredicateInfo(uri="http://x/increases", label="increases", domain="base"),
        ]
        builder = PromptBuilder(reg)
        prompt = builder.build_combined_prompt("text")
        assert "causes" in prompt
        assert "increases" in prompt

    def test_combined_prompt_truncates_text(self):
        reg = _mock_registry()
        reg.all_domain_names.return_value = ["base"]
        reg.get_predicates.return_value = []
        builder = PromptBuilder(reg)
        long_text = "word " * 2000  # ~10000 chars
        prompt = builder.build_combined_prompt(long_text)
        assert len(prompt) < len(long_text)
