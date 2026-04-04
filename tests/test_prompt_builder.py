"""Tests for PromptBuilder — template-based extraction prompt generation."""

from unittest.mock import MagicMock

from knowledge_service.clients.prompt_builder import PromptBuilder, _MAX_TEXT_CHARS
from knowledge_service.ontology.registry import PredicateInfo


def _mock_registry():
    reg = MagicMock()
    reg.get_prompt.return_value = None
    return reg


class TestBuildEntityPrompt:
    def test_returns_string(self):
        reg = _mock_registry()
        builder = PromptBuilder(reg)
        prompt = builder.build_entity_prompt("Some text", title="Test Title")
        assert isinstance(prompt, str)
        assert "Test Title" in prompt
        assert "Some text" in prompt

    def test_uses_override(self):
        reg = _mock_registry()
        reg.get_prompt.return_value = "Custom: {context}{text}"
        builder = PromptBuilder(reg)
        prompt = builder.build_entity_prompt("text here", title="T")
        assert "Custom:" in prompt
        assert "text here" in prompt

    def test_truncates_long_text(self):
        reg = _mock_registry()
        builder = PromptBuilder(reg)
        long_text = "x" * 10000
        prompt = builder.build_entity_prompt(long_text)
        # Only _MAX_TEXT_CHARS of text should appear, plus template overhead
        assert "x" * _MAX_TEXT_CHARS in prompt
        assert "x" * (_MAX_TEXT_CHARS + 1) not in prompt

    def test_includes_source_type(self):
        reg = _mock_registry()
        builder = PromptBuilder(reg)
        prompt = builder.build_entity_prompt("text", source_type="article")
        assert "Source type: article" in prompt

    def test_no_context_when_no_title_or_source(self):
        reg = _mock_registry()
        builder = PromptBuilder(reg)
        prompt = builder.build_entity_prompt("text")
        # Should start directly with "Extract entities"
        assert prompt.startswith("Extract entities")

    def test_queries_registry_for_base_entities(self):
        reg = _mock_registry()
        builder = PromptBuilder(reg)
        builder.build_entity_prompt("text")
        reg.get_prompt.assert_called_with("base_entities")


class TestBuildRelationPrompt:
    def test_includes_entities_and_predicates(self):
        reg = _mock_registry()
        builder = PromptBuilder(reg)
        preds = [
            PredicateInfo(uri="http://x/causes", label="causes", domain="base"),
            PredicateInfo(uri="http://x/increases", label="increases", domain="base"),
        ]
        prompt = builder.build_relation_prompt("text", ["ACME", "Widget"], preds, ["base"])
        assert "ACME" in prompt
        assert "Widget" in prompt
        assert "causes" in prompt
        assert "increases" in prompt

    def test_uses_domain_override(self):
        reg = _mock_registry()

        def side_effect(name):
            if name == "financial_relations":
                return "Custom financial: {entities} {predicates} {text}"
            return None

        reg.get_prompt.side_effect = side_effect
        builder = PromptBuilder(reg)
        preds = [PredicateInfo(uri="http://x/owns", label="owns", domain="financial")]
        prompt = builder.build_relation_prompt("text", ["ACME"], preds, ["financial"])
        assert "Custom financial" in prompt
        assert "ACME" in prompt

    def test_falls_back_to_base_when_no_domain_override(self):
        reg = _mock_registry()
        builder = PromptBuilder(reg)
        preds = [PredicateInfo(uri="http://x/causes", label="causes", domain="base")]
        prompt = builder.build_relation_prompt("text", ["A"], preds, ["base"])
        # Should use the default template which includes "Extract relationships"
        assert "Extract relationships" in prompt

    def test_truncates_long_text(self):
        reg = _mock_registry()
        builder = PromptBuilder(reg)
        long_text = "y" * 10000
        preds = [PredicateInfo(uri="http://x/causes", label="causes", domain="base")]
        prompt = builder.build_relation_prompt(long_text, ["A"], preds, ["base"])
        assert "y" * _MAX_TEXT_CHARS in prompt
        assert "y" * (_MAX_TEXT_CHARS + 1) not in prompt

    def test_includes_context_with_title(self):
        reg = _mock_registry()
        builder = PromptBuilder(reg)
        preds = [PredicateInfo(uri="http://x/causes", label="causes", domain="base")]
        prompt = builder.build_relation_prompt(
            "text", ["A"], preds, ["base"], title="My Title", source_type="paper"
        )
        assert "Title: My Title" in prompt
        assert "Source type: paper" in prompt


class TestBuildCombinedPrompt:
    def test_combined_prompt_includes_entity_and_relation_instructions(self):
        reg = _mock_registry()
        reg.get_domains_for_entity_types.return_value = ["base"]
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
        reg.get_domains_for_entity_types.return_value = ["base"]
        reg.get_predicates.return_value = []
        builder = PromptBuilder(reg)
        hints = [{"text": "dopamine", "label": "CHEMICAL", "wikidata_id": "Q80635"}]
        prompt = builder.build_combined_prompt("text", entity_hints=hints)
        assert "dopamine" in prompt
        assert "CHEMICAL" in prompt

    def test_combined_prompt_includes_predicates(self):
        reg = _mock_registry()
        reg.get_domains_for_entity_types.return_value = ["base"]
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
        reg.get_domains_for_entity_types.return_value = ["base"]
        reg.get_predicates.return_value = []
        builder = PromptBuilder(reg)
        long_text = "word " * 2000  # ~10000 chars
        prompt = builder.build_combined_prompt(long_text)
        assert len(prompt) < len(long_text)


class TestThesisPrompt:
    def test_includes_statement(self):
        reg = _mock_registry()
        builder = PromptBuilder(reg)
        prompt = builder.build_thesis_decomposition_prompt("Cold exposure boosts dopamine")
        assert "Cold exposure boosts dopamine" in prompt

    def test_requests_json_format(self):
        reg = _mock_registry()
        builder = PromptBuilder(reg)
        prompt = builder.build_thesis_decomposition_prompt("test statement")
        assert '{"items": [...]}' in prompt
        assert "subject" in prompt
        assert "predicate" in prompt
