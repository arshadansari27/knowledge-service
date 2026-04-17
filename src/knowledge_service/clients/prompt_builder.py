"""Builds extraction prompts from templates + DomainRegistry."""

from __future__ import annotations

import logging

from knowledge_service.ontology.registry import DomainRegistry, PredicateInfo

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 4000

# ---------------------------------------------------------------------------
# Inline fallback templates (used when no file-based template exists)
# ---------------------------------------------------------------------------

_DEFAULT_ENTITY_TEMPLATE = """{context}Extract entities and events from the text below.
Return ONLY a JSON object: {{"items": [...]}}

Each item must have a knowledge_type field. Supported types and required fields:
- Entity: uri, rdf_type (e.g. "schema:Person", "schema:Thing"), label, properties (dict), confidence
- Event: subject, occurred_at (YYYY-MM-DD), confidence, properties (dict)

Entity naming rules:
- Use canonical, well-known names: "dopamine" not "the neurotransmitter dopamine"
- Use singular form: "neuron" not "neurons"
- Use lowercase snake_case: "cold_exposure" not "Cold Exposure"
- Be specific: "vitamin_d3" not "vitamin_d" when the text specifies D3
- The uri and label should both use the snake_case form

Extract every distinct entity and event mentioned. If nothing found, return {{"items": []}}

Example:
Text: "Regular cold water immersion has been shown to increase dopamine levels by up to 250%."
Output: {{"items": [
  {{"knowledge_type": "Entity", "uri": "cold_water_immersion", "rdf_type": "schema:Thing", "label": "cold_water_immersion", "properties": {{}}, "confidence": 0.95}},
  {{"knowledge_type": "Entity", "uri": "dopamine", "rdf_type": "schema:Thing", "label": "dopamine", "properties": {{}}, "confidence": 0.95}}
]}}

Text:
---
{text}
---"""

_DEFAULT_RELATION_TEMPLATE = """{context}Extract relationships and claims from the text below.
Return ONLY a JSON object: {{"items": [...]}}

Known entities: [{entities}]
Prefer these entities as subjects and objects. If the text clearly mentions an entity not in this list, you may use it — follow the entity naming rules below.

Each item must have a knowledge_type field. Supported types and required fields:
- Claim: subject, predicate, object, object_type, confidence (0.0-0.89)
- Fact: subject, predicate, object, object_type, confidence (0.9-1.0) for verified facts
- Relationship: subject, predicate, object, object_type, confidence
- TemporalState: subject, property, value, valid_from (YYYY-MM-DD), valid_until (YYYY-MM-DD), confidence
- Conclusion: concludes (text), derived_from (list of identifiers), inference_method, confidence

Preferred predicates (use these when applicable):
{predicates}
Only invent a new predicate if none of the above fit.

Entity naming rules (for consistency with the entity list above):
- Use canonical, well-known names: "dopamine" not "the neurotransmitter dopamine"
- Use singular form: "neuron" not "neurons"
- Use lowercase snake_case: "cold_exposure" not "Cold Exposure"
- Be specific: "vitamin_d3" not "vitamin_d" when the text specifies D3

For object values, include object_type ("entity" or "literal"):
- "entity": the object is a thing/concept from the entity list above
- "literal": the object is a measurement, description, or date (e.g. "250%", "2024-01-15")

Use Claim for uncertain assertions, Fact for high-confidence verifiable statements.
Extract 3-8 items. If nothing found, return {{"items": []}}

Example:
Text: "Regular cold water immersion has been shown to increase dopamine levels by up to 250%."
Known entities: [cold_water_immersion, dopamine]
Output: {{"items": [
  {{"knowledge_type": "Claim", "subject": "cold_water_immersion", "predicate": "increases", "object": "dopamine", "object_type": "entity", "confidence": 0.75}},
  {{"knowledge_type": "Claim", "subject": "cold_water_immersion", "predicate": "has_property", "object": "250% dopamine increase", "object_type": "literal", "confidence": 0.7}}
]}}

Text:
---
{text}
---"""


_DEFAULT_COMBINED_TEMPLATE = """{context}You are a knowledge extraction system. Extract entities, events, AND relationships from the text below.
Return ONLY a JSON object: {{"entities": [...], "relations": [...]}}

## Step 1: Extract Entities and Events

Each entity/event item must have a knowledge_type field:
- Entity: uri, rdf_type (e.g. "schema:Person", "schema:Thing"), label, properties (dict), confidence
- Event: subject, occurred_at (YYYY-MM-DD), confidence, properties (dict)

Entity naming rules:
- Use canonical, well-known names: "dopamine" not "the neurotransmitter dopamine"
- Use singular form: "neuron" not "neurons"
- Use lowercase snake_case: "cold_exposure" not "Cold Exposure"
- Be specific: "vitamin_d3" not "vitamin_d" when the text specifies D3
- The uri and label should both use the snake_case form

## Step 2: Extract Relationships Using Those Entities

Each relation item must have a knowledge_type field:
- Claim: subject, predicate, object, object_type, confidence (0.0-0.89)
- Fact: subject, predicate, object, object_type, confidence (0.9-1.0) for verified facts
- Relationship: subject, predicate, object, object_type, confidence
- TemporalState: subject, property, value, valid_from (YYYY-MM-DD), valid_until (YYYY-MM-DD), confidence
- Conclusion: concludes (text), derived_from (list of identifiers), inference_method, confidence

Preferred predicates (use these when applicable):
{predicates}
Only invent a new predicate if none of the above fit.

Use entities from Step 1 as subjects and objects. For object values, include object_type ("entity" or "literal"):
- "entity": the object is a thing/concept
- "literal": the object is a measurement, description, or date (e.g. "250%", "2024-01-15")

Use Claim for uncertain assertions, Fact for high-confidence verifiable statements.

If nothing found, return {{"entities": [], "relations": []}}

Text:
---
{text}
---"""


class PromptBuilder:
    """Builds extraction prompts from DomainRegistry templates with inline fallbacks."""

    def __init__(self, registry: DomainRegistry) -> None:
        self._registry = registry

    def build_entity_prompt(
        self,
        text: str,
        title: str | None = None,
        source_type: str | None = None,
        entity_hints: list[dict] | None = None,
    ) -> str:
        """Build entity/event extraction prompt."""
        template = self._registry.get_prompt("base_entities")
        if template is None:
            template = _DEFAULT_ENTITY_TEMPLATE
        context = ""
        if title:
            context += f"Title: {title}\n"
        if source_type:
            context += f"Source type: {source_type}\n"
        if entity_hints:
            context += "\nNLP-detected entities (confirm, correct, or add to these):\n"
            for hint in entity_hints:
                context += f"- {hint['text']} ({hint['label']})\n"
        return template.format(context=context, text=text[:_MAX_TEXT_CHARS])

    def build_relation_prompt(
        self,
        text: str,
        entity_names: list[str],
        predicates: list[PredicateInfo],
        domains: list[str],
        title: str | None = None,
        source_type: str | None = None,
    ) -> str:
        """Build domain-aware relation extraction prompt."""
        # Check for domain-specific override (skip "base" — that's the default template)
        for domain in domains:
            if domain == "base":
                continue
            override = self._registry.get_prompt(f"{domain}_relations")
            if override:
                return override.format(
                    entities=", ".join(entity_names),
                    predicates=", ".join(p.label for p in predicates),
                    text=text[:_MAX_TEXT_CHARS],
                )
        # Use base template
        template = self._registry.get_prompt("base_relations")
        if template is None:
            template = _DEFAULT_RELATION_TEMPLATE
        context = ""
        if title:
            context += f"Title: {title}\n"
        if source_type:
            context += f"Source type: {source_type}\n"
        return template.format(
            context=context,
            entities=", ".join(entity_names),
            predicates=", ".join(p.label for p in predicates),
            text=text[:_MAX_TEXT_CHARS],
        )

    def build_combined_prompt(
        self,
        text: str,
        title: str | None = None,
        source_type: str | None = None,
        entity_hints: list[dict] | None = None,
        domains: list[str] | None = None,
    ) -> str:
        """Build single-pass prompt for combined entity + relation extraction."""
        template = _DEFAULT_COMBINED_TEMPLATE
        context = ""
        if title:
            context += f"Title: {title}\n"
        if source_type:
            context += f"Source type: {source_type}\n"
        if entity_hints:
            context += "\nNLP-detected entities (confirm, correct, or add to these):\n"
            for hint in entity_hints:
                context += f"- {hint['text']} ({hint['label']})\n"

        active_domains = domains or (
            self._registry.get_domains_for_entity_types([]) if self._registry else ["base"]
        )
        predicates_list = self._registry.get_predicates(active_domains) if self._registry else []
        predicates_str = (
            ", ".join(p.label for p in predicates_list)
            if predicates_list
            else (
                "causes, increases, decreases, inhibits, activates, is_a, part_of, located_in, "
                "created_by, depends_on, related_to, contains, precedes, follows, has_property, "
                "used_for, produced_by, associated_with"
            )
        )

        return template.format(
            context=context,
            predicates=predicates_str,
            text=text[:_MAX_TEXT_CHARS],
        )
