"""Builds extraction prompts from templates + DomainRegistry."""

from __future__ import annotations

import logging

from knowledge_service.ontology.registry import DomainRegistry

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 4000

# ---------------------------------------------------------------------------
# Inline fallback template (used when no file-based template exists)
# ---------------------------------------------------------------------------

_DEFAULT_COMBINED_TEMPLATE = """{context}You are a knowledge extraction system. Extract entities, events, AND relationships from the text below.
Return ONLY a JSON object: {{"entities": [...], "relations": [...]}}

## Step 1: Extract Entities and Events

Each entity/event item must have a knowledge_type field:
- Entity: uri, rdf_type (a schema.org class name like "Person", "Country", "Organization", "Thing" — bare class name, no "schema:" prefix and no other namespace), label, properties (dict), confidence
- Event: subject, occurred_at (YYYY-MM-DD), confidence, properties (dict)

Entity naming rules:
- Use canonical, well-known names: "dopamine" not "the neurotransmitter dopamine"
- Use singular form: "neuron" not "neurons"
- Use lowercase snake_case: "cold_exposure" not "Cold Exposure"
- Be specific: "vitamin_d3" not "vitamin_d" when the text specifies D3
- The uri and label should both use the snake_case form
- The uri must be the entity's own name. NEVER use the literal string "schema" or "schema_<type>" as a uri (or as a subject/object below) — those are not entities.

## Step 2: Extract Relationships Using Those Entities

Each relation item must have a knowledge_type field:
- Claim: subject, predicate, object, object_type, confidence (0.0-0.89)
- Fact: subject, predicate, object, object_type, confidence (0.9-1.0) for verified facts
- Relationship: subject, predicate, object, object_type, confidence

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

    def build_combined_prompt(
        self,
        text: str,
        title: str | None = None,
        source_type: str | None = None,
        entity_hints: list[dict] | None = None,
        domains: list[str] | None = None,
    ) -> str:
        """Build single-pass prompt for combined entity + relation extraction."""
        active_domains = domains or (
            self._registry.all_domain_names() if self._registry else ["base"]
        )

        # Check for domain-specific override first (skip "base" — that's the default)
        template = None
        for domain in active_domains:
            if domain == "base":
                continue
            override = self._registry.get_prompt(f"{domain}_combined") if self._registry else None
            if override:
                template = override
                break
        if template is None:
            template = self._registry.get_prompt("base_combined") if self._registry else None
        if template is None:
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
