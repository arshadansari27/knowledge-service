"""Coreference resolution phase: tier-1 Wikidata QID merging + tier-2 LLM grouping."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from knowledge_service.nlp import NlpResult
from knowledge_service.ontology.uri import slugify, to_entity_uri

logger = logging.getLogger(__name__)

_COREFERENCE_PROMPT_TEMPLATE = """\
Given these entity mentions extracted from a document, group those
that refer to the same real-world entity or concept. Return ONLY a JSON object:
{{"items": [...]}}

Each item: {{"canonical": "snake_case_label", "aliases": ["alias1", "alias2"]}}

Only group entities you are confident refer to the same thing.
If an entity doesn't group with anything, omit it.

Entities: {entity_list}"""


@dataclass
class EntityGroup:
    canonical_label: str
    canonical_uri: str  # via to_entity_uri()
    aliases: list[str] = field(default_factory=list)
    wikidata_id: str | None = None
    rdf_type: str | None = None


@dataclass
class CoreferenceResult:
    groups: list[EntityGroup] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)

    def canonicalize(self, items: list[dict]) -> list[dict]:
        """Rewrite entity labels in knowledge items to canonical forms.

        Builds alias_map from all groups (case-insensitive), then rewrites
        'subject' and 'object' fields in each item dict.
        """
        alias_map: dict[str, str] = {}
        for group in self.groups:
            for alias in group.aliases:
                alias_map[alias.lower()] = group.canonical_label
            # Also map the canonical label to itself (idempotent)
            alias_map[group.canonical_label.lower()] = group.canonical_label

        rewritten = []
        for item in items:
            item = dict(item)
            subject = item.get("subject")
            if subject and isinstance(subject, str):
                item["subject"] = alias_map.get(subject.lower(), subject)
            obj = item.get("object")
            if obj and isinstance(obj, str):
                item["object"] = alias_map.get(obj.lower(), obj)
            rewritten.append(item)
        return rewritten


class CoreferencePhase:
    """Two-tier coreference resolution for knowledge items.

    Tier 1: Entities sharing a Wikidata QID (from NlpResults) are merged
            deterministically — no LLM call needed.
    Tier 2: Remaining unlinked entities are sent to the LLM in a single call
            for grouping by semantic similarity.
    """

    def __init__(self, extraction_client: Any, pg_pool: Any) -> None:
        self._extraction_client = extraction_client
        self._pg_pool = pg_pool

    async def run(
        self,
        knowledge_items: list[dict],
        nlp_results: list[NlpResult],
    ) -> CoreferenceResult:
        """Run coreference resolution and return a CoreferenceResult."""
        result = CoreferenceResult()

        # --- Tier 1: Group by Wikidata QID ---
        qid_to_labels: dict[str, list[str]] = {}
        qid_to_rdf_type: dict[str, str | None] = {}

        for nlp_result in nlp_results:
            for entity in nlp_result.entities:
                if entity.wikidata_id:
                    qid_to_labels.setdefault(entity.wikidata_id, [])
                    label = slugify(entity.text)
                    if label not in qid_to_labels[entity.wikidata_id]:
                        qid_to_labels[entity.wikidata_id].append(label)

        # Collect all linked labels to determine what is "linked"
        linked_labels: set[str] = set()
        for qid, labels in qid_to_labels.items():
            if not labels:
                continue
            canonical_label = labels[0]
            aliases = labels[1:]
            group = EntityGroup(
                canonical_label=canonical_label,
                canonical_uri=to_entity_uri(canonical_label),
                aliases=aliases,
                wikidata_id=qid,
                rdf_type=qid_to_rdf_type.get(qid),
            )
            result.groups.append(group)
            linked_labels.add(canonical_label)
            linked_labels.update(aliases)

        # --- Collect all entity labels from knowledge items ---
        all_item_labels: set[str] = set()
        for item in knowledge_items:
            subject = item.get("subject")
            obj = item.get("object")
            if subject and isinstance(subject, str):
                all_item_labels.add(slugify(subject))
            if obj and isinstance(obj, str) and item.get("object_type") == "entity":
                all_item_labels.add(slugify(obj))

        # --- Tier 2: LLM coreference for unlinked entities ---
        unlinked = sorted(all_item_labels - linked_labels)

        if unlinked:
            entity_list = ", ".join(unlinked)
            prompt = _COREFERENCE_PROMPT_TEMPLATE.format(entity_list=entity_list)

            llm_items = await self._extraction_client._call_llm(prompt)

            if llm_items:
                llm_canonical_labels: set[str] = set()
                for llm_item in llm_items:
                    canonical = llm_item.get("canonical", "").strip()
                    aliases = [a.strip() for a in llm_item.get("aliases", []) if a.strip()]
                    if not canonical:
                        continue
                    canonical_slug = slugify(canonical)
                    alias_slugs = [slugify(a) for a in aliases]
                    group = EntityGroup(
                        canonical_label=canonical_slug,
                        canonical_uri=to_entity_uri(canonical_slug),
                        aliases=alias_slugs,
                    )
                    result.groups.append(group)
                    llm_canonical_labels.add(canonical_slug)
                    llm_canonical_labels.update(alias_slugs)

                # Unmapped = unlinked entities not covered by LLM grouping
                result.unmapped = sorted(set(unlinked) - llm_canonical_labels)
            else:
                result.unmapped = unlinked

        # --- Persist aliases to entity_aliases table ---
        await self._store_aliases(result.groups)

        return result

    async def _store_aliases(self, groups: list[EntityGroup]) -> None:
        """Persist alias → canonical URI mappings to the entity_aliases table."""
        if not groups or self._pg_pool is None:
            return

        source = "spacy_linking"
        rows: list[tuple[str, str, str]] = []
        for group in groups:
            source_tag = "spacy_linking" if group.wikidata_id else "llm_coreference"
            for alias in group.aliases:
                rows.append((alias, group.canonical_uri, source_tag))
            # Also store canonical → canonical_uri mapping
            rows.append((group.canonical_label, group.canonical_uri, source_tag))

        if not rows:
            return

        try:
            async with self._pg_pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO entity_aliases (alias, canonical, source)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (alias) DO UPDATE SET canonical = EXCLUDED.canonical,
                                                       source = EXCLUDED.source
                    """,
                    rows,
                )
        except Exception:
            logger.exception("CoreferencePhase: failed to store aliases")
            _ = source  # suppress unused variable warning
