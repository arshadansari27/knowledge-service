"""Coreference resolution phase: deterministic Wikidata-QID merging."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter, ValidationError

from knowledge_service.models import KnowledgeInput
from knowledge_service.nlp import NlpResult
from knowledge_service.ontology.uri import slugify, to_entity_uri

logger = logging.getLogger(__name__)


_KNOWLEDGE_ADAPTER: TypeAdapter[KnowledgeInput] = TypeAdapter(KnowledgeInput)


def _to_dict(item) -> dict:
    """Convert a Pydantic model or dict to a plain dict."""
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return dict(item)


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

    def canonicalize(self, items: list) -> list:
        """Rewrite entity labels in knowledge items to canonical forms.

        Builds alias_map from all groups (case-insensitive), then rewrites
        'subject' and 'object' fields in each item. Re-validates the rewritten
        item back into a Pydantic ``KnowledgeInput`` so downstream
        ``ProcessPhase`` can call ``to_triples()`` on Triple/Event/Entity
        items uniformly. Items that fail re-validation pass through as dicts
        and a warning is logged.
        """
        alias_map: dict[str, str] = {}
        for group in self.groups:
            for alias in group.aliases:
                alias_map[alias.lower()] = group.canonical_label
            alias_map[group.canonical_label.lower()] = group.canonical_label

        rewritten: list = []
        for item in items:
            d = _to_dict(item)
            subject = d.get("subject")
            if subject and isinstance(subject, str):
                d["subject"] = alias_map.get(subject.lower(), subject)
            obj = d.get("object")
            if obj and isinstance(obj, str):
                d["object"] = alias_map.get(obj.lower(), obj)
            label = d.get("label")
            if label and isinstance(label, str):
                d["label"] = alias_map.get(label.lower(), label)
            uri = d.get("uri")
            if uri and isinstance(uri, str):
                d["uri"] = alias_map.get(uri.lower(), uri)
            try:
                rewritten.append(_KNOWLEDGE_ADAPTER.validate_python(d))
            except ValidationError as exc:
                logger.warning(
                    "Coref canonicalize: re-validation failed (keys=%s): %s",
                    sorted(d.keys()),
                    exc.errors()[:2],
                )
                rewritten.append(d)
        return rewritten


class CoreferencePhase:
    """Deterministic coreference resolution via shared Wikidata QID.

    Entities surfaced by the NLP pre-pass that share a Wikidata QID are merged
    into a single EntityGroup — no LLM call needed. Prior versions ran a
    second LLM-grouping pass over the remaining unlinked entities; that path
    was dropped because it no-op'd silently on LLM failure, depended on spaCy
    being healthy, and produced low-precision groupings.
    """

    def __init__(self, pg_pool: Any) -> None:
        self._pg_pool = pg_pool

    async def run(
        self,
        knowledge_items: list[dict],  # noqa: ARG002 — shape retained for worker stability
        nlp_results: list[NlpResult],
    ) -> CoreferenceResult:
        """Group entities that share a Wikidata QID and persist aliases."""
        result = CoreferenceResult()

        qid_to_labels: dict[str, list[str]] = {}
        for nlp_result in nlp_results:
            for entity in nlp_result.entities:
                if not entity.wikidata_id:
                    continue
                qid_to_labels.setdefault(entity.wikidata_id, [])
                label = slugify(entity.text)
                if label not in qid_to_labels[entity.wikidata_id]:
                    qid_to_labels[entity.wikidata_id].append(label)

        for qid, labels in qid_to_labels.items():
            if not labels:
                continue
            canonical_label = labels[0]
            aliases = labels[1:]
            result.groups.append(
                EntityGroup(
                    canonical_label=canonical_label,
                    canonical_uri=to_entity_uri(canonical_label),
                    aliases=aliases,
                    wikidata_id=qid,
                )
            )

        await self._store_aliases(result.groups)
        return result

    async def _store_aliases(self, groups: list[EntityGroup]) -> None:
        """Persist alias → canonical URI mappings to the entity_aliases table."""
        if not groups or self._pg_pool is None:
            return

        rows: list[tuple[str, str, str]] = []
        for group in groups:
            for alias in group.aliases:
                rows.append((alias, group.canonical_uri, "spacy_linking"))
            rows.append((group.canonical_label, group.canonical_uri, "spacy_linking"))

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
