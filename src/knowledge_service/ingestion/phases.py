"""Ingestion phases: embed, extract, process."""

import logging
from dataclasses import dataclass, field
from typing import Any

from knowledge_service.ingestion.pipeline import IngestContext, ingest_triple
from knowledge_service.ontology.uri import is_uri

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 32


@dataclass
class PhaseResult:
    """Aggregated results from all phases."""

    triples_created: int = 0
    entities_resolved: int = 0
    chunks_failed: int = 0
    chunk_id_map: dict = field(default_factory=dict)
    knowledge_items: list = field(default_factory=list)
    chunk_ids_for_items: list = field(default_factory=list)
    extractor: str = "api"


class EmbedPhase:
    """Phase 1: Embed chunks and store in content table."""

    def __init__(self, embedding_client: Any, content_store: Any):
        self._embedding_client = embedding_client
        self._content_store = content_store

    async def run(
        self,
        content_id: str,
        chunk_records: list[dict],
    ) -> dict[int, str]:
        """Embed all chunks and insert into content table.

        Returns chunk_id_map: {chunk_index: chunk_uuid}.
        """
        texts = [c["chunk_text"] for c in chunk_records]
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            batch_embeddings = await self._embedding_client.embed_batch(batch)
            embeddings.extend(batch_embeddings)

        for rec, emb in zip(chunk_records, embeddings):
            rec["embedding"] = emb

        await self._content_store.delete_chunks(content_id)
        chunk_id_pairs = await self._content_store.insert_chunks(content_id, chunk_records)
        return dict(chunk_id_pairs) if chunk_id_pairs else {}


class ExtractPhase:
    """Phase 2: Extract knowledge items from chunks via LLM."""

    def __init__(self, extraction_client: Any):
        self._extraction_client = extraction_client

    async def run(
        self,
        chunk_records: list[dict],
        chunk_id_map: dict[int, str],
        title: str | None = None,
        source_type: str | None = None,
        nlp_hints: list | None = None,
    ) -> tuple[list[dict], list[str | None], int]:
        """Extract knowledge from chunks.

        Args:
            nlp_hints: Optional list of NlpResult objects from the NLP pre-pass.
                       When provided, entity hints are forwarded to the LLM extraction
                       to improve entity recognition.

        Returns (knowledge_items, chunk_ids_for_items, chunks_failed).
        """
        knowledge: list[dict] = []
        chunk_ids: list[str | None] = []
        chunks_failed = 0

        # Build a lookup from chunk_index → NlpResult for hint injection
        hint_map: dict[int, Any] = {}
        if nlp_hints:
            for hint in nlp_hints:
                hint_map[hint.chunk_index] = hint

        for chunk in chunk_records:
            chunk_index = chunk["chunk_index"]
            cid = chunk_id_map.get(chunk_index)

            # Build entity_hints from NLP results for this chunk
            entity_hints: list[dict] | None = None
            nlp_result = hint_map.get(chunk_index)
            if nlp_result and nlp_result.entities:
                entity_hints = [
                    {
                        "text": e.text,
                        "label": e.label,
                        "wikidata_id": e.wikidata_id,
                    }
                    for e in nlp_result.entities
                ]

            try:
                items = await self._extraction_client.extract(
                    chunk["chunk_text"],
                    title=title,
                    source_type=source_type,
                    entity_hints=entity_hints,
                )
            except Exception:
                logger.exception(
                    "ExtractPhase: chunk %d extraction raised an exception (title=%s)",
                    chunk_index,
                    title,
                )
                chunks_failed += 1
                continue
            if items is None:
                logger.warning(
                    "ExtractPhase: chunk %d extraction returned None — LLM call failed "
                    "(title=%s, chunk_len=%d)",
                    chunk_index,
                    title,
                    len(chunk["chunk_text"]),
                )
                chunks_failed += 1
                continue
            for item in items:
                knowledge.append(item)
                chunk_ids.append(cid)

            # Add fallback EntityInput for NLP-detected entities the LLM missed
            if nlp_result and nlp_result.entities and items is not None:
                from knowledge_service.config import settings  # noqa: PLC0415
                from knowledge_service.models import EntityInput  # noqa: PLC0415

                llm_labels = set()
                for item in items:
                    if hasattr(item, "label"):
                        llm_labels.add(item.label.lower())
                    if hasattr(item, "subject"):
                        llm_labels.add(item.subject.lower())
                    elif isinstance(item, dict):
                        for key in ("label", "subject", "uri"):
                            val = item.get(key)
                            if val:
                                llm_labels.add(val.lower())

                for ent in nlp_result.entities:
                    if ent.text.lower() not in llm_labels:
                        fallback = EntityInput(
                            uri=ent.text,
                            rdf_type=f"schema:{ent.label}" if ent.label else "schema:Thing",
                            label=ent.text,
                            confidence=settings.nlp_entity_confidence,
                        )
                        knowledge.append(fallback)
                        chunk_ids.append(cid)

        return knowledge, chunk_ids, chunks_failed


class ProcessPhase:
    """Phase 3: Resolve entities, expand to triples, ingest."""

    def __init__(self, stores: Any, entity_store: Any | None = None, engine: Any | None = None):
        self._stores = stores
        self._entity_store = entity_store
        self._engine = engine

    async def run(
        self,
        knowledge_items: list[dict],
        chunk_ids_for_items: list[str | None],
        source_url: str,
        source_type: str,
        extractor: str,
        graph: str,
    ) -> tuple[int, int]:
        """Process all knowledge items into triples.

        Returns (triples_created, entities_resolved).
        """
        triples_created = 0
        entities_resolved = 0

        for i, item in enumerate(knowledge_items):
            cid = chunk_ids_for_items[i] if i < len(chunk_ids_for_items) else None
            ctx = IngestContext(
                source_url=source_url,
                source_type=source_type,
                extractor=extractor,
                graph=graph,
                chunk_id=cid,
            )

            # Each knowledge item should have a to_triples() method (new model)
            # or be a raw triple dict
            if hasattr(item, "to_triples"):
                triples = item.to_triples()
            elif isinstance(item, dict) and "subject" in item and "predicate" in item:
                triples = [item]
            else:
                logger.warning("Skipping unrecognized knowledge item: %s", type(item))
                continue

            for triple in triples:
                # Resolve entities via embeddings (if entity_store available)
                if self._entity_store is not None:
                    triple["subject"] = await self._entity_store.resolve_entity(
                        triple["subject"], rdf_type=triple.get("rdf_type")
                    )
                    triple["predicate"] = await self._entity_store.resolve_predicate(
                        triple["predicate"]
                    )
                    if triple.get("object_type") == "entity" or is_uri(triple.get("object", "")):
                        triple["object"] = await self._entity_store.resolve_entity(triple["object"])
                    entities_resolved += 1

                result = await ingest_triple(triple, self._stores, ctx, engine=self._engine)
                if result.is_new:
                    triples_created += 1

        return triples_created, entities_resolved
