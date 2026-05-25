"""Ingestion phases: embed, extract, process."""

import logging
from typing import Any

from knowledge_service._utils import is_object_entity
from knowledge_service.config import settings
from knowledge_service.ingestion.pipeline import IngestContext, ingest_triple
from knowledge_service.ontology.uri import is_uri, to_entity_uri

logger = logging.getLogger(__name__)


class EmbedPhase:
    """Phase 1: Embed chunks and store in content table.

    ``batch_size`` controls the size of each embed_batch() call. Defaults to
    ``settings.embed_batch_size`` (``EMBED_BATCH_SIZE`` env var, default 20)
    so operators tuning the env var get the change without a redeploy of
    code-level constants.
    """

    def __init__(
        self,
        embedding_client: Any,
        content_store: Any,
        batch_size: int | None = None,
    ):
        self._embedding_client = embedding_client
        self._content_store = content_store
        self._batch_size = batch_size if batch_size is not None else settings.embed_batch_size

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
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_embeddings = await self._embedding_client.embed_batch(batch)
            embeddings.extend(batch_embeddings)

        for rec, emb in zip(chunk_records, embeddings):
            rec["embedding"] = emb

        chunk_id_pairs = await self._content_store.replace_chunks(content_id, chunk_records)
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
        domains: list[str] | None = None,
    ) -> tuple[list[dict], list[str | None], int, int]:
        """Extract knowledge from chunks.

        ``domains`` scopes the LLM prompt's predicate list (and prompt
        override, when one exists). Without it, ``PromptBuilder`` falls back
        to every registered domain, which dilutes the prompt for domain-tagged
        ingests.

        Returns (knowledge_items, chunk_ids_for_items, chunks_failed, items_rejected).
        """
        knowledge: list[dict] = []
        chunk_ids: list[str | None] = []
        chunks_failed = 0
        items_rejected = 0

        # Build a lookup from chunk_index → NlpResult for hint injection
        hint_map: dict[int, Any] = {}
        if nlp_hints:
            for hint in nlp_hints:
                hint_map[hint.chunk_index] = hint

        for chunk in chunk_records:
            chunk_index = chunk["chunk_index"]
            cid = chunk_id_map.get(chunk_index)
            nlp_result = hint_map.get(chunk_index)

            entity_hints: list[dict] | None = None
            if nlp_result and nlp_result.entities:
                entity_hints = [
                    {
                        "text": e.text,
                        "label": e.label,
                        "wikidata_id": e.wikidata_id,
                    }
                    for e in nlp_result.entities
                ]

            items, rejected = await self._extraction_client.extract_with_stats(
                chunk["chunk_text"],
                title=title,
                source_type=source_type,
                entity_hints=entity_hints,
                domains=domains,
            )
            items_rejected += rejected
            if items is None:
                chunks_failed += 1
                continue
            for item in items:
                knowledge.append(item)
                chunk_ids.append(cid)

            # Add fallback EntityInput for NLP-detected entities the LLM missed
            if nlp_result and nlp_result.entities and items is not None:
                self._emit_ner_missed(nlp_result, items, cid, knowledge, chunk_ids)

        return knowledge, chunk_ids, chunks_failed, items_rejected

    @staticmethod
    def _emit_ner_missed(
        nlp_result: Any, items: list, cid: str | None, knowledge: list, chunk_ids: list
    ) -> None:
        """Emit NER entities that the LLM missed as fallback items.

        Filters out spaCy NER hits that produce junk triples in production:
        URL-shaped text (every page header gets one), and the numeric/quantity
        labels (``CARDINAL``, ``MONEY``, ``PERCENT``, ``QUANTITY``,
        ``ORDINAL``, ``DATE``, ``TIME``) which are values, not entities.
        Remaining spaCy labels are mapped to schema.org canonical names so
        ``ORG`` doesn't bifurcate against the LLM's ``Organization``.
        """
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
            text = (ent.text or "").strip()
            if not text or text.lower() in llm_labels:
                continue
            if _looks_like_url(text):
                continue
            mapped = _SPACY_LABEL_TO_SCHEMA.get(ent.label, ent.label)
            if mapped is None:
                continue
            fallback = EntityInput(
                uri=text,
                rdf_type=f"schema:{mapped}" if mapped else "schema:Thing",
                label=text,
                confidence=settings.nlp_entity_confidence,
            )
            knowledge.append(fallback)
            chunk_ids.append(cid)


# spaCy ``ner`` model uses UPPERCASE labels (``ORG``, ``PERSON``, ``GPE``,
# ``WORK_OF_ART``, ``NORP``, ``PRODUCT``, ``EVENT``, ``LAW``, ``LANGUAGE``,
# ``FAC``). Map them to schema.org canonical types so they don't bifurcate
# against the LLM's emitted ``Organization`` / ``Person`` / ``Place`` /
# ``CreativeWork``. Labels mapped to ``None`` are dropped — those are
# numeric/quantity classes that describe values, not entities.
_SPACY_LABEL_TO_SCHEMA: dict[str, str | None] = {
    "ORG": "Organization",
    "PERSON": "Person",
    "GPE": "Place",
    "LOC": "Place",
    "FAC": "Place",
    "NORP": "Organization",
    "PRODUCT": "Product",
    "WORK_OF_ART": "CreativeWork",
    "EVENT": "Event",
    "LAW": "Legislation",
    "LANGUAGE": "Language",
    "CARDINAL": None,
    "ORDINAL": None,
    "QUANTITY": None,
    "PERCENT": None,
    "MONEY": None,
    "DATE": None,
    "TIME": None,
}


def _looks_like_url(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("http://", "https://", "www.", "ftp://"))


class ProcessPhase:
    """Phase 3: Resolve entities, expand to triples, ingest."""

    def __init__(
        self,
        stores: Any,
        entity_store: Any | None = None,
        engine: Any | None = None,
        drainer: Any | None = None,
    ):
        self._stores = stores
        self._entity_store = entity_store
        self._engine = engine
        self._drainer = drainer

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
                keys = sorted(item.keys()) if isinstance(item, dict) else None
                logger.warning(
                    "Skipping unrecognized knowledge item: type=%s keys=%s",
                    type(item).__name__,
                    keys,
                )
                continue

            for triple in triples:
                # Normalize object to entity URI when it represents an entity
                # reference (not a literal value). This is essential for inference
                # rules that need URI objects to chain triples.
                if is_object_entity(triple):
                    triple["object"] = to_entity_uri(triple["object"])

                # Resolve entities via embeddings (if entity_store available)
                if self._entity_store is not None:
                    triple["subject"] = await self._entity_store.resolve_entity(
                        triple["subject"], rdf_type=triple.get("rdf_type")
                    )
                    triple["predicate"] = await self._entity_store.resolve_predicate(
                        triple["predicate"]
                    )
                    if is_uri(triple.get("object", "")):
                        triple["object"] = await self._entity_store.resolve_entity(triple["object"])
                    entities_resolved += 1

                result = await ingest_triple(
                    triple, self._stores, ctx, engine=self._engine, drainer=self._drainer
                )
                if result.is_new:
                    triples_created += 1

        return triples_created, entities_resolved
