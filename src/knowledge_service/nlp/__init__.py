"""NLP phase: spaCy NER + entity linking for ingestion pre-processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NlpEntity:
    text: str
    label: str
    start_char: int
    end_char: int
    wikidata_id: str | None = None


@dataclass
class NlpResult:
    chunk_index: int
    entities: list[NlpEntity] = field(default_factory=list)
    sentence_count: int = 0


class NlpPhase:
    """Phase: Run spaCy NER + entity linking on each chunk."""

    def __init__(self, nlp: Any) -> None:
        self._nlp = nlp

    async def run(self, chunk_records: list[dict]) -> list[NlpResult]:
        """Process each chunk through spaCy and return NlpResult per chunk."""
        results: list[NlpResult] = []

        for chunk in chunk_records:
            chunk_index = chunk.get("chunk_index", 0)
            text = chunk.get("chunk_text", "")

            try:
                doc = self._nlp(text)
            except Exception:
                logger.exception("spaCy failed on chunk %d", chunk_index)
                results.append(NlpResult(chunk_index=chunk_index))
                continue

            entities: list[NlpEntity] = []
            for ent in doc.ents:
                wikidata_id: str | None = None
                try:
                    linked = ent._.linkedEntities
                    if linked:
                        wikidata_id = linked[0].get_id()
                except Exception:
                    pass

                entities.append(
                    NlpEntity(
                        text=ent.text,
                        label=ent.label_,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        wikidata_id=wikidata_id,
                    )
                )

            sentence_count = sum(1 for _ in doc.sents)

            results.append(
                NlpResult(
                    chunk_index=chunk_index,
                    entities=entities,
                    sentence_count=sentence_count,
                )
            )

        return results
