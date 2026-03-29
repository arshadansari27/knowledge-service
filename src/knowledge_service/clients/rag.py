"""RAGClient — calls LLM with assembled retrieval context to generate answers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from knowledge_service._utils import _extract_json
from knowledge_service.clients.base import BaseLLMClient
from knowledge_service.stores.rag import RetrievalContext

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    """Parsed LLM response."""

    answer: str
    source_urls_cited: list[str] = field(default_factory=list)


_MAX_PROMPT_CHARS = 48_000


def build_rag_prompt(question: str, context: RetrievalContext) -> str:
    """Build the LLM prompt from a question and retrieval context."""
    sections: list[str] = [
        "You are a knowledge assistant. Answer the question using ONLY the context below.",
        "If the context doesn't contain enough information, say so. Do not fabricate.",
        'Return a JSON object: {"answer": "...", "source_urls_cited": ["..."]}',
        "",
    ]
    running_len = sum(len(s) for s in sections)

    # Content section
    if context.content_results:
        sections.append("## Relevant Content")
        running_len += len(sections[-1])
        for row in context.content_results:
            title = row.get("title", "Untitled")
            source_type = row.get("source_type", "unknown")
            similarity = row.get("similarity", 0.0)
            text = row.get("chunk_text") or row.get("summary") or "No content"
            section = (
                f" [Section: {row.get('section_header')}]" if row.get("section_header") else ""
            )
            line = f'- "{title}" ({source_type}, similarity: {similarity:.2f}){section}: {text}'
            if running_len + len(line) > _MAX_PROMPT_CHARS:
                sections.append("(... additional sources truncated for length ...)")
                break
            sections.append(line)
            running_len += len(line)
        sections.append("")

    # Knowledge triples section
    if context.knowledge_triples:
        sections.append("## Knowledge Graph Facts")
        running_len += len(sections[-1])
        for t in context.knowledge_triples:
            s = t.get("subject", "?")
            p = t.get("predicate", "?")
            o = t.get("object", "?")
            ktype = t.get("knowledge_type", "?")
            conf = t.get("confidence", "?")
            trust = t.get("trust_tier", "unknown")
            line = f"- [{trust}] {s} -> {p} -> {o} ({ktype}, confidence: {conf})"
            if running_len + len(line) > _MAX_PROMPT_CHARS:
                sections.append("(... additional triples truncated for length ...)")
                break
            sections.append(line)
            running_len += len(line)
        sections.append("")

    # Contradictions section
    if context.contradictions:
        sections.append("## Contradictions Found")
        for c in context.contradictions:
            s = c.get("subject", "?")
            p = c.get("predicate", "?")
            o = c.get("object", "?")
            conf = c.get("confidence", "?")
            sections.append(f"- {s} -> {p} -> {o} (confidence: {conf})")
        sections.append("")

    sections.append("## Question")
    sections.append(question)

    return "\n".join(sections)


class RAGClient(BaseLLMClient):
    """Calls the LLM with retrieval context to generate answers."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        super().__init__(base_url, model, api_key, read_timeout=120.0)

    async def answer(self, question: str, context: RetrievalContext) -> RAGAnswer:
        """Generate an answer from the question and retrieval context."""
        prompt = build_rag_prompt(question, context)

        response = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()

        raw = response.json()["choices"][0]["message"]["content"]
        parsed = _extract_json(raw)

        if parsed and isinstance(parsed, dict):
            return RAGAnswer(
                answer=parsed.get("answer", raw),
                source_urls_cited=parsed.get("source_urls_cited", []),
            )
        logger.warning("RAGClient: could not parse JSON response, using raw text")
        return RAGAnswer(answer=raw, source_urls_cited=[])
