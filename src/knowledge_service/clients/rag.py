"""RAGClient — calls LLM with assembled retrieval context to generate answers."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from knowledge_service.stores.rag import RetrievalContext

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    """Parsed LLM response."""

    answer: str
    source_urls_cited: list[str] = field(default_factory=list)


def build_rag_prompt(question: str, context: RetrievalContext) -> str:
    """Build the LLM prompt from a question and retrieval context."""
    sections: list[str] = [
        "You are a knowledge assistant. Answer the question using ONLY the context below.",
        "If the context doesn't contain enough information, say so. Do not fabricate.",
        'Return a JSON object: {"answer": "...", "source_urls_cited": ["..."]}',
        "",
    ]

    # Content section
    if context.content_results:
        sections.append("## Relevant Content")
        for row in context.content_results:
            title = row.get("title", "Untitled")
            source_type = row.get("source_type", "unknown")
            similarity = row.get("similarity", 0.0)
            text = row.get("chunk_text") or row.get("summary") or "No content"
            sections.append(f'- "{title}" ({source_type}, similarity: {similarity:.2f}): {text}')
        sections.append("")

    # Knowledge triples section
    if context.knowledge_triples:
        sections.append("## Knowledge Graph Facts")
        for t in context.knowledge_triples:
            s = t.get("subject", "?")
            p = t.get("predicate", "?")
            o = t.get("object", "?")
            ktype = t.get("knowledge_type", "?")
            conf = t.get("confidence", "?")
            sections.append(f"- {s} -> {p} -> {o} ({ktype}, confidence: {conf})")
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


class RAGClient:
    """Calls the LLM with retrieval context to generate answers."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self._model = model
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )

    async def answer(self, question: str, context: RetrievalContext) -> RAGAnswer:
        """Generate an answer from the question and retrieval context."""
        prompt = build_rag_prompt(question, context)

        response = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()

        raw = response.json()["choices"][0]["message"]["content"]

        # Strip markdown fences (same as ExtractionClient)
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
        stripped = re.sub(r"\n?```\s*$", "", stripped)

        try:
            parsed = json.loads(stripped)
            return RAGAnswer(
                answer=parsed.get("answer", stripped),
                source_urls_cited=parsed.get("source_urls_cited", []),
            )
        except json.JSONDecodeError:
            logger.warning("RAGClient: could not parse JSON response, using raw text")
            return RAGAnswer(answer=raw, source_urls_cited=[])

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if not self._client.is_closed:
            await self._client.aclose()
