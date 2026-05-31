"""RAGClient — calls LLM with assembled retrieval context to generate answers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from knowledge_service._utils import _extract_json
from knowledge_service.clients.base import BaseLLMClient
from knowledge_service.stores.rag import RetrievalContext

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    """Parsed LLM response."""

    answer: str


_MAX_PROMPT_CHARS = 48_000


def _render_triple(t: dict) -> str:
    """Render a triple as a prompt line.

    Verbalized prose when localized labels are present (lever B —
    ``- cold exposure increases dopamine (confidence: 0.70) · source: <url>``),
    else the legacy raw-URI SPO form so callers that don't enrich are unaffected.
    """
    conf = t.get("confidence", "?")
    if t.get("subject_label"):
        line = (
            f"- {t['subject_label']} {t.get('predicate_label', '')} "
            f"{t.get('object_label', '')} (confidence: {conf})"
        )
        src = t.get("source_url")
        if src:
            line += f" · source: {src}"
        return line
    s = t.get("subject", "?")
    p = t.get("predicate", "?")
    o = t.get("object", "?")
    ktype = t.get("knowledge_type", "?")
    trust = t.get("trust_tier", "unknown")
    return f"- [{trust}] {s} -> {p} -> {o} ({ktype}, confidence: {conf})"


def _render_contradictions(context: RetrievalContext) -> list[str]:
    lines = ["## Contradictions Found"]
    for c in context.contradictions:
        s = c.get("subject", "?")
        p = c.get("predicate", "?")
        o = c.get("object", "?")
        conf = c.get("confidence", "?")
        lines.append(f"- {s} -> {p} -> {o} (confidence: {conf})")
    lines.append("")
    return lines


def build_verify_prompt(question: str, draft_answer: str, context: RetrievalContext) -> str:
    """Prompt for the verify pass (lever E): re-check a draft answer against the
    knowledge-graph facts and detected contradictions."""
    sections: list[str] = [
        "You are a fact-checker. Below is a draft answer, followed by knowledge-graph "
        "facts and any detected contradictions.",
        "Verify each claim in the draft against the facts. Correct any claim a fact "
        "contradicts, drop or flag claims the facts do not support, and keep claims "
        "that are consistent. Do not introduce new claims the facts don't support.",
        'Return a JSON object: {"answer": "..."}',
        "",
        "## Draft answer",
        draft_answer,
        "",
    ]
    if context.knowledge_triples:
        sections.append("## Knowledge Graph Facts")
        sections.extend(_render_triple(t) for t in context.knowledge_triples)
        sections.append("")
    if context.contradictions:
        sections.extend(_render_contradictions(context))
    sections.append("## Question")
    sections.append(question)
    return "\n".join(sections)


def build_rag_prompt(question: str, context: RetrievalContext, include_graph: bool = True) -> str:
    """Build the LLM prompt from a question and retrieval context."""
    sections: list[str] = [
        "You are a knowledge assistant. Answer the question using ONLY the context below.",
        "If the context doesn't contain enough information, say so. Do not fabricate.",
        'Return a JSON object: {"answer": "..."}',
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
            # similarity is None for BM25-only hits after the hybrid-search
            # honesty fix (no cosine was computed). Show "n/a" instead of
            # crashing the format spec on NoneType.
            similarity = row.get("similarity")
            sim_str = f"{similarity:.2f}" if isinstance(similarity, (int, float)) else "n/a"
            text = row.get("chunk_text") or row.get("summary") or "No content"
            section = (
                f" [Section: {row.get('section_header')}]" if row.get("section_header") else ""
            )
            line = f'- "{title}" ({source_type}, similarity: {sim_str}){section}: {text}'
            if running_len + len(line) > _MAX_PROMPT_CHARS:
                sections.append("(... additional sources truncated for length ...)")
                break
            sections.append(line)
            running_len += len(line)
        sections.append("")

    # Knowledge triples section (omitted when include_graph is False — the verify
    # pass's first call answers from chunks alone)
    if include_graph and context.knowledge_triples:
        sections.append("## Knowledge Graph Facts")
        running_len += len(sections[-1])
        for t in context.knowledge_triples:
            line = _render_triple(t)
            if running_len + len(line) > _MAX_PROMPT_CHARS:
                sections.append("(... additional triples truncated for length ...)")
                break
            sections.append(line)
            running_len += len(line)
        sections.append("")

    # Contradictions section
    if include_graph and context.contradictions:
        sections.extend(_render_contradictions(context))

    sections.append("## Question")
    sections.append(question)

    return "\n".join(sections)


class RAGClient(BaseLLMClient):
    """Calls the LLM with retrieval context to generate answers."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        super().__init__(base_url, model, api_key, read_timeout=120.0)

    async def _complete(self, prompt: str) -> str:
        """POST a prompt and return the parsed answer string. Raises on transport error."""
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("RAGClient: LLM API returned %s", exc.response.status_code)
            raise
        except httpx.TimeoutException:
            logger.warning("RAGClient: LLM request timed out")
            raise

        raw = response.json()["choices"][0]["message"]["content"]
        parsed = _extract_json(raw)
        if parsed and isinstance(parsed, dict):
            return parsed.get("answer", raw)
        logger.warning("RAGClient: could not parse JSON response, using raw text")
        return raw

    async def answer(self, question: str, context: RetrievalContext) -> RAGAnswer:
        """Generate an answer from the question and retrieval context (direct path)."""
        return RAGAnswer(answer=await self._complete(build_rag_prompt(question, context)))

    async def answer_verified(self, question: str, context: RetrievalContext) -> RAGAnswer:
        """Two-call verify path (lever E): answer from chunks alone, then re-check
        that draft against the graph facts + contradictions and return the revision.

        If the verify call fails, fall back to the draft — a transport error in the
        *checker* must not 502 a request that already has a valid grounded answer.
        """
        draft = await self._complete(build_rag_prompt(question, context, include_graph=False))
        try:
            final = await self._complete(build_verify_prompt(question, draft, context))
        except Exception as exc:
            logger.warning("RAGClient: verify pass failed (%s); returning draft", exc)
            return RAGAnswer(answer=draft)
        return RAGAnswer(answer=final)

    async def answer_auto(
        self, question: str, context: RetrievalContext, answer_mode: str
    ) -> RAGAnswer:
        """Dispatch by answer_mode. ``verify`` runs the two-call path only when the
        graph has something to check (triples or contradictions); with nothing to
        verify against, a second pass would only second-guess a grounded answer, so
        fall back to a single direct call."""
        if answer_mode == "verify" and (context.knowledge_triples or context.contradictions):
            return await self.answer_verified(question, context)
        return await self.answer(question, context)
