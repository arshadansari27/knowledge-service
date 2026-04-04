"""NLP-based chunk scoring and filtering to reduce LLM calls during extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge_service.nlp import NlpResult

# Sections that should always be skipped (case-insensitive match)
_SKIP_SECTIONS = re.compile(
    r"^(references|bibliography|acknowledgements?|table\s+of\s+contents"
    r"|appendix(\s+\w)?|index|about\s+the\s+author|works\s+cited"
    r"|further\s+reading|endnotes|footnotes)$",
    re.IGNORECASE,
)

# First-line detection for sections without proper headers
_SKIP_FIRST_LINE = re.compile(
    r"^(references|bibliography|acknowledgements?|table\s+of\s+contents)\s*$",
    re.IGNORECASE,
)

# Citation patterns: [1], (Author, 2024), (Author et al., 2024)
_CITATION_RE = re.compile(r"\[\d+\]|\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s*\d{4}\)")

# Bare URL pattern
_URL_RE = re.compile(r"https?://\S+")

# Numeric-heavy line (>60% digits/punctuation)
_NUMERIC_LINE_RE = re.compile(r"^[\d\s.,;:|\-/]+$")


def score_chunk(chunk: dict, nlp_result: NlpResult | None) -> float:
    """Score a chunk 0.0-1.0 for extraction value using cheap NLP signals.

    Returns 0.0 for chunks that should always be skipped (references, etc.).
    """
    text = chunk.get("chunk_text", "")
    section = chunk.get("section_header") or ""

    # --- Instant skip: known low-value sections ---
    for part in section.split(" > "):
        if _SKIP_SECTIONS.match(part.strip()):
            return 0.0

    first_line = text.split("\n", 1)[0].strip().rstrip(":")
    if _SKIP_FIRST_LINE.match(first_line):
        return 0.0

    # --- Signal 1: NER entity density (weight 0.4) ---
    entity_count = len(nlp_result.entities) if nlp_result else 0
    text_len = max(len(text), 1)
    entity_density = min(entity_count / max(text_len / 500, 1), 1.0)

    # --- Signal 2: Sentence count (weight 0.2) ---
    sentence_count = (
        nlp_result.sentence_count
        if nlp_result
        else text.count(". ") + text.count(".\n") + (1 if text.rstrip().endswith(".") else 0)
    )
    sentence_score = min(max(sentence_count - 1, 0) / 4.0, 1.0)

    # --- Signal 3: Boilerplate ratio (weight 0.2) ---
    citations = len(_CITATION_RE.findall(text))
    urls = len(_URL_RE.findall(text))
    lines = text.strip().split("\n")
    numeric_lines = sum(1 for line in lines if _NUMERIC_LINE_RE.match(line.strip())) if lines else 0
    total_tokens = max(len(text.split()), 1)
    # Citations count 3x: a chunk with 1/3 citations (33%) should register as nearly all boilerplate
    boilerplate_items = citations * 3 + urls * 3 + numeric_lines * 3
    boilerplate_ratio = min(boilerplate_items / total_tokens, 1.0)
    boilerplate_score = 1.0 - boilerplate_ratio

    # --- Signal 4: Lexical diversity (weight 0.2) ---
    words = text.lower().split()
    if len(words) > 5:
        diversity = len(set(words)) / len(words)
        diversity_score = min(diversity / 0.7, 1.0)
    else:
        # Very short chunks get a near-zero diversity score
        diversity_score = 0.1

    # Sentence score is penalized when boilerplate is high — cited lists look like
    # well-structured prose but contain no extractable knowledge.
    adjusted_sentence_score = sentence_score * boilerplate_score

    # --- Weighted sum ---
    score = (
        0.4 * entity_density
        + 0.2 * adjusted_sentence_score
        + 0.2 * boilerplate_score
        + 0.2 * diversity_score
    )
    return max(0.0, min(1.0, score))
