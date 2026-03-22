"""Markdown-aware text chunking with section header propagation.

Detects markdown content and splits on heading boundaries, preserving the
heading hierarchy as ``section_header`` (e.g. "Title > Section A").  Plain
text falls back to ``RecursiveCharacterTextSplitter`` with improved separators.
"""

from __future__ import annotations

import re

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Heading levels fed to MarkdownHeaderTextSplitter.
_MD_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# Regex: a markdown heading at the start of a line (levels 1-3).
# Negative lookbehind for ``` to avoid matching headings inside code fences.
_HEADING_RE = re.compile(r"^#{1,3}\s", re.MULTILINE)

# Simple code-fence detector: strip fenced blocks before checking for headings.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _is_markdown(text: str) -> bool:
    """Return True if *text* looks like markdown (heading in first 2000 chars)."""
    sample = text[:2000]
    # Strip code fences so ``# comment`` inside fences doesn't match.
    stripped = _CODE_FENCE_RE.sub("", sample)
    return bool(_HEADING_RE.search(stripped))


def _build_section_header(metadata: dict[str, str]) -> str | None:
    """Build a hierarchical section header from heading metadata.

    Example: ``{"h1": "Title", "h2": "Methods"}`` → ``"Title > Methods"``.
    """
    parts = []
    for key in ("h1", "h2", "h3"):
        if key in metadata and metadata[key]:
            parts.append(metadata[key])
    return " > ".join(parts) if parts else None


def chunk_text(
    text: str,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """Split *text* into chunks, returning dicts with keys:

    - ``chunk_text``: the chunk content
    - ``section_header``: hierarchical heading path (or None for plain text)
    - ``char_start``: character offset in *text*
    - ``char_end``: character offset end in *text*
    """
    if _is_markdown(text):
        return _chunk_markdown(text, chunk_size, chunk_overlap)
    return _chunk_plain(text, chunk_size, chunk_overlap)


# ------------------------------------------------------------------
# Markdown path
# ------------------------------------------------------------------


def _chunk_markdown(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Split markdown text on heading boundaries with sub-splitting for large sections."""
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS,
        strip_headers=False,
    )
    docs = md_splitter.split_text(text)

    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    results: list[dict] = []
    for doc in docs:
        section_header = _build_section_header(doc.metadata)
        content = doc.page_content

        if len(content) > chunk_size:
            sub_chunks = sub_splitter.split_text(content)
            for sc in sub_chunks:
                results.append(
                    {
                        "chunk_text": sc,
                        "section_header": section_header,
                        "char_start": 0,  # placeholder — computed below
                        "char_end": 0,
                    }
                )
        else:
            results.append(
                {
                    "chunk_text": content,
                    "section_header": section_header,
                    "char_start": 0,
                    "char_end": 0,
                }
            )

    # Compute char offsets by locating each chunk in the original text.
    _fill_offsets(results, text)
    return results


# ------------------------------------------------------------------
# Plain-text path
# ------------------------------------------------------------------

_PLAIN_SEPARATORS = ["# ", "## ", "### ", "\n\n", "\n", ". ", " ", ""]


def _chunk_plain(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Split plain text with improved separators."""
    if len(text) <= chunk_size:
        return [
            {
                "chunk_text": text,
                "section_header": None,
                "char_start": 0,
                "char_end": len(text),
            }
        ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_PLAIN_SEPARATORS,
    )
    chunks_text = splitter.split_text(text)

    results: list[dict] = []
    for ct in chunks_text:
        results.append(
            {
                "chunk_text": ct,
                "section_header": None,
                "char_start": 0,
                "char_end": 0,
            }
        )

    _fill_offsets(results, text)
    return results


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _fill_offsets(chunks: list[dict], text: str) -> None:
    """Populate ``char_start`` / ``char_end`` by searching for each chunk in *text*."""
    search_start = 0
    for chunk in chunks:
        ct = chunk["chunk_text"]
        # Use first 100 chars as search needle to handle overlap.
        needle = ct[:100]
        pos = text.find(needle, search_start)
        if pos == -1:
            pos = text.find(needle)
        if pos == -1:
            pos = search_start
        chunk["char_start"] = pos
        chunk["char_end"] = pos + len(ct)
        search_start = max(search_start, pos + 1)
