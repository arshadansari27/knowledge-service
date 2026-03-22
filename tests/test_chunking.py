"""Unit tests for the chunking module — markdown-aware text splitting."""

from __future__ import annotations

from knowledge_service.chunking import _is_markdown, chunk_text


# ---------------------------------------------------------------------------
# Tests: _is_markdown detection
# ---------------------------------------------------------------------------


class TestIsMarkdown:
    def test_h1_heading_detected(self):
        text = "# Introduction\n\nSome content about the topic."
        assert _is_markdown(text) is True

    def test_h2_heading_detected(self):
        text = "## Section Two\n\nDetails here."
        assert _is_markdown(text) is True

    def test_h3_heading_detected(self):
        text = "### Subsection\n\nMore details."
        assert _is_markdown(text) is True

    def test_plain_text_not_detected(self):
        text = "This is just a regular paragraph with no markdown headings at all."
        assert _is_markdown(text) is False

    def test_hash_in_code_not_detected(self):
        """A hash inside a code fence should not trigger markdown detection."""
        text = """Here is some code:

```python
# This is a comment, not a heading
x = 42
```

No real headings here.
"""
        assert _is_markdown(text) is False

    def test_heading_beyond_2000_chars_not_detected(self):
        """Only the first 2000 chars are inspected."""
        text = "A" * 2100 + "\n# Late Heading\n\nContent."
        assert _is_markdown(text) is False

    def test_heading_within_2000_chars_detected(self):
        text = "Some intro.\n\n## Methods\n\n" + "B" * 3000
        assert _is_markdown(text) is True


# ---------------------------------------------------------------------------
# Tests: chunk_text with markdown input
# ---------------------------------------------------------------------------


class TestChunkTextMarkdown:
    def test_splits_on_headings(self):
        md = "# Title\n\nIntro paragraph.\n\n## Section A\n\nSection A content.\n\n## Section B\n\nSection B content."
        chunks = chunk_text(md, chunk_size=4000, chunk_overlap=200)
        assert len(chunks) >= 2

    def test_section_header_populated(self):
        md = "# Title\n\nIntro.\n\n## Methods\n\nMethod details here."
        chunks = chunk_text(md, chunk_size=4000, chunk_overlap=200)
        headers = [c["section_header"] for c in chunks]
        # At least one chunk should have a section header containing "Methods"
        assert any(h is not None and "Methods" in h for h in headers)

    def test_hierarchical_section_header(self):
        md = "# Title\n\nIntro.\n\n## Section A\n\n### Subsection A1\n\nDeep content."
        chunks = chunk_text(md, chunk_size=4000, chunk_overlap=200)
        headers = [c["section_header"] for c in chunks if c["section_header"]]
        # Should have a hierarchical header like "Section A > Subsection A1"
        assert any(">" in h for h in headers)

    def test_char_offsets_present(self):
        md = "# Title\n\nIntro.\n\n## Section A\n\nContent A."
        chunks = chunk_text(md, chunk_size=4000, chunk_overlap=200)
        for c in chunks:
            assert "char_start" in c
            assert "char_end" in c
            assert isinstance(c["char_start"], int)
            assert isinstance(c["char_end"], int)
            assert c["char_start"] >= 0
            assert c["char_end"] > c["char_start"]

    def test_large_section_sub_split(self):
        """A markdown section larger than chunk_size should be sub-split."""
        md = "# Title\n\nIntro.\n\n## Big Section\n\n" + "Word " * 2000
        chunks = chunk_text(md, chunk_size=500, chunk_overlap=50)
        # The big section should produce multiple chunks
        assert len(chunks) >= 3

    def test_chunk_text_content_present(self):
        md = "# Title\n\nIntro.\n\n## Methods\n\nMethod details."
        chunks = chunk_text(md, chunk_size=4000, chunk_overlap=200)
        for c in chunks:
            assert "chunk_text" in c
            assert len(c["chunk_text"]) > 0


# ---------------------------------------------------------------------------
# Tests: chunk_text with plain text input
# ---------------------------------------------------------------------------


class TestChunkTextPlain:
    def test_returns_chunks(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=200)
        assert len(chunks) >= 1

    def test_section_header_is_none(self):
        text = "Just plain text with no markdown headings."
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=200)
        for c in chunks:
            assert c["section_header"] is None

    def test_short_text_single_chunk(self):
        text = "Short text."
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=200)
        assert len(chunks) == 1
        assert chunks[0]["chunk_text"] == "Short text."
        assert chunks[0]["char_start"] == 0
        assert chunks[0]["char_end"] == len("Short text.")

    def test_long_plain_text_splits(self):
        text = "Sentence. " * 1000
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 2

    def test_char_offsets_present_plain(self):
        text = "First paragraph.\n\nSecond paragraph."
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=200)
        for c in chunks:
            assert "char_start" in c
            assert "char_end" in c
            assert c["char_start"] >= 0
            assert c["char_end"] > c["char_start"]
