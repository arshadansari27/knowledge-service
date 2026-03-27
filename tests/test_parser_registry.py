"""Tests for the parsing module — ParsedDocument, Parser protocol, ParserRegistry, and TextParser."""

from __future__ import annotations

import pytest

from knowledge_service.parsing import ParsedDocument, ParserRegistry
from knowledge_service.parsing.text import TextParser


# ---------------------------------------------------------------------------
# Tests: ParsedDocument dataclass
# ---------------------------------------------------------------------------


class TestParsedDocument:
    def test_required_fields(self):
        doc = ParsedDocument(text="hello", title="My Doc", metadata={"key": "val"}, source_format="text")
        assert doc.text == "hello"
        assert doc.title == "My Doc"
        assert doc.metadata == {"key": "val"}
        assert doc.source_format == "text"

    def test_images_default_empty(self):
        doc = ParsedDocument(text="hello", title=None, metadata={}, source_format="text")
        assert doc.images == []

    def test_images_field(self):
        doc = ParsedDocument(text="hi", title=None, metadata={}, source_format="image", images=[b"\xff\xd8\xff"])
        assert len(doc.images) == 1
        assert doc.images[0] == b"\xff\xd8\xff"

    def test_title_none(self):
        doc = ParsedDocument(text="plain", title=None, metadata={}, source_format="text")
        assert doc.title is None

    def test_images_not_shared_across_instances(self):
        doc1 = ParsedDocument(text="a", title=None, metadata={}, source_format="text")
        doc2 = ParsedDocument(text="b", title=None, metadata={}, source_format="text")
        doc1.images.append(b"data")
        assert doc2.images == []


# ---------------------------------------------------------------------------
# Tests: ParserRegistry — register & get
# ---------------------------------------------------------------------------


class TestParserRegistryRegisterGet:
    def test_register_and_get(self):
        registry = ParserRegistry()
        parser = TextParser()
        registry.register(parser)
        assert registry.get_parser("text") is parser

    def test_register_multiple_formats(self):
        registry = ParserRegistry()
        parser = TextParser()
        registry.register(parser)
        # TextParser supports "text" and "md"
        assert registry.get_parser("text") is parser
        assert registry.get_parser("md") is parser

    def test_get_unknown_format_returns_none(self):
        registry = ParserRegistry()
        assert registry.get_parser("pdf") is None

    def test_register_second_parser_overwrites(self):
        registry = ParserRegistry()
        parser1 = TextParser()
        parser2 = TextParser()
        registry.register(parser1)
        registry.register(parser2)
        assert registry.get_parser("text") is parser2


# ---------------------------------------------------------------------------
# Tests: ParserRegistry.detect_format — content-type
# ---------------------------------------------------------------------------


class TestParserRegistryDetectFormatContentType:
    def test_pdf_content_type(self):
        registry = ParserRegistry()
        assert registry.detect_format(content_type="application/pdf") == "pdf"

    def test_html_content_type(self):
        registry = ParserRegistry()
        assert registry.detect_format(content_type="text/html") == "html"

    def test_csv_content_type(self):
        registry = ParserRegistry()
        assert registry.detect_format(content_type="text/csv") == "csv"

    def test_json_content_type(self):
        registry = ParserRegistry()
        assert registry.detect_format(content_type="application/json") == "json"

    def test_plain_text_content_type(self):
        registry = ParserRegistry()
        assert registry.detect_format(content_type="text/plain") == "text"

    def test_image_wildcard_content_type(self):
        registry = ParserRegistry()
        assert registry.detect_format(content_type="image/png") == "image"

    def test_image_jpeg_content_type(self):
        registry = ParserRegistry()
        assert registry.detect_format(content_type="image/jpeg") == "image"

    def test_content_type_with_charset(self):
        registry = ParserRegistry()
        assert registry.detect_format(content_type="text/html; charset=utf-8") == "html"


# ---------------------------------------------------------------------------
# Tests: ParserRegistry.detect_format — URL extension
# ---------------------------------------------------------------------------


class TestParserRegistryDetectFormatURL:
    def test_pdf_extension(self):
        registry = ParserRegistry()
        assert registry.detect_format(url="https://example.com/doc.pdf") == "pdf"

    def test_html_extension(self):
        registry = ParserRegistry()
        assert registry.detect_format(url="https://example.com/page.html") == "html"

    def test_htm_extension(self):
        registry = ParserRegistry()
        assert registry.detect_format(url="https://example.com/page.htm") == "html"

    def test_csv_extension(self):
        registry = ParserRegistry()
        assert registry.detect_format(url="https://example.com/data.csv") == "csv"

    def test_json_extension(self):
        registry = ParserRegistry()
        assert registry.detect_format(url="https://example.com/data.json") == "json"

    def test_txt_extension(self):
        registry = ParserRegistry()
        assert registry.detect_format(url="https://example.com/notes.txt") == "text"

    def test_md_extension(self):
        registry = ParserRegistry()
        assert registry.detect_format(url="https://example.com/README.md") == "text"

    def test_png_extension(self):
        registry = ParserRegistry()
        assert registry.detect_format(url="https://example.com/photo.png") == "image"

    def test_jpg_extension(self):
        registry = ParserRegistry()
        assert registry.detect_format(url="https://example.com/photo.jpg") == "image"

    def test_jpeg_extension(self):
        registry = ParserRegistry()
        assert registry.detect_format(url="https://example.com/photo.jpeg") == "image"

    def test_gif_extension(self):
        registry = ParserRegistry()
        assert registry.detect_format(url="https://example.com/anim.gif") == "image"

    def test_webp_extension(self):
        registry = ParserRegistry()
        assert registry.detect_format(url="https://example.com/photo.webp") == "image"


# ---------------------------------------------------------------------------
# Tests: ParserRegistry.detect_format — magic bytes
# ---------------------------------------------------------------------------


class TestParserRegistryDetectFormatMagicBytes:
    def test_pdf_magic_bytes(self):
        registry = ParserRegistry()
        assert registry.detect_format(data=b"%PDF-1.4 rest of file") == "pdf"

    def test_png_magic_bytes(self):
        registry = ParserRegistry()
        assert registry.detect_format(data=b"\x89PNG\r\n\x1a\n rest") == "image"

    def test_jpeg_magic_bytes(self):
        registry = ParserRegistry()
        assert registry.detect_format(data=b"\xff\xd8\xff\xe0 jpeg data") == "image"

    def test_gif87a_magic_bytes(self):
        registry = ParserRegistry()
        assert registry.detect_format(data=b"GIF87a rest of file") == "image"

    def test_gif89a_magic_bytes(self):
        registry = ParserRegistry()
        assert registry.detect_format(data=b"GIF89a rest of file") == "image"


# ---------------------------------------------------------------------------
# Tests: ParserRegistry.detect_format — fallback and priority
# ---------------------------------------------------------------------------


class TestParserRegistryDetectFormatFallback:
    def test_fallback_to_text(self):
        registry = ParserRegistry()
        assert registry.detect_format() == "text"

    def test_fallback_unknown_extension(self):
        registry = ParserRegistry()
        # Unknown extension, no content-type, no magic bytes
        assert registry.detect_format(url="https://example.com/file.xyz") == "text"

    def test_content_type_takes_priority_over_url(self):
        registry = ParserRegistry()
        # content-type says PDF, URL says HTML — content-type wins
        result = registry.detect_format(
            content_type="application/pdf",
            url="https://example.com/page.html",
        )
        assert result == "pdf"

    def test_content_type_takes_priority_over_magic_bytes(self):
        registry = ParserRegistry()
        # content-type says text/plain, magic bytes look like PDF
        result = registry.detect_format(
            content_type="text/plain",
            data=b"%PDF-1.4",
        )
        assert result == "text"

    def test_url_takes_priority_over_magic_bytes(self):
        registry = ParserRegistry()
        # URL says PDF, magic bytes look like PNG
        result = registry.detect_format(
            url="https://example.com/file.pdf",
            data=b"\x89PNG\r\n\x1a\n",
        )
        assert result == "pdf"


# ---------------------------------------------------------------------------
# Tests: TextParser
# ---------------------------------------------------------------------------


class TestTextParser:
    def test_supported_formats(self):
        parser = TextParser()
        assert "text" in parser.supported_formats
        assert "md" in parser.supported_formats

    async def test_parse_string(self):
        parser = TextParser()
        doc = await parser.parse("Hello, world!")
        assert doc.text == "Hello, world!"
        assert doc.source_format == "text"
        assert doc.title is None

    async def test_parse_bytes(self):
        parser = TextParser()
        doc = await parser.parse(b"Hello in bytes")
        assert doc.text == "Hello in bytes"
        assert doc.source_format == "text"

    async def test_parse_utf8_bytes(self):
        parser = TextParser()
        doc = await parser.parse("Héllo wörld".encode("utf-8"))
        assert doc.text == "Héllo wörld"

    async def test_parse_returns_parsed_document(self):
        parser = TextParser()
        doc = await parser.parse("test content")
        assert isinstance(doc, ParsedDocument)

    async def test_parse_metadata_empty(self):
        parser = TextParser()
        doc = await parser.parse("some text")
        assert doc.metadata == {}

    async def test_parse_images_empty(self):
        parser = TextParser()
        doc = await parser.parse("some text")
        assert doc.images == []

    async def test_parse_content_type_ignored(self):
        parser = TextParser()
        doc = await parser.parse("data", content_type="text/plain")
        assert doc.text == "data"

    async def test_parse_markdown_string(self):
        parser = TextParser()
        md = "# Heading\n\nSome **bold** text."
        doc = await parser.parse(md)
        assert doc.text == md
        assert doc.source_format == "text"
