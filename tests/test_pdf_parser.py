"""Tests for PdfParser."""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import pytest

from knowledge_service.parsing.pdf import PdfParser

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def parser() -> PdfParser:
    return PdfParser()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return (FIXTURES / "sample.pdf").read_bytes()


async def test_extract_text(parser: PdfParser, sample_pdf_bytes: bytes) -> None:
    doc = await parser.parse(sample_pdf_bytes)
    assert "Sample Research Paper" in doc.text
    assert "cold exposure" in doc.text
    assert "Participants were exposed" in doc.text


async def test_page_count(parser: PdfParser, sample_pdf_bytes: bytes) -> None:
    doc = await parser.parse(sample_pdf_bytes)
    assert doc.metadata["page_count"] == 2


async def test_source_format(parser: PdfParser, sample_pdf_bytes: bytes) -> None:
    doc = await parser.parse(sample_pdf_bytes)
    assert doc.source_format == "pdf"


async def test_parse_empty_pdf(parser: PdfParser) -> None:
    """A PDF with a single blank page should return empty text and page_count=1."""
    blank_doc = pymupdf.open()
    blank_doc.new_page()  # pymupdf requires at least one page to save
    buf = io.BytesIO()
    blank_doc.save(buf)
    blank_doc.close()

    result = await parser.parse(buf.getvalue())
    assert result.text.strip() == ""
    assert result.metadata["page_count"] == 1


async def test_supported_formats(parser: PdfParser) -> None:
    assert "pdf" in parser.supported_formats


async def test_corrupt_bytes_raises(parser: PdfParser) -> None:
    with pytest.raises(Exception):
        await parser.parse(b"not a pdf at all %GARBAGE")
