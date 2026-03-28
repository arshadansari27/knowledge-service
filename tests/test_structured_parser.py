"""Tests for StructuredParser and ImageParser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_service.parsing.image import ImageParser
from knowledge_service.parsing.structured import StructuredParser

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# StructuredParser — CSV
# ---------------------------------------------------------------------------


@pytest.fixture
def structured_parser() -> StructuredParser:
    return StructuredParser()


async def test_parse_csv_file(structured_parser: StructuredParser) -> None:
    csv_bytes = (FIXTURES / "sample.csv").read_bytes()
    doc = await structured_parser.parse(csv_bytes, content_type="text/csv")
    assert doc.source_format == "csv"
    assert "cold_exposure" in doc.text
    assert "increases_dopamine" in doc.text
    assert "vitamin_d3" in doc.text
    assert "5000_iu" in doc.text


async def test_parse_csv_row_format(structured_parser: StructuredParser) -> None:
    csv_text = "name,value\nfoo,bar\nbaz,qux"
    doc = await structured_parser.parse(csv_text, content_type="text/csv")
    assert "name: foo" in doc.text
    assert "value: bar" in doc.text


# ---------------------------------------------------------------------------
# StructuredParser — JSON
# ---------------------------------------------------------------------------


async def test_parse_json_bytes(structured_parser: StructuredParser) -> None:
    data = {"subject": "cold_exposure", "predicate": "increases", "object": "dopamine"}
    json_bytes = json.dumps(data).encode()
    doc = await structured_parser.parse(json_bytes)
    assert doc.source_format == "json"
    assert "cold_exposure" in doc.text
    assert "dopamine" in doc.text


async def test_parse_json_string(structured_parser: StructuredParser) -> None:
    data = [{"key": "value"}, {"key2": "value2"}]
    doc = await structured_parser.parse(json.dumps(data))
    assert doc.source_format == "json"
    assert '"key": "value"' in doc.text


async def test_json_pretty_printed(structured_parser: StructuredParser) -> None:
    doc = await structured_parser.parse('{"a":1}')
    # pretty-printed JSON has newlines and indentation
    assert "\n" in doc.text


async def test_json_fallback_to_csv(structured_parser: StructuredParser) -> None:
    """Non-JSON content without csv hint should fall back to CSV parsing."""
    csv_text = "col1,col2\nalpha,beta"
    doc = await structured_parser.parse(csv_text)
    assert doc.source_format == "csv"
    assert "alpha" in doc.text


async def test_structured_supported_formats(structured_parser: StructuredParser) -> None:
    assert "json" in structured_parser.supported_formats
    assert "csv" in structured_parser.supported_formats


# ---------------------------------------------------------------------------
# ImageParser
# ---------------------------------------------------------------------------


@pytest.fixture
def image_parser() -> ImageParser:
    return ImageParser()


async def test_image_returns_empty_text(image_parser: ImageParser) -> None:
    fake_png = b"\x89PNG\r\n\x1a\nfake image data"
    doc = await image_parser.parse(fake_png)
    assert doc.text == ""


async def test_image_stores_bytes(image_parser: ImageParser) -> None:
    fake_png = b"\x89PNG\r\n\x1a\nfake image data"
    doc = await image_parser.parse(fake_png)
    assert len(doc.images) == 1
    assert doc.images[0] == fake_png


async def test_image_source_format(image_parser: ImageParser) -> None:
    doc = await image_parser.parse(b"someimagedata")
    assert doc.source_format == "image"


async def test_image_supported_formats(image_parser: ImageParser) -> None:
    assert "image" in image_parser.supported_formats
