"""Tests for HtmlParser."""

from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_service.parsing.html import HtmlParser

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def parser() -> HtmlParser:
    return HtmlParser()


@pytest.fixture
def sample_html() -> str:
    return (FIXTURES / "sample.html").read_text()


async def test_extracts_article_text(parser: HtmlParser, sample_html: str) -> None:
    doc = await parser.parse(sample_html)
    assert "Cold Exposure and Dopamine" in doc.text
    assert "dopamine levels" in doc.text
    assert "Participants were exposed" in doc.text


async def test_strips_scripts(parser: HtmlParser, sample_html: str) -> None:
    doc = await parser.parse(sample_html)
    assert "console.log" not in doc.text
    assert "tracking" not in doc.text


async def test_extracts_title(parser: HtmlParser, sample_html: str) -> None:
    doc = await parser.parse(sample_html)
    assert doc.title is not None
    assert "Cold Exposure" in doc.title


async def test_parses_bytes_input(parser: HtmlParser, sample_html: str) -> None:
    doc = await parser.parse(sample_html.encode("utf-8"))
    assert "Cold Exposure and Dopamine" in doc.text


async def test_handles_malformed_html(parser: HtmlParser) -> None:
    malformed = "<html><body><p>Unclosed paragraph<div>Mixed tags</p></body>"
    doc = await parser.parse(malformed)
    assert "Unclosed paragraph" in doc.text or doc.text == ""


async def test_supported_formats(parser: HtmlParser) -> None:
    assert "html" in parser.supported_formats


async def test_source_format(parser: HtmlParser, sample_html: str) -> None:
    doc = await parser.parse(sample_html)
    assert doc.source_format == "html"
