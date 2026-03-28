"""HtmlParser — extract article text from HTML using readability-lxml and BeautifulSoup4."""

from __future__ import annotations

from bs4 import BeautifulSoup
from readability import Document

from knowledge_service.parsing import ParsedDocument


class HtmlParser:
    """Parser for HTML documents.

    Uses readability-lxml to isolate the main article body, then
    BeautifulSoup4 to strip residual tags and extract clean text.
    Scripts and style elements are removed before text extraction.
    """

    supported_formats: set[str] = {"html"}

    async def parse(self, source: bytes | str, content_type: str | None = None) -> ParsedDocument:
        if isinstance(source, bytes):
            html = source.decode("utf-8", errors="replace")
        else:
            html = source

        readable = Document(html)
        title: str | None = readable.short_title() or None

        article_html = readable.summary(html_partial=False)

        soup = BeautifulSoup(article_html, "lxml")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        return ParsedDocument(
            text=text,
            title=title,
            metadata={},
            source_format="html",
        )
