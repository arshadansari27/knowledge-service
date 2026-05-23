"""PdfParser — extract text from PDF files using PyMuPDF."""

from __future__ import annotations


import pymupdf

from knowledge_service.parsing import ParsedDocument


class PdfParser:
    """Parser for PDF documents using PyMuPDF (fitz).

    Extracts text page by page, joined with double newlines. Title is read
    from PDF metadata; falls back to None.
    """

    supported_formats: set[str] = {"pdf"}

    async def parse(self, source: bytes | str, content_type: str | None = None) -> ParsedDocument:
        if isinstance(source, str):
            source = source.encode("utf-8")

        doc = pymupdf.open(stream=source, filetype="pdf")

        pages: list[str] = [page.get_text() for page in doc]
        text = "\n\n".join(pages)

        meta = doc.metadata or {}
        title: str | None = meta.get("title") or None

        return ParsedDocument(
            text=text,
            title=title,
            metadata={"page_count": doc.page_count},
            source_format="pdf",
        )
