"""PdfParser — extract text and images from PDF files using PyMuPDF."""

from __future__ import annotations

import io

import pymupdf

from knowledge_service.parsing import ParsedDocument


class PdfParser:
    """Parser for PDF documents using PyMuPDF (fitz).

    Extracts text page by page, joined with double newlines.
    Extracts embedded images as PNG bytes for future OCR use.
    Title is read from PDF metadata; falls back to None.
    """

    supported_formats: set[str] = {"pdf"}

    async def parse(self, source: bytes | str, content_type: str | None = None) -> ParsedDocument:
        if isinstance(source, str):
            source = source.encode("utf-8")

        doc = pymupdf.open(stream=source, filetype="pdf")

        pages: list[str] = []
        images: list[bytes] = []

        for page in doc:
            pages.append(page.get_text())

            for img_info in page.get_images(full=True):
                xref = img_info[0]
                pix = pymupdf.Pixmap(doc, xref)
                if pix.colorspace and pix.colorspace.n > 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                images.append(pix.tobytes("png"))

        text = "\n\n".join(pages)

        meta = doc.metadata or {}
        title: str | None = meta.get("title") or None

        return ParsedDocument(
            text=text,
            title=title,
            metadata={"page_count": doc.page_count},
            source_format="pdf",
            images=images,
        )
