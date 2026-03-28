"""ImageParser — stub that stores raw image bytes for future OCR."""

from __future__ import annotations

from knowledge_service.parsing import ParsedDocument


class ImageParser:
    """Stub parser for image files.

    Text extraction is not yet implemented.  The raw source bytes are
    stored in ``ParsedDocument.images`` so a future OCR step can process them.
    """

    supported_formats: set[str] = {"image"}

    async def parse(self, source: bytes | str, content_type: str | None = None) -> ParsedDocument:
        if isinstance(source, str):
            source = source.encode("utf-8")

        return ParsedDocument(
            text="",
            title=None,
            metadata={},
            source_format="image",
            images=[source],
        )
