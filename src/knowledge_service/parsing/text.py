"""TextParser — passthrough parser for plain text and markdown."""

from __future__ import annotations

from knowledge_service.parsing import ParsedDocument


class TextParser:
    """Parser for plain text and markdown content.

    Returns the source unchanged (bytes are decoded as UTF-8).
    """

    supported_formats: set[str] = {"text", "md"}

    async def parse(self, source: bytes | str, content_type: str | None = None) -> ParsedDocument:
        if isinstance(source, bytes):
            text = source.decode("utf-8")
        else:
            text = source

        return ParsedDocument(
            text=text,
            title=None,
            metadata={},
            source_format="text",
        )
