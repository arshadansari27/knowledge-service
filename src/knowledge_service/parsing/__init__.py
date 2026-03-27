"""Parsing module — document format detection, parser registry, and parsed document model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse


@dataclass
class ParsedDocument:
    """Normalised output from any parser."""

    text: str
    title: str | None
    metadata: dict
    source_format: str
    images: list[bytes] = field(default_factory=list)


class Parser(Protocol):
    """Protocol that all parsers must implement."""

    supported_formats: set[str]

    async def parse(self, source: bytes | str, content_type: str | None = None) -> ParsedDocument:
        ...


# ---------------------------------------------------------------------------
# Content-type → format map
# ---------------------------------------------------------------------------

_CONTENT_TYPE_MAP: dict[str, str] = {
    "application/pdf": "pdf",
    "text/html": "html",
    "text/csv": "csv",
    "application/json": "json",
    "text/plain": "text",
}

# ---------------------------------------------------------------------------
# URL extension → format map
# ---------------------------------------------------------------------------

_EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".csv": "csv",
    ".json": "json",
    ".txt": "text",
    ".md": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
}

# ---------------------------------------------------------------------------
# Magic byte signatures → format
# ---------------------------------------------------------------------------

_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"%PDF", "pdf"),
    (b"\x89PNG", "image"),
    (b"\xff\xd8\xff", "image"),  # JPEG
    (b"GIF87a", "image"),
    (b"GIF89a", "image"),
]


class ParserRegistry:
    """Registry that maps source formats to parser instances and detects formats."""

    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}

    def register(self, parser: Parser) -> None:
        """Register a parser for all its supported_formats."""
        for fmt in parser.supported_formats:
            self._parsers[fmt] = parser

    def get_parser(self, source_format: str) -> Parser | None:
        """Return the registered parser for *source_format*, or None."""
        return self._parsers.get(source_format)

    def detect_format(
        self,
        content_type: str | None = None,
        url: str | None = None,
        data: bytes | None = None,
    ) -> str:
        """Detect source format using priority: content-type > URL extension > magic bytes > fallback "text"."""

        # 1. Content-type header
        if content_type:
            mime = content_type.split(";")[0].strip().lower()
            if mime in _CONTENT_TYPE_MAP:
                return _CONTENT_TYPE_MAP[mime]
            if mime.startswith("image/"):
                return "image"

        # 2. URL file extension
        if url:
            path = urlparse(url).path.lower()
            for ext, fmt in _EXTENSION_MAP.items():
                if path.endswith(ext):
                    return fmt

        # 3. Magic bytes
        if data:
            for magic, fmt in _MAGIC_BYTES:
                if data.startswith(magic):
                    return fmt

        return "text"
