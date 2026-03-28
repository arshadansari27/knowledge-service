"""StructuredParser — parse JSON and CSV content into plain text."""

from __future__ import annotations

import csv
import io
import json

from knowledge_service.parsing import ParsedDocument


class StructuredParser:
    """Parser for JSON and CSV documents.

    Format detection priority:
    1. If content_type contains "csv" → parse as CSV.
    2. Otherwise try JSON first; fall back to CSV on failure.

    JSON output is pretty-printed.
    CSV output joins each row as "key: value" lines, rows separated by blank lines.
    """

    supported_formats: set[str] = {"json", "csv"}

    async def parse(self, source: bytes | str, content_type: str | None = None) -> ParsedDocument:
        if isinstance(source, bytes):
            text_in = source.decode("utf-8", errors="replace")
        else:
            text_in = source

        is_csv_hint = content_type is not None and "csv" in content_type.lower()

        if is_csv_hint:
            return self._parse_csv(text_in)

        # Try JSON first, fall back to CSV
        try:
            return self._parse_json(text_in)
        except (json.JSONDecodeError, ValueError):
            return self._parse_csv(text_in)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_json(self, text: str) -> ParsedDocument:
        data = json.loads(text)
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        return ParsedDocument(
            text=pretty,
            title=None,
            metadata={},
            source_format="json",
        )

    def _parse_csv(self, text: str) -> ParsedDocument:
        reader = csv.DictReader(io.StringIO(text))
        rows: list[str] = []
        for row in reader:
            row_lines = "\n".join(f"{k}: {v}" for k, v in row.items())
            rows.append(row_lines)
        return ParsedDocument(
            text="\n\n".join(rows),
            title=None,
            metadata={},
            source_format="csv",
        )
