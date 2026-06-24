"""Shared utility helpers reused across API modules."""

from __future__ import annotations

import json
import re


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from freeform LLM output.

    Handles markdown code fences, qwen3 <think> tags, and trailing text.
    Returns None if no valid JSON object is found.
    """
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL).strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    start = stripped.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(stripped, start)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        start = stripped.find("{", start + 1)
    return None
