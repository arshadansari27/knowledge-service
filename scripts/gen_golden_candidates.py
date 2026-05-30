#!/usr/bin/env python3
"""Generate golden-set CANDIDATES from snapshot chunks (auto-then-curate workflow).

Samples chunks across source_types, asks the LLM to produce a question answerable
from each chunk, and emits candidate golden items (relevance label = that chunk's
content_id). Output is a CANDIDATE file you curate by hand before saving to
src/knowledge_service/eval/golden.json.

Usage:
    python scripts/gen_golden_candidates.py --per-type 10 --out golden_candidates.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from knowledge_service._utils import _extract_json
from knowledge_service.clients.llm import ExtractionClient
from knowledge_service.config import settings

_Q_PROMPT = """Given this document chunk, write ONE natural question a user would ask
that is answerable ONLY from this chunk, plus a one-sentence reference answer.
Do not quote the chunk verbatim in the question.

Return JSON: {{"question": "...", "reference_answer": "..."}}

CHUNK:
{chunk}
"""


async def _sample_chunks(pool, per_type: int) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT c.content_id::text AS content_id, c.chunk_text, cm.source_type
        FROM content c
        JOIN content_metadata cm ON cm.id = c.content_id
        WHERE c.chunk_text <> ''
        ORDER BY cm.source_type, random()
        """
    )
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        bucket = by_type.setdefault(r["source_type"], [])
        if len(bucket) < per_type:
            bucket.append(dict(r))
    sampled: list[dict] = []
    for items in by_type.values():
        sampled.extend(items)
    return sampled


async def _amain(args) -> None:
    import asyncpg

    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=4)
    client = ExtractionClient(settings.llm_base_url, settings.llm_chat_model, settings.llm_api_key)
    chunks = await _sample_chunks(pool, args.per_type)
    candidates = []
    for i, ch in enumerate(chunks):
        resp = await client.client.post(
            "/v1/chat/completions",
            json={
                "model": client.model,
                "messages": [
                    {"role": "user", "content": _Q_PROMPT.format(chunk=ch["chunk_text"][:3000])}
                ],
            },
        )
        resp.raise_for_status()
        parsed = _extract_json(resp.json()["choices"][0]["message"]["content"])
        if not isinstance(parsed, dict) or "question" not in parsed:
            continue
        candidates.append(
            {
                "id": f"auto-{ch['source_type']}-{i:03d}",
                "question": parsed["question"],
                "query_type": "semantic",
                "relevant_source_ids": [ch["content_id"]],
                "reference_answer": parsed.get("reference_answer", ""),
                "notes": f"auto-generated from source_type={ch['source_type']}; CURATE ME",
            }
        )
    Path(args.out).write_text(json.dumps(candidates, indent=2))
    print(f"wrote {len(candidates)} candidates to {args.out}")
    await client.close()
    await pool.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-type", type=int, default=10)
    p.add_argument("--out", default="golden_candidates.json")
    asyncio.run(_amain(p.parse_args()))


if __name__ == "__main__":
    main()
