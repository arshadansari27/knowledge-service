"""Generate a golden eval set from the live corpus (chunks → LLM question/answer).

Lives in ``src/`` (unlike scripts/gen_golden_candidates.py) so it ships inside the
Docker image and can run in a swarm one-off container that has DATABASE_URL + LLM
access:

    python -m knowledge_service.eval.gen_golden --per-type 6 --out /tmp/golden.json

Writes validated GoldenItem-shaped JSON directly (empty/garbage rows dropped), so
the output is ready for ``load_golden`` with no manual curation step.
"""

from __future__ import annotations

import argparse
import asyncio
import json

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


async def _amain(args: argparse.Namespace) -> None:
    import asyncpg

    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=4)
    try:
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
            if len(bucket) < args.per_type:
                bucket.append(dict(r))
        sampled = [x for items in by_type.values() for x in items]

        client = ExtractionClient(
            settings.llm_base_url, settings.llm_chat_model, settings.llm_api_key
        )
        golden: list[dict] = []
        for i, ch in enumerate(sampled):
            try:
                resp = await client.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": client.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": _Q_PROMPT.format(chunk=ch["chunk_text"][:3000]),
                            }
                        ],
                    },
                )
                resp.raise_for_status()
                parsed = _extract_json(resp.json()["choices"][0]["message"]["content"])
            except Exception:  # one bad chunk/LLM hiccup shouldn't sink the whole set
                continue
            if not isinstance(parsed, dict):
                continue
            q = str(parsed.get("question", "")).strip()
            a = str(parsed.get("reference_answer", "")).strip()
            if not q or not a:
                continue
            golden.append(
                {
                    "id": f"auto-{i:03d}",
                    "question": q,
                    "query_type": "semantic",
                    "relevant_source_ids": [ch["content_id"]],
                    "reference_answer": a,
                }
            )
        with open(args.out, "w") as f:
            json.dump(golden, f, indent=2)
        print(f"GOLDEN_N={len(golden)} -> {args.out}")
        await client.close()
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a golden eval set from the corpus.")
    parser.add_argument("--per-type", type=int, default=6)
    parser.add_argument("--out", default="/tmp/golden.json")
    asyncio.run(_amain(parser.parse_args()))


if __name__ == "__main__":
    main()
