"""Generate a golden eval set from the live corpus (chunks → LLM question/answer).

Lives in ``src/`` (unlike scripts/gen_golden_candidates.py) so it ships inside the
Docker image and can run in a swarm one-off container that has DATABASE_URL + LLM
access:

    python -m knowledge_service.eval.gen_golden --per-type 8 --out /tmp/golden.json

Produces a mix of the three query types the retriever routes on — ``semantic``,
``entity``, and ``graph`` (relationship/multi-hop). The entity/graph items are the
ones ``retrieval_mode=auto`` actually sends through the knowledge graph, so a
golden set needs them to measure the graph path (a semantic-only set only exercises
chunks_only). Writes validated GoldenItem-shaped JSON directly (empty/garbage rows
dropped), ready for ``load_golden`` with no manual curation step.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from knowledge_service._utils import _extract_json
from knowledge_service.clients.llm import ExtractionClient
from knowledge_service.config import settings

# Per query-type question prompts. The answer must still be findable from the
# chunk (so document-level relevance holds), but the phrasing pushes the query
# classifier toward the intended intent.
_PROMPTS = {
    "semantic": """Given this document chunk, write ONE natural question a user would ask
that is answerable ONLY from this chunk, plus a one-sentence reference answer.
Do not quote the chunk verbatim in the question.

Return JSON: {{"question": "...", "reference_answer": "..."}}

CHUNK:
{chunk}
""",
    "entity": """Given this document chunk, write ONE natural question about a SPECIFIC NAMED
ENTITY (a person, organization, product, place, or concept) mentioned in the chunk
— e.g. "What is X?" or "Tell me about X" — answerable ONLY from this chunk, plus a
one-sentence reference answer. The question MUST name the entity explicitly.

Return JSON: {{"question": "...", "reference_answer": "..."}}

CHUNK:
{chunk}
""",
    "graph": """Given this document chunk, write ONE natural question about a RELATIONSHIP or
CONNECTION between two things mentioned in the chunk — e.g. "How is X related to Y?",
"What causes X?", "What does X depend on?" — answerable ONLY from this chunk, plus a
one-sentence reference answer. The question MUST reference how things connect, not a
single isolated fact.

Return JSON: {{"question": "...", "reference_answer": "..."}}

CHUNK:
{chunk}
""",
}


async def _amain(args: argparse.Namespace) -> None:
    import asyncpg

    qtypes = [t.strip() for t in args.query_types.split(",") if t.strip() in _PROMPTS]
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=4)
    try:
        # Sample enough distinct chunks to cover per_type items for EACH query type.
        need = args.per_type * len(qtypes)
        rows = await pool.fetch(
            """
            SELECT c.content_id::text AS content_id, c.chunk_text, cm.source_type
            FROM content c
            JOIN content_metadata cm ON cm.id = c.content_id
            WHERE length(c.chunk_text) > 200
            ORDER BY random()
            LIMIT $1
            """,
            need * 2,  # overfetch so LLM failures/empties still leave enough
        )
        chunks = [dict(r) for r in rows]

        client = ExtractionClient(
            settings.llm_base_url, settings.llm_chat_model, settings.llm_api_key
        )
        golden: list[dict] = []
        ci = 0
        for qtype in qtypes:
            made = 0
            while made < args.per_type and ci < len(chunks):
                ch = chunks[ci]
                ci += 1
                try:
                    resp = await client.client.post(
                        "/v1/chat/completions",
                        json={
                            "model": client.model,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": _PROMPTS[qtype].format(
                                        chunk=ch["chunk_text"][:3000]
                                    ),
                                }
                            ],
                        },
                    )
                    resp.raise_for_status()
                    parsed = _extract_json(resp.json()["choices"][0]["message"]["content"])
                except Exception:  # one bad chunk/LLM hiccup shouldn't sink the set
                    continue
                if not isinstance(parsed, dict):
                    continue
                q = str(parsed.get("question", "")).strip()
                a = str(parsed.get("reference_answer", "")).strip()
                if not q or not a:
                    continue
                golden.append(
                    {
                        "id": f"auto-{qtype}-{made:03d}",
                        "question": q,
                        "query_type": qtype,
                        "relevant_source_ids": [ch["content_id"]],
                        "reference_answer": a,
                    }
                )
                made += 1
        with open(args.out, "w") as f:
            json.dump(golden, f, indent=2)
        counts: dict[str, int] = {}
        for g in golden:
            counts[g["query_type"]] = counts.get(g["query_type"], 0) + 1
        print(f"GOLDEN_N={len(golden)} BY_TYPE={counts} -> {args.out}")
        await client.close()
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a golden eval set from the corpus.")
    parser.add_argument("--per-type", type=int, default=8)
    parser.add_argument("--query-types", default="semantic,entity,graph")
    parser.add_argument("--out", default="/tmp/golden.json")
    asyncio.run(_amain(parser.parse_args()))


if __name__ == "__main__":
    main()
