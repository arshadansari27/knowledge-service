"""Diagnostic: dump question → retrieved triples → generated answer → judge rationale
for a few items, in both verbalize-OFF and verbalize-ON arms.

Run as a one-off so we can SEE why faithfulness is near-zero (is it the raw URIs in
the prompt, or is the judge/golden set the problem?). Does not write reports; prints
everything to stdout.

    python -m knowledge_service.eval.diagnose --per-type 3 --query-types entity,graph
"""

from __future__ import annotations

import argparse
import asyncio

from knowledge_service.config import settings
from knowledge_service.eval.gen_golden import _PROMPTS  # reuse the type prompts
from knowledge_service.eval.judge import Judge
from knowledge_service.eval.runner import _build_components, _format_context_for_judge


async def _amain(args: argparse.Namespace) -> None:
    import asyncpg

    from knowledge_service._utils import _extract_json
    from knowledge_service.clients.llm import ExtractionClient

    qtypes = [t.strip() for t in args.query_types.split(",") if t.strip() in _PROMPTS]

    # --- build a tiny golden set inline (so we control the items) ---
    pool0 = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=2)
    rows = await pool0.fetch(
        "SELECT c.content_id::text AS content_id, c.chunk_text FROM content c "
        "WHERE length(c.chunk_text) > 200 ORDER BY random() LIMIT $1",
        args.per_type * len(qtypes) * 2,
    )
    await pool0.close()
    gen = ExtractionClient(settings.llm_base_url, settings.llm_chat_model, settings.llm_api_key)
    items: list[dict] = []
    ci = 0
    for qtype in qtypes:
        made = 0
        while made < args.per_type and ci < len(rows):
            ch = rows[ci]
            ci += 1
            try:
                r = await gen.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": gen.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": _PROMPTS[qtype].format(chunk=ch["chunk_text"][:3000]),
                            }
                        ],
                    },
                )
                r.raise_for_status()
                p = _extract_json(r.json()["choices"][0]["message"]["content"])
            except Exception:
                continue
            if not isinstance(p, dict) or not p.get("question") or not p.get("reference_answer"):
                continue
            items.append(
                {"qtype": qtype, "question": p["question"], "reference": p["reference_answer"]}
            )
            made += 1
    await gen.close()

    # --- components (retriever + rag client) ---
    pool, knowledge_store, retriever, rag_client, clients = await _build_components()
    judge = Judge(
        base_url=settings.eval_judge_base_url,
        model=settings.eval_judge_model,
        api_key=settings.eval_judge_api_key,
    )
    try:
        for it in items:
            intent = await retriever.classify(it["question"])
            for verbalize in (False, True):
                settings.rag_verbalize_triples = verbalize
                ctx = await retriever.retrieve(
                    it["question"],
                    max_sources=5,
                    min_confidence=0.0,
                    intent=intent,
                    retrieval_mode="full",
                )
                ans = await rag_client.answer_auto(it["question"], ctx, "direct")
                js = await judge.score_one(
                    question=it["question"],
                    reference_answer=it["reference"],
                    generated_answer=ans.answer,
                    retrieved_context=_format_context_for_judge(ctx),
                )
                print("=" * 88)
                print(f"[{it['qtype']}] verbalize={verbalize} intent={intent.intent}")
                print(f"Q: {it['question']}")
                print(f"REF: {it['reference']}")
                print(f"#triples={len(ctx.knowledge_triples)} #chunks={len(ctx.content_results)}")
                for t in ctx.knowledge_triples[:4]:
                    s = t.get("subject_label") or t.get("subject")
                    p = t.get("predicate_label") or t.get("predicate")
                    o = t.get("object_label") or t.get("object")
                    print(f"   T: {s} | {p} | {o}")
                print(f"ANSWER: {ans.answer[:600]}")
                print(f"JUDGE: faith={js.faithfulness} correct={js.correctness} :: {js.rationale}")
    finally:
        await judge.close()
        for c in clients:
            await c.close()
        await pool.close()
        knowledge_store.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump eval internals for a few items.")
    parser.add_argument("--per-type", type=int, default=3)
    parser.add_argument("--query-types", default="entity,graph")
    asyncio.run(_amain(parser.parse_args()))


if __name__ == "__main__":
    main()
