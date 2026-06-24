"""Eval runner: load golden -> run each mode -> score -> report.

The corpus-dependent run (real Postgres, oxigraph, LLMs) is invoked via the CLI.
`score_query` is the pure scoring seam, unit-tested in isolation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from knowledge_service.config import settings
from knowledge_service.eval.golden import GoldenItem, load_golden
from knowledge_service.eval.judge import Judge, JudgeScore
from knowledge_service.eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from knowledge_service.eval.report import QueryResult, aggregate, format_summary_table
from knowledge_service.stores.rag import RetrievalContext

logger = logging.getLogger(__name__)

_DEFAULT_GOLDEN = Path(__file__).with_name("golden.json")
_REPORTS_DIR = Path(__file__).with_name("reports")


def _retrieved_ids(context: RetrievalContext) -> list[str]:
    """Ordered, de-duplicated content ids from a retrieval context.

    Several retrieved chunks can share one ``content_id`` (a document is chunked
    N ways). Golden relevance is labelled at the content level, so the ranked id
    list must be de-duplicated by content — keeping first-occurrence order — or a
    single relevant document counted once per chunk inflates DCG above the ideal
    (nDCG > 1). Falls back to the chunk ``id`` when ``content_id`` is absent.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for row in context.content_results:
        cid = row.get("content_id") or row.get("id")
        if cid is None:
            continue
        cid = str(cid)
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


def _format_context_for_judge(context: RetrievalContext) -> str:
    parts: list[str] = []
    for row in context.content_results:
        parts.append(str(row.get("chunk_text") or row.get("summary") or ""))
    for t in context.knowledge_triples:
        parts.append(f"{t.get('subject')} -> {t.get('predicate')} -> {t.get('object')}")
    return "\n".join(parts)


def score_query(
    item: GoldenItem,
    mode: str,
    context: RetrievalContext,
    generated_answer: str,
    judge_score: JudgeScore,
    k: int,
) -> QueryResult:
    """Pure: combine retrieval metrics + judge scores into a QueryResult."""
    retrieved = _retrieved_ids(context)
    relevant = item.relevant_source_ids
    metrics = {
        "recall_at_k": recall_at_k(retrieved, relevant, k),
        "precision_at_k": precision_at_k(retrieved, relevant, k),
        "mrr": mrr(retrieved, relevant),
        "ndcg_at_k": ndcg_at_k(retrieved, relevant, k),
        "faithfulness": judge_score.faithfulness,
        "correctness": judge_score.correctness,
    }
    return QueryResult(
        query_id=item.id,
        mode=mode,
        query_type=item.query_type,
        metrics=metrics,
        answer=generated_answer,
        rationale=judge_score.rationale,
    )


async def _build_components():
    """Construct stores + clients + retriever against the configured snapshot.

    Mirrors main.py lifespan wiring (chunk-only, graph removed).
    """
    import asyncpg

    from knowledge_service.clients.llm import EmbeddingClient
    from knowledge_service.clients.rag import RAGClient
    from knowledge_service.stores.content import ContentStore
    from knowledge_service.stores.rag import RAGRetriever

    pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=10)

    embedding_client = EmbeddingClient(
        settings.llm_base_url, settings.llm_embed_model, settings.llm_api_key
    )
    content_store = ContentStore(pool, exclude_inflight=settings.reader_exclude_inflight)

    rag_model = settings.llm_rag_model or settings.llm_chat_model
    rag_client = RAGClient(settings.llm_base_url, rag_model, settings.llm_api_key)

    # ponytail: chunk-only retriever (graph layer removed)
    retriever = RAGRetriever(
        embedding_client=embedding_client,
        embedding_store=content_store,
    )
    clients = (embedding_client, rag_client)
    # Return None for knowledge_store (graph removed); adjust shutdown handling
    return pool, None, retriever, rag_client, clients


async def run_eval(modes: list[str], k: int, golden_path: Path) -> list[QueryResult]:
    items = load_golden(golden_path)
    judge = Judge(
        base_url=settings.eval_judge_base_url,
        model=settings.eval_judge_model,
        api_key=settings.eval_judge_api_key,
    )
    pool, knowledge_store, retriever, rag_client, clients = await _build_components()
    sem = asyncio.Semaphore(settings.eval_concurrency)

    async def _one(item: GoldenItem, mode: str) -> QueryResult:
        async with sem:
            # ponytail: chunk-only retrieval; intent classification (graph) removed.
            context = await retriever.retrieve(
                item.question,
                max_sources=k,
                min_confidence=0.0,
            )
            answer_obj = await rag_client.answer(item.question, context)
            judge_score = await judge.score_one(
                question=item.question,
                reference_answer=item.reference_answer,
                generated_answer=answer_obj.answer,
                retrieved_context=_format_context_for_judge(context),
            )
            return score_query(item, mode, context, answer_obj.answer, judge_score, k)

    try:
        tasks = [_one(item, mode) for mode in modes for item in items]
        results = await asyncio.gather(*tasks)
    finally:
        await judge.close()
        for c in clients:
            await c.close()
        await pool.close()
        # ponytail: knowledge_store (graph layer) is now None
    return list(results)


def _write_report(results: list[QueryResult], k: int, timestamp: str) -> Path:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    agg = aggregate(results)
    payload = {
        "k": k,
        "timestamp": timestamp,
        "summary": {f"{m}::{t}": cell for (m, t), cell in agg.items()},
        "queries": [
            {
                "query_id": r.query_id,
                "mode": r.mode,
                "query_type": r.query_type,
                "metrics": r.metrics,
                "answer": r.answer,
                "rationale": r.rationale,
            }
            for r in results
        ],
    }
    path = _REPORTS_DIR / f"{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


async def _amain(args: argparse.Namespace) -> None:
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    golden_path = Path(args.golden) if args.golden else _DEFAULT_GOLDEN
    results = await run_eval(modes=modes, k=args.k, golden_path=golden_path)
    print(format_summary_table(aggregate(results)))
    path = _write_report(results, args.k, args.timestamp)
    print(f"\nReport written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the KS eval harness.")
    parser.add_argument("--modes", default="full,chunks_only")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--golden", default="")
    parser.add_argument(
        "--timestamp",
        required=True,
        help="Report filename stamp, e.g. 2026-05-30T1200 (passed in; the harness "
        "does not call the clock so runs stay reproducible).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
