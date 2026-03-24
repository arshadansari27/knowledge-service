"""Admin endpoint for community rebuild."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Request

from knowledge_service.stores.community import CommunityDetector, CommunitySummarizer

router = APIRouter()


@router.post("/rebuild-communities")
async def rebuild_communities(request: Request):
    """Trigger a full community detection + summarization rebuild."""
    knowledge_store = request.app.state.knowledge_store
    community_store = request.app.state.community_store

    start = time.time()

    # Step 1: Detect communities
    detector = CommunityDetector(knowledge_store)
    communities = await asyncio.to_thread(detector.detect)

    # Step 2: Summarize each community via LLM (uses RAG model for quality)
    summarizer = CommunitySummarizer(
        request.app.state.rag_client._client,
        knowledge_store,
        model=request.app.state.rag_client._model,
    )

    summarized = []
    for c in communities:
        result = await summarizer.summarize_one(c)
        summarized.append(result)

    # Step 3: Store
    count = await community_store.replace_all(summarized)

    duration = time.time() - start
    level_counts: dict[str, int] = {}
    for c in summarized:
        key = f"level_{c['level']}"
        level_counts[key] = level_counts.get(key, 0) + 1

    return {
        "communities_built": count,
        "levels": level_counts,
        "duration_seconds": round(duration, 2),
    }
