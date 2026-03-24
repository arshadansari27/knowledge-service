"""Admin endpoint for community rebuild."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Request

from knowledge_service.stores.community import CommunityDetector

router = APIRouter()


@router.post("/rebuild-communities")
async def rebuild_communities(request: Request):
    """Trigger a full community detection rebuild."""
    knowledge_store = request.app.state.knowledge_store
    community_store = request.app.state.community_store

    start = time.time()

    # Step 1: Detect communities
    detector = CommunityDetector(knowledge_store)
    communities = await asyncio.to_thread(detector.detect)

    # Step 2: Store (without summaries for now -- summarization is Task 7)
    count = await community_store.replace_all(communities)

    duration = time.time() - start
    level_counts: dict[str, int] = {}
    for c in communities:
        key = f"level_{c['level']}"
        level_counts[key] = level_counts.get(key, 0) + 1

    return {
        "communities_built": count,
        "levels": level_counts,
        "duration_seconds": round(duration, 2),
    }
