"""Admin trigger for the maintenance sweep.

Lets an operator run cleanup immediately rather than waiting for the next
scheduled tick — useful right after deploying a fix that introduces a new
normalization rule, or for re-verifying convergence after re-ingest.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from knowledge_service.maintenance.normalizer import run_all

router = APIRouter()


@router.post("/maintenance/run")
async def trigger_maintenance(request: Request) -> dict:
    """Run every normalization sweep once. Returns the per-operation stats."""
    stores = request.app.state.stores
    return await run_all(stores)
