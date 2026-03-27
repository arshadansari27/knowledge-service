"""Admin thesis views."""

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/theses", tags=["admin-theses"])


@router.get("")
async def admin_list_theses(request: Request):
    stores = request.app.state.stores
    theses = await stores.theses.list()
    return {"theses": theses, "count": len(theses)}


@router.get("/{thesis_id}")
async def admin_get_thesis(thesis_id: str, request: Request):
    stores = request.app.state.stores
    thesis = await stores.theses.get(thesis_id)
    if not thesis:
        raise HTTPException(status_code=404, detail="Thesis not found")
    return thesis


@router.post("/{thesis_id}/activate")
async def admin_activate_thesis(thesis_id: str, request: Request):
    stores = request.app.state.stores
    thesis = await stores.theses.get(thesis_id)
    if not thesis:
        raise HTTPException(status_code=404, detail="Thesis not found")
    await stores.theses.update(thesis_id, status="active")
    return {"status": "active"}


@router.post("/{thesis_id}/archive")
async def admin_archive_thesis(thesis_id: str, request: Request):
    stores = request.app.state.stores
    thesis = await stores.theses.get(thesis_id)
    if not thesis:
        raise HTTPException(status_code=404, detail="Thesis not found")
    await stores.theses.update(thesis_id, status="archived")
    return {"status": "archived"}
