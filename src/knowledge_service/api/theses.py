"""Thesis API — create, manage, and monitor investment theses."""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from knowledge_service.ingestion.pipeline import compute_hash
from knowledge_service.ontology.uri import to_entity_uri, to_predicate_uri

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/theses", tags=["theses"])


class ThesisCreate(BaseModel):
    name: str
    description: str  # NL statement to decompose


class ThesisPatch(BaseModel):
    status: str | None = None
    add_claims: list[dict] | None = None  # [{triple_hash, subject, predicate, object, role}]
    remove_claims: list[str] | None = None  # [triple_hash, ...]


@router.post("")
async def create_thesis(body: ThesisCreate, request: Request):
    stores = request.app.state.stores
    thesis_id = await stores.theses.create(body.name, body.description)

    # Decompose description into claims via LLM
    extraction_client = getattr(request.app.state, "extraction_client", None)
    claims = []
    if extraction_client:
        raw_claims = await extraction_client.decompose_thesis(body.description)
        if raw_claims:
            for claim in raw_claims:
                s = claim.get("subject", "")
                p = claim.get("predicate", "")
                o = claim.get("object", "")
                s_uri = to_entity_uri(s)
                p_uri = to_predicate_uri(p)
                triple_hash = compute_hash({"subject": s_uri, "predicate": p_uri, "object": o})
                await stores.theses.add_claim(thesis_id, triple_hash, s, p, o, "supporting")
                claims.append(
                    {
                        "triple_hash": triple_hash,
                        "subject": s,
                        "predicate": p,
                        "object": o,
                        "role": "supporting",
                    }
                )

    return {"id": thesis_id, "status": "draft", "claims": claims}


@router.get("")
async def list_theses(request: Request, status: str | None = None):
    stores = request.app.state.stores
    return await stores.theses.list(status=status)


@router.get("/{thesis_id}")
async def get_thesis(thesis_id: str, request: Request):
    stores = request.app.state.stores
    thesis = await stores.theses.get(thesis_id)
    if not thesis:
        raise HTTPException(status_code=404, detail="Thesis not found")
    return thesis


@router.patch("/{thesis_id}")
async def patch_thesis(thesis_id: str, body: ThesisPatch, request: Request):
    stores = request.app.state.stores
    # Verify thesis exists
    thesis = await stores.theses.get(thesis_id)
    if not thesis:
        raise HTTPException(status_code=404, detail="Thesis not found")

    if body.status:
        await stores.theses.update(thesis_id, status=body.status)

    if body.add_claims:
        for claim in body.add_claims:
            await stores.theses.add_claim(
                thesis_id,
                claim["triple_hash"],
                claim["subject"],
                claim["predicate"],
                claim["object"],
                claim.get("role", "supporting"),
            )

    if body.remove_claims:
        for triple_hash in body.remove_claims:
            await stores.theses.remove_claim(thesis_id, triple_hash)

    return await stores.theses.get(thesis_id)


@router.delete("/{thesis_id}")
async def delete_thesis(thesis_id: str, request: Request):
    stores = request.app.state.stores
    thesis = await stores.theses.get(thesis_id)
    if not thesis:
        raise HTTPException(status_code=404, detail="Thesis not found")
    await stores.theses.update(thesis_id, status="archived")
    return {"status": "archived"}


@router.get("/{thesis_id}/breaks")
async def get_thesis_breaks(thesis_id: str, request: Request):
    stores = request.app.state.stores
    thesis = await stores.theses.get(thesis_id)
    if not thesis:
        raise HTTPException(status_code=404, detail="Thesis not found")
    # Find contradictions for all claims in this thesis
    claim_hashes = {c["triple_hash"] for c in thesis.get("claims", [])}
    if not claim_hashes:
        return {"breaks": []}
    breaks = await stores.theses.find_by_hashes(claim_hashes, status="active")
    return {"breaks": breaks}
