"""POST /api/content/upload — file upload endpoint with format detection and parsing."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from knowledge_service.api.content import _accept_content_request, _run_ingestion_worker
from knowledge_service.config import settings
from knowledge_service.models import ContentAcceptedResponse, ContentRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/content/upload")
async def upload_content(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    url: str | None = Form(None),
    title: str | None = Form(None),
    source_type: str | None = Form(None),
    tags: str | None = Form(None),
    domains: str | None = Form(None),
    metadata: str | None = Form(None),
):
    """Upload a file for ingestion.

    Accepts multipart/form-data with a file and optional metadata fields.
    Detects format, parses the file, and feeds into the content ingestion pipeline.
    """
    from knowledge_service.api.content import _parser_registry  # noqa: PLC0415

    # Read file content
    data = await file.read()

    # Check file size
    if len(data) > settings.max_upload_size:
        return JSONResponse(
            status_code=413,
            content={
                "detail": f"File too large: {len(data)} bytes exceeds {settings.max_upload_size} byte limit"
            },
        )

    # Detect format
    if _parser_registry is None:
        return JSONResponse(
            status_code=500,
            content={"detail": "Parser registry not initialized"},
        )

    filename = file.filename or ""
    content_type = file.content_type
    fmt = _parser_registry.detect_format(
        content_type=content_type,
        url=filename,
        data=data,
    )

    parser = _parser_registry.get_parser(fmt)
    if parser is None:
        return JSONResponse(
            status_code=422,
            content={"detail": f"No parser available for format: {fmt}"},
        )

    # Parse
    try:
        parsed = await parser.parse(data, content_type=content_type)
    except Exception as exc:
        logger.warning("Failed to parse uploaded file %s: %s", filename, exc)
        return JSONResponse(
            status_code=422,
            content={"detail": f"Failed to parse file: {exc}"},
        )

    if not parsed.text or not parsed.text.strip():
        return JSONResponse(
            status_code=422,
            content={"detail": "No text could be extracted from the uploaded file"},
        )

    # Parse optional JSON fields
    parsed_tags: list[str] = []
    if tags:
        try:
            parsed_tags = json.loads(tags)
        except (json.JSONDecodeError, TypeError):
            parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]

    parsed_domains: list[str] | None = None
    if domains:
        try:
            parsed_domains = json.loads(domains)
        except (json.JSONDecodeError, TypeError):
            parsed_domains = [d.strip() for d in domains.split(",") if d.strip()]

    parsed_metadata: dict = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            pass

    # Build ContentRequest
    body = ContentRequest(
        url=url or f"upload://{filename}",
        title=title or parsed.title or filename,
        raw_text=parsed.text,
        source_type=source_type or fmt,
        tags=parsed_tags,
        domains=parsed_domains,
        metadata=parsed_metadata,
    )

    stores = request.app.state.stores
    result = await _accept_content_request(body, stores)

    if result.get("status_code") == 422:
        return JSONResponse(status_code=422, content={"detail": result["error"]})
    if result.get("conflict"):
        return JSONResponse(
            status_code=409,
            content={"detail": "Active ingestion job exists for this content"},
        )

    background_tasks.add_task(
        _run_ingestion_worker,
        result["job_id"],
        result["content_id"],
        result["body"],
        result["chunk_records"],
        request.app.state,
    )

    return JSONResponse(
        ContentAcceptedResponse(
            content_id=result["content_id"],
            job_id=result["job_id"],
            chunks_total=result["chunks_total"],
            chunks_capped_from=result["chunks_capped_from"],
        ).model_dump(),
        status_code=202,
    )
