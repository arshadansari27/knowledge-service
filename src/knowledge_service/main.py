"""FastAPI application factory and lifespan management for the Knowledge Service."""

import logging
from contextlib import asynccontextmanager
from importlib.metadata import version as pkg_version
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from knowledge_service.clients.llm import EmbeddingClient
from knowledge_service.clients.rag import RAGClient
from knowledge_service.config import settings
from knowledge_service.stores import Stores
from knowledge_service.stores.content import ContentStore
from knowledge_service.stores.migrations import run_migrations
from knowledge_service.stores.rag import RAGRetriever
from knowledge_service.api import (
    health,
    content,
    search,
    ask,
    upload as upload_api,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown.

    Startup:
    - Creates an asyncpg connection pool for PostgreSQL.
    - Runs pending database migrations.
    - Creates LLM API clients for embeddings and RAG.

    Shutdown:
    - Closes the asyncpg pool and LLM clients.
    """
    # --- Startup ---
    import asyncpg  # noqa: PLC0415 — deferred to avoid import cost in tests

    pg_pool = await asyncpg.create_pool(settings.database_url)
    await run_migrations(pg_pool)

    # LLM clients
    embedding_client = EmbeddingClient(
        base_url=settings.llm_base_url,
        model=settings.llm_embed_model,
        api_key=settings.llm_api_key,
    )

    # Build Stores dataclass (chunk + embed path only)
    stores = Stores(
        content=ContentStore(pg_pool, exclude_inflight=settings.reader_exclude_inflight),
        pg_pool=pg_pool,
    )
    app.state.stores = stores

    # Mark orphaned ingestion jobs as failed (lost on restart)
    async with pg_pool.acquire() as conn:
        updated = await conn.execute(
            """UPDATE ingestion_jobs SET status = 'failed',
                      error = '{"type": "ServiceRestart", "message": "interrupted by service restart", "phase": "unknown"}'
               WHERE status NOT IN ('completed', 'failed')"""
        )
        if updated != "UPDATE 0":
            logger.info("Marked orphaned ingestion jobs as failed: %s", updated)

    # RAG components
    rag_model = settings.llm_rag_model or settings.llm_chat_model
    app.state.rag_client = RAGClient(
        base_url=settings.llm_base_url,
        model=rag_model,
        api_key=settings.llm_api_key,
    )

    # RAGRetriever — chunk-only path (embedding_store only)
    app.state.rag_retriever = RAGRetriever(
        embedding_client=embedding_client,
        embedding_store=stores.content,
    )

    # State references read by admin/* and api/health.py.
    app.state.embedding_client = embedding_client
    app.state.pg_pool = pg_pool

    yield

    # --- Shutdown ---
    await pg_pool.close()
    await app.state.rag_client.close()
    await embedding_client.close()


def create_app(use_lifespan: bool = True) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        use_lifespan: When True (default), attach the production lifespan
            context manager that connects to real services.  Pass False in
            tests to skip service initialisation and set ``app.state``
            manually instead.

    Returns:
        Configured FastAPI application instance.
    """
    lf = lifespan if use_lifespan else None
    app = FastAPI(
        title="Knowledge Service",
        version=pkg_version("knowledge-service"),
        lifespan=lf,
    )
    # Surviving routers: health, content (ingest/upload), search, ask
    app.include_router(health.router)
    app.include_router(content.router, prefix="/api")
    app.include_router(upload_api.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(ask.router, prefix="/api")

    # Admin panel — store credentials on app.state so both middleware and login route use the same
    from knowledge_service.admin.auth import AuthMiddleware, login_router
    from knowledge_service.admin.routes import router as admin_router

    app.state.admin_password = settings.admin_password
    app.state.secret_key = settings.secret_key

    from knowledge_service.admin.jobs import router as jobs_router
    from knowledge_service.admin.stats import router as stats_router

    app.include_router(login_router)
    app.include_router(admin_router)
    app.include_router(stats_router, prefix="/api/admin")
    app.include_router(jobs_router, prefix="/api/admin")

    app.add_middleware(
        AuthMiddleware,
        admin_password=settings.admin_password,
        secret_key=settings.secret_key,
    )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled exception on %s %s (query=%s)",
            request.method,
            request.url.path,
            request.url.query or "",
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "type": type(exc).__name__},
        )

    return app


app = create_app()
