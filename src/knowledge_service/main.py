"""FastAPI application factory and lifespan management for the Knowledge Service."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from knowledge_service.clients.federation import FederationClient
from knowledge_service.clients.llm import EmbeddingClient, ExtractionClient
from knowledge_service.config import settings
from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.reasoning.engine import ReasoningEngine
from knowledge_service.stores.embedding import EmbeddingStore
from knowledge_service.stores.entity_resolver import EntityResolver
from knowledge_service.stores.knowledge import KnowledgeStore
from knowledge_service.api import health, content, claims, search, knowledge, contradictions


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown.

    Startup:
    - Initialises the pyoxigraph KnowledgeStore and loads the base ontology.
    - Creates an asyncpg connection pool for PostgreSQL.
    - Creates LLM API clients for embeddings and knowledge extraction.

    Shutdown:
    - Flushes the KnowledgeStore to disk.
    - Closes the asyncpg pool.
    - Closes the httpx client.
    """
    # --- Startup ---
    Path(settings.oxigraph_data_dir).mkdir(parents=True, exist_ok=True)
    app.state.knowledge_store = KnowledgeStore(data_dir=settings.oxigraph_data_dir)
    bootstrap_ontology(app.state.knowledge_store._store)

    import asyncpg  # noqa: PLC0415 — deferred to avoid import cost in tests

    app.state.pg_pool = await asyncpg.create_pool(settings.database_url)

    app.state.embedding_client = EmbeddingClient(
        base_url=settings.llm_base_url,
        model=settings.llm_embed_model,
        api_key=settings.llm_api_key,
    )

    app.state.extraction_client = ExtractionClient(
        base_url=settings.llm_base_url,
        model=settings.llm_chat_model,
        api_key=settings.llm_api_key,
    )

    app.state.reasoning_engine = ReasoningEngine(rules_dir=settings.problog_rules_dir)

    # Federation client (optional)
    federation_client = None
    if settings.federation_enabled:
        federation_client = FederationClient(timeout=settings.federation_timeout)
    app.state.federation_client = federation_client

    # Shared EmbeddingStore and EntityResolver
    embedding_store = EmbeddingStore(app.state.pg_pool)
    app.state.embedding_store = embedding_store
    app.state.entity_resolver = EntityResolver(
        app.state.knowledge_store,
        embedding_store,
        app.state.embedding_client,
        federation_client=federation_client,
    )

    yield

    # --- Shutdown ---
    app.state.knowledge_store.flush()
    await app.state.pg_pool.close()
    if app.state.federation_client is not None:
        await app.state.federation_client.close()
    await app.state.embedding_client.close()
    await app.state.extraction_client.close()


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
        version="0.1.0",
        lifespan=lf,
    )
    app.include_router(health.router)
    app.include_router(content.router, prefix="/api")
    app.include_router(claims.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(knowledge.router, prefix="/api")
    app.include_router(contradictions.router, prefix="/api")
    return app


app = create_app()
