"""FastAPI application factory and lifespan management for the Knowledge Service."""

import asyncio
import logging
from contextlib import asynccontextmanager
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import AsyncIterator

import asyncpg.exceptions
from fastapi import FastAPI

from knowledge_service.clients.federation import FederationClient
from knowledge_service.clients.llm import EmbeddingClient, ExtractionClient, LLMClientError
from knowledge_service.clients.rag import RAGClient
from knowledge_service.config import settings
from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.reasoning.engine import ReasoningEngine
from knowledge_service.stores.embedding import EmbeddingStore
from knowledge_service.stores.entity_resolver import EntityResolver
from knowledge_service.stores.knowledge import KnowledgeStore
from knowledge_service.stores.rag import RAGRetriever
from knowledge_service.api import (
    health,
    content,
    claims,
    search,
    knowledge,
    contradictions,
    ask,
    changes,
)

logger = logging.getLogger(__name__)


async def run_migrations(pool: object, migrations_dir: str | Path = "migrations") -> None:
    """Run pending SQL migrations, tracked by schema_migrations table.

    Uses advisory lock to prevent concurrent runs.
    """
    import logging  # noqa: PLC0415

    log = logging.getLogger(__name__)
    migrations_path = Path(migrations_dir)
    if not migrations_path.exists():
        log.warning("Migrations dir not found: %s", migrations_path)
        return

    sql_files = sorted(migrations_path.glob("*.sql"))
    if not sql_files:
        log.info("No migrations found")
        return

    async with pool.acquire() as conn:
        await conn.execute("SELECT pg_advisory_lock(hashtext('knowledge_migrations'))")
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            applied = {
                row["filename"]
                for row in await conn.fetch("SELECT filename FROM schema_migrations")
            }
            pending = [f for f in sql_files if f.name not in applied]
            if not pending:
                log.info("Migrations up to date (%d total)", len(sql_files))
                return

            for sql_file in pending:
                sql = sql_file.read_text()
                try:
                    await conn.execute(sql)
                except asyncpg.exceptions.DuplicateTableError as exc:
                    log.warning(
                        "Migration %s: table already exists (applied externally): %s",
                        sql_file.name,
                        exc,
                    )
                except asyncpg.exceptions.DuplicateObjectError as exc:
                    log.warning(
                        "Migration %s: object already exists (applied externally): %s",
                        sql_file.name,
                        exc,
                    )
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES ($1)", sql_file.name
                )
                log.info("Migration applied: %s", sql_file.name)

            log.info("Migrations complete: %d applied, %d total", len(pending), len(sql_files))
        finally:
            await conn.execute("SELECT pg_advisory_unlock(hashtext('knowledge_migrations'))")


async def _seed_predicate_embeddings(embedding_store: EmbeddingStore, embedding_client) -> None:
    """Seed the predicate_embeddings table with canonical predicates.

    Embeds each canonical predicate label and upserts into the table so
    that predicate resolution has a warm vocabulary from the start.
    """
    import logging  # noqa: PLC0415

    from knowledge_service.ontology.namespaces import KS  # noqa: PLC0415  # noqa: F811

    log = logging.getLogger(__name__)

    # Use DomainRegistry predicates if available, otherwise fall back to hardcoded list
    registry = getattr(app.state, "domain_registry", None) if "app" in dir() else None
    if registry is not None:
        all_preds = registry.get_predicates()
        pred_labels = [p.label for p in all_preds]
        pred_uris = [p.uri for p in all_preds]
    else:
        _fallback = [
            "causes",
            "increases",
            "decreases",
            "inhibits",
            "activates",
            "is_a",
            "part_of",
            "located_in",
            "created_by",
            "depends_on",
            "related_to",
            "contains",
            "precedes",
            "follows",
            "has_property",
            "used_for",
            "produced_by",
            "associated_with",
        ]
        pred_labels = [p.replace("_", " ") for p in _fallback]
        pred_uris = [f"{KS}{p}" for p in _fallback]

    try:
        embeddings = await embedding_client.embed_batch(pred_labels)
    except (LLMClientError, OSError) as exc:
        log.warning("Failed to seed predicate embeddings — embedding API unavailable: %s", exc)
        return

    for uri, label, embedding in zip(pred_uris, pred_labels, embeddings):
        await embedding_store.insert_predicate_embedding(uri=uri, label=label, embedding=embedding)
    log.info("Seeded %d canonical predicate embeddings", len(pred_labels))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown.

    Startup:
    - Initialises the pyoxigraph KnowledgeStore and loads the base ontology.
    - Creates an asyncpg connection pool for PostgreSQL.
    - Runs pending database migrations.
    - Creates LLM API clients for embeddings and knowledge extraction.

    Shutdown:
    - Flushes the KnowledgeStore to disk.
    - Closes the asyncpg pool.
    - Closes the httpx client.
    """
    # --- Startup ---
    Path(settings.oxigraph_data_dir).mkdir(parents=True, exist_ok=True)
    app.state.knowledge_store = KnowledgeStore(data_dir=settings.oxigraph_data_dir)
    ontology_dir = Path(__file__).resolve().parent / "ontology"
    bootstrap_ontology(app.state.knowledge_store, ontology_dir)

    import asyncpg  # noqa: PLC0415 — deferred to avoid import cost in tests

    app.state.pg_pool = await asyncpg.create_pool(settings.database_url)
    await run_migrations(app.state.pg_pool)

    # Mark orphaned ingestion jobs as failed (lost on restart)
    async with app.state.pg_pool.acquire() as conn:
        updated = await conn.execute(
            """UPDATE ingestion_jobs SET status = 'failed',
                      error = '{"type": "ServiceRestart", "message": "interrupted by service restart", "phase": "unknown"}'
               WHERE status NOT IN ('completed', 'failed')"""
        )
        if updated != "UPDATE 0":
            logger.info("Marked orphaned ingestion jobs as failed: %s", updated)

    from knowledge_service.stores.graph_migration import migrate_to_named_graphs  # noqa: PLC0415

    await migrate_to_named_graphs(app.state.knowledge_store.store, app.state.pg_pool)

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

    # Load inverse predicate pairs from ontology for inverse_holds rule
    from knowledge_service.ontology.namespaces import KS_INVERSE_PREDICATE  # noqa: PLC0415

    inverse_quads = list(
        app.state.knowledge_store.store.quads_for_pattern(None, KS_INVERSE_PREDICATE, None, None)
    )
    inverse_pairs = [
        (q.subject.value.split("/")[-1], q.object.value.split("/")[-1]) for q in inverse_quads
    ]
    app.state.reasoning_engine.set_inverse_pairs(inverse_pairs)

    # Federation client (optional)
    federation_client = None
    if settings.federation_enabled:
        federation_client = FederationClient(timeout=settings.federation_timeout)
    app.state.federation_client = federation_client

    # Shared EmbeddingStore and EntityResolver
    embedding_store = EmbeddingStore(app.state.pg_pool)
    app.state.embedding_store = embedding_store

    # Community store for global search
    from knowledge_service.stores.community import CommunityStore  # noqa: PLC0415

    app.state.community_store = CommunityStore(app.state.pg_pool)
    app.state.entity_resolver = EntityResolver(
        app.state.knowledge_store,
        embedding_store,
        app.state.embedding_client,
        federation_client=federation_client,
    )

    # Seed canonical predicate embeddings (idempotent — upserts)
    await _seed_predicate_embeddings(embedding_store, app.state.embedding_client)

    # Query classifier
    from knowledge_service.clients.classifier import QueryClassifier  # noqa: PLC0415

    app.state.query_classifier = QueryClassifier(
        base_url=settings.llm_base_url,
        model=settings.llm_chat_model,
        api_key=settings.llm_api_key,
    )

    # RAG components
    rag_model = settings.llm_rag_model or settings.llm_chat_model
    app.state.rag_client = RAGClient(
        base_url=settings.llm_base_url,
        model=rag_model,
        api_key=settings.llm_api_key,
    )
    app.state.rag_retriever = RAGRetriever(
        embedding_client=app.state.embedding_client,
        embedding_store=embedding_store,
        knowledge_store=app.state.knowledge_store,
        community_store=app.state.community_store,
    )

    # Optional periodic community rebuild
    _rebuild_task = None
    if settings.community_rebuild_interval > 0:

        async def _community_rebuild_loop() -> None:
            while True:
                await asyncio.sleep(settings.community_rebuild_interval)
                try:
                    from knowledge_service.stores.community import (  # noqa: PLC0415
                        CommunityDetector,
                        CommunitySummarizer,
                    )

                    detector = CommunityDetector(app.state.knowledge_store)
                    communities = await asyncio.to_thread(detector.detect)
                    # Summarize each community via LLM (uses extraction model — high volume)
                    summarizer = CommunitySummarizer(
                        app.state.extraction_client.client,
                        app.state.knowledge_store,
                        model=app.state.extraction_client.model,
                    )
                    summarized = []
                    for c in communities:
                        summarized.append(await summarizer.summarize_one(c))
                    await app.state.community_store.replace_all(summarized)
                    logger.info("Periodic community rebuild: %d communities", len(summarized))
                except Exception as exc:
                    logger.warning("Periodic community rebuild failed: %s", exc)

        _rebuild_task = asyncio.create_task(_community_rebuild_loop())
        app.state._community_rebuild_task = _rebuild_task

    yield

    # --- Shutdown ---
    if hasattr(app.state, "_community_rebuild_task") and app.state._community_rebuild_task:
        app.state._community_rebuild_task.cancel()
        try:
            await app.state._community_rebuild_task
        except asyncio.CancelledError:
            pass
    app.state.knowledge_store.flush()
    await app.state.pg_pool.close()
    if app.state.federation_client is not None:
        await app.state.federation_client.close()
    await app.state.query_classifier.close()
    await app.state.rag_client.close()
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
        version=pkg_version("knowledge-service"),
        lifespan=lf,
    )
    app.include_router(health.router)
    app.include_router(content.router, prefix="/api")
    app.include_router(claims.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(knowledge.router, prefix="/api")
    app.include_router(contradictions.router, prefix="/api")
    app.include_router(ask.router, prefix="/api")
    app.include_router(changes.router)

    # Admin panel — store credentials on app.state so both middleware and login route use the same values
    from knowledge_service.admin.auth import AuthMiddleware, login_router
    from knowledge_service.admin.routes import router as admin_router

    app.state.admin_password = settings.admin_password
    app.state.secret_key = settings.secret_key

    from knowledge_service.admin.stats import router as stats_router
    from knowledge_service.admin.communities import router as communities_router
    from knowledge_service.admin.jobs import router as jobs_router

    app.include_router(login_router)
    app.include_router(admin_router)
    app.include_router(stats_router, prefix="/api/admin")
    app.include_router(communities_router, prefix="/api/admin")
    app.include_router(jobs_router, prefix="/api/admin")

    app.add_middleware(
        AuthMiddleware,
        admin_password=settings.admin_password,
        secret_key=settings.secret_key,
    )

    return app


app = create_app()
