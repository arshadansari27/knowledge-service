"""FastAPI application factory and lifespan management for the Knowledge Service."""

import asyncio
import logging
from contextlib import asynccontextmanager
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from knowledge_service.clients.llm import EmbeddingClient, ExtractionClient
from knowledge_service.clients.rag import RAGClient
from knowledge_service.config import settings
from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.ontology.registry import DomainRegistry
from knowledge_service.stores import Stores
from knowledge_service.stores.content import ContentStore
from knowledge_service.stores.entities import EntityStore
from knowledge_service.stores.migrations import run_migrations
from knowledge_service.stores.provenance import ProvenanceStore
from knowledge_service.stores.rag import RAGRetriever
from knowledge_service.stores.theses import ThesisStore
from knowledge_service.stores.triples import TripleStore
from knowledge_service.api import (
    health,
    content,
    claims,
    search,
    knowledge,
    contradictions,
    ask,
    changes,
    upload as upload_api,
)
from knowledge_service.api.theses import router as theses_router
from knowledge_service.admin.theses import router as admin_theses_router

logger = logging.getLogger(__name__)


def _canonical_predicate_entries(
    domain_registry: DomainRegistry | None,
) -> list[tuple[str, str]]:
    """Return canonical (uri, label) pairs to seed predicate_embeddings with.

    Prefers predicates declared in the loaded ontology; falls back to a
    hard-coded list matching ontology/domains/base.ttl when the registry is
    absent (e.g. minimal test setups).
    """
    from knowledge_service.ontology.namespaces import KS  # noqa: PLC0415

    if domain_registry is not None:
        return [(p.uri, p.label) for p in domain_registry.get_predicates()]

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
    return [(f"{KS}{p}", p.replace("_", " ")) for p in _fallback]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown.

    Startup:
    - Initialises the pyoxigraph TripleStore and loads the base ontology.
    - Creates an asyncpg connection pool for PostgreSQL.
    - Runs pending database migrations.
    - Creates LLM API clients for embeddings and knowledge extraction.

    Shutdown:
    - Flushes the TripleStore to disk.
    - Closes the asyncpg pool.
    - Closes LLM clients.
    """
    # --- Startup ---
    Path(settings.oxigraph_data_dir).mkdir(parents=True, exist_ok=True)
    triple_store = TripleStore(data_dir=settings.oxigraph_data_dir)
    ontology_dir = Path(__file__).resolve().parent / "ontology"
    bootstrap_ontology(triple_store, ontology_dir)

    import asyncpg  # noqa: PLC0415 — deferred to avoid import cost in tests

    pg_pool = await asyncpg.create_pool(settings.database_url)
    await run_migrations(pg_pool)

    # Mark orphaned ingestion jobs as failed (lost on restart)
    async with pg_pool.acquire() as conn:
        updated = await conn.execute(
            """UPDATE ingestion_jobs SET status = 'failed',
                      error = '{"type": "ServiceRestart", "message": "interrupted by service restart", "phase": "unknown"}'
               WHERE status NOT IN ('completed', 'failed')"""
        )
        if updated != "UPDATE 0":
            logger.info("Marked orphaned ingestion jobs as failed: %s", updated)

    from knowledge_service.stores.graph_migration import migrate_to_named_graphs  # noqa: PLC0415

    await migrate_to_named_graphs(triple_store.store, pg_pool)

    # LLM clients
    embedding_client = EmbeddingClient(
        base_url=settings.llm_base_url,
        model=settings.llm_embed_model,
        api_key=settings.llm_api_key,
    )

    # Build Stores dataclass
    from knowledge_service.ingestion.outbox import OutboxStore, OutboxDrainer  # noqa: PLC0415

    entity_store = EntityStore(pg_pool, embedding_client)
    stores = Stores(
        triples=triple_store,
        content=ContentStore(pg_pool, exclude_inflight=settings.reader_exclude_inflight),
        entities=entity_store,
        provenance=ProvenanceStore(pg_pool),
        theses=ThesisStore(pg_pool),
        outbox=OutboxStore(),
        pg_pool=pg_pool,
    )
    app.state.stores = stores
    app.state.outbox_drainer = OutboxDrainer(pg_pool, triple_store)

    drained = await app.state.outbox_drainer.drain_pending()
    if drained:
        logger.info(
            "Startup drain: applied %d pending outbox rows from prior run",
            len(drained),
        )

    # DomainRegistry
    prompts_dir = ontology_dir / "prompts"
    domain_registry = DomainRegistry(triple_store, prompts_dir)
    domain_registry.load()
    app.state.domain_registry = domain_registry

    # Inference engine
    from knowledge_service.reasoning.engine import (  # noqa: PLC0415
        InferenceEngine,
        InverseRule,
        TransitiveRule,
        TypeInheritanceRule,
    )

    inference_rules = [InverseRule(), TransitiveRule(), TypeInheritanceRule()]
    inference_engine = InferenceEngine(triple_store, inference_rules, max_depth=3)
    inference_engine.configure()
    app.state.inference_engine = inference_engine
    logger.info("InferenceEngine initialized with %d rules", len(inference_rules))

    # ExtractionClient with registry
    extraction_client = ExtractionClient(
        base_url=settings.llm_base_url,
        model=settings.llm_chat_model,
        api_key=settings.llm_api_key,
        registry=domain_registry,
    )
    app.state.extraction_client = extraction_client
    app.state.embedding_client = embedding_client

    # Federation client (optional enrichment)
    federation_client = None
    if settings.federation_enabled:
        from knowledge_service.clients.federation import FederationClient  # noqa: PLC0415

        federation_client = FederationClient(timeout=settings.federation_timeout)
        app.state.federation_client = federation_client

    # Parser registry
    from knowledge_service.parsing import ParserRegistry  # noqa: PLC0415
    from knowledge_service.parsing.text import TextParser  # noqa: PLC0415
    from knowledge_service.parsing.pdf import PdfParser  # noqa: PLC0415
    from knowledge_service.parsing.html import HtmlParser  # noqa: PLC0415
    from knowledge_service.parsing.structured import StructuredParser  # noqa: PLC0415
    from knowledge_service.parsing.image import ImageParser  # noqa: PLC0415

    parser_registry = ParserRegistry()
    parser_registry.register(TextParser())
    parser_registry.register(PdfParser())
    parser_registry.register(HtmlParser())
    parser_registry.register(StructuredParser())
    parser_registry.register(ImageParser())
    app.state.parser_registry = parser_registry

    # Make parser_registry available to content endpoint module
    import knowledge_service.api.content as _content_mod  # noqa: PLC0415

    _content_mod._parser_registry = parser_registry

    # spaCy NLP pipeline (optional — graceful degradation if unavailable)
    from knowledge_service.nlp.bootstrap import load_spacy_nlp  # noqa: PLC0415

    nlp = load_spacy_nlp(settings.spacy_data_dir)
    app.state.nlp = nlp
    if nlp:
        pipe_names = [name for name, _ in nlp.pipeline]
        if "entityLinker" in pipe_names:
            app.state.nlp_status = "ok"
        else:
            app.state.nlp_status = "degraded: entity linker not loaded"
        logger.info("spaCy NLP pipeline loaded — NLP pre-pass enabled")
    else:
        app.state.nlp_status = "unavailable: spaCy not loaded"
        logger.info("spaCy unavailable — NLP pre-pass disabled, LLM-only extraction")

    # Register canonical predicates for lazy seeding. EntityStore will seed on
    # first resolve_predicate() call (or retry per-call if the embedding backend
    # is temporarily down). We kick off one best-effort attempt now so the
    # warm-path stays warm when everything is healthy.
    entity_store.set_predicate_seed(_canonical_predicate_entries(domain_registry))
    try:
        await entity_store.ensure_predicates_seeded()
    except Exception as exc:
        logger.warning("Startup predicate seed attempt raised — lazy retry will cover: %s", exc)

    # RAG components
    rag_model = settings.llm_rag_model or settings.llm_chat_model
    app.state.rag_client = RAGClient(
        base_url=settings.llm_base_url,
        model=rag_model,
        api_key=settings.llm_api_key,
    )

    # Classify client (reuses extraction_client as the LLM backend for query classification)
    from knowledge_service.clients.base import BaseLLMClient  # noqa: PLC0415

    classify_client = BaseLLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_chat_model,
        api_key=settings.llm_api_key,
    )

    # Community store
    from knowledge_service.stores.community import CommunityStore  # noqa: PLC0415

    app.state.community_store = CommunityStore(pg_pool)
    app.state._last_community_rebuild = 0.0

    app.state.rag_retriever = RAGRetriever(
        embedding_client=embedding_client,
        embedding_store=stores.content,
        knowledge_store=triple_store,
        community_store=app.state.community_store,
        entity_store=entity_store,
        classify_client=classify_client,
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

                    detector = CommunityDetector(triple_store)
                    communities = await asyncio.to_thread(detector.detect)
                    summarizer = CommunitySummarizer(
                        extraction_client.client,
                        triple_store,
                        model=extraction_client.model,
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

    # BACKWARD COMPAT: Keep old state references for any code not yet migrated
    app.state.knowledge_store = triple_store
    app.state.embedding_store = stores.content
    app.state.pg_pool = pg_pool
    app.state.entity_resolver = None  # Removed — entity resolution is in EntityStore now

    yield

    # --- Shutdown ---
    if hasattr(app.state, "_community_rebuild_task") and app.state._community_rebuild_task:
        app.state._community_rebuild_task.cancel()
        try:
            await app.state._community_rebuild_task
        except asyncio.CancelledError:
            pass
    triple_store.flush()
    await pg_pool.close()
    await classify_client.close()
    await app.state.rag_client.close()
    await embedding_client.close()
    await extraction_client.close()
    if federation_client is not None:
        await federation_client.close()


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
    app.include_router(upload_api.router, prefix="/api")
    app.include_router(claims.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(knowledge.router, prefix="/api")
    app.include_router(contradictions.router, prefix="/api")
    app.include_router(ask.router, prefix="/api")
    app.include_router(changes.router)
    app.include_router(theses_router)
    app.include_router(admin_theses_router)

    # Admin panel — store credentials on app.state so both middleware and login route use the same
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
