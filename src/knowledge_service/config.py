from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    llm_embed_model: str = "nomic-embed-text"
    llm_chat_model: str = "qwen3:14b"
    llm_rag_model: str = ""
    oxigraph_data_dir: str = "./data/oxigraph"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    admin_password: str  # Required — no default; also accepted as X-API-Key for m2m calls
    secret_key: str  # Required — no default; must be set via SECRET_KEY env var

    # Operational limits
    chunk_size: int = 4000
    chunk_overlap: int = 200
    max_chunks: int = 50
    embed_batch_size: int = 20
    entity_cache_max_size: int = 1000
    # Cap concurrent extraction LLM calls. Without this, a burst of N ingestion
    # jobs (typical aegis daily arxiv pull is 30–50) fires N parallel requests
    # at qwen3, which serves ~2–4 at a time on asif; the tail queues inside
    # Ollama past the 600s read timeout and the whole batch falls into a
    # retry-cascade. Pick 4 to match observed parallelism.
    extraction_max_concurrent: int = 4

    # Ingestion pipeline
    spacy_data_dir: str = "/app/data/spacy"
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    url_fetch_timeout: int = 30
    nlp_entity_confidence: float = 0.5

    # Reader-side status filtering
    reader_exclude_inflight: bool = True  # env: READER_EXCLUDE_INFLIGHT

    # Eval harness — judge runs via litellm's OpenAI-compatible chat-completions
    # endpoint (kimi-k2.5 by default; a different family from the gpt-oss SUT, so it
    # doesn't grade itself). Set base_url + api_key via env at eval time (same
    # litellm proxy + key as LLM_BASE_URL/LLM_API_KEY).
    eval_judge_base_url: str = ""
    eval_judge_model: str = "kimi-k2.5"
    eval_judge_api_key: str = ""  # litellm key; required only when running the eval judge
    eval_concurrency: int = 4

    # Background maintenance sweep (lowercases knowledge_type, remaps
    # spaCy NER labels to schema.org canonical). Set interval to 0 to disable.
    maintenance_interval_seconds: float = 21600.0  # 6h
    maintenance_initial_delay_seconds: float = 60.0

    # Retrieval: cap on knowledge-graph triples passed into the RAG prompt.
    # The 2026-05-31 eval showed graph-on flooded the prompt with ~97 triples
    # (max 239), which lowered answer faithfulness. Keep only the top-N highest
    # confidence triples. env: RAG_MAX_TRIPLES
    rag_max_triples: int = 15

    # Default retrieval mode for /api/ask when the request doesn't specify one:
    #   "auto"        — route by query intent (graph for entity/relationship
    #                   questions, chunks-only for plain semantic search). After
    #                   relevance-ranking the triples, the 2026-05-31 eval showed
    #                   the graph wins on entity questions and lifts correctness on
    #                   relationship questions while mildly hurting semantic search,
    #                   so route it where it helps. See
    #                   docs/kg-vs-rag-eval-findings.md.
    #   "full"        — always use the KG-augmented path.
    #   "chunks_only" — never use the graph (pure hybrid vector+BM25).
    # A per-request retrieval_mode always overrides this. env: RAG_DEFAULT_RETRIEVAL_MODE
    rag_default_retrieval_mode: str = "auto"

    # --- Graph quality levers (2026-05-31). All default to current behavior so a
    # deploy is a no-op until the eval picks winners; each is flipped via env per A/B.
    # See docs/superpowers/specs/2026-05-31-graph-quality-improvements-design.md
    # A — minimum cosine relevance a triple must clear to reach the prompt (0.0 = off).
    rag_triple_relevance_floor: float = 0.35  # env: RAG_TRIPLE_RELEVANCE_FLOOR
    # B — render triples as localized prose (+ source url) instead of raw-URI SPO.
    # Default OFF so a deploy is a true no-op (no prompt-shape change, no provenance
    # lookup) until the eval validates it. Flip on once measured.
    rag_verbalize_triples: bool = True  # env: RAG_VERBALIZE_TRIPLES
    # C — drop triples whose source document is already among the retrieved chunks.
    rag_triple_novelty_filter: bool = True  # env: RAG_TRIPLE_NOVELTY_FILTER
    # D — predicate-similarity triple lookup (relevance-blind). Off = skip it.
    rag_predicate_lookup_enabled: bool = False  # env: RAG_PREDICATE_LOOKUP_ENABLED
    # E — default answer strategy: "direct" (single call) or "verify" (graph fact-check).
    rag_default_answer_mode: str = "direct"  # env: RAG_DEFAULT_ANSWER_MODE

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
