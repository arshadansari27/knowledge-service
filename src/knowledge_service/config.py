from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    llm_embed_model: str = "nomic-embed-text"
    llm_chat_model: str = "qwen3:14b"
    llm_rag_model: str = ""
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

    # Ingestion pipeline
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    url_fetch_timeout: int = 30

    # Reader-side status filtering
    reader_exclude_inflight: bool = True  # env: READER_EXCLUDE_INFLIGHT

    # Eval harness
    eval_judge_base_url: str = "https://api.anthropic.com"
    eval_judge_model: str = "claude-opus-4-8"
    eval_judge_api_key: str = ""  # Anthropic key; required only when running the eval judge
    eval_concurrency: int = 4

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
