import secrets

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    llm_embed_model: str = "nomic-embed-text"
    llm_chat_model: str = "qwen3:14b"
    llm_rag_model: str = ""
    oxigraph_data_dir: str = "./data/oxigraph"
    problog_rules_dir: str = "./src/knowledge_service/reasoning/rules"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    federation_enabled: bool = True
    federation_timeout: float = 3.0
    community_rebuild_interval: int = 0  # seconds, 0 = disabled
    admin_password: str  # Required — no default; also accepted as X-API-Key for m2m calls
    secret_key: str = secrets.token_hex(32)

    model_config = {"env_file": ".env"}


settings = Settings()
