from dataclasses import dataclass

from knowledge_service.stores.content import ContentStore


@dataclass
class Stores:
    """Surviving stores: content (chunks + embeddings) and the DB pool."""

    content: ContentStore
    pg_pool: object  # asyncpg.Pool
