"""Community detection, storage, and summarization for global search."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CommunityStore:
    """Asyncpg-backed store for community data."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def replace_all(self, communities: list[dict]) -> int:
        """Delete all communities and insert new ones atomically."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM communities")
                for c in communities:
                    await conn.execute(
                        """INSERT INTO communities (level, label, summary, member_entities, member_count)
                           VALUES ($1, $2, $3, $4, $5)""",
                        c["level"],
                        c.get("label"),
                        c.get("summary"),
                        c["member_entities"],
                        c["member_count"],
                    )
        return len(communities)

    async def get_by_level(self, level: int) -> list[dict]:
        sql = "SELECT * FROM communities WHERE level = $1 ORDER BY member_count DESC"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, level)
        return [dict(r) for r in rows]

    async def get_all(self) -> list[dict]:
        sql = "SELECT * FROM communities ORDER BY level, member_count DESC"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)
        return [dict(r) for r in rows]

    async def get_member_entities(self) -> set[str]:
        """Return all entity URIs that belong to any community."""
        sql = "SELECT member_entities FROM communities"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)
        entities: set[str] = set()
        for r in rows:
            entities.update(r["member_entities"])
        return entities
