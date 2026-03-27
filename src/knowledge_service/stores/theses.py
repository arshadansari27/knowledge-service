"""Thesis storage — named collections of claims with break detection."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ALLOWED_UPDATE_FIELDS = {"name", "description", "status", "owner"}


class ThesisStore:
    def __init__(self, pool):
        self._pool = pool

    async def create(self, name: str, description: str, owner: str | None = None) -> str:
        async with self._pool.acquire() as conn:
            return str(
                await conn.fetchval(
                    "INSERT INTO theses (name, description, owner) VALUES ($1, $2, $3) RETURNING id",
                    name,
                    description,
                    owner,
                )
            )

    async def get(self, thesis_id: str) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM theses WHERE id = $1", thesis_id)
            if not row:
                return None
            claims = await conn.fetch(
                "SELECT * FROM thesis_claims WHERE thesis_id = $1 ORDER BY added_at", thesis_id
            )
            return {**dict(row), "claims": [dict(c) for c in claims]}

    async def list(self, status: str | None = None) -> list[dict]:
        async with self._pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    "SELECT * FROM theses WHERE status = $1 ORDER BY updated_at DESC", status
                )
            else:
                rows = await conn.fetch("SELECT * FROM theses ORDER BY updated_at DESC")
            return [dict(r) for r in rows]

    async def update(self, thesis_id: str, **fields) -> None:
        if not fields:
            return
        invalid = set(fields) - _ALLOWED_UPDATE_FIELDS
        if invalid:
            raise ValueError(f"Invalid fields: {invalid}")
        sets = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(fields))
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE theses SET {sets} WHERE id = $1", thesis_id, *fields.values()
            )

    async def add_claim(
        self,
        thesis_id: str,
        triple_hash: str,
        subject: str,
        predicate: str,
        object_: str,
        role: str = "supporting",
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO thesis_claims (thesis_id, triple_hash, subject, predicate, object, role)
                   VALUES ($1, $2, $3, $4, $5, $6)
                   ON CONFLICT (thesis_id, triple_hash) DO NOTHING""",
                thesis_id,
                triple_hash,
                subject,
                predicate,
                object_,
                role,
            )

    async def remove_claim(self, thesis_id: str, triple_hash: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM thesis_claims WHERE thesis_id = $1 AND triple_hash = $2",
                thesis_id,
                triple_hash,
            )

    async def find_by_hashes(self, hashes: set[str], status: str = "active") -> list[dict]:
        if not hashes:
            return []
        placeholders = ", ".join(f"${i + 2}" for i in range(len(hashes)))
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT tc.*, t.name FROM thesis_claims tc
                    JOIN theses t ON tc.thesis_id = t.id
                    WHERE t.status = $1 AND tc.triple_hash IN ({placeholders})""",
                status,
                *hashes,
            )
            return [dict(r) for r in rows]

    async def find_breaks_for_entity(self, entity_uri: str, since) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT tc.*, t.name FROM thesis_claims tc
                   JOIN theses t ON tc.thesis_id = t.id
                   WHERE t.status = 'active' AND tc.subject = $1 AND tc.added_at >= $2""",
                entity_uri,
                since,
            )
            return [dict(r) for r in rows]
