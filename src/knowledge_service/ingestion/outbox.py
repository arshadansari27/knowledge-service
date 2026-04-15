"""Outbox for coordinating pyoxigraph writes with PG transactions.

See docs/superpowers/specs/2026-04-15-processphase-2pc-outbox-design.md
for the invariant and recovery semantics.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class OutboxStore:
    """Staging surface for pyoxigraph writes, written inside a PG transaction."""

    _STAGE_SQL = """
        INSERT INTO triple_outbox (
            triple_hash, operation, subject, predicate, object,
            confidence, knowledge_type, valid_from, valid_until,
            graph, payload
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING id
    """

    async def stage(
        self,
        conn: Any,
        *,
        operation: str,
        triple_hash: str,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float | None = None,
        knowledge_type: str | None = None,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        graph: str,
        payload: dict | None = None,
    ) -> int:
        """Insert a pending outbox row using the caller's connection/transaction.

        Returns the assigned id so the caller can drain exactly these rows.
        """
        payload_json = json.dumps(payload) if payload is not None else None
        return await conn.fetchval(
            self._STAGE_SQL,
            triple_hash,
            operation,
            subject,
            predicate,
            object_,
            confidence,
            knowledge_type,
            valid_from,
            valid_until,
            graph,
            payload_json,
        )


@dataclass
class AppliedEntry:
    id: int
    operation: str
    triple_hash: str
    is_new: bool | None  # None for ops where is_new is not meaningful


class OutboxDrainer:
    """Applies pending outbox rows to pyoxigraph and marks them applied in PG."""

    _SELECT_BY_IDS = """
        SELECT id, triple_hash, operation, subject, predicate, object,
               confidence, knowledge_type, valid_from, valid_until,
               graph, payload
        FROM triple_outbox
        WHERE applied_at IS NULL AND id = ANY($1::bigint[])
        ORDER BY id
        FOR UPDATE SKIP LOCKED
    """

    _SELECT_PENDING = """
        SELECT id, triple_hash, operation, subject, predicate, object,
               confidence, knowledge_type, valid_from, valid_until,
               graph, payload
        FROM triple_outbox
        WHERE applied_at IS NULL
        ORDER BY id
        FOR UPDATE SKIP LOCKED
    """

    _MARK_APPLIED = "UPDATE triple_outbox SET applied_at = NOW() WHERE id = $1"

    def __init__(self, pool: Any, triple_store: Any) -> None:
        self._pool = pool
        self._triples = triple_store

    async def drain_ids(self, ids: list[int]) -> list[AppliedEntry]:
        if not ids:
            return []
        return await self._drain_rows(self._SELECT_BY_IDS, ids)

    async def drain_pending(self, limit: int | None = None) -> list[AppliedEntry]:
        sql = self._SELECT_PENDING
        if limit is not None:
            sql = sql + f" LIMIT {int(limit)}"
        return await self._drain_rows(sql, None)

    async def _drain_rows(self, sql: str, arg: Any) -> list[AppliedEntry]:
        applied: list[AppliedEntry] = []
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(sql, arg) if arg is not None else await conn.fetch(sql)
                for row in rows:
                    result = await self._apply_row(dict(row))
                    if result is None:
                        continue
                    await conn.execute(self._MARK_APPLIED, row["id"])
                    applied.append(result)
        return applied

    async def _apply_row(self, row: dict) -> AppliedEntry | None:
        op = row["operation"]
        if op == "insert":
            return await self._apply_insert(row)
        logger.warning("OutboxDrainer: unknown operation %r (row id=%s)", op, row["id"])
        return None

    async def _apply_insert(self, row: dict) -> AppliedEntry:
        triple_hash, is_new = await asyncio.to_thread(
            self._triples.insert,
            row["subject"],
            row["predicate"],
            row["object"],
            row["confidence"],
            row["knowledge_type"],
            row["valid_from"],
            row["valid_until"],
            row["graph"],
        )
        return AppliedEntry(
            id=row["id"],
            operation="insert",
            triple_hash=triple_hash,
            is_new=is_new,
        )
