"""Outbox for coordinating pyoxigraph writes with PG transactions.

See docs/superpowers/specs/2026-04-15-processphase-2pc-outbox-design.md
for the invariant and recovery semantics.
"""

from __future__ import annotations

import json
import logging
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
