"""Periodic background runner for ``maintenance.normalizer.run_all``.

Started from ``main.py:lifespan``. One asyncio task per process — that's
fine because each operation is a single SPARQL UPDATE pass over pyoxigraph
and runs in seconds even at hundreds-of-thousands of triples.

Failure policy: log and continue. A failed sweep doesn't crash the
service; the next interval retries.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from knowledge_service.maintenance.normalizer import run_all

logger = logging.getLogger(__name__)


async def _loop(stores: Any, interval_seconds: float, initial_delay_seconds: float) -> None:
    if initial_delay_seconds > 0:
        await asyncio.sleep(initial_delay_seconds)
    while True:
        try:
            stats = await run_all(stores)
            logger.info("maintenance: sweep complete %s", stats)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("maintenance: sweep failed; will retry next interval")
        await asyncio.sleep(interval_seconds)


def start(
    stores: Any,
    interval_seconds: float,
    initial_delay_seconds: float = 60.0,
) -> asyncio.Task:
    """Spawn the background sweep loop. Returns the Task so the lifespan
    handler can cancel it on shutdown.

    The initial delay (default 60s) lets the rest of the service warm up
    before the first cleanup pass — at startup we're competing with
    migrations, the outbox drain, and the spaCy KB load."""
    task = asyncio.create_task(
        _loop(stores, interval_seconds, initial_delay_seconds),
        name="maintenance-sweep",
    )
    logger.info(
        "maintenance: scheduler started, interval=%.0fs initial_delay=%.0fs",
        interval_seconds,
        initial_delay_seconds,
    )
    return task
