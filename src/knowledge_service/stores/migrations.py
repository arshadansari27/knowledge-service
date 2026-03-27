"""SQL migration runner. Moved from main.py."""

import logging
from pathlib import Path

import asyncpg.exceptions

logger = logging.getLogger(__name__)


async def run_migrations(pool, migrations_dir: Path | str = "migrations"):
    migrations_dir = Path(migrations_dir)
    if not migrations_dir.exists():
        logger.warning("Migrations dir not found: %s", migrations_dir)
        return

    sql_files = sorted(migrations_dir.glob("*.sql"))
    if not sql_files:
        logger.info("No migrations found")
        return

    async with pool.acquire() as conn:
        await conn.execute("SELECT pg_advisory_lock(hashtext('knowledge_migrations'))")
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            applied = {
                row["filename"]
                for row in await conn.fetch("SELECT filename FROM schema_migrations")
            }
            pending = [f for f in sql_files if f.name not in applied]
            if not pending:
                logger.info("Migrations up to date (%d total)", len(sql_files))
                return

            for sql_file in pending:
                sql = sql_file.read_text()
                try:
                    await conn.execute(sql)
                except asyncpg.exceptions.DuplicateTableError as exc:
                    logger.warning("Migration %s: table already exists: %s", sql_file.name, exc)
                except asyncpg.exceptions.DuplicateObjectError as exc:
                    logger.warning("Migration %s: object already exists: %s", sql_file.name, exc)
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES ($1)", sql_file.name
                )
                logger.info("Migration applied: %s", sql_file.name)

            logger.info("Migrations complete: %d applied, %d total", len(pending), len(sql_files))
        finally:
            await conn.execute("SELECT pg_advisory_unlock(hashtext('knowledge_migrations'))")
