"""Periodic, idempotent cleanup of accumulated data quality drift.

Why this exists: extraction is noisy, and even fixed pipelines leave
historical residue. We don't drop and re-ingest 141k triples to undo a
casing bug — we run a janitor.

See ``normalizer.py`` for the operations; ``scheduler.py`` for the
background task; ``src/knowledge_service/admin/maintenance.py`` for the
manual trigger endpoint.
"""
