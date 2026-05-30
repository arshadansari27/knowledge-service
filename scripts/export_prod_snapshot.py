#!/usr/bin/env python3
"""Export a prod knowledge-service snapshot (Postgres + oxigraph) for the eval harness.

Reads connection details from env vars only (no hardcoded credentials):
    KS_PROD_DATABASE_URL   postgres URL to dump (pg_dump is online/read-only safe)
    KS_PROD_OXIGRAPH_DIR   path to a COPY of the prod oxigraph data dir (offline dump)
    KS_SNAPSHOT_DIR        output directory (default ./data/snapshot)

Usage:
    KS_PROD_DATABASE_URL=postgresql://... \\
    KS_PROD_OXIGRAPH_DIR=/path/to/oxigraph-copy \\
    python scripts/export_prod_snapshot.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"error: required environment variable {name} is not set")
    return value


def _dump_postgres(database_url: str, out_dir: Path) -> Path:
    dump_path = out_dir / "knowledge.dump"
    subprocess.run(
        ["pg_dump", "--format=custom", "--no-owner", "--file", str(dump_path), database_url],
        check=True,
    )
    return dump_path


def _dump_oxigraph(oxigraph_dir: str, out_dir: Path) -> Path:
    """Dump a COPY of the prod oxigraph store to N-Quads (offline, read-only open)."""
    import pyoxigraph

    nq_path = out_dir / "oxigraph.nq"
    store = pyoxigraph.Store(oxigraph_dir)
    with open(nq_path, "wb") as fh:
        store.dump(output=fh, format=pyoxigraph.RdfFormat.N_QUADS)
    return nq_path


def main() -> None:
    database_url = _require_env("KS_PROD_DATABASE_URL")
    oxigraph_dir = _require_env("KS_PROD_OXIGRAPH_DIR")
    snapshot_dir = Path(os.environ.get("KS_SNAPSHOT_DIR", "./data/snapshot"))
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    pg_path = _dump_postgres(database_url, snapshot_dir)
    nq_path = _dump_oxigraph(oxigraph_dir, snapshot_dir)

    manifest = {
        "postgres_dump": pg_path.name,
        "oxigraph_nquads": nq_path.name,
        "postgres_dump_bytes": pg_path.stat().st_size,
        "oxigraph_nquads_bytes": nq_path.stat().st_size,
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"snapshot written to {snapshot_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
