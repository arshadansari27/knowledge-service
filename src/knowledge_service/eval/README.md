# KS Eval Harness

Measures retrieval + answer quality and compares graph-on (`full`) vs graph-off
(`chunks_only`). Spec: `docs/superpowers/specs/2026-05-30-ks-eval-harness-design.md`.

## One-time: build the snapshot

```bash
# 1. Copy the prod oxigraph data dir locally (offline dump needs a copy, not the live store).
# 2. Export Postgres + oxigraph to ./data/snapshot:
KS_PROD_DATABASE_URL=postgresql://... \
KS_PROD_OXIGRAPH_DIR=/path/to/oxigraph-copy \
python scripts/export_prod_snapshot.py

# 3. Restore Postgres locally and load oxigraph N-Quads into a local store dir:
createdb knowledge_eval
pg_restore --no-owner --dbname "postgresql://localhost/knowledge_eval" data/snapshot/knowledge.dump
python -c "import pyoxigraph as o; s=o.Store('./data/oxigraph-eval'); s.bulk_load(open('data/snapshot/oxigraph.nq','rb'), o.RdfFormat.N_QUADS)"
```

## Build the golden set

```bash
python scripts/gen_golden_candidates.py --per-type 10 --out golden_candidates.json
# curate by hand -> src/knowledge_service/eval/golden.json (see plan Task 9)
```

## Run

```bash
export DATABASE_URL=postgresql://localhost/knowledge_eval
export OXIGRAPH_DATA_DIR=./data/oxigraph-eval
export EVAL_JUDGE_API_KEY=...   # Anthropic key for the judge
uv run python -m knowledge_service.eval --modes full,chunks_only --k 5 --timestamp 2026-05-30T1200
```

Prints a `mode x query_type` summary table and writes
`src/knowledge_service/eval/reports/<timestamp>.json`.

## Reading the result

Graph-on "does no harm" if `full >= chunks_only` (within run-to-run noise) on
aggregate faithfulness/correctness. Per-query-type rows show where `full` clearly
wins — candidates for intent-based routing rather than cutting.
