#!/usr/bin/env python3
"""Five-minute demo of the knowledge service.

Ingests the bundled OpenAI Nov 2023 corpus, waits for the pipeline to finish,
then runs a small tour of the read-side APIs to show off the parts of the
system that are hard to see in a flat docs read: multi-source consolidation,
contradiction detection, and RAG with chunk-level provenance.

Usage:
    uv run python scripts/demo.py
    uv run python scripts/demo.py --corpus examples/openai-nov-2023
    uv run python scripts/demo.py --skip-ingest          # just rerun the queries
    KNOWLEDGE_URL=https://knowledge.hikmahtech.in \\
    KNOWLEDGE_API_KEY=... uv run python scripts/demo.py  # against a deployed instance

Defaults assume a docker-compose stack at http://localhost:8000 with
ADMIN_PASSWORD set; pass --api-key or KNOWLEDGE_API_KEY to authenticate.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

DEFAULT_CORPUS = "examples/openai-nov-2023"
POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 600


@dataclass
class CorpusDocument:
    """A demo document parsed from a markdown file with YAML-ish frontmatter."""

    path: Path
    title: str
    url: str
    source_type: str
    publication: str | None
    published_at: str | None
    tags: list[str]
    body: str


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Parse a minimal YAML-ish frontmatter block.

    Supports: `key: value` (string), `key: [a, b, c]` (inline list), and `key:` followed
    by `  - item` lines. Designed for the bundled demo corpus, not a general YAML parser.
    """
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    block = text[4:end]
    rest = text[end + 5 :]

    fields: dict[str, object] = {}
    current_list_key: str | None = None
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line:
            current_list_key = None
            continue
        if line.startswith("  - ") and current_list_key:
            existing = fields.setdefault(current_list_key, [])
            assert isinstance(existing, list)
            existing.append(line[4:].strip())
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not value:
            fields[key] = []
            current_list_key = key
            continue
        if value.startswith("[") and value.endswith("]"):
            fields[key] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
            current_list_key = None
            continue
        fields[key] = value
        current_list_key = None
    return fields, rest.lstrip("\n")


def load_corpus(directory: Path) -> list[CorpusDocument]:
    """Read all `.md` files in `directory` (excluding README.md) into CorpusDocument."""
    docs: list[CorpusDocument] = []
    for path in sorted(directory.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        if not meta.get("title") or not meta.get("url"):
            print(f"  skipping {path.name}: missing title or url in frontmatter", file=sys.stderr)
            continue
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        docs.append(
            CorpusDocument(
                path=path,
                title=str(meta["title"]),
                url=str(meta["url"]),
                source_type=str(meta.get("source_type", "article")),
                publication=str(meta["publication"]) if meta.get("publication") else None,
                published_at=str(meta["published_at"]) if meta.get("published_at") else None,
                tags=[str(t) for t in tags],
                body=body,
            )
        )
    return docs


def ingest(client: httpx.Client, doc: CorpusDocument) -> str:
    """POST a document and return the new content_id."""
    payload = {
        "url": doc.url,
        "title": doc.title,
        "raw_text": doc.body,
        "source_type": doc.source_type,
        "tags": doc.tags,
        "metadata": {
            k: v
            for k, v in {
                "publication": doc.publication,
                "published_at": doc.published_at,
            }.items()
            if v
        },
    }
    resp = client.post("/api/content", json=payload)
    resp.raise_for_status()
    return resp.json()["content_id"]


def wait_for_completion(client: httpx.Client, content_ids: list[str]) -> dict[str, str]:
    """Poll every `POLL_INTERVAL_SECONDS` until all jobs terminate. Returns id → status."""
    statuses: dict[str, str] = {}
    started = time.time()
    pending = set(content_ids)
    last_summary = ""
    while pending and time.time() - started < POLL_TIMEOUT_SECONDS:
        for cid in list(pending):
            resp = client.get(f"/api/content/{cid}/status")
            if resp.status_code != 200:
                continue
            payload = resp.json()
            state = payload.get("status", "unknown")
            statuses[cid] = state
            if state in ("completed", "failed"):
                pending.discard(cid)
        summary = "  " + "  ".join(
            f"{state}={sum(1 for s in statuses.values() if s == state)}"
            for state in (
                "embedding",
                "analyzing",
                "extracting",
                "resolving",
                "processing",
                "completed",
                "failed",
            )
            if any(s == state for s in statuses.values())
        )
        if summary != last_summary:
            print(summary)
            last_summary = summary
        if pending:
            time.sleep(POLL_INTERVAL_SECONDS)
    for cid in pending:
        statuses[cid] = statuses.get(cid, "timeout")
    return statuses


def section(title: str) -> None:
    print()
    print("─" * 78)
    print(f"  {title}")
    print("─" * 78)


def show_graph_stats(client: httpx.Client) -> None:
    """Counts of triples per named graph — gives a sense of trust-tier composition."""
    section("Triple counts per named graph")
    query = textwrap.dedent("""
        SELECT ?g (COUNT(*) AS ?count) WHERE {
          GRAPH ?g { ?s ?p ?o }
        }
        GROUP BY ?g
        ORDER BY DESC(?count)
    """).strip()
    resp = client.post("/api/knowledge/sparql", json={"query": query})
    if resp.status_code != 200:
        print(f"  (sparql endpoint returned {resp.status_code}; skipping)")
        return
    rows = resp.json()
    if not rows:
        print("  (graph is empty)")
        return
    width = max(len(str(row.get("g", ""))) for row in rows)
    for row in rows:
        graph_uri = str(row.get("g", "?"))
        count = row.get("count", "?")
        print(f"  {graph_uri.ljust(width)}  {count}")


def show_contradictions(client: httpx.Client, limit: int = 8) -> None:
    """List the contradictions surfaced by the engine."""
    section("Contradictions detected")
    resp = client.get("/api/knowledge/contradictions", params={"min_confidence": 0.0})
    if resp.status_code != 200:
        print(f"  (contradictions endpoint returned {resp.status_code}; skipping)")
        return
    rows = resp.json()
    if not rows:
        print("  none detected — extraction may not have produced overlapping predicates yet")
        return
    print(f"  {len(rows)} pair(s) found; showing up to {limit}\n")
    for row in rows[:limit]:
        a = row.get("claim_a", {})
        b = row.get("claim_b", {})
        prob = row.get("contradiction_probability", 0.0)
        print(f"  • subj : {short_uri(a.get('subject', '?'))}")
        print(f"    pred : {short_uri(a.get('predicate', '?'))}")
        print(f"    A    : {short_uri(a.get('object', '?'))}  (conf {a.get('confidence', 0):.2f})")
        print(f"    B    : {short_uri(b.get('object', '?'))}  (conf {b.get('confidence', 0):.2f})")
        print(f"    P(contra) = {prob:.2f}")
        print()


def ask_and_show(client: httpx.Client, question: str) -> None:
    """Run /api/ask and print the answer + the evidence snippets that backed it."""
    section(f"Q: {question}")
    resp = client.post(
        "/api/ask",
        json={"question": question, "max_sources": 8, "min_confidence": 0.0},
        timeout=120,
    )
    if resp.status_code != 200:
        print(f"  (ask endpoint returned {resp.status_code}; skipping)")
        print(f"  body: {resp.text[:300]}")
        return
    payload = resp.json()
    answer = payload.get("answer", "(no answer)")
    print()
    for line in textwrap.wrap(answer, width=78):
        print(f"  {line}")
    print()
    sources = payload.get("sources", [])
    if sources:
        print("  Sources cited:")
        for src in sources:
            title = src.get("title") or src.get("url") or "?"
            print(f"    • {title}  ({src.get('source_type', '?')})")
    contradictions = payload.get("contradictions", [])
    if contradictions:
        print(f"\n  ⚠  {len(contradictions)} contradiction(s) surfaced in the retrieved triples")
    evidence = payload.get("evidence", []) or []
    if evidence:
        print(f"\n  First evidence snippet (of {len(evidence)}):")
        first = evidence[0]
        chunk_text = first.get("chunk_text", "")[:240]
        print(f"    “{chunk_text}{'…' if len(first.get('chunk_text', '')) > 240 else ''}”")
        if first.get("source_url"):
            print(f"    — {first['source_url']}")


def short_uri(value: object) -> str:
    """Compact URI for terminal display."""
    text = str(value)
    if "/" in text:
        tail = text.rsplit("/", 1)[-1]
        return tail or text
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--corpus", default=DEFAULT_CORPUS, help=f"Corpus directory (default: {DEFAULT_CORPUS})"
    )
    parser.add_argument(
        "--server",
        default=os.getenv("KNOWLEDGE_URL", "http://localhost:8000"),
        help="Server URL (default: KNOWLEDGE_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("KNOWLEDGE_API_KEY", ""),
        help="API key (default: KNOWLEDGE_API_KEY env var)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion and run only the read-side tour",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        print(f"ERROR: corpus directory {corpus_dir} not found", file=sys.stderr)
        return 1

    docs = load_corpus(corpus_dir)
    if not docs:
        print(f"ERROR: no documents found in {corpus_dir}", file=sys.stderr)
        return 1

    client = httpx.Client(
        base_url=args.server,
        timeout=60,
        headers={"X-API-Key": args.api_key} if args.api_key else {},
    )

    health = client.get("/health")
    if health.status_code != 200:
        print(f"ERROR: {args.server}/health returned {health.status_code}", file=sys.stderr)
        print(f"       body: {health.text[:200]}", file=sys.stderr)
        return 1
    print(f"Connected to {args.server}")
    print(f"Corpus: {corpus_dir} ({len(docs)} documents)")

    if not args.skip_ingest:
        section(f"Ingesting {len(docs)} documents")
        content_ids: list[str] = []
        for doc in docs:
            try:
                cid = ingest(client, doc)
                content_ids.append(cid)
                print(f"  accepted  {doc.path.name}  → {cid}")
            except httpx.HTTPStatusError as exc:
                print(
                    f"  FAIL      {doc.path.name}  ({exc.response.status_code}: {exc.response.text[:120]})"
                )
            except httpx.HTTPError as exc:
                print(f"  FAIL      {doc.path.name}  ({exc})")

        if content_ids:
            print()
            print(
                f"  waiting for {len(content_ids)} jobs to finish (poll every {POLL_INTERVAL_SECONDS}s)…"
            )
            statuses = wait_for_completion(client, content_ids)
            completed = sum(1 for s in statuses.values() if s == "completed")
            failed = sum(1 for s in statuses.values() if s == "failed")
            other = len(statuses) - completed - failed
            print(f"  done: {completed} completed, {failed} failed, {other} unresolved")
            if completed == 0:
                print("\n  no documents completed — skipping query tour", file=sys.stderr)
                return 1

    show_graph_stats(client)
    show_contradictions(client)

    questions = [
        "Who was the CEO of OpenAI between 17 and 22 November 2023?",
        "What role did Microsoft play in the OpenAI board events?",
        "Why did the OpenAI board say it removed Sam Altman?",
    ]
    for question in questions:
        ask_and_show(client, question)

    print()
    print("Demo complete. The interesting bits:")
    print("  • The CEO predicate has multiple object values across the corpus (contradictions).")
    print(
        "  • Several documents agree on stable facts (Brockman is president, OpenAI built ChatGPT)"
    )
    print("    — those agreements feed Noisy-OR consolidation.")
    print("  • Each answer above cites the source chunks it was grounded in.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
