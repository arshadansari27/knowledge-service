"""Community detection, storage, and summarization for global search."""

from __future__ import annotations

import logging
from typing import Any

import httpx
import igraph

from knowledge_service._utils import _extract_json, _rdf_value_to_str

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


class CommunityDetector:
    """Extract entity graph from KnowledgeStore and run Leiden community detection."""

    def __init__(self, knowledge_store) -> None:
        self._ks = knowledge_store

    def detect(self) -> list[dict]:
        """Run Leiden at 2 resolution levels, return community assignments."""
        edges = self._extract_graph()
        if not edges:
            return []

        # Build igraph Graph
        entities: set[str] = set()
        for e in edges:
            entities.add(e["source"])
            entities.add(e["target"])

        entity_list = sorted(entities)
        entity_idx = {uri: i for i, uri in enumerate(entity_list)}

        g = igraph.Graph(n=len(entity_list), directed=False)
        g.vs["name"] = entity_list

        edge_list = []
        weights = []
        seen_edges: set[tuple[int, int]] = set()
        for e in edges:
            pair = (
                min(entity_idx[e["source"]], entity_idx[e["target"]]),
                max(entity_idx[e["source"]], entity_idx[e["target"]]),
            )
            if pair not in seen_edges:
                seen_edges.add(pair)
                edge_list.append(pair)
                weights.append(e["weight"])

        g.add_edges(edge_list)
        g.es["weight"] = weights

        communities: list[dict] = []

        # Level 0: fine-grained (resolution=1.0)
        partition_0 = g.community_leiden(weights="weight", resolution=1.0)
        for cluster_members in partition_0:
            if len(cluster_members) > 0:
                member_uris = [entity_list[i] for i in cluster_members]
                communities.append(
                    {
                        "level": 0,
                        "member_entities": member_uris,
                        "member_count": len(member_uris),
                    }
                )

        # Level 1: coarse (resolution=0.5)
        partition_1 = g.community_leiden(weights="weight", resolution=0.5)
        for cluster_members in partition_1:
            if len(cluster_members) > 0:
                member_uris = [entity_list[i] for i in cluster_members]
                communities.append(
                    {
                        "level": 1,
                        "member_entities": member_uris,
                        "member_count": len(member_uris),
                    }
                )

        return communities

    def _extract_graph(self) -> list[dict]:
        """Extract entity-to-entity edges from the knowledge store.

        Only includes edges between domain entities (ks-data: namespace).
        Filters out ontology/external URIs in Python to avoid slow SPARQL
        FILTER + RDF-star OPTIONAL combinations in pyoxigraph.
        """
        from knowledge_service.ontology.namespaces import KS_CONFIDENCE, KS_DATA

        sparql = f"""
            SELECT DISTINCT ?s ?o ?conf WHERE {{
                GRAPH ?g {{
                    ?s ?p ?o .
                }}
                OPTIONAL {{
                    GRAPH ?g {{
                        << ?s ?p ?o >> <{KS_CONFIDENCE.value}> ?conf .
                    }}
                }}
                FILTER(isIRI(?o))
            }}
        """
        rows = self._ks.query(sparql)
        edges = []
        for r in rows:
            s = r["s"].value if hasattr(r["s"], "value") else str(r["s"])
            o = r["o"].value if hasattr(r["o"], "value") else str(r["o"])
            # Only keep edges between domain entities
            if not s.startswith(KS_DATA) or not o.startswith(KS_DATA):
                continue
            conf = float(r["conf"].value) if r.get("conf") and r["conf"] else 0.5
            edges.append({"source": s, "target": o, "weight": conf})
        return edges


class CommunitySummarizer:
    """Generate LLM summaries for communities."""

    def __init__(
        self, llm_client: httpx.AsyncClient, knowledge_store: Any, model: str = ""
    ) -> None:
        self._client = llm_client
        self._ks = knowledge_store
        self._model = model

    async def summarize_one(self, community: dict) -> dict:
        """Generate label + summary for a single community. Returns updated community dict."""
        community = dict(community)
        try:
            context = self._build_context(community)
            prompt = (
                "Given these entities and their relationships, generate:\n"
                "1. A short label (3-5 words) for this community's theme\n"
                "2. A 2-3 sentence summary of what this community represents\n"
                "\n"
                f"{context}\n"
                "\n"
                'Return JSON: {"label": "...", "summary": "..."}'
            )

            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            parsed = _extract_json(raw)
            if parsed is None:
                raise ValueError(f"No JSON found in LLM response: {raw[:100]}")
            community["label"] = parsed.get("label")
            community["summary"] = parsed.get("summary")
        except Exception as exc:
            logger.warning("CommunitySummarizer: failed for community: %s", exc)
            community["label"] = None
            community["summary"] = None
        return community

    def _build_context(self, community: dict) -> str:
        members = community["member_entities"][:10]
        lines = [f"Entities: {', '.join(m.rsplit('/', 1)[-1] for m in members)}"]

        relationships: list[str] = []
        for uri in members[:5]:
            try:
                triples = self._ks.get_triples_by_subject(uri)
            except Exception:
                continue
            for t in triples[:5]:
                s = uri.rsplit("/", 1)[-1]
                p = _rdf_value_to_str(t.get("predicate", "")).rsplit("/", 1)[-1]
                o = _rdf_value_to_str(t.get("object", "")).rsplit("/", 1)[-1]
                relationships.append(f"{s} -> {p} -> {o}")

        if relationships:
            lines.append("Relationships:\n" + "\n".join(f"  - {r}" for r in relationships[:15]))

        return "\n".join(lines)
