"""FederationClient: SPARQL-over-HTTP client for DBpedia and Wikidata federation."""

from __future__ import annotations

import logging
from xml.etree import ElementTree

import httpx

logger = logging.getLogger(__name__)

SPARQL_NS = "http://www.w3.org/2005/sparql-results#"


def _parse_sparql_xml(xml_text: str) -> list[dict[str, str]]:
    """Parse SPARQL XML results into a list of dicts."""
    root = ElementTree.fromstring(xml_text)
    results = []
    for result in root.findall(f".//{{{SPARQL_NS}}}result"):
        row: dict[str, str] = {}
        for binding in result.findall(f"{{{SPARQL_NS}}}binding"):
            name = binding.attrib["name"]
            uri_el = binding.find(f"{{{SPARQL_NS}}}uri")
            lit_el = binding.find(f"{{{SPARQL_NS}}}literal")
            if uri_el is not None and uri_el.text:
                row[name] = uri_el.text
            elif lit_el is not None and lit_el.text:
                row[name] = lit_el.text
        results.append(row)
    return results


class FederationClient:
    """SPARQL-over-HTTP client for DBpedia and Wikidata federation."""

    ENDPOINTS = {
        "dbpedia": "https://dbpedia.org/sparql",
        "wikidata": "https://query.wikidata.org/sparql",
    }
    USER_AGENT = "KnowledgeService/0.1.0"

    def __init__(self, timeout: float = 3.0) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": self.USER_AGENT},
        )

    async def _sparql_query(self, endpoint: str, query: str) -> list[dict[str, str]]:
        """Execute a SPARQL SELECT query and return parsed results."""
        try:
            response = await self._client.get(
                endpoint,
                params={"query": query, "format": "application/sparql-results+xml"},
            )
            response.raise_for_status()
            return _parse_sparql_xml(response.text)
        except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
            logger.debug("Federation query to %s failed: %s", endpoint, exc)
            return []
        except ElementTree.ParseError as exc:
            logger.debug("Failed to parse SPARQL XML from %s: %s", endpoint, exc)
            return []

    async def lookup_entity(self, label: str) -> dict | None:
        """Search DBpedia then Wikidata for a matching entity.

        Returns {uri, rdf_type, description} if found, None otherwise.
        """
        result = await self._lookup_dbpedia(label)
        if result:
            return result
        return await self._lookup_wikidata(label)

    async def _lookup_dbpedia(self, label: str) -> dict | None:
        escaped = label.replace('"', '\\"')
        query = f"""
            SELECT ?entity ?type ?comment WHERE {{
                ?entity rdfs:label "{escaped}"@en .
                OPTIONAL {{ ?entity rdf:type ?type .
                           FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/")) }}
                OPTIONAL {{ ?entity rdfs:comment ?comment . FILTER(LANG(?comment) = "en") }}
            }} LIMIT 1
        """
        rows = await self._sparql_query(self.ENDPOINTS["dbpedia"], query)
        if not rows:
            return None
        row = rows[0]
        return {
            "uri": row.get("entity", ""),
            "rdf_type": row.get("type", ""),
            "description": row.get("comment", ""),
        }

    async def _lookup_wikidata(self, label: str) -> dict | None:
        escaped = label.replace('"', '\\"')
        query = f"""
            SELECT ?entity ?type ?description WHERE {{
                ?entity rdfs:label "{escaped}"@en .
                OPTIONAL {{ ?entity wdt:P31 ?type }}
                OPTIONAL {{ ?entity schema:description ?description .
                           FILTER(LANG(?description) = "en") }}
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
            }} LIMIT 1
        """
        rows = await self._sparql_query(self.ENDPOINTS["wikidata"], query)
        if not rows:
            return None
        row = rows[0]
        return {
            "uri": row.get("entity", ""),
            "rdf_type": row.get("type", ""),
            "description": row.get("description", ""),
        }

    async def enrich_entity(self, uri: str) -> list[dict]:
        """Fetch type, description, sameAs links for a known external URI.

        Returns list of triples: [{subject, predicate, object}].
        Returns [] on timeout or error.
        """
        if "dbpedia.org" in uri:
            return await self._enrich_dbpedia(uri)
        if "wikidata.org" in uri:
            return await self._enrich_wikidata(uri)
        return []

    async def _enrich_dbpedia(self, uri: str) -> list[dict]:
        query = f"""
            SELECT ?p ?o WHERE {{
                <{uri}> ?p ?o .
                FILTER(?p IN (
                    rdf:type,
                    rdfs:comment,
                    owl:sameAs,
                    <http://dbpedia.org/ontology/abstract>
                ))
                FILTER(
                    !isLiteral(?o) ||
                    LANG(?o) = "" ||
                    LANG(?o) = "en"
                )
            }} LIMIT 20
        """
        rows = await self._sparql_query(self.ENDPOINTS["dbpedia"], query)
        return [
            {"subject": uri, "predicate": row["p"], "object": row["o"]}
            for row in rows
            if "p" in row and "o" in row
        ]

    async def _enrich_wikidata(self, uri: str) -> list[dict]:
        query = f"""
            SELECT ?p ?o WHERE {{
                <{uri}> ?p ?o .
                FILTER(?p IN (
                    wdt:P31,
                    schema:description,
                    wdt:P279
                ))
                FILTER(
                    !isLiteral(?o) ||
                    LANG(?o) = "" ||
                    LANG(?o) = "en"
                )
            }} LIMIT 20
        """
        rows = await self._sparql_query(self.ENDPOINTS["wikidata"], query)
        return [
            {"subject": uri, "predicate": row["p"], "object": row["o"]}
            for row in rows
            if "p" in row and "o" in row
        ]

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if not self._client.is_closed:
            await self._client.aclose()
