import re

from knowledge_service.clients.federation import FederationClient

DBPEDIA_SPARQL = re.compile(r"https://dbpedia\.org/sparql.*")
WIKIDATA_SPARQL = re.compile(r"https://query\.wikidata\.org/sparql.*")


def _sparql_xml_response(bindings_xml: str) -> str:
    """Wrap SPARQL result bindings in the standard XML envelope."""
    return f"""<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <results>
    {bindings_xml}
  </results>
</sparql>"""


DBPEDIA_LOOKUP_RESPONSE = _sparql_xml_response("""
    <result>
      <binding name="entity"><uri>http://dbpedia.org/resource/PostgreSQL</uri></binding>
      <binding name="type"><uri>http://dbpedia.org/ontology/Software</uri></binding>
      <binding name="comment"><literal xml:lang="en">PostgreSQL is a relational database.</literal></binding>
    </result>
""")

DBPEDIA_EMPTY_RESPONSE = _sparql_xml_response("")

DBPEDIA_ENRICH_RESPONSE = _sparql_xml_response("""
    <result>
      <binding name="p"><uri>http://www.w3.org/1999/02/22-rdf-syntax-ns#type</uri></binding>
      <binding name="o"><uri>http://dbpedia.org/ontology/Software</uri></binding>
    </result>
    <result>
      <binding name="p"><uri>http://www.w3.org/2002/07/owl#sameAs</uri></binding>
      <binding name="o"><uri>http://www.wikidata.org/entity/Q192490</uri></binding>
    </result>
""")


class TestLookupEntity:
    async def test_dbpedia_returns_entity(self, httpx_mock):
        httpx_mock.add_response(url=DBPEDIA_SPARQL, text=DBPEDIA_LOOKUP_RESPONSE)
        client = FederationClient(timeout=3.0)
        result = await client.lookup_entity("PostgreSQL")
        assert result is not None
        assert result["uri"] == "http://dbpedia.org/resource/PostgreSQL"
        assert result["rdf_type"] == "http://dbpedia.org/ontology/Software"
        assert "PostgreSQL" in result["description"]
        await client.close()

    async def test_dbpedia_no_match_returns_none(self, httpx_mock):
        httpx_mock.add_response(url=DBPEDIA_SPARQL, text=DBPEDIA_EMPTY_RESPONSE)
        httpx_mock.add_response(url=WIKIDATA_SPARQL, text=DBPEDIA_EMPTY_RESPONSE)
        client = FederationClient(timeout=3.0)
        result = await client.lookup_entity("xyznonexistent12345")
        assert result is None
        await client.close()

    async def test_timeout_returns_none(self, httpx_mock):
        import httpx

        httpx_mock.add_exception(httpx.ReadTimeout("timeout"), url=DBPEDIA_SPARQL)
        httpx_mock.add_exception(httpx.ReadTimeout("timeout"), url=WIKIDATA_SPARQL)
        client = FederationClient(timeout=0.1)
        result = await client.lookup_entity("PostgreSQL")
        assert result is None
        await client.close()

    async def test_http_error_returns_none(self, httpx_mock):
        httpx_mock.add_response(url=DBPEDIA_SPARQL, status_code=500)
        httpx_mock.add_response(url=WIKIDATA_SPARQL, status_code=500)
        client = FederationClient(timeout=3.0)
        result = await client.lookup_entity("PostgreSQL")
        assert result is None
        await client.close()

    async def test_user_agent_header_sent(self, httpx_mock):
        httpx_mock.add_response(url=DBPEDIA_SPARQL, text=DBPEDIA_LOOKUP_RESPONSE)
        client = FederationClient(timeout=3.0)
        await client.lookup_entity("PostgreSQL")
        request = httpx_mock.get_requests()[0]
        assert "KnowledgeService" in request.headers["user-agent"]
        await client.close()

    async def test_wikidata_fallback_when_dbpedia_empty(self, httpx_mock):
        """When DBpedia returns nothing, Wikidata is tried."""
        httpx_mock.add_response(url=DBPEDIA_SPARQL, text=DBPEDIA_EMPTY_RESPONSE)
        wikidata_response = _sparql_xml_response("""
            <result>
              <binding name="entity"><uri>http://www.wikidata.org/entity/Q192490</uri></binding>
              <binding name="type"><uri>http://www.wikidata.org/entity/Q7397</uri></binding>
              <binding name="description"><literal xml:lang="en">relational database management system</literal></binding>
            </result>
        """)
        httpx_mock.add_response(url=WIKIDATA_SPARQL, text=wikidata_response)
        client = FederationClient(timeout=3.0)
        result = await client.lookup_entity("PostgreSQL")
        assert result is not None
        assert "wikidata.org" in result["uri"]
        await client.close()

    async def test_close_is_idempotent(self, httpx_mock):
        httpx_mock.add_response(url=DBPEDIA_SPARQL, text=DBPEDIA_LOOKUP_RESPONSE)
        client = FederationClient(timeout=3.0)
        await client.lookup_entity("PostgreSQL")
        await client.close()
        await client.close()  # should not raise


class TestEnrichEntity:
    async def test_enrich_dbpedia_returns_triples(self, httpx_mock):
        httpx_mock.add_response(url=DBPEDIA_SPARQL, text=DBPEDIA_ENRICH_RESPONSE)
        client = FederationClient(timeout=3.0)
        triples = await client.enrich_entity("http://dbpedia.org/resource/PostgreSQL")
        assert len(triples) == 2
        assert triples[0]["subject"] == "http://dbpedia.org/resource/PostgreSQL"
        assert triples[0]["predicate"] == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        assert triples[0]["object"] == "http://dbpedia.org/ontology/Software"
        await client.close()

    async def test_enrich_unknown_uri_returns_empty(self, httpx_mock):
        client = FederationClient(timeout=3.0)
        triples = await client.enrich_entity("http://unknown.example.com/thing")
        assert triples == []
        await client.close()

    async def test_enrich_timeout_returns_empty(self, httpx_mock):
        import httpx as httpx_module

        httpx_mock.add_exception(httpx_module.ReadTimeout("timeout"), url=DBPEDIA_SPARQL)
        client = FederationClient(timeout=0.1)
        triples = await client.enrich_entity("http://dbpedia.org/resource/PostgreSQL")
        assert triples == []
        await client.close()
