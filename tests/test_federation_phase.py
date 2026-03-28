"""Test federation enrichment phase."""

from unittest.mock import AsyncMock

from knowledge_service.ingestion.federation import FederationPhase
from knowledge_service.stores.triples import TripleStore


class TestFederationPhase:
    def _make_phase(self, lookup_results=None):
        federation_client = AsyncMock()
        if lookup_results is None:
            lookup_results = {}
        federation_client.lookup_entity = AsyncMock(
            side_effect=lambda label: lookup_results.get(label)
        )
        triple_store = TripleStore(data_dir=None)
        return (
            FederationPhase(
                federation_client=federation_client,
                triple_store=triple_store,
                max_lookups=10,
                delay=0,  # no delay in tests
            ),
            federation_client,
            triple_store,
        )

    async def test_enriches_entity_with_dbpedia_match(self):
        phase, client, ts = self._make_phase(
            lookup_results={
                "dopamine": {
                    "uri": "http://dbpedia.org/resource/Dopamine",
                    "rdf_type": "http://dbpedia.org/ontology/ChemicalSubstance",
                    "description": "A neurotransmitter",
                }
            }
        )
        entities = [{"label": "dopamine", "uri": "http://knowledge.local/data/dopamine"}]
        result = await phase.run(entities)
        assert result.entities_enriched == 1
        client.lookup_entity.assert_called_once_with("dopamine")

    async def test_skips_when_no_match(self):
        phase, client, ts = self._make_phase(lookup_results={})
        entities = [
            {"label": "obscure_thing", "uri": "http://knowledge.local/data/obscure_thing"}
        ]
        result = await phase.run(entities)
        assert result.entities_enriched == 0

    async def test_respects_max_lookups(self):
        phase, client, ts = self._make_phase()
        phase._max_lookups = 2
        entities = [
            {"label": f"entity_{i}", "uri": f"http://knowledge.local/data/entity_{i}"}
            for i in range(5)
        ]
        await phase.run(entities)
        assert client.lookup_entity.call_count == 2

    async def test_handles_lookup_error_gracefully(self):
        phase, client, ts = self._make_phase()
        client.lookup_entity = AsyncMock(side_effect=Exception("network error"))
        entities = [{"label": "test", "uri": "http://knowledge.local/data/test"}]
        result = await phase.run(entities)
        assert result.entities_enriched == 0  # no crash
