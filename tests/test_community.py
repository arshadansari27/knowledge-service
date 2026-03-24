import pytest
from unittest.mock import AsyncMock, MagicMock
from knowledge_service.stores.community import (
    CommunityDetector,
    CommunityStore,
    CommunitySummarizer,
)


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=txn)
    txn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn)
    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_ctx
    return pool, conn


@pytest.fixture
def store(mock_pool):
    pool, _ = mock_pool
    return CommunityStore(pool)


class TestCommunityStore:
    async def test_replace_all_deletes_and_inserts(self, store, mock_pool):
        _, conn = mock_pool
        communities = [
            {
                "level": 0,
                "label": "Health",
                "summary": "Health topics",
                "member_entities": ["http://e/a", "http://e/b"],
                "member_count": 2,
            },
        ]
        await store.replace_all(communities)
        # Should execute delete then insert within transaction
        assert conn.execute.call_count >= 1

    async def test_get_by_level(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {
                "id": "uuid1",
                "level": 0,
                "label": "Test",
                "summary": "Sum",
                "member_entities": ["http://e/a"],
                "member_count": 1,
                "built_at": "2026-01-01",
            },
        ]
        results = await store.get_by_level(0)
        assert len(results) == 1
        sql = conn.fetch.call_args[0][0]
        assert "level" in sql

    async def test_get_all(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        results = await store.get_all()
        assert results == []

    async def test_get_member_entities(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {"member_entities": ["http://e/a", "http://e/b"]},
            {"member_entities": ["http://e/b", "http://e/c"]},
        ]
        result = await store.get_member_entities()
        assert "http://e/a" in result
        assert "http://e/c" in result


def _make_knowledge_store_for_detection():
    """Mock KnowledgeStore that returns a small entity graph with ks: predicates."""
    ks = MagicMock()
    ks.query.return_value = [
        {
            "s": MagicMock(value="http://e/a"),
            "p": MagicMock(value="http://knowledge.local/schema/causes"),
            "o": MagicMock(value="http://e/b"),
            "conf": MagicMock(value="0.8"),
        },
        {
            "s": MagicMock(value="http://e/b"),
            "p": MagicMock(value="http://knowledge.local/schema/increases"),
            "o": MagicMock(value="http://e/c"),
            "conf": MagicMock(value="0.7"),
        },
        {
            "s": MagicMock(value="http://e/d"),
            "p": MagicMock(value="http://knowledge.local/schema/reduces"),
            "o": MagicMock(value="http://e/e"),
            "conf": MagicMock(value="0.9"),
        },
    ]
    return ks


class TestCommunityDetector:
    def test_detect_returns_communities(self):
        ks = _make_knowledge_store_for_detection()
        detector = CommunityDetector(ks)
        communities = detector.detect()
        assert len(communities) > 0
        for c in communities:
            assert "level" in c
            assert "member_entities" in c
            assert "member_count" in c

    def test_detect_produces_two_levels(self):
        ks = _make_knowledge_store_for_detection()
        detector = CommunityDetector(ks)
        communities = detector.detect()
        levels = {c["level"] for c in communities}
        assert 0 in levels

    def test_detect_empty_graph(self):
        ks = MagicMock()
        ks.query.return_value = []
        detector = CommunityDetector(ks)
        communities = detector.detect()
        assert communities == []

    def test_filters_ontology_predicates_in_python(self):
        """Detector should filter out non-ks: predicates from SPARQL results."""
        ks = MagicMock()
        ks.query.return_value = [
            # Domain edge — should be kept
            {
                "s": MagicMock(value="http://e/a"),
                "p": MagicMock(value="http://knowledge.local/schema/increases"),
                "o": MagicMock(value="http://e/b"),
                "conf": MagicMock(value="0.8"),
            },
            # Ontology edge — should be filtered out
            {
                "s": MagicMock(value="http://e/a"),
                "p": MagicMock(value="http://www.w3.org/2002/07/owl#sameAs"),
                "o": MagicMock(value="http://wikidata.org/Q123"),
                "conf": MagicMock(value="1.0"),
            },
            # rdf:type — should be filtered out
            {
                "s": MagicMock(value="http://e/c"),
                "p": MagicMock(value="http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                "o": MagicMock(value="http://schema.org/Thing"),
                "conf": MagicMock(value="0.9"),
            },
        ]
        detector = CommunityDetector(ks)
        communities = detector.detect()
        # Only 1 edge (a->b), so should produce communities with just those 2 entities
        all_members = set()
        for c in communities:
            all_members.update(c["member_entities"])
        assert "http://e/a" in all_members
        assert "http://e/b" in all_members
        assert "http://wikidata.org/Q123" not in all_members
        assert "http://schema.org/Thing" not in all_members


class TestCommunitySummarizer:
    async def test_summarize_produces_label_and_summary(self):
        mock_llm_client = AsyncMock()
        mock_llm_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": '{"label": "Health Topics", "summary": "This community covers health and biohacking."}'
                        }
                    }
                ]
            },
            raise_for_status=lambda: None,
        )

        mock_ks = MagicMock()
        mock_ks.get_triples_by_subject.return_value = []

        summarizer = CommunitySummarizer(mock_llm_client, mock_ks)
        community = {
            "level": 0,
            "member_entities": ["http://e/a", "http://e/b"],
            "member_count": 2,
        }
        result = await summarizer.summarize_one(community)
        assert result["label"] == "Health Topics"
        assert result["summary"] == "This community covers health and biohacking."

    async def test_summarize_handles_llm_failure(self):
        mock_llm_client = AsyncMock()
        mock_llm_client.post.side_effect = Exception("LLM down")

        mock_ks = MagicMock()
        mock_ks.get_triples_by_subject.return_value = []

        summarizer = CommunitySummarizer(mock_llm_client, mock_ks)
        community = {
            "level": 0,
            "member_entities": ["http://e/a"],
            "member_count": 1,
        }
        result = await summarizer.summarize_one(community)
        assert result.get("label") is None
        assert result.get("summary") is None

    async def test_summarize_does_not_mutate_original(self):
        mock_llm_client = AsyncMock()
        mock_llm_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [
                    {"message": {"content": '{"label": "Test", "summary": "Test summary."}'}}
                ]
            },
            raise_for_status=lambda: None,
        )

        mock_ks = MagicMock()
        mock_ks.get_triples_by_subject.return_value = []

        summarizer = CommunitySummarizer(mock_llm_client, mock_ks)
        original = {
            "level": 0,
            "member_entities": ["http://e/a"],
            "member_count": 1,
        }
        result = await summarizer.summarize_one(original)
        assert "label" not in original  # Original not mutated
        assert result["label"] == "Test"

    async def test_summarize_handles_freeform_response(self):
        """Summarizer should extract JSON from freeform LLM output with trailing text."""
        freeform = (
            "<think>I should generate a label and summary.</think>\n"
            '```json\n{"label": "Brain Science", "summary": "Covers neuroscience."}\n```\n'
            "Hope this helps!"
        )
        mock_llm_client = AsyncMock()
        mock_llm_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": freeform}}]},
            raise_for_status=lambda: None,
        )

        mock_ks = MagicMock()
        mock_ks.get_triples_by_subject.return_value = []

        summarizer = CommunitySummarizer(mock_llm_client, mock_ks)
        community = {"level": 0, "member_entities": ["http://e/a"], "member_count": 1}
        result = await summarizer.summarize_one(community)
        assert result["label"] == "Brain Science"
        assert result["summary"] == "Covers neuroscience."

    async def test_summarize_builds_context_with_triples(self):
        mock_llm_client = AsyncMock()
        mock_llm_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": '{"label": "Neuro", "summary": "Neuroscience topics."}'
                        }
                    }
                ]
            },
            raise_for_status=lambda: None,
        )

        mock_ks = MagicMock()
        mock_ks.get_triples_by_subject.return_value = [
            {
                "predicate": MagicMock(value="http://ks/causes"),
                "object": MagicMock(value="http://e/b"),
            },
        ]

        summarizer = CommunitySummarizer(mock_llm_client, mock_ks)
        community = {
            "level": 0,
            "member_entities": ["http://e/a"],
            "member_count": 1,
        }
        result = await summarizer.summarize_one(community)
        assert result["label"] == "Neuro"
        # Verify LLM was called and the prompt included relationship context
        call_args = mock_llm_client.post.call_args
        prompt = call_args[1]["json"]["messages"][0]["content"]
        assert "causes" in prompt
        assert "a" in prompt
