import pytest
from unittest.mock import AsyncMock, MagicMock
from knowledge_service.stores.embedding import EmbeddingStore


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_ctx
    return pool, conn


@pytest.fixture
def store(mock_pool):
    pool, _ = mock_pool
    return EmbeddingStore(pool)


class TestInsertContentMetadata:
    async def test_returns_content_id(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = {"id": "metadata-uuid-123"}
        result = await store.insert_content_metadata(
            url="https://example.com/article",
            title="Test Article",
            summary="A test summary",
            raw_text="Full text content",
            source_type="article",
            tags=["test", "example"],
            metadata={},
        )
        conn.fetchrow.assert_called_once()
        assert result == "metadata-uuid-123"

    async def test_sql_targets_content_metadata(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = {"id": "some-uuid"}
        await store.insert_content_metadata(
            url="https://example.com/article",
            title="Test",
            summary="Sum",
            raw_text="Text",
            source_type="article",
            tags=[],
            metadata={},
        )
        sql = conn.fetchrow.call_args[0][0]
        assert "INSERT INTO content_metadata" in sql
        assert "ON CONFLICT" in sql

    async def test_passes_url_param(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = {"id": "uuid-abc"}
        await store.insert_content_metadata(
            url="https://example.com/unique",
            title="Title",
            summary="Sum",
            raw_text="Text",
            source_type="article",
            tags=[],
            metadata={},
        )
        args = conn.fetchrow.call_args[0]
        assert "https://example.com/unique" in args


class TestSearch:
    async def test_search_returns_chunk_results(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {
                "id": "chunk-uuid-1",
                "chunk_text": "relevant chunk text",
                "chunk_index": 0,
                "content_id": "metadata-uuid-1",
                "url": "https://a.com",
                "title": "A",
                "summary": "S",
                "source_type": "article",
                "tags": ["t"],
                "ingested_at": "2025-01-01",
                "similarity": 0.95,
            }
        ]
        results = await store.search(query_embedding=[0.1] * 768, limit=10)
        assert len(results) == 1
        assert results[0]["chunk_text"] == "relevant chunk text"
        assert results[0]["content_id"] == "metadata-uuid-1"
        assert results[0]["similarity"] == 0.95

    async def test_search_calls_fetch(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search(query_embedding=[0.1] * 768, limit=5)
        conn.fetch.assert_called_once()

    async def test_search_sql_joins_content_metadata(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search(query_embedding=[0.1] * 768, limit=5)
        sql = conn.fetch.call_args[0][0]
        assert "content_metadata" in sql
        assert "JOIN" in sql.upper()
        assert "<=>" in sql
        assert "halfvec" in sql

    async def test_search_with_source_type_filter(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search(
            query_embedding=[0.1] * 768,
            limit=5,
            source_type="article",
        )
        sql = conn.fetch.call_args[0][0]
        assert "source_type" in sql

    async def test_search_with_tags_filter(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search(
            query_embedding=[0.1] * 768,
            limit=5,
            tags=["python", "database"],
        )
        sql = conn.fetch.call_args[0][0]
        assert "tags" in sql

    async def test_search_returns_empty_list(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        results = await store.search(query_embedding=[0.1] * 768, limit=10)
        assert results == []

    async def test_search_passes_limit_param(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search(query_embedding=[0.1] * 768, limit=42)
        args = conn.fetch.call_args[0]
        assert 42 in args


class TestDeleteChunks:
    async def test_calls_execute_with_delete(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "DELETE 3"
        await store.delete_chunks("content-uuid-123")
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "DELETE FROM content" in sql
        assert "content_id" in sql

    async def test_passes_content_id_param(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "DELETE 0"
        await store.delete_chunks("my-uuid")
        args = conn.execute.call_args[0]
        assert "my-uuid" in args


class TestInsertChunks:
    async def test_inserts_multiple_chunks(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "INSERT 0 1"
        chunks = [
            {
                "chunk_index": 0,
                "chunk_text": "First chunk",
                "embedding": [0.1] * 768,
                "char_start": 0,
                "char_end": 100,
            },
            {
                "chunk_index": 1,
                "chunk_text": "Second chunk",
                "embedding": [0.2] * 768,
                "char_start": 80,
                "char_end": 200,
            },
        ]
        await store.insert_chunks("content-uuid-123", chunks)
        assert conn.execute.call_count == 2

    async def test_sql_targets_content_table(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "INSERT 0 1"
        chunks = [
            {
                "chunk_index": 0,
                "chunk_text": "chunk",
                "embedding": [0.1] * 768,
                "char_start": 0,
                "char_end": 50,
            },
        ]
        await store.insert_chunks("uuid-1", chunks)
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO content" in sql

    async def test_passes_content_id(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "INSERT 0 1"
        chunks = [
            {
                "chunk_index": 0,
                "chunk_text": "chunk",
                "embedding": [0.1] * 768,
                "char_start": 0,
                "char_end": 50,
            },
        ]
        await store.insert_chunks("my-content-id", chunks)
        args = conn.execute.call_args[0]
        assert "my-content-id" in args

    async def test_no_chunks_no_execute(self, store, mock_pool):
        _, conn = mock_pool
        await store.insert_chunks("uuid-1", [])
        conn.execute.assert_not_called()


class TestSearchEntities:
    async def test_search_entities_returns_matches(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {"uri": "http://knowledge.local/data/pg", "label": "PostgreSQL", "similarity": 0.92}
        ]
        results = await store.search_entities(
            query_embedding=[0.1] * 768,
            limit=3,
        )
        assert len(results) == 1
        assert results[0]["uri"] == "http://knowledge.local/data/pg"

    async def test_search_entities_calls_fetch(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search_entities(query_embedding=[0.1] * 768, limit=3)
        conn.fetch.assert_called_once()

    async def test_search_entities_sql_targets_entity_embeddings_table(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search_entities(query_embedding=[0.1] * 768, limit=3)
        sql = conn.fetch.call_args[0][0]
        assert "entity_embeddings" in sql
        assert "<=>" in sql
        assert "halfvec" in sql

    async def test_search_entities_passes_limit(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search_entities(query_embedding=[0.1] * 768, limit=7)
        args = conn.fetch.call_args[0]
        assert 7 in args


class TestInsertEntityEmbedding:
    async def test_insert_entity_embedding(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = None
        await store.insert_entity_embedding(
            uri="http://knowledge.local/data/pg",
            label="PostgreSQL",
            rdf_type="schema:SoftwareApplication",
            embedding=[0.1] * 768,
        )
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "entity_embeddings" in sql

    async def test_insert_entity_embedding_uses_upsert(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = None
        await store.insert_entity_embedding(
            uri="http://knowledge.local/data/pg",
            label="PostgreSQL",
            rdf_type="schema:SoftwareApplication",
            embedding=[0.1] * 768,
        )
        sql = conn.execute.call_args[0][0]
        assert "ON CONFLICT" in sql

    async def test_insert_entity_embedding_passes_uri(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = None
        await store.insert_entity_embedding(
            uri="http://knowledge.local/data/pg",
            label="PostgreSQL",
            rdf_type="schema:SoftwareApplication",
            embedding=[0.1] * 768,
        )
        args = conn.execute.call_args[0]
        assert "http://knowledge.local/data/pg" in args

    async def test_insert_entity_embedding_passes_label(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = None
        await store.insert_entity_embedding(
            uri="http://knowledge.local/data/pg",
            label="PostgreSQL",
            rdf_type="schema:SoftwareApplication",
            embedding=[0.1] * 768,
        )
        args = conn.execute.call_args[0]
        assert "PostgreSQL" in args


class TestGetEntityByUri:
    async def test_returns_entity_when_found(self, store, mock_pool):
        """get_entity_by_uri returns label and rdf_type for a known URI."""
        _, conn = mock_pool
        conn.fetchrow.return_value = {
            "label": "PostgreSQL",
            "rdf_type": "http://dbpedia.org/ontology/Software",
        }
        result = await store.get_entity_by_uri("http://knowledge.local/data/postgresql")
        assert result is not None
        assert result["label"] == "PostgreSQL"
        assert result["rdf_type"] == "http://dbpedia.org/ontology/Software"
        conn.fetchrow.assert_called_once()

    async def test_returns_none_when_not_found(self, store, mock_pool):
        """get_entity_by_uri returns None for an unknown URI."""
        _, conn = mock_pool
        conn.fetchrow.return_value = None
        result = await store.get_entity_by_uri("http://knowledge.local/data/nonexistent")
        assert result is None

    async def test_sql_targets_entity_embeddings_table(self, store, mock_pool):
        """SQL query should target entity_embeddings table."""
        _, conn = mock_pool
        conn.fetchrow.return_value = None
        await store.get_entity_by_uri("http://knowledge.local/data/pg")
        sql = conn.fetchrow.call_args[0][0]
        assert "entity_embeddings" in sql
