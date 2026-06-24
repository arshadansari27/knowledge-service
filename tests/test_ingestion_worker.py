# tests/test_ingestion_worker.py
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from knowledge_service.ingestion.phases import ExtractPhase
from knowledge_service.ingestion.worker import JobTracker, run_ingestion


class TestJobTracker:
    async def test_complete_sets_status(self):
        pool = MagicMock()
        conn = AsyncMock()
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _acquire():
            yield conn

        pool.acquire = _acquire
        tracker = JobTracker("job-id", pool)
        await tracker.complete(triples_created=5, entities_resolved=3, chunks_failed=0)
        conn.execute.assert_called()

    async def test_fail_sets_error(self):
        pool = MagicMock()
        conn = AsyncMock()
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _acquire():
            yield conn

        pool.acquire = _acquire
        tracker = JobTracker("job-id", pool)
        await tracker.fail(Exception("boom"))
        call_args = conn.execute.call_args
        assert "failed" in str(call_args).lower()


def _make_mock_pool():
    """Create a mock asyncpg pool with working acquire context manager."""
    pool = MagicMock()
    conn = AsyncMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool, conn


class TestRunIngestionWithNlp:
    """Test that run_ingestion is now embed-only (NLP/extraction phases are skipped)."""

    async def test_run_ingestion_accepts_nlp_pipeline(self, monkeypatch):
        # ponytail: NLP and extraction phases are now skipped. Ingest is embed-only.
        # This test verifies that even with nlp/extraction_client passed, they're
        # not called. Job completes after embed.
        pool, conn = _make_mock_pool()

        # Mock stores
        stores = MagicMock()
        stores.pg_pool = pool
        stores.content = AsyncMock()
        stores.content.replace_chunks = AsyncMock(return_value=[(0, "chunk-uuid-0")])

        # Mock embedding client
        embedding_client = AsyncMock()
        embedding_client.embed_batch = AsyncMock(return_value=[[0.1] * 768])

        # Mock extraction client (should NOT be called)
        extraction_client = AsyncMock()
        extraction_client.extract_with_stats = AsyncMock(return_value=([], 0))

        # Mock spaCy nlp pipeline (should NOT be called)
        mock_nlp = MagicMock()

        chunk_records = [{"chunk_text": "Test sentence.", "chunk_index": 0}]

        await run_ingestion(
            job_id="test-job-id",
            content_id="test-content-id",
            chunk_records=chunk_records,
            raw_text="Test sentence.",
            knowledge=None,
            title="Test",
            source_url="http://example.com",
            source_type="article",
            stores=stores,
            embedding_client=embedding_client,
            extraction_client=extraction_client,
            nlp=mock_nlp,
        )

        # Verify NLP was NOT called (embed-only ingest)
        mock_nlp.assert_not_called()

        # Verify extraction was NOT called
        extraction_client.extract_with_stats.assert_not_called()

        # Verify job was marked complete (not failed)
        calls = conn.execute.call_args_list
        final_sql = str(calls[-1])
        assert "completed" in final_sql.lower()


class TestExtractPhaseFiltering:
    async def test_no_filtering_when_no_nlp_hints(self):
        """Without NLP hints, all chunks go to LLM (no filtering)."""
        extraction_client = AsyncMock()
        extraction_client.extract = AsyncMock(return_value=[])
        extraction_client.extract_with_stats = AsyncMock(return_value=([], 0))

        phase = ExtractPhase(extraction_client)

        chunk_records = [
            {"chunk_text": "Text one.", "chunk_index": 0, "section_header": None},
            {"chunk_text": "Text two.", "chunk_index": 1, "section_header": None},
        ]
        chunk_id_map = {0: "uuid-0", 1: "uuid-1"}

        knowledge, chunk_ids, chunks_failed, items_rejected = await phase.run(
            chunk_records,
            chunk_id_map,
            nlp_hints=None,
        )

        assert extraction_client.extract_with_stats.call_count == 2
        assert chunks_failed == 0
        assert items_rejected == 0

    async def test_domains_threaded_to_extractor(self):
        """Regression guard: ContentRequest.domains must reach
        extract_with_stats so PromptBuilder can scope predicates. See audit
        finding A2."""
        extraction_client = AsyncMock()
        extraction_client.extract_with_stats = AsyncMock(return_value=([], 0))

        phase = ExtractPhase(extraction_client)

        chunk_records = [
            {"chunk_text": "Text.", "chunk_index": 0, "section_header": None},
        ]

        await phase.run(
            chunk_records,
            {0: "uuid-0"},
            domains=["health", "research"],
        )

        kwargs = extraction_client.extract_with_stats.call_args.kwargs
        assert kwargs["domains"] == ["health", "research"]


class TestNerFallbackFiltering:
    """``_emit_ner_missed`` previously produced ~12k polluted entities in
    production where the spaCy NER text was a URL (the page header) and
    the rdf_type was an UPPERCASE spaCy label. Verify filters."""

    def _entities_from_ner(self, ner_entities, llm_items=None):
        from knowledge_service.nlp import NlpResult

        knowledge: list = []
        chunk_ids: list = []
        nlp_result = NlpResult(entities=ner_entities, chunk_index=0)
        ExtractPhase._emit_ner_missed(nlp_result, llm_items or [], "chunk-1", knowledge, chunk_ids)
        return knowledge

    def test_url_text_is_skipped(self):
        from knowledge_service.nlp import NlpEntity

        out = self._entities_from_ner(
            [
                NlpEntity(text="https://example.com/foo", label="ORG"),
                NlpEntity(text="Apple", label="ORG"),
            ]
        )
        labels = {e.label for e in out}
        assert "https://example.com/foo" not in labels
        assert "Apple" in labels

    def test_numeric_labels_dropped(self):
        """``CARDINAL``, ``MONEY``, ``PERCENT``, ``QUANTITY``, ``DATE``,
        ``TIME``, ``ORDINAL`` describe values, not entities; they get
        dropped."""
        from knowledge_service.nlp import NlpEntity

        out = self._entities_from_ner(
            [
                NlpEntity(text=text, label=label)
                for text, label in [
                    ("169", "CARDINAL"),
                    ("$1.5M", "MONEY"),
                    ("50%", "PERCENT"),
                    ("2026-05-26", "DATE"),
                    ("12pm", "TIME"),
                    ("first", "ORDINAL"),
                    ("3 kg", "QUANTITY"),
                ]
            ]
        )
        assert out == []

    def test_spacy_labels_mapped_to_schema_org(self):
        from knowledge_service.nlp import NlpEntity

        out = self._entities_from_ner(
            [
                NlpEntity(text="Apple", label="ORG"),
                NlpEntity(text="London", label="GPE"),
                NlpEntity(text="Alice", label="PERSON"),
                NlpEntity(text="Hamlet", label="WORK_OF_ART"),
            ]
        )
        rdf_types = {e.rdf_type for e in out}
        assert rdf_types == {
            "schema:Organization",
            "schema:Place",
            "schema:Person",
            "schema:CreativeWork",
        }
