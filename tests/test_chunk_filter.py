from knowledge_service.ingestion.chunk_filter import filter_chunks, score_chunk
from knowledge_service.nlp import NlpEntity, NlpResult


class TestScoreChunk:
    def test_references_section_returns_zero(self):
        chunk = {"chunk_text": "Some references here.", "section_header": "References"}
        assert score_chunk(chunk, nlp_result=None) == 0.0

    def test_bibliography_section_returns_zero(self):
        chunk = {"chunk_text": "Some text.", "section_header": "Bibliography"}
        assert score_chunk(chunk, nlp_result=None) == 0.0

    def test_acknowledgements_section_returns_zero(self):
        chunk = {"chunk_text": "Thanks to everyone.", "section_header": "Acknowledgements"}
        assert score_chunk(chunk, nlp_result=None) == 0.0

    def test_table_of_contents_returns_zero(self):
        chunk = {
            "chunk_text": "Chapter 1 ... 5\nChapter 2 ... 10",
            "section_header": "Table of Contents",
        }
        assert score_chunk(chunk, nlp_result=None) == 0.0

    def test_appendix_section_returns_zero(self):
        chunk = {"chunk_text": "Appendix data.", "section_header": "Appendix A"}
        assert score_chunk(chunk, nlp_result=None) == 0.0

    def test_hierarchical_references_header_returns_zero(self):
        chunk = {"chunk_text": "Some text.", "section_header": "Paper > References"}
        assert score_chunk(chunk, nlp_result=None) == 0.0

    def test_references_detected_in_first_line(self):
        chunk = {"chunk_text": "References\n[1] Author, A. (2024)...", "section_header": None}
        assert score_chunk(chunk, nlp_result=None) == 0.0

    def test_entity_rich_chunk_scores_high(self):
        entities = [
            NlpEntity(text="dopamine", label="CHEMICAL", start_char=0, end_char=8),
            NlpEntity(text="serotonin", label="CHEMICAL", start_char=20, end_char=29),
            NlpEntity(text="cold exposure", label="EVENT", start_char=40, end_char=53),
        ]
        nlp_result = NlpResult(chunk_index=0, entities=entities, sentence_count=5)
        chunk = {
            "chunk_text": "Dopamine and serotonin levels rise after cold exposure. " * 5,
            "section_header": "Results",
        }
        score = score_chunk(chunk, nlp_result=nlp_result)
        assert score > 0.5

    def test_boilerplate_heavy_chunk_scores_low(self):
        text = "[1] Smith (2024). [2] Jones (2023). [3] Lee (2022). [4] Kim (2021). [5] Wang (2020). More refs follow."
        chunk = {"chunk_text": text, "section_header": None}
        score = score_chunk(chunk, nlp_result=None)
        assert score < 0.3

    def test_low_lexical_diversity_scores_low(self):
        text = "data data data data data data data data data data value value value value value"
        chunk = {"chunk_text": text, "section_header": None}
        score = score_chunk(chunk, nlp_result=None)
        assert score < 0.3

    def test_short_chunk_low_sentences_scores_low(self):
        chunk = {"chunk_text": "Page 42.", "section_header": None}
        score = score_chunk(chunk, nlp_result=None)
        assert score < 0.3

    def test_normal_prose_without_nlp_scores_moderate(self):
        text = (
            "Cold water immersion has been studied extensively in recent years. "
            "Researchers found significant increases in dopamine levels. "
            "The protocol involved three-minute exposures at eleven degrees Celsius. "
            "Participants reported enhanced mood and alertness afterward."
        )
        chunk = {"chunk_text": text, "section_header": "Methods"}
        score = score_chunk(chunk, nlp_result=None)
        assert 0.2 <= score <= 0.8

    def test_score_is_clamped_zero_to_one(self):
        entities = [
            NlpEntity(text=f"e{i}", label="MISC", start_char=i, end_char=i + 2) for i in range(50)
        ]
        nlp_result = NlpResult(chunk_index=0, entities=entities, sentence_count=20)
        chunk = {"chunk_text": "x " * 100, "section_header": None}
        score = score_chunk(chunk, nlp_result=nlp_result)
        assert 0.0 <= score <= 1.0


class TestFilterChunks:
    def test_partitions_by_threshold(self):
        chunks = [
            {
                "chunk_text": "Some references here.",
                "chunk_index": 0,
                "section_header": "References",
            },
            {
                "chunk_text": (
                    "Cold water immersion has been studied extensively in recent years. "
                    "Researchers found significant increases in dopamine levels. "
                    "The protocol involved three-minute exposures. "
                    "Participants reported enhanced mood and alertness."
                ),
                "chunk_index": 1,
                "section_header": "Results",
            },
        ]
        entities = [
            NlpEntity(text="dopamine", label="CHEMICAL", start_char=0, end_char=8),
            NlpEntity(text="cold water", label="EVENT", start_char=10, end_char=20),
        ]
        nlp_results = [
            NlpResult(chunk_index=0, entities=[], sentence_count=1),
            NlpResult(chunk_index=1, entities=entities, sentence_count=4),
        ]
        extract_indices, skip_indices = filter_chunks(chunks, nlp_results)
        assert 0 in skip_indices
        assert 1 in extract_indices

    def test_all_chunks_skipped_when_all_low_value(self):
        chunks = [
            {"chunk_text": "Refs.", "chunk_index": 0, "section_header": "References"},
            {"chunk_text": "Thanks.", "chunk_index": 1, "section_header": "Acknowledgements"},
        ]
        nlp_results = [
            NlpResult(chunk_index=0, entities=[], sentence_count=1),
            NlpResult(chunk_index=1, entities=[], sentence_count=1),
        ]
        extract_indices, skip_indices = filter_chunks(chunks, nlp_results)
        assert len(extract_indices) == 0
        assert len(skip_indices) == 2

    def test_no_nlp_results_still_works(self):
        chunks = [
            {
                "chunk_text": "Some normal text with multiple sentences. Another one here. And another.",
                "chunk_index": 0,
                "section_header": None,
            },
        ]
        extract_indices, skip_indices = filter_chunks(chunks, nlp_results=None)
        assert isinstance(extract_indices, list)
        assert isinstance(skip_indices, list)
        assert len(extract_indices) + len(skip_indices) == 1
