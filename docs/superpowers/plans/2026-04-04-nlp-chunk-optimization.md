# NLP Chunk Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce LLM calls during ingestion by ~70-80% via NLP-scored chunk gating and single-pass extraction.

**Architecture:** A new `chunk_filter` module scores chunks using NLP signals (entity density, section headers, boilerplate ratio, lexical diversity, sentence count) and gates them before LLM extraction. The two-phase LLM extraction (entities → relations) merges into a single call. Skipped chunks still contribute spaCy NER entities as fallbacks.

**Tech Stack:** Python 3.12, spaCy (existing), regex, pytest, pytest-httpx

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/knowledge_service/ingestion/chunk_filter.py` | **New.** `score_chunk()` and `filter_chunks()` — NLP-based chunk scoring and partitioning |
| `src/knowledge_service/clients/llm.py` | **Modify.** Merge two-phase `extract()` into single-pass; add `_build_combined_extraction_prompt_fallback()` |
| `src/knowledge_service/clients/prompt_builder.py` | **Modify.** Add `build_combined_prompt()` method and `_DEFAULT_COMBINED_TEMPLATE` |
| `src/knowledge_service/ingestion/phases.py` | **Modify.** Integrate chunk filtering in `ExtractPhase.run()`; handle skip path with NER fallback |
| `src/knowledge_service/ingestion/worker.py` | **Modify.** Add `chunks_skipped` to `_ALLOWED_JOB_COLUMNS`; pass counter through |
| `migrations/013_chunks_skipped.sql` | **New.** Add `chunks_skipped` column to `ingestion_jobs` |
| `tests/test_chunk_filter.py` | **New.** Unit tests for chunk scoring and filtering |
| `tests/test_extraction_client.py` | **Modify.** Update for single-pass extraction (1 LLM call instead of 2) |
| `tests/test_prompt_builder.py` | **Modify.** Add tests for `build_combined_prompt()` |
| `tests/test_ingestion_worker.py` | **Modify.** Add test for `chunks_skipped` tracking |

---

### Task 1: Chunk Filter Module — Scoring

**Files:**
- Create: `src/knowledge_service/ingestion/chunk_filter.py`
- Create: `tests/test_chunk_filter.py`

- [ ] **Step 1: Write failing tests for `score_chunk()`**

```python
# tests/test_chunk_filter.py
import pytest

from knowledge_service.ingestion.chunk_filter import score_chunk
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
        chunk = {"chunk_text": "Chapter 1 ... 5\nChapter 2 ... 10", "section_header": "Table of Contents"}
        assert score_chunk(chunk, nlp_result=None) == 0.0

    def test_appendix_section_returns_zero(self):
        chunk = {"chunk_text": "Appendix data.", "section_header": "Appendix A"}
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
        entities = [NlpEntity(text=f"e{i}", label="MISC", start_char=i, end_char=i + 2) for i in range(50)]
        nlp_result = NlpResult(chunk_index=0, entities=entities, sentence_count=20)
        chunk = {"chunk_text": "x " * 100, "section_header": None}
        score = score_chunk(chunk, nlp_result=nlp_result)
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunk_filter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'knowledge_service.ingestion.chunk_filter'`

- [ ] **Step 3: Implement `score_chunk()`**

```python
# src/knowledge_service/ingestion/chunk_filter.py
"""NLP-based chunk scoring and filtering to reduce LLM calls during extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge_service.nlp import NlpResult

# Sections that should always be skipped (case-insensitive match)
_SKIP_SECTIONS = re.compile(
    r"^(references|bibliography|acknowledgements?|table\s+of\s+contents"
    r"|appendix(\s+\w)?|index|about\s+the\s+author|works\s+cited"
    r"|further\s+reading|endnotes|footnotes)$",
    re.IGNORECASE,
)

# First-line detection for sections without proper headers
_SKIP_FIRST_LINE = re.compile(
    r"^(references|bibliography|acknowledgements?|table\s+of\s+contents)\s*$",
    re.IGNORECASE,
)

# Citation patterns: [1], (Author, 2024), (Author et al., 2024)
_CITATION_RE = re.compile(r"\[\d+\]|\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s*\d{4}\)")

# Bare URL pattern
_URL_RE = re.compile(r"https?://\S+")

# Numeric-heavy line (>60% digits/punctuation)
_NUMERIC_LINE_RE = re.compile(r"^[\d\s.,;:|\-/]+$")


def score_chunk(chunk: dict, nlp_result: NlpResult | None) -> float:
    """Score a chunk 0.0-1.0 for extraction value using cheap NLP signals.

    Returns 0.0 for chunks that should always be skipped (references, etc.).
    """
    text = chunk.get("chunk_text", "")
    section = chunk.get("section_header") or ""

    # --- Instant skip: known low-value sections ---
    # Check each part of hierarchical headers like "Paper > References"
    for part in section.split(" > "):
        if _SKIP_SECTIONS.match(part.strip()):
            return 0.0

    # Check first line of text for section headers without metadata
    first_line = text.split("\n", 1)[0].strip().rstrip(":")
    if _SKIP_FIRST_LINE.match(first_line):
        return 0.0

    # --- Signal 1: NER entity density (weight 0.4) ---
    entity_count = len(nlp_result.entities) if nlp_result else 0
    text_len = max(len(text), 1)
    # entities per 500 chars, capped at 1.0
    entity_density = min(entity_count / max(text_len / 500, 1), 1.0)

    # --- Signal 2: Sentence count (weight 0.2) ---
    sentence_count = nlp_result.sentence_count if nlp_result else text.count(". ") + text.count(".\n") + (1 if text.rstrip().endswith(".") else 0)
    # 0 for <2 sentences, scales up to 1.0 at 5+ sentences
    sentence_score = min(max(sentence_count - 1, 0) / 4.0, 1.0)

    # --- Signal 3: Boilerplate ratio (weight 0.2) ---
    citations = len(_CITATION_RE.findall(text))
    urls = len(_URL_RE.findall(text))
    lines = text.strip().split("\n")
    numeric_lines = sum(1 for line in lines if _NUMERIC_LINE_RE.match(line.strip())) if lines else 0
    total_tokens = max(len(text.split()), 1)
    boilerplate_items = citations + urls + numeric_lines * 3
    boilerplate_ratio = min(boilerplate_items / total_tokens, 1.0)
    boilerplate_score = 1.0 - boilerplate_ratio

    # --- Signal 4: Lexical diversity (weight 0.2) ---
    words = text.lower().split()
    if len(words) > 5:
        diversity = len(set(words)) / len(words)
    else:
        diversity = 0.5  # too short to judge
    diversity_score = min(diversity / 0.7, 1.0)  # normalize: 0.7 diversity = 1.0

    # --- Weighted sum ---
    score = (
        0.4 * entity_density
        + 0.2 * sentence_score
        + 0.2 * boilerplate_score
        + 0.2 * diversity_score
    )
    return max(0.0, min(1.0, score))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunk_filter.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/ingestion/chunk_filter.py tests/test_chunk_filter.py
git commit -m "feat: add chunk scoring for NLP-based extraction filtering"
```

---

### Task 2: Chunk Filter Module — `filter_chunks()`

**Files:**
- Modify: `src/knowledge_service/ingestion/chunk_filter.py`
- Modify: `tests/test_chunk_filter.py`

- [ ] **Step 1: Write failing tests for `filter_chunks()`**

Add to `tests/test_chunk_filter.py`:

```python
from knowledge_service.ingestion.chunk_filter import filter_chunks


class TestFilterChunks:
    def test_partitions_by_threshold(self):
        chunks = [
            {"chunk_text": "Some references here.", "chunk_index": 0, "section_header": "References"},
            {"chunk_text": (
                "Cold water immersion has been studied extensively in recent years. "
                "Researchers found significant increases in dopamine levels. "
                "The protocol involved three-minute exposures. "
                "Participants reported enhanced mood and alertness."
            ), "chunk_index": 1, "section_header": "Results"},
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
            {"chunk_text": "Some normal text with multiple sentences. Another one here. And another.", "chunk_index": 0, "section_header": None},
        ]
        extract_indices, skip_indices = filter_chunks(chunks, nlp_results=None)
        # Without NLP, entity density is 0 but other signals still work
        assert isinstance(extract_indices, list)
        assert isinstance(skip_indices, list)
        assert len(extract_indices) + len(skip_indices) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunk_filter.py::TestFilterChunks -v`
Expected: FAIL — `ImportError: cannot import name 'filter_chunks'`

- [ ] **Step 3: Implement `filter_chunks()`**

Add to `src/knowledge_service/ingestion/chunk_filter.py`:

```python
_SCORE_THRESHOLD = 0.3


def filter_chunks(
    chunk_records: list[dict],
    nlp_results: list[NlpResult] | None,
) -> tuple[list[int], list[int]]:
    """Partition chunk indices into extract vs skip sets based on scoring.

    Returns (extract_indices, skip_indices) where each list contains chunk_index values.
    """
    nlp_map: dict[int, NlpResult] = {}
    if nlp_results:
        for r in nlp_results:
            nlp_map[r.chunk_index] = r

    extract_indices: list[int] = []
    skip_indices: list[int] = []

    for chunk in chunk_records:
        idx = chunk.get("chunk_index", 0)
        nlp_result = nlp_map.get(idx)
        score = score_chunk(chunk, nlp_result)
        if score < _SCORE_THRESHOLD:
            skip_indices.append(idx)
        else:
            extract_indices.append(idx)

    return extract_indices, skip_indices
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunk_filter.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/ingestion/chunk_filter.py tests/test_chunk_filter.py
git commit -m "feat: add filter_chunks() to partition chunks by extraction value"
```

---

### Task 3: Single-Pass Combined Prompt Template

**Files:**
- Modify: `src/knowledge_service/clients/prompt_builder.py`
- Modify: `tests/test_prompt_builder.py`

- [ ] **Step 1: Write failing tests for `build_combined_prompt()`**

Add to `tests/test_prompt_builder.py`:

```python
class TestBuildCombinedPrompt:
    def test_combined_prompt_includes_entity_and_relation_instructions(self):
        registry = _make_mock_registry()
        builder = PromptBuilder(registry)
        prompt = builder.build_combined_prompt("Some text", title="Test", source_type="article")
        assert "Entity" in prompt
        assert "Event" in prompt
        assert "Claim" in prompt
        assert "Relationship" in prompt
        assert "snake_case" in prompt
        assert "Some text" in prompt

    def test_combined_prompt_includes_nlp_hints(self):
        registry = _make_mock_registry()
        builder = PromptBuilder(registry)
        hints = [{"text": "dopamine", "label": "CHEMICAL", "wikidata_id": "Q80635"}]
        prompt = builder.build_combined_prompt("text", entity_hints=hints)
        assert "dopamine" in prompt
        assert "CHEMICAL" in prompt

    def test_combined_prompt_includes_predicates(self):
        registry = _make_mock_registry()
        builder = PromptBuilder(registry)
        prompt = builder.build_combined_prompt("text")
        # Should include domain predicates
        assert "causes" in prompt or "increases" in prompt

    def test_combined_prompt_truncates_text(self):
        registry = _make_mock_registry()
        builder = PromptBuilder(registry)
        long_text = "word " * 2000  # ~10000 chars
        prompt = builder.build_combined_prompt(long_text)
        assert len(prompt) < len(long_text)  # text was truncated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompt_builder.py::TestBuildCombinedPrompt -v`
Expected: FAIL — `AttributeError: 'PromptBuilder' object has no attribute 'build_combined_prompt'`

- [ ] **Step 3: Read existing test file to find `_make_mock_registry` helper**

Run: `uv run pytest tests/test_prompt_builder.py -v --co` to check existing test structure. Then read `tests/test_prompt_builder.py` to find the mock helper and use the same pattern.

- [ ] **Step 4: Implement `build_combined_prompt()` and template**

Add the template constant and method to `src/knowledge_service/clients/prompt_builder.py`:

```python
_DEFAULT_COMBINED_TEMPLATE = """{context}You are a knowledge extraction system. Extract entities, events, AND relationships from the text below.
Return ONLY a JSON object: {{"entities": [...], "relations": [...]}}

## Step 1: Extract Entities and Events

Each entity/event item must have a knowledge_type field:
- Entity: uri, rdf_type (e.g. "schema:Person", "schema:Thing"), label, properties (dict), confidence
- Event: subject, occurred_at (YYYY-MM-DD), confidence, properties (dict)

Entity naming rules:
- Use canonical, well-known names: "dopamine" not "the neurotransmitter dopamine"
- Use singular form: "neuron" not "neurons"
- Use lowercase snake_case: "cold_exposure" not "Cold Exposure"
- Be specific: "vitamin_d3" not "vitamin_d" when the text specifies D3
- The uri and label should both use the snake_case form

## Step 2: Extract Relationships Using Those Entities

Each relation item must have a knowledge_type field:
- Claim: subject, predicate, object, object_type, confidence (0.0-0.89)
- Fact: subject, predicate, object, object_type, confidence (0.9-1.0) for verified facts
- Relationship: subject, predicate, object, object_type, confidence
- TemporalState: subject, property, value, valid_from (YYYY-MM-DD), valid_until (YYYY-MM-DD), confidence
- Conclusion: concludes (text), derived_from (list of identifiers), inference_method, confidence

Preferred predicates (use these when applicable):
{predicates}
Only invent a new predicate if none of the above fit.

Use entities from Step 1 as subjects and objects. For object values, include object_type ("entity" or "literal"):
- "entity": the object is a thing/concept
- "literal": the object is a measurement, description, or date (e.g. "250%", "2024-01-15")

Use Claim for uncertain assertions, Fact for high-confidence verifiable statements.

If nothing found, return {{"entities": [], "relations": []}}

Text:
---
{text}
---"""
```

Add to the `PromptBuilder` class:

```python
    def build_combined_prompt(
        self,
        text: str,
        title: str | None = None,
        source_type: str | None = None,
        entity_hints: list[dict] | None = None,
        domains: list[str] | None = None,
    ) -> str:
        """Build single-pass prompt for combined entity + relation extraction."""
        template = _DEFAULT_COMBINED_TEMPLATE
        context = ""
        if title:
            context += f"Title: {title}\n"
        if source_type:
            context += f"Source type: {source_type}\n"
        if entity_hints:
            context += "\nNLP-detected entities (confirm, correct, or add to these):\n"
            for hint in entity_hints:
                context += f"- {hint['text']} ({hint['label']})\n"

        active_domains = domains or (
            self._registry.get_domains_for_entity_types([]) if self._registry else ["base"]
        )
        predicates_list = self._registry.get_predicates(active_domains) if self._registry else []
        predicates_str = ", ".join(p.label for p in predicates_list) if predicates_list else (
            "causes, increases, decreases, inhibits, activates, is_a, part_of, located_in, "
            "created_by, depends_on, related_to, contains, precedes, follows, has_property, "
            "used_for, produced_by, associated_with"
        )

        return template.format(
            context=context,
            predicates=predicates_str,
            text=text[:_MAX_TEXT_CHARS],
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompt_builder.py -v`
Expected: All PASS (existing + new)

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/clients/prompt_builder.py tests/test_prompt_builder.py
git commit -m "feat: add build_combined_prompt() for single-pass extraction"
```

---

### Task 4: Single-Pass `ExtractionClient.extract()`

**Files:**
- Modify: `src/knowledge_service/clients/llm.py`
- Modify: `tests/test_extraction_client.py`

- [ ] **Step 1: Update tests to expect single-pass behavior**

Replace the two-phase tests in `tests/test_extraction_client.py`. The key changes:
- `mock_llm` fixture now returns a single combined response instead of two sequential responses
- `TestTwoPhaseExtract` becomes `TestSinglePassExtract` — expects 1 LLM call, not 2
- The combined response uses `{"entities": [...], "relations": [...]}` format

Update `_make_chat_response` and add a combined variant:

```python
def _make_combined_response(entities: list, relations: list) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"entities": entities, "relations": relations}),
                }
            }
        ],
    }
```

Update `mock_llm` fixture:

```python
@pytest.fixture
def mock_llm(httpx_mock):
    # Single combined response
    httpx_mock.add_response(
        url=_CHAT_URL,
        json=_make_combined_response(
            entities=[
                {
                    "knowledge_type": "Entity",
                    "uri": "cold_exposure",
                    "rdf_type": "schema:Thing",
                    "label": "cold_exposure",
                    "properties": {},
                    "confidence": 0.9,
                },
            ],
            relations=[
                {
                    "knowledge_type": "Claim",
                    "subject": "cold_exposure",
                    "predicate": "increases",
                    "object": "dopamine",
                    "confidence": 0.7,
                },
            ],
        ),
    )
    return httpx_mock
```

Replace `TestTwoPhaseExtract` with:

```python
class TestSinglePassExtract:
    async def test_makes_one_llm_call(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_combined_response(
                entities=[
                    {
                        "knowledge_type": "Entity",
                        "uri": "dopamine",
                        "rdf_type": "schema:Thing",
                        "label": "dopamine",
                        "properties": {},
                        "confidence": 0.9,
                    },
                ],
                relations=[
                    {
                        "knowledge_type": "Claim",
                        "subject": "cold_exposure",
                        "predicate": "increases",
                        "object": "dopamine",
                        "confidence": 0.7,
                    },
                ],
            ),
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("Cold exposure increases dopamine.")
        assert len(httpx_mock.get_requests()) == 1
        assert len(result) == 2
        await client.close()

    async def test_returns_entities_only_when_no_relations(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_combined_response(
                entities=[
                    {
                        "knowledge_type": "Entity",
                        "uri": "x",
                        "rdf_type": "schema:Thing",
                        "label": "x",
                        "properties": {},
                        "confidence": 0.9,
                    },
                ],
                relations=[],
            ),
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("text")
        assert len(result) == 1
        assert isinstance(result[0], EntityInput)
        await client.close()

    async def test_returns_none_on_http_error(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, status_code=500)
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("text")
        assert result is None
        assert len(httpx_mock.get_requests()) == 1
        await client.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extraction_client.py -v`
Expected: Multiple FAIL — `extract()` still makes 2 calls and expects `{"items": [...]}` format

- [ ] **Step 3: Implement single-pass `extract()`**

Rewrite `ExtractionClient.extract()` in `src/knowledge_service/clients/llm.py`:

```python
    async def extract(
        self,
        text: str,
        title: str | None = None,
        source_type: str | None = None,
        domains: list[str] | None = None,
        entity_hints: list[dict] | None = None,
    ) -> list | None:
        """Extract KnowledgeInput items from raw text using single-pass extraction.

        Extracts entities, events, and relations in a single LLM call.
        Returns None if LLM call failed (distinguishable from [] = nothing found).
        """
        from knowledge_service.models import KnowledgeInput  # noqa: PLC0415

        adapter = TypeAdapter(KnowledgeInput)

        # Build combined prompt
        if self._prompt_builder:
            prompt = self._prompt_builder.build_combined_prompt(
                text, title, source_type, entity_hints=entity_hints, domains=domains
            )
        else:
            prompt = _build_combined_extraction_prompt_fallback(
                text, title, source_type, entity_hints=entity_hints
            )

        raw = await self._call_llm_combined(prompt)
        if raw is None:
            return None

        items = []
        for item_dict in raw:
            try:
                items.append(adapter.validate_python(item_dict))
            except ValidationError as exc:
                logger.warning("ExtractionClient: skipping invalid item %s: %s", item_dict, exc)

        return items
```

Add `_call_llm_combined()` method (parses the `{"entities": [...], "relations": [...]}` format):

```python
    async def _call_llm_combined(self, prompt: str) -> list[dict] | None:
        """Send a combined prompt and return merged entity + relation dicts.

        Accepts responses in either format:
        - {"entities": [...], "relations": [...]}  (combined format)
        - {"items": [...]}  (legacy format)
        Returns None on failure.
        """
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("ExtractionClient: LLM API returned %s", exc.response.status_code)
            return None
        except httpx.TimeoutException as exc:
            logger.warning("ExtractionClient: LLM API request timed out: %s", exc)
            return None

        raw_text = response.json()["choices"][0]["message"]["content"]
        from knowledge_service._utils import _extract_json  # noqa: PLC0415

        parsed = _extract_json(raw_text)
        if parsed is None:
            logger.warning("ExtractionClient: could not parse JSON from response")
            return None

        # Accept both combined and legacy format
        if "entities" in parsed or "relations" in parsed:
            entities = parsed.get("entities", [])
            relations = parsed.get("relations", [])
            return entities + relations

        return parsed.get("items", [])
```

Add the fallback prompt builder:

```python
def _build_combined_extraction_prompt_fallback(
    text: str,
    title: str | None,
    source_type: str | None,
    entity_hints: list[dict] | None = None,
) -> str:
    """Build a single-pass combined extraction prompt (no DomainRegistry)."""
    context = ""
    if title:
        context += f"Title: {title}\n"
    if source_type:
        context += f"Source type: {source_type}\n"
    if entity_hints:
        context += "\nNLP-detected entities (confirm, correct, or add to these):\n"
        for hint in entity_hints:
            context += f"- {hint['text']} ({hint['label']})\n"
    return f"""{context}You are a knowledge extraction system. Extract entities, events, AND relationships from the text below.
Return ONLY a JSON object: {{"entities": [...], "relations": [...]}}

## Step 1: Extract Entities and Events

Each entity/event item must have a knowledge_type field:
- Entity: uri, rdf_type (e.g. "schema:Person", "schema:Thing"), label, properties (dict), confidence
- Event: subject, occurred_at (YYYY-MM-DD), confidence, properties (dict)

Entity naming rules:
- Use canonical, well-known names: "dopamine" not "the neurotransmitter dopamine"
- Use singular form: "neuron" not "neurons"
- Use lowercase snake_case: "cold_exposure" not "Cold Exposure"
- Be specific: "vitamin_d3" not "vitamin_d" when the text specifies D3
- The uri and label should both use the snake_case form

## Step 2: Extract Relationships Using Those Entities

Each relation item must have a knowledge_type field:
- Claim: subject, predicate, object, object_type, confidence (0.0-0.89)
- Fact: subject, predicate, object, object_type, confidence (0.9-1.0) for verified facts
- Relationship: subject, predicate, object, object_type, confidence
- TemporalState: subject, property, value, valid_from (YYYY-MM-DD), valid_until (YYYY-MM-DD), confidence
- Conclusion: concludes (text), derived_from (list of identifiers), inference_method, confidence

Preferred predicates (use these when applicable):
{_FALLBACK_PREDICATES}
Only invent a new predicate if none of the above fit.

Use entities from Step 1 as subjects and objects. For object values, include object_type ("entity" or "literal"):
- "entity": the object is a thing/concept
- "literal": the object is a measurement, description, or date

Use Claim for uncertain assertions, Fact for high-confidence verifiable statements.

If nothing found, return {{"entities": [], "relations": []}}

Text:
---
{text[:_MAX_TEXT_CHARS]}
---"""
```

Update the class docstring on `ExtractionClient`:

```python
class ExtractionClient(BaseLLMClient):
    """Single-pass extraction using DomainRegistry for domain-aware prompts.

    Extracts entities, events, and relations in a single LLM call per chunk.
    """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extraction_client.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -v --ignore=tests/e2e`
Expected: All PASS (some tests in other files may need the mock fixture updated — fix if needed)

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/clients/llm.py tests/test_extraction_client.py
git commit -m "feat: merge two-phase extraction into single-pass LLM call"
```

---

### Task 5: Integrate Chunk Filtering into ExtractPhase

**Files:**
- Modify: `src/knowledge_service/ingestion/phases.py`
- Modify: `tests/test_ingestion_worker.py`

- [ ] **Step 1: Write failing test for skip path**

Add to `tests/test_ingestion_worker.py`:

```python
from knowledge_service.ingestion.phases import ExtractPhase
from knowledge_service.nlp import NlpEntity, NlpResult
from unittest.mock import AsyncMock


class TestExtractPhaseFiltering:
    async def test_skips_low_value_chunks_and_emits_ner_fallback(self):
        """Chunks in skip set should not trigger LLM calls but should emit NER entities."""
        extraction_client = AsyncMock()
        extraction_client.extract = AsyncMock(return_value=[
            {
                "knowledge_type": "Entity",
                "uri": "dopamine",
                "rdf_type": "schema:Thing",
                "label": "dopamine",
                "properties": {},
                "confidence": 0.9,
            },
        ])

        phase = ExtractPhase(extraction_client)

        chunk_records = [
            {"chunk_text": "[1] Smith (2024). [2] Jones (2023).", "chunk_index": 0, "section_header": "References"},
            {"chunk_text": "Cold exposure significantly increases dopamine release in the brain. Multiple studies confirm this. The effect is dose-dependent. Results were consistent across participants.", "chunk_index": 1, "section_header": "Results"},
        ]
        chunk_id_map = {0: "uuid-0", 1: "uuid-1"}

        # NLP results: chunk 0 has no entities, chunk 1 has entities
        nlp_results = [
            NlpResult(chunk_index=0, entities=[], sentence_count=1),
            NlpResult(chunk_index=1, entities=[
                NlpEntity(text="dopamine", label="CHEMICAL", start_char=0, end_char=8),
            ], sentence_count=4),
        ]

        knowledge, chunk_ids, chunks_failed, chunks_skipped = await phase.run(
            chunk_records, chunk_id_map,
            title="Test", source_type="article",
            nlp_hints=nlp_results,
        )

        # LLM should only be called for chunk 1 (not chunk 0)
        assert extraction_client.extract.call_count == 1
        # chunks_skipped should be 1 (the references chunk)
        assert chunks_skipped == 1

    async def test_no_filtering_when_no_nlp_hints(self):
        """Without NLP hints, all chunks go to LLM (no filtering)."""
        extraction_client = AsyncMock()
        extraction_client.extract = AsyncMock(return_value=[])

        phase = ExtractPhase(extraction_client)

        chunk_records = [
            {"chunk_text": "Text one.", "chunk_index": 0, "section_header": None},
            {"chunk_text": "Text two.", "chunk_index": 1, "section_header": None},
        ]
        chunk_id_map = {0: "uuid-0", 1: "uuid-1"}

        knowledge, chunk_ids, chunks_failed, chunks_skipped = await phase.run(
            chunk_records, chunk_id_map,
            nlp_hints=None,
        )

        # All chunks sent to LLM when no NLP hints
        assert extraction_client.extract.call_count == 2
        assert chunks_skipped == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingestion_worker.py::TestExtractPhaseFiltering -v`
Expected: FAIL — `ExtractPhase.run()` returns 3-tuple, not 4-tuple

- [ ] **Step 3: Update `ExtractPhase.run()` to integrate filtering**

Modify `src/knowledge_service/ingestion/phases.py`:

```python
class ExtractPhase:
    """Phase 2: Extract knowledge items from chunks via LLM."""

    def __init__(self, extraction_client: Any):
        self._extraction_client = extraction_client

    async def run(
        self,
        chunk_records: list[dict],
        chunk_id_map: dict[int, str],
        title: str | None = None,
        source_type: str | None = None,
        nlp_hints: list | None = None,
    ) -> tuple[list[dict], list[str | None], int, int]:
        """Extract knowledge from chunks.

        Returns (knowledge_items, chunk_ids_for_items, chunks_failed, chunks_skipped).
        """
        knowledge: list[dict] = []
        chunk_ids: list[str | None] = []
        chunks_failed = 0
        chunks_skipped = 0

        # Build a lookup from chunk_index → NlpResult for hint injection
        hint_map: dict[int, Any] = {}
        if nlp_hints:
            for hint in nlp_hints:
                hint_map[hint.chunk_index] = hint

        # Filter chunks if NLP results are available
        skip_set: set[int] = set()
        if nlp_hints:
            from knowledge_service.ingestion.chunk_filter import filter_chunks  # noqa: PLC0415

            _, skip_indices = filter_chunks(chunk_records, nlp_hints)
            skip_set = set(skip_indices)

        for chunk in chunk_records:
            chunk_index = chunk["chunk_index"]
            cid = chunk_id_map.get(chunk_index)
            nlp_result = hint_map.get(chunk_index)

            # --- Skip path: NER fallback only ---
            if chunk_index in skip_set:
                chunks_skipped += 1
                if nlp_result and nlp_result.entities:
                    self._emit_ner_fallback(nlp_result, cid, knowledge, chunk_ids)
                continue

            # --- Extract path: LLM call ---
            entity_hints: list[dict] | None = None
            if nlp_result and nlp_result.entities:
                entity_hints = [
                    {
                        "text": e.text,
                        "label": e.label,
                        "wikidata_id": e.wikidata_id,
                    }
                    for e in nlp_result.entities
                ]

            items = await self._extraction_client.extract(
                chunk["chunk_text"],
                title=title,
                source_type=source_type,
                entity_hints=entity_hints,
            )
            if items is None:
                chunks_failed += 1
                continue
            for item in items:
                knowledge.append(item)
                chunk_ids.append(cid)

            # Add fallback EntityInput for NLP-detected entities the LLM missed
            if nlp_result and nlp_result.entities and items is not None:
                self._emit_ner_missed(nlp_result, items, cid, knowledge, chunk_ids)

        return knowledge, chunk_ids, chunks_failed, chunks_skipped

    @staticmethod
    def _emit_ner_fallback(
        nlp_result: Any, cid: str | None, knowledge: list, chunk_ids: list
    ) -> None:
        """Emit all NER entities as fallback EntityInput items."""
        from knowledge_service.config import settings  # noqa: PLC0415
        from knowledge_service.models import EntityInput  # noqa: PLC0415

        for ent in nlp_result.entities:
            fallback = EntityInput(
                uri=ent.text,
                rdf_type=f"schema:{ent.label}" if ent.label else "schema:Thing",
                label=ent.text,
                confidence=settings.nlp_entity_confidence,
            )
            knowledge.append(fallback)
            chunk_ids.append(cid)

    @staticmethod
    def _emit_ner_missed(
        nlp_result: Any, items: list, cid: str | None, knowledge: list, chunk_ids: list
    ) -> None:
        """Emit NER entities that the LLM missed as fallback items."""
        from knowledge_service.config import settings  # noqa: PLC0415
        from knowledge_service.models import EntityInput  # noqa: PLC0415

        llm_labels = set()
        for item in items:
            if hasattr(item, "label"):
                llm_labels.add(item.label.lower())
            if hasattr(item, "subject"):
                llm_labels.add(item.subject.lower())
            elif isinstance(item, dict):
                for key in ("label", "subject", "uri"):
                    val = item.get(key)
                    if val:
                        llm_labels.add(val.lower())

        for ent in nlp_result.entities:
            if ent.text.lower() not in llm_labels:
                fallback = EntityInput(
                    uri=ent.text,
                    rdf_type=f"schema:{ent.label}" if ent.label else "schema:Thing",
                    label=ent.text,
                    confidence=settings.nlp_entity_confidence,
                )
                knowledge.append(fallback)
                chunk_ids.append(cid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingestion_worker.py::TestExtractPhaseFiltering -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/ingestion/phases.py tests/test_ingestion_worker.py
git commit -m "feat: integrate chunk filtering into ExtractPhase with NER fallback"
```

---

### Task 6: Update Worker to Handle `chunks_skipped`

**Files:**
- Modify: `src/knowledge_service/ingestion/worker.py`
- Create: `migrations/013_chunks_skipped.sql`
- Modify: `tests/test_ingestion_worker.py`

- [ ] **Step 1: Write failing test for chunks_skipped tracking**

Add to `tests/test_ingestion_worker.py`:

```python
class TestJobTrackerChunksSkipped:
    async def test_update_status_accepts_chunks_skipped(self):
        pool, conn = _make_mock_pool()
        tracker = JobTracker("job-id", pool)
        await tracker.update_status("extracting", chunks_skipped=5)
        call_args = conn.execute.call_args
        assert "chunks_skipped" in str(call_args)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingestion_worker.py::TestJobTrackerChunksSkipped -v`
Expected: FAIL — `ValueError: Invalid job columns: {'chunks_skipped'}`

- [ ] **Step 3: Add `chunks_skipped` to allowed columns and update worker**

In `src/knowledge_service/ingestion/worker.py`, add `"chunks_skipped"` to `_ALLOWED_JOB_COLUMNS`:

```python
_ALLOWED_JOB_COLUMNS = frozenset(
    {
        "chunks_embedded",
        "chunks_extracted",
        "chunks_failed",
        "chunks_skipped",
        "triples_created",
        "entities_resolved",
        "entities_linked",
        "entities_coref",
        "error",
    }
)
```

Update `run_ingestion()` extraction phase to unpack the 4-tuple and pass `chunks_skipped`:

```python
        # Phase 3: Extract
        current_phase = "extracting"
        await tracker.update_status("extracting")

        chunks_failed = 0
        chunks_skipped = 0
        if not knowledge and raw_text and extraction_client:
            extract = ExtractPhase(extraction_client)
            knowledge_items, chunk_ids_for_items, chunks_failed, chunks_skipped = await extract.run(
                chunk_records,
                chunk_id_map,
                title=title,
                source_type=source_type,
                nlp_hints=nlp_results,
            )
            extractor = "llm"
            chunks_extracted = len(chunk_records) - chunks_failed - chunks_skipped
        else:
            knowledge_items = list(knowledge or [])
            chunk_ids_for_items = [None] * len(knowledge_items)
            extractor = "api"
            chunks_extracted = 0

        await tracker.update_status(
            "extracting",
            chunks_extracted=chunks_extracted,
            chunks_failed=chunks_failed,
            chunks_skipped=chunks_skipped,
        )
```

- [ ] **Step 4: Create migration**

```sql
-- migrations/013_chunks_skipped.sql
ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS chunks_skipped INTEGER DEFAULT 0;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingestion_worker.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/e2e`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_service/ingestion/worker.py migrations/013_chunks_skipped.sql tests/test_ingestion_worker.py
git commit -m "feat: track chunks_skipped in ingestion jobs"
```

---

### Task 7: Lint, Format, and Final Verification

**Files:**
- All modified files

- [ ] **Step 1: Run ruff check and format**

```bash
uv run ruff check .
uv run ruff format --check .
```

Fix any issues found.

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest tests/ -v --ignore=tests/e2e
```

Expected: All PASS

- [ ] **Step 3: Fix any failures**

If any tests fail, investigate and fix. Common issues:
- Other test files that import from `ExtractionClient` and mock 2 LLM calls (need updating to 1)
- `test_ingestion_worker.py::TestRunIngestionWithNlp` may need updating for 4-tuple return

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -u
git commit -m "fix: resolve test regressions from single-pass extraction"
```

---

## Summary

| Task | What it does | LLM call reduction |
|------|-------------|-------------------|
| 1-2 | Chunk scoring + filtering | Skip 40-60% of chunks |
| 3-4 | Single-pass extraction | Halve calls for remaining chunks |
| 5 | Wire filtering into ExtractPhase | Connects 1-4 together |
| 6 | Track skipped chunks | Observability |
| 7 | Lint + regression fixes | Cleanup |

**Net effect:** ~70-80% fewer LLM calls per document.
