# Ingestion Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve extraction quality via entity-first two-phase LLM extraction and improve chunking quality via markdown-aware splitting with section metadata.

**Architecture:** Phase 5 replaces single-pass extraction in `clients/llm.py` with two LLM calls (entities first, relations second). Phase 6 extracts chunking into a new `chunking.py` module with markdown-aware splitting, adds `section_header` column to the content table, and propagates it through search and RAG.

**Tech Stack:** OpenAI-compatible chat API (qwen3:14b via LiteLLM), langchain_text_splitters (MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter), PostgreSQL

**Spec:** `docs/superpowers/specs/2026-03-22-ingestion-improvements-design.md`

---

## File Map

### Phase 5: Two-Phase Extraction

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/knowledge_service/clients/llm.py:236-348` | Replace `_build_extraction_prompt` with two-phase prompts, rewrite `extract()` |
| Modify | `tests/test_extraction_client.py` | Rewrite prompt tests, add two-phase tests |

### Phase 6: Structure-Aware Chunking

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/knowledge_service/chunking.py` | Markdown-aware chunking module |
| Create | `migrations/006_add_section_header.sql` | Add section_header column |
| Modify | `src/knowledge_service/api/content.py:125-148` | Use new chunking module |
| Modify | `src/knowledge_service/stores/embedding.py:120-154,165-216,217-270` | insert_chunks + search + search_bm25 include section_header |
| Modify | `src/knowledge_service/api/search.py:42-55` | Pass section_header to SearchResult |
| Modify | `src/knowledge_service/clients/rag.py:37-42` | Show section_header in prompt |
| Modify | `src/knowledge_service/models.py:316-326` | SearchResult gains section_header |
| Create | `tests/test_chunking.py` | Chunking module tests |

---

## Phase 5: Two-Phase Extraction

### Task 1: Entity extraction prompt and tests

**Files:**
- Modify: `src/knowledge_service/clients/llm.py:236-282`
- Modify: `tests/test_extraction_client.py`

- [ ] **Step 1: Write failing tests for entity extraction prompt**

Add to `tests/test_extraction_client.py`:

```python
def test_entity_extraction_prompt_focuses_on_entities():
    from knowledge_service.clients.llm import _build_entity_extraction_prompt

    prompt = _build_entity_extraction_prompt("Some text", title=None, source_type=None)
    assert "Entity" in prompt
    assert "Event" in prompt
    # Phase 1 should focus on Entity/Event, not relation types
    assert "Claim:" not in prompt
    assert "Relationship:" not in prompt
    assert "snake_case" in prompt
    assert "singular" in prompt.lower()


def test_entity_extraction_prompt_includes_text():
    from knowledge_service.clients.llm import _build_entity_extraction_prompt

    prompt = _build_entity_extraction_prompt("Cold exposure boosts dopamine.", title="Test", source_type="article")
    assert "Cold exposure boosts dopamine" in prompt
    assert "Title: Test" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extraction_client.py::test_entity_extraction_prompt_focuses_on_entities -v`
Expected: FAIL — `ImportError: cannot import name '_build_entity_extraction_prompt'`

- [ ] **Step 3: Implement _build_entity_extraction_prompt**

In `clients/llm.py`, add before the `ExtractionClient` class (replace `_build_extraction_prompt`):

```python
def _build_entity_extraction_prompt(text: str, title: str | None, source_type: str | None) -> str:
    context = ""
    if title:
        context += f"Title: {title}\n"
    if source_type:
        context += f"Source type: {source_type}\n"
    return f"""{context}Extract entities and events from the text below.
Return ONLY a JSON object: {{"items": [...]}}

Supported types and required fields:
- Entity: uri, rdf_type (e.g. "schema:Person"), label, properties (dict), confidence
- Event: subject, occurred_at (YYYY-MM-DD), confidence, properties (dict)

Entity naming rules:
- Use canonical, well-known names: "dopamine" not "the neurotransmitter dopamine"
- Use singular form: "neuron" not "neurons"
- Use lowercase snake_case: "cold_exposure" not "Cold Exposure"
- Be specific: "vitamin_d3" not "vitamin_d" when the text specifies D3

Extract 2-6 entities/events. If nothing found, return {{"items": []}}

Example:
Text: "Regular cold water immersion has been shown to increase dopamine levels."
Output: {{"items": [
  {{"knowledge_type": "Entity", "uri": "cold_water_immersion", "rdf_type": "schema:Thing", "label": "cold_water_immersion", "properties": {{}}, "confidence": 0.9}},
  {{"knowledge_type": "Entity", "uri": "dopamine", "rdf_type": "schema:Thing", "label": "dopamine", "properties": {{}}, "confidence": 0.95}}
]}}

Text:
---
{text[:_MAX_TEXT_CHARS]}
---"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extraction_client.py::test_entity_extraction_prompt_focuses_on_entities tests/test_extraction_client.py::test_entity_extraction_prompt_includes_text -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/clients/llm.py tests/test_extraction_client.py
git commit -m "feat: add entity extraction prompt (phase 1 of two-phase extraction)"
```

### Task 2: Relation extraction prompt and tests

**Files:**
- Modify: `src/knowledge_service/clients/llm.py`
- Modify: `tests/test_extraction_client.py`

- [ ] **Step 1: Write failing tests for relation extraction prompt**

```python
def test_relation_extraction_prompt_includes_entity_list():
    from knowledge_service.clients.llm import _build_relation_extraction_prompt

    prompt = _build_relation_extraction_prompt(
        "Cold exposure boosts dopamine.",
        title=None, source_type=None,
        entities=["cold_exposure", "dopamine"],
    )
    assert "cold_exposure" in prompt
    assert "dopamine" in prompt
    assert "Claim" in prompt
    assert "Fact" in prompt
    assert "causes" in prompt  # canonical predicates


def test_relation_extraction_prompt_constrains_to_entities():
    from knowledge_service.clients.llm import _build_relation_extraction_prompt

    prompt = _build_relation_extraction_prompt(
        "text", title=None, source_type=None,
        entities=["entity_a", "entity_b"],
    )
    assert "entity_a" in prompt
    assert "entity_b" in prompt
    # Should constrain subjects/objects to these entities
    assert "Only use" in prompt or "only use" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extraction_client.py::test_relation_extraction_prompt_includes_entity_list -v`
Expected: FAIL

- [ ] **Step 3: Implement _build_relation_extraction_prompt**

```python
def _build_relation_extraction_prompt(
    text: str,
    title: str | None,
    source_type: str | None,
    entities: list[str],
) -> str:
    context = ""
    if title:
        context += f"Title: {title}\n"
    if source_type:
        context += f"Source type: {source_type}\n"
    predicates_csv = ", ".join(CANONICAL_PREDICATES)
    entity_list = ", ".join(entities)
    return f"""{context}Given these entities: [{entity_list}]

Extract relationships between them from the text below.
Return ONLY a JSON object: {{"items": [...]}}

Supported types and required fields:
- Claim: subject, predicate, object, object_type, confidence (0.0-0.89)
- Fact: subject, predicate, object, object_type, confidence (0.9-1.0) for verified facts
- Relationship: subject, predicate, object, object_type, confidence
- TemporalState: subject, property, value, valid_from (YYYY-MM-DD), valid_until (YYYY-MM-DD), confidence
- Conclusion: concludes (text), derived_from (list of identifiers), inference_method, confidence

Preferred predicates (use these when applicable):
{predicates_csv}
Only invent a new predicate if none of the above fit.

For object values, include object_type ("entity" or "literal"):
- "entity": the object is a thing/concept (e.g. "dopamine", "postgresql")
- "literal": the object is a measurement, description, or date (e.g. "250%", "2024-01-15")

Only use entities from the list above as subjects and objects (entity references).
Literal objects (measurements, dates, descriptions) do not need to be from the list.

Use Claim for uncertain assertions, Fact for high-confidence verifiable statements.
Extract 3-8 relationships. If nothing found, return {{"items": []}}

Text:
---
{text[:_MAX_TEXT_CHARS]}
---"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extraction_client.py::test_relation_extraction_prompt_includes_entity_list tests/test_extraction_client.py::test_relation_extraction_prompt_constrains_to_entities -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/clients/llm.py tests/test_extraction_client.py
git commit -m "feat: add relation extraction prompt (phase 2 of two-phase extraction)"
```

### Task 3: Rewrite ExtractionClient.extract() for two-phase flow

**Files:**
- Modify: `src/knowledge_service/clients/llm.py:285-343`
- Modify: `tests/test_extraction_client.py`

- [ ] **Step 1: Write failing tests for two-phase extract()**

```python
class TestTwoPhaseExtract:
    async def test_makes_two_llm_calls(self, httpx_mock):
        """extract() should make 2 LLM calls: entities then relations."""
        # Phase 1 response: entities
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_chat_response([
                {"knowledge_type": "Entity", "uri": "dopamine", "rdf_type": "schema:Thing",
                 "label": "dopamine", "properties": {}, "confidence": 0.9},
            ]),
        )
        # Phase 2 response: relations
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_chat_response([
                {"knowledge_type": "Claim", "subject": "cold_exposure",
                 "predicate": "increases", "object": "dopamine", "confidence": 0.7},
            ]),
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("Cold exposure increases dopamine.")
        assert len(httpx_mock.get_requests()) == 2
        assert len(result) == 2  # 1 entity + 1 claim
        await client.close()

    async def test_returns_entities_only_when_phase2_fails(self, httpx_mock):
        """If phase 2 fails, return phase 1 entities only."""
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_chat_response([
                {"knowledge_type": "Entity", "uri": "x", "rdf_type": "schema:Thing",
                 "label": "x", "properties": {}, "confidence": 0.9},
            ]),
        )
        httpx_mock.add_response(url=_CHAT_URL, status_code=500)
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("text")
        assert len(result) == 1
        assert result[0].knowledge_type.value == "Entity"
        await client.close()

    async def test_returns_empty_when_phase1_fails(self, httpx_mock):
        """If phase 1 fails, return empty (don't attempt phase 2)."""
        httpx_mock.add_response(url=_CHAT_URL, status_code=500)
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("text")
        assert result == []
        assert len(httpx_mock.get_requests()) == 1  # only phase 1 attempted
        await client.close()

    async def test_phase2_prompt_contains_phase1_entities(self, httpx_mock):
        """Phase 2 prompt should contain entity names from phase 1."""
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_chat_response([
                {"knowledge_type": "Entity", "uri": "cold_exposure", "rdf_type": "schema:Thing",
                 "label": "cold_exposure", "properties": {}, "confidence": 0.9},
                {"knowledge_type": "Entity", "uri": "dopamine", "rdf_type": "schema:Thing",
                 "label": "dopamine", "properties": {}, "confidence": 0.95},
            ]),
        )
        httpx_mock.add_response(url=_CHAT_URL, json=_make_chat_response([]))
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        await client.extract("Cold exposure increases dopamine.")
        phase2_body = json.loads(httpx_mock.get_requests()[1].content)
        phase2_prompt = phase2_body["messages"][0]["content"]
        assert "cold_exposure" in phase2_prompt
        assert "dopamine" in phase2_prompt
        await client.close()

    async def test_skips_phase2_when_no_entities(self, httpx_mock):
        """If phase 1 returns no entities, skip phase 2."""
        httpx_mock.add_response(url=_CHAT_URL, json=_make_chat_response([]))
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("text")
        assert result == []
        assert len(httpx_mock.get_requests()) == 1
        await client.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extraction_client.py::TestTwoPhaseExtract -v`
Expected: FAIL — extract() still makes 1 call

- [ ] **Step 3: Rewrite extract() method**

Extract a helper `_call_llm` to avoid duplication, then rewrite `extract()`:

```python
async def _call_llm(self, prompt: str) -> list[dict]:
    """Send prompt to LLM, return parsed items list. Returns [] on failure."""
    try:
        response = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("ExtractionClient: LLM API returned %s", exc.response.status_code)
        return []
    except httpx.TimeoutException as exc:
        logger.warning("ExtractionClient: LLM API request timed out: %s", exc)
        return []

    raw = response.json()["choices"][0]["message"]["content"]
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        logger.warning("ExtractionClient: could not parse JSON response: %s", exc)
        return []

    return parsed.get("items", [])

async def extract(
    self,
    text: str,
    title: str | None = None,
    source_type: str | None = None,
) -> list:
    """Extract KnowledgeInput items via two-phase extraction.

    Phase 1: Extract entities and events.
    Phase 2: Extract relations between discovered entities.
    Returns combined list. Returns [] on phase 1 failure.
    """
    from knowledge_service.models import KnowledgeInput  # noqa: PLC0415

    adapter = TypeAdapter(KnowledgeInput)

    # Phase 1: Entity extraction
    phase1_prompt = _build_entity_extraction_prompt(text, title, source_type)
    phase1_raw = await self._call_llm(phase1_prompt)
    if not phase1_raw:
        return []

    phase1_items = []
    for item_dict in phase1_raw:
        try:
            phase1_items.append(adapter.validate_python(item_dict))
        except ValidationError as exc:
            logger.warning("ExtractionClient: skipping invalid phase1 item: %s", exc)

    if not phase1_items:
        return []

    # Collect entity names for phase 2 constraint
    entity_names: list[str] = []
    for item in phase1_items:
        if hasattr(item, "label"):
            entity_names.append(item.label)
        elif hasattr(item, "subject"):
            entity_names.append(item.subject)

    # Phase 2: Relation extraction
    phase2_prompt = _build_relation_extraction_prompt(text, title, source_type, entity_names)
    phase2_raw = await self._call_llm(phase2_prompt)

    phase2_items = []
    for item_dict in phase2_raw:
        try:
            phase2_items.append(adapter.validate_python(item_dict))
        except ValidationError as exc:
            logger.warning("ExtractionClient: skipping invalid phase2 item: %s", exc)

    return phase1_items + phase2_items
```

- [ ] **Step 4: Remove old `_build_extraction_prompt` function**

Delete lines 236-282 (the old single prompt function). It's now replaced by the two new functions.

- [ ] **Step 5: Update mock_llm fixture and rewrite old prompt tests**

The `mock_llm` fixture only mocks 1 response but two-phase extract needs 2. Update it:

```python
@pytest.fixture
def mock_llm(httpx_mock):
    # Phase 1: entity response
    httpx_mock.add_response(
        url=_CHAT_URL,
        json=_make_chat_response([
            {"knowledge_type": "Entity", "uri": "cold_exposure", "rdf_type": "schema:Thing",
             "label": "cold_exposure", "properties": {}, "confidence": 0.9},
        ]),
    )
    # Phase 2: relation response
    httpx_mock.add_response(
        url=_CHAT_URL,
        json=_make_chat_response([
            {"knowledge_type": "Claim", "subject": "cold_exposure",
             "predicate": "increases", "object": "dopamine", "confidence": 0.7},
        ]),
    )
    return httpx_mock
```

Update existing `TestExtract` tests that use `mock_llm`:
- `test_returns_claim_from_valid_response`: now expects 2 items (1 entity + 1 claim), assert `len(result) == 2`
- `test_extract_returns_raw_labels_not_uris`: check entity item `result[0].label == "cold_exposure"`

Delete the 5 standalone tests that import `_build_extraction_prompt` and replace with:

```python
def test_relation_prompt_includes_relation_types():
    from knowledge_service.clients.llm import _build_relation_extraction_prompt
    prompt = _build_relation_extraction_prompt("text", None, None, ["a", "b"])
    for t in ("Claim", "Fact", "Relationship", "TemporalState", "Conclusion"):
        assert t in prompt

def test_relation_prompt_includes_predicates():
    from knowledge_service.clients.llm import _build_relation_extraction_prompt
    prompt = _build_relation_extraction_prompt("text", None, None, ["a"])
    for pred in ("causes", "increases", "decreases"):
        assert pred in prompt

def test_relation_prompt_includes_object_type():
    from knowledge_service.clients.llm import _build_relation_extraction_prompt
    prompt = _build_relation_extraction_prompt("text", None, None, ["a"])
    assert "object_type" in prompt

def test_entity_prompt_includes_naming_rules():
    from knowledge_service.clients.llm import _build_entity_extraction_prompt
    prompt = _build_entity_extraction_prompt("text", None, None)
    assert "snake_case" in prompt
    assert "singular" in prompt.lower()

def test_entity_prompt_includes_example():
    from knowledge_service.clients.llm import _build_entity_extraction_prompt
    prompt = _build_entity_extraction_prompt("text", None, None)
    assert "Example:" in prompt
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_service/clients/llm.py tests/test_extraction_client.py
git commit -m "feat: two-phase extraction — entities first, relations second"
```

---

## Phase 6: Structure-Aware Chunking

### Task 4: Chunking module

**Files:**
- Create: `src/knowledge_service/chunking.py`
- Create: `tests/test_chunking.py`

- [ ] **Step 1: Write failing tests for chunking module**

Create `tests/test_chunking.py`:

```python
import pytest
from knowledge_service.chunking import chunk_text, _is_markdown


class TestIsMarkdown:
    def test_heading_detected(self):
        assert _is_markdown("# Title\nSome content") is True

    def test_h2_detected(self):
        assert _is_markdown("## Section\nContent here") is True

    def test_plain_text_not_markdown(self):
        assert _is_markdown("Just some plain text without headings.") is False

    def test_hash_in_code_not_detected(self):
        # Hash mid-line (e.g., in code) should not trigger
        assert _is_markdown("Use color #FF0000 for red.") is False


class TestChunkTextMarkdown:
    def test_splits_on_headings(self):
        text = "# Intro\nIntro content.\n\n## Methods\nMethod details here."
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=200)
        assert len(chunks) >= 2
        assert any("Intro content" in c["chunk_text"] for c in chunks)
        assert any("Method details" in c["chunk_text"] for c in chunks)

    def test_section_header_populated(self):
        text = "# Title\n## Section A\nContent A.\n\n## Section B\nContent B."
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=200)
        headers = [c["section_header"] for c in chunks if c["section_header"]]
        assert len(headers) >= 1

    def test_char_offsets_present(self):
        text = "# Heading\nSome content here."
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=200)
        for c in chunks:
            assert "char_start" in c
            assert "char_end" in c
            assert c["char_start"] >= 0
            assert c["char_end"] > c["char_start"]


class TestChunkTextPlain:
    def test_plain_text_returns_chunks(self):
        text = "A " * 3000  # 6000 chars
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=200)
        assert len(chunks) >= 2

    def test_plain_text_section_header_is_none(self):
        text = "Just some plain text."
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=200)
        assert all(c["section_header"] is None for c in chunks)

    def test_short_text_single_chunk(self):
        text = "Short text."
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=200)
        assert len(chunks) == 1
        assert chunks[0]["chunk_text"] == "Short text."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'knowledge_service.chunking'`

- [ ] **Step 3: Implement chunking module**

Create `src/knowledge_service/chunking.py`:

```python
"""Structure-aware text chunking with markdown heading support."""

from __future__ import annotations

import re

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def _is_markdown(text: str) -> bool:
    """Detect if text contains markdown headings."""
    sample = text[:2000]
    return bool(re.search(r"^#{1,3}\s", sample, re.MULTILINE))


def chunk_text(
    text: str,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """Split text into chunks with section headers and char offsets.

    Returns list of {"chunk_text": str, "section_header": str | None,
                      "char_start": int, "char_end": int}.
    """
    if not text or not text.strip():
        return []

    if _is_markdown(text):
        return _chunk_markdown(text, chunk_size, chunk_overlap)
    return _chunk_plain(text, chunk_size, chunk_overlap)


def _chunk_markdown(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[dict]:
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    md_docs = md_splitter.split_text(text)
    results: list[dict] = []

    for doc in md_docs:
        content = doc.page_content
        # Build section header from metadata
        header_parts = []
        for key in ("h1", "h2", "h3"):
            if key in doc.metadata:
                header_parts.append(doc.metadata[key])
        section_header = " > ".join(header_parts) if header_parts else None

        if len(content) > chunk_size:
            sub_chunks = sub_splitter.split_text(content)
            for sc in sub_chunks:
                offset = text.find(sc[:80])
                if offset == -1:
                    offset = 0
                results.append({
                    "chunk_text": sc,
                    "section_header": section_header,
                    "char_start": offset,
                    "char_end": offset + len(sc),
                })
        else:
            offset = text.find(content[:80])
            if offset == -1:
                offset = 0
            results.append({
                "chunk_text": content,
                "section_header": section_header,
                "char_start": offset,
                "char_end": offset + len(content),
            })

    return results if results else _chunk_plain(text, chunk_size, chunk_overlap)


def _chunk_plain(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["# ", "## ", "### ", "\n\n", "\n", ". ", " ", ""],
    )

    if len(text) < chunk_size:
        return [{"chunk_text": text, "section_header": None, "char_start": 0, "char_end": len(text)}]

    chunks = splitter.split_text(text)
    results: list[dict] = []
    search_start = 0
    for ct in chunks:
        offset = text.find(ct[:80], search_start)
        if offset == -1:
            offset = search_start
        results.append({
            "chunk_text": ct,
            "section_header": None,
            "char_start": offset,
            "char_end": offset + len(ct),
        })
        search_start = max(search_start, offset + 1)

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/chunking.py tests/test_chunking.py
git commit -m "feat: markdown-aware chunking module with section headers"
```

### Task 5: Migration and schema changes

**Files:**
- Create: `migrations/006_add_section_header.sql`
- Modify: `src/knowledge_service/stores/embedding.py:120-154` (insert_chunks)
- Modify: `src/knowledge_service/models.py:316-326` (SearchResult)

- [ ] **Step 1: Create migration**

```sql
ALTER TABLE content ADD COLUMN section_header TEXT;
```

- [ ] **Step 2: Update insert_chunks to include section_header**

In `embedding.py`, update the SQL in `insert_chunks()`:

```sql
INSERT INTO content (
    content_id, chunk_index, chunk_text, embedding, char_start, char_end, section_header
)
VALUES ($1, $2, $3, $4::vector(768), $5, $6, $7)
RETURNING id
```

Add `chunk.get("section_header")` as the 7th parameter:

```python
row = await conn.fetchrow(
    sql,
    content_id,
    chunk["chunk_index"],
    chunk["chunk_text"],
    embedding_str,
    chunk["char_start"],
    chunk["char_end"],
    chunk.get("section_header"),
)
```

- [ ] **Step 3: Add section_header to SearchResult model**

In `models.py`, add to `SearchResult`:

```python
class SearchResult(BaseModel):
    content_id: str
    url: str
    title: str
    summary: str | None
    similarity: float
    source_type: str
    tags: list[str]
    ingested_at: datetime
    chunk_text: str
    chunk_index: int
    section_header: str | None = None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add migrations/006_add_section_header.sql src/knowledge_service/stores/embedding.py src/knowledge_service/models.py
git commit -m "feat: section_header column, insert_chunks support, SearchResult field"
```

### Task 6: Propagate section_header through search and RAG

**Files:**
- Modify: `src/knowledge_service/stores/embedding.py:165-216,217-270` (search + search_bm25 SELECT)
- Modify: `src/knowledge_service/api/search.py:42-55`
- Modify: `src/knowledge_service/clients/rag.py:37-42`

- [ ] **Step 1: Add c.section_header to search() SQL**

In `embedding.py`, in the `search()` method's SQL SELECT clause, add `c.section_header` after `c.chunk_index`:

```sql
SELECT
    c.id, c.chunk_text, c.chunk_index, c.section_header,
    m.id AS content_id, ...
```

- [ ] **Step 2: Add c.section_header to search_bm25() SQL**

Same change in `search_bm25()`:

```sql
SELECT
    c.id, c.chunk_text, c.chunk_index, c.section_header,
    m.id AS content_id, ...
```

- [ ] **Step 3: Update /api/search to pass section_header**

In `api/search.py`, add `section_header` to the SearchResult construction:

```python
SearchResult(
    ...
    chunk_index=row["chunk_index"],
    section_header=row.get("section_header"),
)
```

- [ ] **Step 4: Update RAG prompt to show section header**

In `clients/rag.py`, update the content section (around line 42):

```python
section = f" [Section: {row.get('section_header')}]" if row.get("section_header") else ""
sections.append(f'- "{title}" ({source_type}, similarity: {similarity:.2f}){section}: {text}')
```

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/stores/embedding.py src/knowledge_service/api/search.py src/knowledge_service/clients/rag.py
git commit -m "feat: propagate section_header through search and RAG prompt"
```

### Task 7: Wire chunking module into content.py

**Files:**
- Modify: `src/knowledge_service/api/content.py:125-148`

- [ ] **Step 1: Replace chunking logic with new module**

In `content.py`, replace the chunking block (lines 125-148) with:

```python
from knowledge_service.chunking import chunk_text as split_into_chunks

# Step 2: Chunk and embed
text = body.raw_text or body.summary or body.title
raw_chunks = split_into_chunks(text, chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)

chunk_records: list[dict] = []
for i, rc in enumerate(raw_chunks):
    chunk_records.append({
        "chunk_index": i,
        "chunk_text": rc["chunk_text"],
        "char_start": rc["char_start"],
        "char_end": rc["char_end"],
        "section_header": rc.get("section_header"),
    })
```

Remove the old `_splitter` module-level constant and the `langchain_text_splitters` import from content.py (it's now in `chunking.py`).

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Run lint**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_service/api/content.py
git commit -m "feat: wire markdown-aware chunking into content ingestion"
```

### Task 8: Final integration test and lint

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Run lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean

- [ ] **Step 3: Fix any issues and commit**

```bash
git add -A && git commit -m "chore: lint fixes for ingestion improvements"
```
