"""E2E test configuration — requires real PostgreSQL, Ollama, and spaCy KB."""

import os
import pytest

E2E_DB_URL = os.getenv(
    "E2E_DB_URL", "postgresql://knowledge:knowledge@localhost:5433/knowledge_test"
)
E2E_LLM_URL = os.getenv("E2E_LLM_URL", "http://localhost:11434")
E2E_SPACY_DATA = os.getenv("E2E_SPACY_DATA", "/tmp/spacy_test_data")


def pytest_collection_modifyitems(items):
    """Auto-mark all tests in e2e/ directory."""
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


@pytest.fixture
def e2e_db_url():
    return E2E_DB_URL


@pytest.fixture
def e2e_llm_url():
    return E2E_LLM_URL


@pytest.fixture
def e2e_spacy_data():
    return E2E_SPACY_DATA
