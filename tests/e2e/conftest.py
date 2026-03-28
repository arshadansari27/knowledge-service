"""E2E test configuration — starts the real service as a subprocess.

Requires: PostgreSQL (docker compose up -d postgres), Ollama running.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

E2E_DB_URL = os.getenv("E2E_DB_URL", "postgresql://knowledge:knowledge@localhost:5433/knowledge")
E2E_LLM_URL = os.getenv("E2E_LLM_URL", "http://localhost:11434")
E2E_LLM_CHAT_MODEL = os.getenv("E2E_LLM_CHAT_MODEL", "llama3.1:8b-instruct-q4_K_M")
E2E_LLM_EMBED_MODEL = os.getenv("E2E_LLM_EMBED_MODEL", "nomic-embed-text")
E2E_SPACY_DATA = os.getenv("E2E_SPACY_DATA", "/tmp/spacy_e2e_data")
E2E_ADMIN_PASSWORD = os.getenv("E2E_ADMIN_PASSWORD", "changeme")
E2E_SECRET_KEY = os.getenv("E2E_SECRET_KEY", "e2e_test_secret_key_not_for_production")
E2E_PORT = int(os.getenv("E2E_PORT", "8199"))

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def pytest_collection_modifyitems(items):
    """Auto-mark all tests in e2e/ directory."""
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session")
def e2e_config():
    return {
        "db_url": E2E_DB_URL,
        "llm_url": E2E_LLM_URL,
        "chat_model": E2E_LLM_CHAT_MODEL,
        "embed_model": E2E_LLM_EMBED_MODEL,
        "spacy_data": E2E_SPACY_DATA,
        "admin_password": E2E_ADMIN_PASSWORD,
        "secret_key": E2E_SECRET_KEY,
        "fixtures_dir": str(FIXTURES_DIR),
        "base_url": f"http://localhost:{E2E_PORT}",
        "port": E2E_PORT,
    }


@pytest.fixture(scope="session")
def e2e_service(e2e_config):
    """Start the knowledge-service as a subprocess for E2E testing."""
    env = {
        **os.environ,
        "DATABASE_URL": e2e_config["db_url"],
        "LLM_BASE_URL": e2e_config["llm_url"],
        "LLM_CHAT_MODEL": e2e_config["chat_model"],
        "LLM_EMBED_MODEL": e2e_config["embed_model"],
        "SPACY_DATA_DIR": e2e_config["spacy_data"],
        "ADMIN_PASSWORD": e2e_config["admin_password"],
        "SECRET_KEY": e2e_config["secret_key"],
        "OXIGRAPH_DATA_DIR": "/tmp/e2e_oxigraph",
        "API_HOST": "127.0.0.1",
        "API_PORT": str(e2e_config["port"]),
    }

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "knowledge_service.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(e2e_config["port"]),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for health check
    base_url = e2e_config["base_url"]
    for attempt in range(30):
        try:
            resp = httpx.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                break
        except (httpx.ConnectError, httpx.ReadError):
            pass
        time.sleep(2)
    else:
        # Dump output for debugging
        proc.terminate()
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        pytest.fail(f"Service failed to start within 60s. Output:\n{stdout}")

    yield proc

    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture
def client(e2e_service, e2e_config):
    """Provide a sync HTTP client pointed at the running service."""
    return httpx.Client(base_url=e2e_config["base_url"], timeout=30)


@pytest.fixture
def api_headers(e2e_config):
    """Return headers with API key for authenticated endpoints."""
    return {"X-API-Key": e2e_config["admin_password"]}
