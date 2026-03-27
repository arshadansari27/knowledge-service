"""Tests for Settings configuration."""

import pytest
from pydantic import ValidationError

from knowledge_service.config import Settings


def test_ingestion_pipeline_config_defaults(monkeypatch):
    """New ingestion pipeline fields should have correct default values."""
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")

    settings = Settings()

    assert settings.spacy_data_dir == "/app/data/spacy"
    assert settings.max_upload_size == 50 * 1024 * 1024
    assert settings.url_fetch_timeout == 30
    assert settings.nlp_entity_confidence == 0.5


def test_secret_key_is_required_no_default(monkeypatch):
    """secret_key must be explicitly provided — no auto-generated default.

    If secret_key had a default (e.g. secrets.token_hex(32)), every restart
    would generate a different key, invalidating all active sessions.
    """
    # Remove SECRET_KEY from environment so Settings cannot find it
    monkeypatch.delenv("SECRET_KEY", raising=False)

    # Also ensure ADMIN_PASSWORD is present so it doesn't mask the error
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")

    # Import inside the test to avoid module-level side effects;
    # use a fresh Settings() instantiation rather than the module singleton.
    from knowledge_service.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    missing_fields = [e["loc"][0] for e in errors if e["type"] == "missing"]
    assert "secret_key" in missing_fields, (
        f"Expected 'secret_key' to be a required field with no default, "
        f"but validation errors were: {errors}"
    )
