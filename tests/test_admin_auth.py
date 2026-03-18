"""Tests for admin authentication and configuration."""

import pytest


def test_config_requires_admin_password(monkeypatch):
    """Settings should fail validation when ADMIN_PASSWORD is not set."""
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    from pydantic import ValidationError
    from pydantic_settings import BaseSettings

    class TestSettings(BaseSettings):
        admin_password: str
        model_config = {"env_file": ".env"}

    with pytest.raises(ValidationError):
        TestSettings()


def test_config_accepts_admin_password(monkeypatch):
    """Settings should accept ADMIN_PASSWORD when set."""
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password-123")
    from knowledge_service.config import Settings

    s = Settings()
    assert s.admin_password == "test-password-123"


def test_secret_key_has_default(monkeypatch):
    """SECRET_KEY should have a generated default when not set."""
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password-123")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    from knowledge_service.config import Settings

    s = Settings()
    assert s.secret_key is not None
    assert len(s.secret_key) >= 32
