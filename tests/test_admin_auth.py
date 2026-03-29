"""Tests for admin authentication and configuration."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.responses import PlainTextResponse

from knowledge_service.admin.auth import AuthMiddleware, login_router


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
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-123")
    from knowledge_service.config import Settings

    s = Settings()
    assert s.admin_password == "test-password-123"


def test_secret_key_is_required(monkeypatch):
    """SECRET_KEY must be explicitly set — no auto-generated default."""
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password-123")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    from pydantic import ValidationError
    from knowledge_service.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    errors = exc_info.value.errors()
    missing_fields = [e["loc"][0] for e in errors if e["type"] == "missing"]
    assert "secret_key" in missing_fields


@pytest.fixture
def auth_app():
    """Create a minimal FastAPI app with auth middleware for testing."""
    app = FastAPI()
    # Store credentials on app.state so login_submit can access them
    app.state.admin_password = "testpass123"
    app.state.secret_key = "testsecretkey123"

    app.include_router(login_router)

    @app.get("/admin")
    async def admin_page():
        return PlainTextResponse("admin content")

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok")

    @app.get("/api/test")
    async def api_test():
        return {"status": "ok"}

    @app.get("/docs")
    async def docs_page():
        return PlainTextResponse("swagger")

    app.add_middleware(
        AuthMiddleware,
        admin_password="testpass123",
        secret_key="testsecretkey123",
    )
    return app


@pytest.fixture
async def auth_client(auth_app):
    transport = ASGITransport(app=auth_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_health_is_unauthenticated(auth_client):
    resp = await auth_client.get("/health")
    assert resp.status_code == 200


async def test_docs_is_unauthenticated(auth_client):
    resp = await auth_client.get("/docs")
    assert resp.status_code == 200


async def test_admin_redirects_to_login(auth_client):
    resp = await auth_client.get("/admin", follow_redirects=False)
    assert resp.status_code == 307
    assert "/login" in resp.headers["location"]


async def test_api_returns_401_without_auth(auth_client):
    resp = await auth_client.get("/api/test")
    assert resp.status_code == 401


async def test_login_page_renders(auth_client):
    resp = await auth_client.get("/login")
    assert resp.status_code == 200


async def test_login_with_correct_password(auth_client):
    resp = await auth_client.post(
        "/login",
        data={"password": "testpass123"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "ks_session" in resp.cookies


async def test_login_with_wrong_password(auth_client):
    resp = await auth_client.post(
        "/login",
        data={"password": "wrongpassword"},
        follow_redirects=False,
    )
    assert resp.status_code == 200  # Re-renders login page with error
    assert "ks_session" not in resp.cookies


async def test_authenticated_access(auth_client):
    # Login first
    login_resp = await auth_client.post(
        "/login",
        data={"password": "testpass123"},
        follow_redirects=False,
    )
    cookies = login_resp.cookies

    # Access admin with session cookie
    resp = await auth_client.get("/admin", cookies=cookies)
    assert resp.status_code == 200


async def test_logout_clears_session(auth_client):
    # Login
    login_resp = await auth_client.post(
        "/login",
        data={"password": "testpass123"},
        follow_redirects=False,
    )
    cookies = login_resp.cookies

    # Logout
    logout_resp = await auth_client.post("/logout", cookies=cookies, follow_redirects=False)
    assert logout_resp.status_code == 303

    # Verify session is cleared — admin should redirect to login
    resp = await auth_client.get("/admin", follow_redirects=False)
    assert resp.status_code == 307
