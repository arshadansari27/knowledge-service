"""Authentication middleware and login/logout routes."""

from __future__ import annotations

import hmac
import time
from collections import defaultdict

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner
from starlette.middleware.base import BaseHTTPMiddleware

login_router = APIRouter()

# Paths that do not require authentication
_PUBLIC_PATHS = frozenset({"/health", "/login", "/docs", "/openapi.json", "/redoc"})

# Rate limiting: {ip: [timestamp, ...]}
_login_attempts: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT = 5  # attempts
_RATE_WINDOW = 60  # seconds


def _is_rate_limited(ip: str) -> bool:
    """Check if an IP has exceeded the login attempt rate limit."""
    now = time.monotonic()
    attempts = _login_attempts[ip]
    # Remove expired entries
    _login_attempts[ip] = [t for t in attempts if now - t < _RATE_WINDOW]
    return len(_login_attempts[ip]) >= _RATE_LIMIT


def _record_attempt(ip: str) -> None:
    """Record a failed login attempt."""
    _login_attempts[ip].append(time.monotonic())


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that protects all routes behind a session cookie.

    Public paths (health, login, docs) are excluded.
    API paths receive 401; UI paths are redirected to /login.
    """

    def __init__(self, app, admin_password: str, secret_key: str):
        super().__init__(app)
        self.admin_password = admin_password
        self.signer = TimestampSigner(secret_key)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow public paths
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        # Check session cookie
        session_cookie = request.cookies.get("ks_session")
        if session_cookie:
            try:
                self.signer.unsign(session_cookie, max_age=86400)  # 24h
                return await call_next(request)
            except (BadSignature, SignatureExpired):
                pass

        # Not authenticated
        if path.startswith("/api/"):
            return Response(status_code=401, content="Unauthorized")

        return RedirectResponse(url="/login", status_code=307)


_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login — Knowledge Service</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 flex items-center justify-center min-h-screen">
    <div class="bg-gray-800 p-8 rounded-lg shadow-lg w-full max-w-sm">
        <h1 class="text-2xl font-bold text-white mb-6 text-center">Knowledge Service</h1>
        {error}
        <form method="POST" action="/login">
            <label class="block text-gray-300 text-sm mb-2" for="password">Password</label>
            <input type="password" name="password" id="password" required autofocus
                   class="w-full p-3 rounded bg-gray-700 text-white border border-gray-600 focus:border-blue-500 focus:outline-none mb-4">
            <button type="submit"
                    class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded transition">
                Sign In
            </button>
        </form>
    </div>
</body>
</html>"""


@login_router.get("/login", response_class=HTMLResponse)
async def login_page():
    """Render the login page."""
    return _LOGIN_HTML.format(error="")


@login_router.post("/login")
async def login_submit(request: Request):
    """Validate the password and set a session cookie.

    Reads admin_password and secret_key from app.state (set during create_app)
    to stay consistent with the AuthMiddleware's configured credentials.
    """
    admin_password = request.app.state.admin_password
    secret_key = request.app.state.secret_key

    client_ip = request.client.host if request.client else "unknown"

    if _is_rate_limited(client_ip):
        error_html = '<p class="text-red-400 text-sm mb-4">Too many attempts. Try again later.</p>'
        return HTMLResponse(_LOGIN_HTML.format(error=error_html))

    form = await request.form()
    password = form.get("password", "")

    if not hmac.compare_digest(str(password), admin_password):
        _record_attempt(client_ip)
        error_html = '<p class="text-red-400 text-sm mb-4">Incorrect password.</p>'
        return HTMLResponse(_LOGIN_HTML.format(error=error_html))

    # Create signed session cookie
    signer = TimestampSigner(secret_key)
    session_value = signer.sign("authenticated").decode()

    response = RedirectResponse(url="/admin", status_code=303)
    response.set_cookie(
        key="ks_session",
        value=session_value,
        httponly=True,
        samesite="lax",
        max_age=86400,
    )
    return response


@login_router.post("/logout")
async def logout():
    """Clear the session cookie and redirect to login."""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("ks_session")
    return response
