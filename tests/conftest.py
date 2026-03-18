import os

os.environ.setdefault("ADMIN_PASSWORD", "test-suite-password")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing")

import pytest
from itsdangerous import TimestampSigner


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_test_session_cookie() -> str:
    """Generate a valid ks_session cookie for test clients using create_app()."""
    signer = TimestampSigner(os.environ["SECRET_KEY"])
    return signer.sign("authenticated").decode()
