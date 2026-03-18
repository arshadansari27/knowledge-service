import os

os.environ.setdefault("ADMIN_PASSWORD", "test-suite-password")

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
