"""Fixtures for integration tests that talk to real services (NFR-TEST4).

Integration tests set ``pytestmark = pytest.mark.integration``; they are excluded by default
(``addopts = -m 'not integration'``) and run with ``pytest -m integration``. Each service
fixture skips the test when its URL environment variable is not set.
"""

import os

import pytest


def _service_url(env_var: str) -> str:
    url = os.environ.get(env_var)
    if not url:
        pytest.skip(f"{env_var} tanımlı değil; entegrasyon testi atlandı")
    return url


@pytest.fixture
def qdrant_url() -> str:
    """URL of a running Qdrant instance (``QDRANT_URL``)."""
    return _service_url("QDRANT_URL")


@pytest.fixture
def ollama_url() -> str:
    """URL of a running Ollama instance (``OLLAMA_URL``)."""
    return _service_url("OLLAMA_URL")
