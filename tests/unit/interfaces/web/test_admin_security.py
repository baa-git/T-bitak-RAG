"""Untrusted requests must not start administrative work (ingest, experiment runs).

Exercise the real application and PipelineBackend, replacing only the job launcher with
a spy so that a security test cannot start ingest, alter an index or use the GPU.
"""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from tubitak_rag.config import AppConfig
from tubitak_rag.interfaces.web.api import ADMIN_TOKEN_ENV
from tubitak_rag.interfaces.web.app import create_app
from tubitak_rag.interfaces.web.pipeline_backend import PipelineBackend
from tubitak_rag.interfaces.web.schemas import JobStarted

ADMIN_OPERATIONS = [
    pytest.param("/api/ingest", None, id="ingest-without-body"),
    pytest.param("/api/ingest", {"full": False}, id="incremental-ingest"),
    pytest.param("/api/ingest", {"full": True}, id="full-index-rebuild"),
    pytest.param("/api/runs", {"config": "v3"}, id="unlimited-experiment"),
    pytest.param("/api/runs", {"config": "v3", "limit": 1}, id="limited-experiment"),
]
ADMIN_TOKEN = "correct-admin-token"


@pytest.fixture
def admin_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[TestClient, Mock]]:
    monkeypatch.setenv(ADMIN_TOKEN_ENV, ADMIN_TOKEN)
    cfg = AppConfig(paths={"artifacts_dir": tmp_path / "artifacts"})
    backend = PipelineBackend(cfg, config_path=tmp_path / "server.yaml", probes={})
    launcher = Mock(return_value=JobStarted(job_id="review-job"))
    monkeypatch.setattr(backend, "start_ingest", launcher)
    monkeypatch.setattr(backend, "start_run", launcher)
    with TestClient(create_app(backend=backend)) as client:
        yield client, launcher


@pytest.mark.parametrize(("path", "payload"), ADMIN_OPERATIONS)
@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="anonymous"),
        pytest.param({"Authorization": "Bearer invalid-review-token"}, id="invalid-token"),
    ],
)
def test_untrusted_request_is_denied_before_starting_administrative_work(
    admin_client: tuple[TestClient, Mock],
    path: str,
    payload: dict[str, object] | None,
    headers: dict[str, str],
) -> None:
    client, launcher = admin_client

    response = client.post(path, json=payload, headers=headers)

    assert (response.status_code in {401, 403}) and not launcher.called, (
        f"Administrative request must be denied before launching work; "
        f"status={response.status_code}, launcher_calls={launcher.call_count}"
    )


@pytest.mark.parametrize(("path", "payload"), ADMIN_OPERATIONS)
def test_forged_localhost_headers_cannot_authorize_administrative_work(
    admin_client: tuple[TestClient, Mock], path: str, payload: dict[str, object] | None
) -> None:
    client, launcher = admin_client

    response = client.post(
        path,
        json=payload,
        headers={"X-Forwarded-For": "127.0.0.1", "X-Real-IP": "127.0.0.1"},
    )

    assert (response.status_code in {401, 403}) and not launcher.called, (
        f"Client-supplied IP headers must not grant administrative access; "
        f"status={response.status_code}, launcher_calls={launcher.call_count}"
    )


@pytest.mark.parametrize(("path", "expected_status"), [("/api/ingest", 405), ("/api/runs", 200)])
def test_read_request_never_starts_administrative_work(
    admin_client: tuple[TestClient, Mock], path: str, expected_status: int
) -> None:
    client, launcher = admin_client

    response = client.get(path)

    assert response.status_code == expected_status
    launcher.assert_not_called()


@pytest.mark.parametrize(("path", "payload"), ADMIN_OPERATIONS)
def test_admin_routes_are_closed_without_a_server_token(
    admin_client: tuple[TestClient, Mock],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    payload: dict[str, object] | None,
) -> None:
    client, launcher = admin_client
    monkeypatch.delenv(ADMIN_TOKEN_ENV)

    response = client.post(path, json=payload, headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})

    assert response.status_code == 403 and ADMIN_TOKEN_ENV in response.json()["detay"]
    launcher.assert_not_called()


@pytest.mark.parametrize(("path", "payload"), ADMIN_OPERATIONS)
def test_valid_admin_token_starts_the_job(
    admin_client: tuple[TestClient, Mock], path: str, payload: dict[str, object] | None
) -> None:
    client, launcher = admin_client

    response = client.post(path, json=payload, headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})

    assert response.status_code == 202
    launcher.assert_called_once()
