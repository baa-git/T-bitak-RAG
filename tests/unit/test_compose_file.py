"""Static checks of the Docker Compose files (NFR-DOCKER1/3, NFR-HW2, NFR-PRIV1)."""

from pathlib import Path
from typing import Any

import pytest
import yaml


def _load(repo_root: Path, name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((repo_root / name).read_text(encoding="utf-8"))
    return data


@pytest.fixture
def compose(repo_root: Path) -> dict[str, Any]:
    return _load(repo_root, "docker-compose.yml")


def test_stack_has_app_qdrant_ollama_and_init(compose: dict[str, Any]) -> None:
    assert compose["name"] == "tubitak-rag"  # own project: no clash with host containers
    assert set(compose["services"]) == {"app", "qdrant", "ollama", "init"}
    assert compose["services"]["qdrant"]["image"] == "qdrant/qdrant:v1.19.0"


def test_only_the_web_app_publishes_a_port(compose: dict[str, Any]) -> None:
    services = compose["services"]
    assert "ports" not in services["qdrant"] and "ports" not in services["ollama"]
    (port,) = services["app"]["ports"]
    assert port.startswith("${APP_BIND:-127.0.0.1}:${APP_PORT:-8000}:8000")


def test_named_volumes_persist_index_models_and_runs(compose: dict[str, Any]) -> None:
    assert set(compose["volumes"]) == {
        "qdrant_storage",
        "ollama_models",
        "hf_cache",
        "artifacts",
        "raw_data",
    }
    app_mounts = {m.split(":")[1] for m in compose["services"]["app"]["volumes"]}
    assert {"/app/artifacts", "/app/data/raw", "/hf_cache"} <= app_mounts
    assert compose["services"]["qdrant"]["volumes"] == ["qdrant_storage:/qdrant/storage"]


def test_app_waits_for_init_and_healthy_services(compose: dict[str, Any]) -> None:
    services = compose["services"]
    depends = services["app"]["depends_on"]
    assert depends["init"]["condition"] == "service_completed_successfully"
    assert depends["qdrant"]["condition"] == depends["ollama"]["condition"] == "service_healthy"
    assert services["init"]["entrypoint"][-1] == "tubitak_rag.interfaces.bootstrap"
    assert "healthcheck" in services["qdrant"] and "healthcheck" in services["ollama"]


def test_default_stack_is_cpu_and_local(compose: dict[str, Any]) -> None:
    app = compose["services"]["app"]
    assert "deploy" not in app and "deploy" not in compose["services"]["ollama"]
    assert app["build"]["args"]["TORCH_INDEX_URL"].endswith("/whl/cpu}")
    assert "ANTHROPIC_API_KEY" not in str(compose)  # NFR-PRIV1: no API key is passed in


def test_gpu_override_reserves_nvidia_devices(repo_root: Path) -> None:
    gpu = _load(repo_root, "docker-compose.gpu.yml")["services"]
    for name in ("app", "init", "ollama"):
        device = gpu[name]["deploy"]["resources"]["reservations"]["devices"][0]
        assert device["driver"] == "nvidia" and device["capabilities"] == ["gpu"]
    assert "cu12" in gpu["app"]["build"]["args"]["TORCH_INDEX_URL"]
    assert gpu["app"]["image"] == "tubitak-rag-app:gpu"


def test_init_reads_the_prebuilt_index_bundle(compose: dict[str, Any]) -> None:
    init = compose["services"]["init"]
    assert "${INDEX_BUNDLE_DIR:-./release}:/app/release:ro" in init["volumes"]
    assert {"raw_data:/app/data/raw", "artifacts:/app/artifacts", "hf_cache:/hf_cache"} <= set(
        init["volumes"]
    )
    env = init["environment"]
    # Varsayılan: projenin Hugging Face veri seti; .env ile ezilebilir.
    assert env["INDEX_BUNDLE_URL"].startswith(
        "${INDEX_BUNDLE_URL:-https://huggingface.co/datasets/lolalol57/"
    )
    assert env["INDEX_BUNDLE_URL"].endswith("/resolve/main/}")
    assert env["TUBITAK_RAG_USE_BUNDLE"] == "${TUBITAK_RAG_USE_BUNDLE:-1}"
    assert "release" not in str(compose["services"]["app"].get("volumes"))


def test_hf_token_is_passed_through_only_when_set(compose: dict[str, Any]) -> None:
    for name in ("init", "app"):
        assert compose["services"][name]["environment"]["HF_TOKEN"] == "${HF_TOKEN:-}"


def test_openai_compatible_settings_default_to_the_local_ollama(compose: dict[str, Any]) -> None:
    for name in ("init", "app"):
        env = compose["services"][name]["environment"]
        assert env["LLM_BASE_URL"] == "${LLM_BASE_URL:-http://ollama:11434/v1}"
        assert env["LLM_API_KEY"] == "${LLM_API_KEY:-}"  # empty unless set in .env


def test_release_folder_is_kept_out_of_git_and_the_image(repo_root: Path) -> None:
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    dockerignore = (repo_root / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "release/*" in gitignore and "!release/.gitkeep" in gitignore
    assert "release/" in dockerignore
    assert (repo_root / "release" / ".gitkeep").is_file()
