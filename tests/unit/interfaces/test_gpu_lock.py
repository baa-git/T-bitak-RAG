"""Tests for the GPU lock file ("GPU kullanım kuralı")."""

from pathlib import Path

import pytest

from tubitak_rag.config import AppConfig
from tubitak_rag.interfaces.gpu_lock import (
    GpuLockError,
    busy_message,
    gpu_lock,
    lock_holder,
    uses_gpu,
)


def test_lock_file_holds_owner_while_held_and_is_removed_after(tmp_path: Path) -> None:
    lock = tmp_path / "artifacts" / ".gpu.lock"

    with gpu_lock(lock, "rag ingest (v1)"):
        assert lock.read_text(encoding="utf-8").startswith("rag ingest (v1)")

    assert not lock.exists()


def test_held_lock_is_refused_in_turkish_and_left_alone(tmp_path: Path) -> None:
    lock = tmp_path / ".gpu.lock"
    lock.write_text("deney koşusu v3\n2026-10-15T10:00:00\n", encoding="utf-8")

    with pytest.raises(GpuLockError, match="GPU kilidi") as caught, gpu_lock(lock, "rag ingest"):
        pass

    assert "deney koşusu v3" in str(caught.value)
    assert lock.exists()


def test_lock_is_released_when_the_work_fails(tmp_path: Path) -> None:
    lock = tmp_path / ".gpu.lock"

    with pytest.raises(RuntimeError), gpu_lock(lock, "rag ingest"):
        raise RuntimeError("CUDA OOM")

    assert not lock.exists()


def test_lock_holder_describes_a_held_lock_and_is_none_when_free(tmp_path: Path) -> None:
    lock = tmp_path / ".gpu.lock"
    assert lock_holder(lock) is None

    lock.write_text("rag run (v3) pid=7\n2026-10-16T01:00:00\n", encoding="utf-8")

    assert lock_holder(lock) == "rag run (v3) pid=7 · 2026-10-16T01:00:00"


def test_busy_message_is_turkish_and_names_the_holder(tmp_path: Path) -> None:
    message = busy_message(tmp_path / ".gpu.lock", "rag run (v3)")
    assert "GPU kilidi dolu" in message and "rag run (v3)" in message


def _cfg(embedder: str, llm: str, reranker: str = "fake", enabled: bool = True) -> AppConfig:
    return AppConfig.model_validate(
        {
            "embedder": {"provider": embedder},
            "llm": {"provider": llm},
            "reranker": {"provider": reranker, "enabled": enabled},
        }
    )


def test_uses_gpu_for_real_embedder_always() -> None:
    assert uses_gpu(_cfg("sentence_transformers", "fake"), answering=False)


def test_all_fake_config_never_uses_gpu() -> None:
    assert not uses_gpu(_cfg("fake", "fake"), answering=True)


def test_answering_uses_gpu_for_real_llm_or_reranker_only() -> None:
    assert uses_gpu(_cfg("fake", "ollama"), answering=True)
    assert not uses_gpu(_cfg("fake", "ollama"), answering=False)
    assert uses_gpu(_cfg("fake", "fake", "cross_encoder"), answering=True)
    assert not uses_gpu(_cfg("fake", "fake", "cross_encoder", enabled=False), answering=True)
