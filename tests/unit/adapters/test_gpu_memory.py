"""Releasing cached GPU memory after a pipeline is dropped (web pipeline cache)."""

import sys
import types

import pytest

from tubitak_rag.adapters.gpu_memory import release_gpu_memory


def _fake_torch(available: bool, calls: list[str]) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: available, empty_cache=lambda: calls.append("empty_cache")
    )
    return torch


def test_empties_cuda_cache_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, calls))
    release_gpu_memory()
    assert calls == ["empty_cache"]


def test_skips_cuda_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, calls))
    release_gpu_memory()
    assert calls == []


def test_works_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    release_gpu_memory()
