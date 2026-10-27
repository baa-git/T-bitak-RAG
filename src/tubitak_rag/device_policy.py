"""Where the local models run, and how the reranker adapts to a CPU-only machine.

Two environment switches sit on top of the YAML config (Docker sets them in compose):

* ``TUBITAK_RAG_DEVICE=auto|cpu|cuda`` overrides ``embedder.device`` and ``reranker.device``.
  ``auto`` picks the GPU when PyTorch sees one; ``cuda`` without a GPU is a config error
  (asking for the GPU explicitly must not silently fall back to a 30x slower CPU run).
* ``RERANKER_CPU_CANDIDATES=<n>|none`` overrides ``reranker.cpu_max_candidates``.

The cross-encoder needs one full model pass per (question, candidate) pair: measured
~0.6 s per pair on the host CPU (30 candidates ≈ 19 s) and 0.6 s for all 30 on the GPU;
inside Docker Desktop's VM 8 candidates took 5.7–6.4 s and 6 took 4.7 s.
NFR-PERF1 asks for retrieval + rerank under 5 s on CPU, so when the reranker
ends up on the CPU only the best ``cpu_max_candidates`` retrieval results are reranked
(default 6 ≈ 4.7 s + 0.2 s retrieval); ``0`` switches the reranker off on CPU, ``none``
keeps every candidate. On the GPU nothing changes.

This module holds only the decisions; the GPU probe is injected (``factory`` passes the
PyTorch-backed one from the adapters), so it is unit-tested without torch.
"""

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from tubitak_rag.config import AppConfig, ConfigError

logger = logging.getLogger(__name__)

DEVICE_ENV = "TUBITAK_RAG_DEVICE"
CPU_CANDIDATES_ENV = "RERANKER_CPU_CANDIDATES"
DEVICES = ("auto", "cpu", "cuda")
_UNBOUNDED = {"none", "null", "yok", "sinirsiz"}

CudaProbe = Callable[[], bool]


def device_setting(configured: str, environ: Mapping[str, str] | None = None) -> str:
    """The device setting in effect: ``TUBITAK_RAG_DEVICE`` when set, else ``configured``."""
    env = os.environ if environ is None else environ
    value = env.get(DEVICE_ENV, "").strip().lower()
    return value or configured


def choose_device(setting: str, cuda_available: CudaProbe) -> str:
    """Map ``auto|cpu|cuda`` to a concrete ``cpu`` or ``cuda``; probes the GPU lazily.

    Raises:
        ConfigError: for an unknown setting, or ``cuda`` on a machine without a usable GPU.
    """
    if setting not in DEVICES:
        raise ConfigError(
            f"Geçersiz cihaz: {setting!r} (izin verilenler: {', '.join(DEVICES)}; "
            f"{DEVICE_ENV} ortam değişkenini ya da yapılandırmadaki device alanını düzeltin)."
        )
    if setting == "cpu":
        return "cpu"
    available = cuda_available()
    if setting == "cuda" and not available:
        raise ConfigError(
            "device: cuda istendi ama kullanılabilir bir GPU yok (PyTorch CUDA göremiyor). "
            "GPU sürücüsünü / Docker GPU profilini kontrol edin ya da auto/cpu seçin."
        )
    return "cuda" if available else "cpu"


def cpu_candidates(configured: int | None, environ: Mapping[str, str] | None = None) -> int | None:
    """``reranker.cpu_max_candidates`` after the ``RERANKER_CPU_CANDIDATES`` override.

    Raises:
        ConfigError: when the variable is neither a non-negative integer nor ``none``.
    """
    env = os.environ if environ is None else environ
    raw = env.get(CPU_CANDIDATES_ENV, "").strip().lower()
    if not raw:
        return configured
    if raw in _UNBOUNDED:
        return None
    if raw.isdigit():
        return int(raw)
    raise ConfigError(
        f"{CPU_CANDIDATES_ENV}={raw!r} geçersiz: 0 veya pozitif bir tam sayı ya da 'none' olmalı."
    )


@dataclass(frozen=True)
class RerankerPlan:
    """How the reranker runs: on or off, on which device, over how many candidates."""

    enabled: bool
    device: str
    k_candidates: int
    note: str | None = None  # Turkish explanation when the config was adjusted


def plan_reranker(
    *, enabled: bool, device: str, k_candidates: int, cpu_max_candidates: int | None
) -> RerankerPlan:
    """Decide the reranker setup for a concrete ``device`` (``cpu`` or ``cuda``)."""
    if not enabled or device != "cpu" or cpu_max_candidates is None:
        return RerankerPlan(enabled, device, k_candidates)
    if cpu_max_candidates == 0:
        return RerankerPlan(
            False,
            device,
            k_candidates,
            "Reranker CPU'da kapatıldı (cpu_max_candidates=0); sonuçlar hibrit arama sırasıyla.",
        )
    if k_candidates <= cpu_max_candidates:
        return RerankerPlan(True, device, k_candidates)
    return RerankerPlan(
        True,
        device,
        cpu_max_candidates,
        f"Reranker CPU'da çalışıyor: aday sayısı {k_candidates} → {cpu_max_candidates} "
        f"(~0,7 s/aday; NFR-PERF1 < 5 s). GPU'da sınır uygulanmaz.",
    )


def apply_device_policy(
    cfg: AppConfig, *, cuda_available: CudaProbe, environ: Mapping[str, str] | None = None
) -> AppConfig:
    """Return ``cfg`` with concrete model devices and the CPU reranker plan applied.

    Fake providers are left untouched (no GPU probe). The result is idempotent: applying
    the policy again changes nothing.
    """
    embedder, reranker, retriever = cfg.embedder, cfg.reranker, cfg.retriever
    if embedder.provider == "sentence_transformers":
        device = choose_device(device_setting(embedder.device, environ), cuda_available)
        embedder = embedder.model_copy(update={"device": device})
    if reranker.enabled and reranker.provider == "cross_encoder":
        device = choose_device(device_setting(reranker.device, environ), cuda_available)
        plan = plan_reranker(
            enabled=True,
            device=device,
            k_candidates=retriever.k_candidates,
            cpu_max_candidates=cpu_candidates(reranker.cpu_max_candidates, environ),
        )
        if plan.note:
            logger.warning(plan.note)
        reranker = reranker.model_copy(update={"enabled": plan.enabled, "device": plan.device})
        retriever = retriever.model_copy(update={"k_candidates": plan.k_candidates})
    return cfg.model_copy(
        update={"embedder": embedder, "reranker": reranker, "retriever": retriever}
    )
