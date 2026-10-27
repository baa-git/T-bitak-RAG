"""Exclusive GPU lock file ("GPU kullanım kuralı").

Only one job may use the GPU at a time (real embedding, reranker, local LLM runs). The job
creates ``artifacts/.gpu.lock`` atomically before it starts and deletes it when it ends;
if the file already exists another job holds the GPU and this one must not start.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from tubitak_rag.config import AppConfig

GPU_LOCK_FILE = ".gpu.lock"  # inside ``paths.artifacts_dir``


class GpuLockError(RuntimeError):
    """The GPU lock is held by another job (message in Turkish)."""


def uses_gpu(cfg: AppConfig, *, answering: bool) -> bool:
    """Whether a real model runs: the embedder, and when answering also the LLM or reranker.

    ``answering=False`` (ingest) only uses the embedder; all-fake configs never need the lock.
    """
    if cfg.embedder.provider != "fake":
        return True
    if not answering:
        return False
    reranker = cfg.reranker.enabled and cfg.reranker.provider != "fake"
    return cfg.llm.provider != "fake" or reranker


def lock_holder(path: Path) -> str | None:
    """Who holds the lock at ``path`` (its lines joined), or None when the GPU is free."""
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "bilinmiyor"
    return " · ".join(line.strip() for line in lines if line.strip()) or "bilinmiyor"


def busy_message(path: Path, holder: str) -> str:
    """The Turkish "GPU is busy" message shown by the CLI and the web interface."""
    return (
        f"GPU kilidi dolu: {path} ({holder}). Başka bir GPU işi sürüyor; o iş bitince "
        "tekrar deneyin. İş çökmüş ve kilit sahipsiz kalmışsa dosyayı elle silin."
    )


@contextmanager
def gpu_lock(path: Path, owner: str) -> Iterator[None]:
    """Hold the GPU lock at ``path`` for the duration of the ``with`` block.

    The file is created with exclusive mode (``"x"``), which fails atomically when it
    already exists, and holds the owner, the process id and the start time.

    Raises:
        GpuLockError: if the lock file already exists; it is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{owner}\npid={os.getpid()}\n{stamp}\n")
    except FileExistsError:
        raise GpuLockError(busy_message(path, lock_holder(path) or "bilinmiyor")) from None
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
