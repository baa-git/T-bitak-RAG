"""Background jobs started from the web interface (ingest, experiment run).

A request must answer in milliseconds, but an ingest or a run takes minutes to hours. The
endpoint therefore only *starts* the work in a background thread and answers ``202`` with a
job id; the registry keeps the job's state in memory. GPU jobs run the ``rag`` CLI in a
separate process: its models are freed when the process exits, a crash cannot take the web
server down, and the CLI takes the GPU lock itself exactly as it does from the terminal.
"""

import itertools
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

JobState = Literal["suruyor", "tamam", "hata"]
_LOG_TAIL_LINES = 8


class JobStatus(BaseModel):
    """State of one background job."""

    job_id: str
    tur: str  # "ingest" | "run"
    durum: JobState
    baslangic: str
    bitis: str | None = None
    hata: str | None = None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class JobRegistry:
    """Runs jobs in daemon threads and remembers their state (lost on restart)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._jobs: dict[str, JobStatus] = {}
        self._threads: dict[str, threading.Thread] = {}

    def start(self, kind: str, work: Callable[[str], None]) -> str:
        """Run ``work(job_id)`` in the background and return the new job id at once."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        with self._lock:
            job_id = f"{kind}-{stamp}-{next(self._counter)}"
            self._jobs[job_id] = JobStatus(
                job_id=job_id, tur=kind, durum="suruyor", baslangic=_now()
            )
            thread = threading.Thread(
                target=self._run, args=(job_id, work), name=job_id, daemon=True
            )
            self._threads[job_id] = thread
        thread.start()
        return job_id

    def _run(self, job_id: str, work: Callable[[str], None]) -> None:
        try:
            work(job_id)
        except Exception as exc:  # the job's error is its result, never the server's
            self._finish(job_id, "hata", str(exc))
        else:
            self._finish(job_id, "tamam", None)

    def _finish(self, job_id: str, state: JobState, error: str | None) -> None:
        with self._lock:
            self._jobs[job_id] = self._jobs[job_id].model_copy(
                update={"durum": state, "bitis": _now(), "hata": error}
            )

    def status(self, job_id: str) -> JobStatus | None:
        """Current state of a job, or None for an unknown id."""
        with self._lock:
            return self._jobs.get(job_id)

    def busy(self) -> bool:
        """Whether any job is still running."""
        with self._lock:
            return any(job.durum == "suruyor" for job in self._jobs.values())

    def wait(self, job_id: str, timeout: float | None = None) -> JobStatus | None:
        """Block until the job ends (or ``timeout``); returns its state."""
        thread = self._threads.get(job_id)
        if thread is None:
            return None
        thread.join(timeout)
        return self.status(job_id)


class CommandError(RuntimeError):
    """A CLI subprocess ended with a non-zero exit code (message in Turkish)."""


def cli_command(*args: str) -> list[str]:
    """The ``rag`` CLI run with this interpreter (works without the console script)."""
    return [sys.executable, "-m", "tubitak_rag.interfaces.cli", *args]


def run_command(argv: Sequence[str], log_path: Path) -> None:
    """Run ``argv`` to completion, writing stdout and stderr to ``log_path``.

    Raises:
        CommandError: for a non-zero exit code, with the last lines of the log.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            list(argv), stdout=log, stderr=subprocess.STDOUT, env=env, check=False
        )
    if result.returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-_LOG_TAIL_LINES:])
        raise CommandError(
            f"Komut {result.returncode} çıkış koduyla bitti (günlük: {log_path}).\n{tail}"
        )
