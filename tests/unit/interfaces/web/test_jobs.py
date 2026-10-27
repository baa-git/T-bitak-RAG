"""Background jobs of the web interface: in-memory registry and CLI subprocesses."""

import sys
import threading
from pathlib import Path

import pytest

from tubitak_rag.interfaces.web.jobs import CommandError, JobRegistry, cli_command, run_command


def test_successful_job_ends_as_tamam() -> None:
    jobs = JobRegistry()
    seen: list[str] = []
    job_id = jobs.start("ingest", seen.append)
    status = jobs.wait(job_id, timeout=5)
    assert status is not None and status.durum == "tamam" and status.hata is None
    assert seen == [job_id]
    assert job_id.startswith("ingest-") and status.tur == "ingest"


def test_failing_job_records_the_error_in_turkish_status() -> None:
    jobs = JobRegistry()

    def boom(_job_id: str) -> None:
        raise RuntimeError("CUDA OOM")

    status = jobs.wait(jobs.start("run", boom), timeout=5)
    assert status is not None and status.durum == "hata" and status.hata == "CUDA OOM"
    assert status.bitis is not None


def test_busy_while_a_job_runs() -> None:
    jobs = JobRegistry()
    release = threading.Event()
    job_id = jobs.start("run", lambda _id: release.wait(5) and None)
    assert jobs.busy()
    status = jobs.status(job_id)
    assert status is not None and status.durum == "suruyor"
    release.set()
    jobs.wait(job_id, timeout=5)
    assert not jobs.busy()


def test_job_ids_are_unique_and_unknown_id_has_no_status() -> None:
    jobs = JobRegistry()
    first, second = jobs.start("ingest", lambda _id: None), jobs.start("ingest", lambda _id: None)
    assert first != second
    assert jobs.status("yok") is None and jobs.wait("yok") is None


def test_cli_command_runs_the_rag_cli_module_with_this_python() -> None:
    assert cli_command("ingest", "--full") == [
        sys.executable,
        "-m",
        "tubitak_rag.interfaces.cli",
        "ingest",
        "--full",
    ]


def test_run_command_writes_output_to_the_log(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "is.log"
    run_command([sys.executable, "-c", "print('ingest bitti ğüş')"], log)
    assert "ingest bitti ğüş" in log.read_text(encoding="utf-8")


def test_run_command_failure_raises_with_exit_code_and_log_tail(tmp_path: Path) -> None:
    log = tmp_path / "is.log"
    code = "import sys; print('GPU kilidi dolu'); sys.exit(3)"
    with pytest.raises(CommandError, match="3") as caught:
        run_command([sys.executable, "-c", code], log)
    assert "GPU kilidi dolu" in str(caught.value)
