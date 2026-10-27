"""The single seam between the web layer (pages + REST API) and the RAG core.

Pages and API endpoints only talk to a ``WebBackend``. ``FakeBackend`` serves demo data
(tests, design work); ``PipelineBackend`` wires the real pipeline, vector store and run
artifacts. Swapping one for the other changes no route and no template.

Errors a backend may raise, and the HTTP status the API maps them to:

* ``UnknownPipelineError`` — no such experiment config (422)
* ``GpuLockError`` (``interfaces/gpu_lock.py``) — another job holds the GPU (409)
* ``ServiceUnavailableError`` — Qdrant/Ollama/index not reachable or not built (503)
"""

from typing import Protocol, runtime_checkable

from tubitak_rag.domain.evaluation import RunMetrics
from tubitak_rag.interfaces.web.schemas import (
    AboutInfo,
    AskResult,
    DocumentPreview,
    DocumentsResponse,
    HealthStatus,
    JobStarted,
    PipelineOption,
    RunDetail,
    RunSummary,
)


class UnknownPipelineError(ValueError):
    """The requested experiment config is not one of the selectable pipelines."""


class ServiceUnavailableError(RuntimeError):
    """A service the request needs is down or not prepared (message in Turkish)."""


@runtime_checkable
class WebBackend(Protocol):
    """Everything the pages and the REST API need from the system."""

    is_demo: bool  # True -> pages show a "demo data" notice

    def pipelines(self) -> list[PipelineOption]:
        """Selectable pipelines (experiment configs); exactly one is the default."""
        ...

    def ask(
        self,
        soru: str,
        program_kodu: str | None,
        doc_turu: str | None,
        yapilandirma: str | None = None,
    ) -> AskResult:
        """Answer one question with the named pipeline (None: the default one)."""
        ...

    def disclaimer(self) -> str:
        """The disclaimer every answer carries (shown once on the query page)."""
        ...

    def about(self) -> AboutInfo:
        """Corpus counts and pipeline components for the Hakkında page; builds no pipeline."""
        ...

    def documents(self) -> DocumentsResponse:
        """List corpus documents with their index state."""
        ...

    def document(self, doc_id: str) -> DocumentPreview | None:
        """Preview of one document (parsed sections, counts), or None if unknown."""
        ...

    def start_ingest(self, full: bool = False) -> JobStarted:
        """Start an incremental (or full) ingest in the background."""
        ...

    def start_run(self, config: str, limit: int | None = None) -> JobStarted:
        """Start an experiment run of the named config in the background."""
        ...

    def runs(self) -> list[RunSummary]:
        """Experiment runs, newest first."""
        ...

    def run_metrics(self, run_id: str) -> RunMetrics | None:
        """``metrics.json`` of one run, or None if the run does not exist."""
        ...

    def run_detail(self, run_id: str) -> RunDetail | None:
        """Metrics plus per-question results of a finished run, or None."""
        ...

    def run_report(self, run_id: str) -> str | None:
        """``report.md`` of one run, or None if it does not exist."""
        ...

    def summary_report(self) -> str | None:
        """``runs/ozet.md`` (all runs side by side), or None if not generated yet."""
        ...

    def health(self) -> HealthStatus:
        """Reachability of the services behind the pipeline."""
        ...
