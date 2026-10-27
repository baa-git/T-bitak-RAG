"""Request and response models shared by the web pages and the REST API .

Field names are Turkish because the API/UI contract fixes them (NFR-LANG1). The
request models forbid unknown fields, so a typo in a client becomes a 422, not a silent
default (FR-A3).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, computed_field

from tubitak_rag.domain.answer import Answer
from tubitak_rag.domain.enums import DocType
from tubitak_rag.domain.evaluation import QuestionResult, RunMetrics

MAX_QUESTION_CHARS = 1000

DocumentState = Literal["indeksli", "taranmis", "eksik", "guncel_degil"]
RunState = Literal["tamam", "suruyor"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class DocumentStatus(_Frozen):
    """One corpus document and its index state."""

    doc_id: str
    baslik: str
    program_kodu: str | None
    doc_turu: str
    kaynak_url: str
    surum_tarihi: str | None  # ISO date "2026-09-25"
    dil: str
    durum: DocumentState
    chunk_sayisi: int


class DocumentsResponse(_Frozen):
    """The corpus table plus the collection it is indexed in."""

    belgeler: list[DocumentStatus]
    koleksiyon: str
    toplam_chunk: int


class JobStarted(_Frozen):
    """A background job (ingest or experiment run) was accepted."""

    job_id: str
    durum: Literal["basladi"] = "basladi"


class RunSummary(_Frozen):
    """One row of the experiments table; ``metrikler`` holds the headline numbers."""

    run_id: str
    deney: str
    tarih: str  # ISO timestamp of ``started_at``
    durum: RunState
    metrikler: dict[str, float]


class SourceContext(_Frozen):
    """Extra location data for a source, read from the stored chunk.

    ``SourceView`` (domain contract) only carries ``konum``; the source panel also shows the
    section hierarchy, page and version date, which the backend looks up by ``chunk_id``.
    """

    hiyerarsi: str | None = None  # "İkinci Bölüm – Destek Kapsamı"
    madde_baslik: str | None = None  # "Madde 7 – Destek oranı ve sınırı"
    sayfa: int | None = None
    surum_tarihi: str | None = None
    program_kodu: str | None = None
    doc_id: str | None = None  # links the source to the document preview …
    bolum_id: str | None = None  # … anchored at this parsed section ("#bolum-<id>")


class AskResult(_Frozen):
    """What the query page renders: the pipeline answer plus per-source context."""

    answer: Answer
    baglam: dict[str, SourceContext] = {}  # chunk_id -> context
    yapilandirma: str | None = None  # experiment config that answered ("v3")


class MethodLine(_Frozen):
    """One labelled method or model: ``("Arama", "hibrit (BM25 + dense, RRF k=60)")``."""

    etiket: str
    deger: str


class PipelineOption(_Frozen):
    """One selectable pipeline: an experiment config from ``configs/experiments``."""

    ad: str  # experiment name, "v3"
    etiket: str  # what the selector shows: the plain name, "v3"
    aciklama: str  # the config's description (not rendered in the UI)
    varsayilan: bool = False
    yontemler: list[MethodLine] = []  # derived from the config (about.describe_methods)
    hazir: bool = True  # False: its collection / BM25 index is missing (shown disabled)
    durum_notu: str | None = None  # why it is not selectable, "indeks yok"


class CorpusGroup(_Frozen):
    """One source of the corpus on the Hakkında page, counted from the manifests."""

    kaynak_grubu: str  # manifest group, "tubitak_mevzuat"
    ad: str
    kaynak: str  # where the documents were downloaded from
    kaynak_url: str | None
    belge_sayisi: int
    ozet: str  # "105 sayfa + 115 dosya (114 PDF, 1 Excel)"
    ayrinti: list[str] = []  # e.g. the law numbers and titles


class AboutInfo(_Frozen):
    """Hakkında page: corpus, counts and the components of the pipelines."""

    gruplar: list[CorpusGroup]
    belge_sayisi: int
    bolum_sayisi: int | None  # parsed sections (parse cache); None if unknown
    parca_sayisi: int | None  # chunks in the default collection; None if unknown
    koleksiyon: str
    bilesenler: list[MethodLine]
    guardrailler: list[MethodLine]
    yapilandirmalar: list[PipelineOption]


class RunsResponse(_Frozen):
    """``GET /api/runs``: every run, newest first."""

    kosular: list[RunSummary]


class RunDetail(_Frozen):
    """``GET /api/runs/{run_id}``: aggregated metrics and the per-question results."""

    metrics: RunMetrics
    sorular: list[QuestionResult]


class ReportResponse(_Frozen):
    """``GET /api/runs/{run_id}/report``: the Markdown report."""

    markdown: str


class HealthStatus(_Frozen):
    """``GET /api/health``: can the services the pipeline needs be reached?"""

    durum: Literal["ok", "sorunlu"]
    qdrant: bool
    ollama: bool
    koleksiyon: str
    surum: str


class SectionView(_Frozen):
    """One parsed section in the document preview."""

    bolum_id: str  # Section.section_id, also the HTML anchor "bolum-<id>"
    baslik: str | None
    hiyerarsi: list[str]
    madde_no: str | None
    sayfa: int | None
    metin: str
    mulga: bool
    degisiklik_notlari: list[str]


class DocumentPreview(_Frozen):
    """``/belgeler/{doc_id}``: identity, counts and the parsed sections of one document."""

    belge: DocumentStatus
    kaynak_sayfa_url: str | None
    ayristirildi: bool  # False: no parsed text yet (missing or never ingested)
    bolumler: list[SectionView]
    chunk_sayilari: dict[str, int]  # collection -> chunks of this document

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bolum_sayisi(self) -> int:
        return len(self.bolumler)


# --- requests ------------------------------------------------------------------------------

Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
]
OptionalText = Annotated[str | None, StringConstraints(strip_whitespace=True, max_length=100)]


class _Request(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AskRequest(_Request):
    """``POST /api/ask``."""

    soru: Question
    program_kodu: OptionalText = None
    doc_turu: DocType | None = None
    config: OptionalText = Field(
        default=None, description="Deney yapılandırması (ör. v2); boşsa varsayılan pipeline."
    )


class IngestRequest(_Request):
    """``POST /api/ingest``."""

    full: bool = False


class RunRequest(_Request):
    """``POST /api/runs``: an experiment config by name ("v3") or path."""

    config: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    limit: Annotated[int | None, Field(ge=1)] = None
