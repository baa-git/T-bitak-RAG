"""Answer models returned to CLI/API users (field names fixed in Turkish by the spec)."""

from tubitak_rag.domain.base import DomainModel
from tubitak_rag.domain.enums import AnswerStatus


class Citation(DomainModel):
    """A numbered reference used inside the answer text as ``[no]``."""

    no: int
    chunk_id: str
    doc_id: str
    doc_title: str
    konum: str  # "Madde 7 (2)" | "Sayfa 6" | "SSS: …"
    kaynak_url: str


class SourceView(DomainModel):
    """A source shown in the numbered source panel."""

    no: int
    chunk_id: str
    doc_title: str
    konum: str
    kaynak_url: str
    text: str
    score: float


class Answer(DomainModel):
    """Final pipeline output for one question."""

    soru: str
    durum: AnswerStatus
    cevaplanabilir: bool
    cevap: str
    atiflar: list[Citation]
    kaynaklar: list[SourceView]
    uyari: str
    gecikme_ms: dict[str, float]  # {"scope", "retrieval", "rerank", "llm", "total"}
    llm_cagri_sayisi: int
    gate_key: str | None
    gate_score: float | None
    atif_onarimi: list[int] = []  # source numbers added by the number guardrail's citation repair
