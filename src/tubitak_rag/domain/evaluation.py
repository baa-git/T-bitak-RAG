"""Evaluation models: test-set rows, per-question results and run-level metrics."""

from typing import Any, ClassVar, Literal

from tubitak_rag.domain.base import DomainModel
from tubitak_rag.domain.enums import QuestionType


class EvidenceQuote(DomainModel):
    """A short verbatim quote that proves the expected answer."""

    doc_id: str
    alinti: str


class TestQuestion(DomainModel):
    """One row of ``eval/testset_v1.jsonl``."""

    __test__: ClassVar[bool] = False  # not a pytest test class despite the name

    id: str
    soru: str
    tur: QuestionType
    zorluk: Literal["kolay", "orta", "zor"]
    beklenen_cevap: str
    kanit_alintilari: list[EvidenceQuote]
    beklenen_belgeler: list[str]
    beklenen_sayilar: list[str] = []
    program_kodu: list[str] = []
    dogrulandi: bool = False
    review_notes: str = ""
    uretim: Literal["llm", "manuel"] = "manuel"


class QuestionResult(DomainModel):
    """One line of ``per_question.jsonl`` written by the experiment runner."""

    id: str
    tur: QuestionType
    soru: str
    retrieved: list[dict[str, Any]]  # {chunk_id, doc_id, rank, score, scores, doc_title, …}
    evidence_hits: list[int | None]  # first 1-based hit rank per evidence quote, or None
    first_hit_rank: int | None
    doc_hit_at_5: bool
    doc_hit_at_10: bool
    gate_key: str | None
    gate_score: float | None
    answer: dict[str, Any]  # {durum, cevap, atiflar: [chunk_id], red_nedeni, …}
    judge: dict[str, Any]  # {dogru, sadik, atif_dogru, gerekce}
    rule_check: dict[str, Any]  # {sayilar_bulundu, atif_kesinligi, programlara_ayrildi}
    latency_ms: dict[str, float]
    llm_calls: int


class RunMetrics(DomainModel):
    """Aggregated metrics of one experiment run (``metrics.json``)."""

    run_id: str
    experiment: str
    started_at: str
    finished_at: str
    wall_clock_s: float
    judge_model: str
    llm_model: str
    embedder: str
    chunker: str
    retriever: str
    reranker: str | None
    counts: dict[str, int]  # {questions, chunks, docs, judge_errors, …}
    retrieval: dict[str, float]  # recall_at_5, recall_at_10, mrr, doc_hit_at_5, doc_hit_at_10
    generation: dict[str, float]  # answer_accuracy, faithfulness, citation_precision, …
    latency_ms: dict[str, float]  # retrieval_p50, retrieval_p95, rerank_p50, llm_p50, total_*
    by_type: dict[str, dict[str, float]]
    stage_times_s: dict[str, float]  # ingest (optional), questions, judge
    refusals: dict[str, int] = {}  # refusal reason code -> count (e.g. "dayanaksiz_sayi")
