"""Builders for evaluation tests: test questions, answers and per-question result rows."""

import json
from pathlib import Path
from typing import Any

from tubitak_rag.domain.answer import Answer, Citation
from tubitak_rag.domain.enums import AnswerStatus, QuestionType
from tubitak_rag.domain.evaluation import EvidenceQuote, QuestionResult, RunMetrics, TestQuestion

QUOTE = "Destek oranı her dönem için en fazla %75 olarak belirlenmiştir"


def make_question(qid: str = "q001", **overrides: Any) -> TestQuestion:
    """An answerable single-document question whose quote is in ``make_chunk``'s text."""
    fields: dict[str, Any] = {
        "id": qid,
        "soru": f"Soru {qid}: 1507 programında destek oranı en fazla yüzde kaçtır?",
        "tur": QuestionType.TEK_BELGE,
        "zorluk": "kolay",
        "beklenen_cevap": "En fazla %75.",
        "kanit_alintilari": [EvidenceQuote(doc_id="mevzuat-031", alinti=QUOTE)],
        "beklenen_belgeler": ["mevzuat-031"],
        "beklenen_sayilar": ["%75"],
    }
    fields.update(overrides)
    return TestQuestion(**fields)


def make_unanswerable(qid: str, tur: QuestionType = QuestionType.CEVAPSIZ) -> TestQuestion:
    """A ``cevapsiz`` / ``kapsam_disi`` question (no quotes, no expected documents)."""
    expected = "bilgi bulunamadı" if tur == QuestionType.CEVAPSIZ else "kapsam dışı"
    return make_question(
        qid,
        tur=tur,
        beklenen_cevap=expected,
        kanit_alintilari=[],
        beklenen_belgeler=[],
        beklenen_sayilar=[],
    )


def make_answer(
    soru: str = "soru",
    durum: AnswerStatus = AnswerStatus.CEVAPLANDI,
    cevap: str = "Destek oranı en fazla %75'tir [1].",
    atiflar: list[Citation] | None = None,
    **overrides: Any,
) -> Answer:
    """A pipeline answer; answered answers cite ``make_chunk``'s chunk by default."""
    answered = durum == AnswerStatus.CEVAPLANDI
    if atiflar is None:
        atiflar = (
            [
                Citation(
                    no=1,
                    chunk_id="mevzuat-031:0001:00",
                    doc_id="mevzuat-031",
                    doc_title="1507 Uygulama Esasları",
                    konum="Madde 7",
                    kaynak_url="https://tubitak.gov.tr/x.pdf",
                )
            ]
            if answered
            else []
        )
    fields: dict[str, Any] = {
        "soru": soru,
        "durum": durum,
        "cevaplanabilir": answered,
        "cevap": cevap
        if answered
        else "Elimdeki mevzuat kaynaklarında bu soruya dayanak bulamadım.",
        "atiflar": atiflar,
        "kaynaklar": [],
        "uyari": "",
        "gecikme_ms": {
            "scope": 1.0,
            "retrieval": 10.0,
            "rerank": 0.0,
            "llm": 100.0,
            "total": 111.0,
        },
        "llm_cagri_sayisi": 1 if answered else 0,
        "gate_key": "dense",
        "gate_score": 0.7,
    }
    fields.update(overrides)
    return Answer(**fields)


def make_result(qid: str = "q001", **overrides: Any) -> QuestionResult:
    """A per-question row: answered, evidence at rank 1, judged correct and faithful."""
    fields: dict[str, Any] = {
        "id": qid,
        "tur": QuestionType.TEK_BELGE,
        "soru": f"Soru {qid}",
        "retrieved": [
            {
                "chunk_id": "mevzuat-031:0001:00",
                "doc_id": "mevzuat-031",
                "rank": 1,
                "score": 0.7,
                "scores": {"dense": 0.7},
                "doc_title": "1507 Uygulama Esasları",
                "konum": "Madde 7",
                "metin": QUOTE,
            }
        ],
        "evidence_hits": [1],
        "first_hit_rank": 1,
        "doc_hit_at_5": True,
        "doc_hit_at_10": True,
        "gate_key": "dense",
        "gate_score": 0.7,
        "answer": {
            "durum": "cevaplandi",
            "cevap": "En fazla %75 [1].",
            "atiflar": ["mevzuat-031:0001:00"],
            "red_nedeni": None,
            "sayi_ayiklandi": False,
        },
        "judge": {"dogru": True, "sadik": True, "atif_dogru": True, "gerekce": "Doğru."},
        "rule_check": {"sayilar_bulundu": True, "atif_kesinligi": 1.0, "programlara_ayrildi": None},
        "latency_ms": {"retrieval": 10.0, "rerank": 0.0, "llm": 100.0, "total": 111.0},
        "llm_calls": 1,
    }
    fields.update(overrides)
    return QuestionResult(**fields)


def refused(
    qid: str, tur: QuestionType, reason: str, durum: str = "bilgi_bulunamadi", **overrides: Any
) -> QuestionResult:
    """A refused row (no answer, no judge) with the given refusal reason."""
    fields: dict[str, Any] = {
        "tur": tur,
        "answer": {
            "durum": durum,
            "cevap": "Elimdeki mevzuat kaynaklarında bu soruya dayanak bulamadım.",
            "atiflar": [],
            "red_nedeni": reason,
            "sayi_ayiklandi": False,
        },
        "judge": {"dogru": None, "sadik": None, "atif_dogru": None, "gerekce": ""},
        "rule_check": {
            "sayilar_bulundu": None,
            "atif_kesinligi": None,
            "programlara_ayrildi": None,
        },
        "llm_calls": 0,
    }
    fields.update(overrides)
    return make_result(qid, **fields)


def make_metrics(experiment: str = "v1", **overrides: Any) -> RunMetrics:
    """Run metrics with every headline key filled."""
    fields: dict[str, Any] = {
        "run_id": f"{experiment}_20261016-120000",
        "experiment": experiment,
        "started_at": "2026-10-16T12:00:00",
        "finished_at": "2026-10-16T12:30:00",
        "wall_clock_s": 1805.0,
        "judge_model": "qwen3:4b",
        "llm_model": "qwen3:4b",
        "embedder": "google/embeddinggemma-2",
        "chunker": "structural_ctx",
        "retriever": "dense",
        "reranker": None,
        "counts": {"questions": 8, "docs": 120, "chunks": 4000},
        "retrieval": {
            "recall_at_5": 0.6,
            "recall_at_10": 0.7,
            "mrr": 0.5,
            "doc_hit_at_5": 0.8,
            "doc_hit_at_10": 0.9,
        },
        "generation": {
            "answer_accuracy": 0.5,
            "faithfulness": 0.8,
            "citation_precision": 0.9,
            "correct_rejection_rate": 0.7,
            "false_rejection_rate": 0.2,
        },
        "latency_ms": {"retrieval_p50": 20.0, "total_p50": 1500.0},
        "by_type": {"tek_belge": {"recall_at_5": 0.6, "answer_accuracy": 0.5}},
        "stage_times_s": {"questions": 1800.0, "judge": 5.0},
    }
    fields.update(overrides)
    return RunMetrics(**fields)


# Manual review (Claude) of ``tests/fixtures/run_example``: accuracy 1/5, faithfulness 2/2
# (q005 undecided), correct rejection 2/3.
REVIEW_ANSWERS: list[dict[str, Any]] = [
    {"id": "q001", "dogru": True, "sadik": True, "not": "Oran doğru."},
    {"id": "q002", "dogru": False, "sadik": True, "not": "Süre eksik verilmiş."},
    {"id": "q005", "dogru": False, "sadik": None, "not": "Kaynak kesik."},
]
REVIEW_OTHERS: list[dict[str, Any]] = [
    {"id": "q010", "karar": "gereksiz_ret", "hata_yeri": "retrieval", "not": "Kanıt yok."},
    {"id": "q020", "karar": "gereksiz_ret", "hata_yeri": "uretim", "not": "Sayı yazıyla."},
    {"id": "q030", "karar": "dogru_ret", "hata_yeri": None, "not": ""},
    {"id": "q031", "karar": "uydurma", "hata_yeri": None, "not": "Noter kuralı kaynakta yok."},
    {"id": "q040", "karar": "dogru_ret", "hata_yeri": None, "not": ""},
]


def write_review(
    judge_dir: Path,
    experiment: str = "v1",
    answers: list[dict[str, Any]] | None = None,
    others: list[dict[str, Any]] | None = None,
) -> None:
    """Write ``<experiment>_claude.json`` and ``<experiment>_rest_claude.json``."""
    judge_dir.mkdir(parents=True, exist_ok=True)
    for suffix, rows in (
        ("_claude.json", REVIEW_ANSWERS if answers is None else answers),
        ("_rest_claude.json", REVIEW_OTHERS if others is None else others),
    ):
        (judge_dir / f"{experiment}{suffix}").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
