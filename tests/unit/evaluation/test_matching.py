"""Tests for evidence quote matching (FR-E5)."""

from tests.builders import make_chunk, make_scored
from tests.eval_builders import QUOTE, make_question
from tubitak_rag.domain.evaluation import EvidenceQuote
from tubitak_rag.evaluation.matching import (
    doc_hit,
    evidence_hits,
    first_hit_rank,
    quote_in_chunk,
)


def test_quote_matches_across_whitespace_and_case() -> None:
    text = "MADDE 7 – (1)  Destek ORANI her\ndönem için en fazla %75 olarak belirlenmiştir."
    assert quote_in_chunk("destek oranı her dönem için en fazla %75", text)
    assert not quote_in_chunk("destek oranı her dönem için en fazla %60", text)


def test_quote_split_across_two_chunks_is_miss() -> None:
    question = make_question()
    first = make_scored(make_chunk("c1", text="Destek oranı her dönem için"), 0.9)
    second = make_scored(make_chunk("c2", text="en fazla %75 olarak belirlenmiştir."), 0.8)
    assert evidence_hits(question, [first, second]) == [None]
    whole = make_scored(make_chunk("c3", text=f"Giriş. {QUOTE}. Son."), 0.7)
    assert evidence_hits(question, [first, second, whole]) == [3]


def test_hits_require_same_doc() -> None:
    question = make_question()
    other_doc = make_scored(make_chunk("x1", doc_id="mevzuat-099", text=QUOTE), 0.9)
    right_doc = make_scored(make_chunk("c1", text=QUOTE), 0.8)
    assert evidence_hits(question, [other_doc]) == [None]
    assert evidence_hits(question, [other_doc, right_doc]) == [2]


def test_each_quote_gets_its_own_rank() -> None:
    question = make_question(
        kanit_alintilari=[
            EvidenceQuote(doc_id="mevzuat-031", alinti=QUOTE),
            EvidenceQuote(doc_id="mevzuat-031", alinti="proje süresi en fazla 18 aydır"),
        ]
    )
    retrieved = [
        make_scored(make_chunk("c1", text="Proje süresi en fazla 18 aydır."), 0.9),
        make_scored(make_chunk("c2", text=QUOTE), 0.8),
    ]
    assert evidence_hits(question, retrieved) == [2, 1]


def test_first_hit_rank() -> None:
    assert first_hit_rank([3, None, 2]) == 2
    assert first_hit_rank([None, None]) is None
    assert first_hit_rank([]) is None


def test_doc_hit_k() -> None:
    question = make_question()
    others = [make_scored(make_chunk(f"o{i}", doc_id=f"diger-{i}"), 0.5) for i in range(5)]
    expected = make_scored(make_chunk("c1"), 0.4)
    retrieved = [*others, expected]
    assert not doc_hit(question, retrieved, 5)
    assert doc_hit(question, retrieved, 6)
