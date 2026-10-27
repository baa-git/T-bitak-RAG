"""Tests for the number-grounding guardrail: every number in an answer must be in the sources."""

import pytest

from tests.builders import make_chunk
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.generation.schema import LlmOutput, cited_numbers
from tubitak_rag.guardrails.numbers import (
    STRIPPED_NOTE,
    UNGROUNDED_NUMBER,
    NumberVerdict,
    check_numbers,
    extract_numbers,
    extract_quantities,
    grounding_tokens,
)

RATE = make_chunk(
    "oran",
    doc_title="TÜBİTAK 1507 Uygulama Esasları",
    madde_no="7",
    hiyerarsi="İkinci Bölüm › Madde 7 – Destek oranı ve sınırı",
    surum_tarihi="2026-09-25",
    text="Bu programda uygulanacak destek oranı her dönem için en fazla %75 olarak belirlenmiştir.",
)
BUDGET = make_chunk(
    "butce",
    doc_title="1507 Program Sayfası",
    madde_no=None,
    hiyerarsi="Sıkça Sorulan Sorular",
    surum_tarihi=None,
    text="Projenin toplam bütçesi 3.500.000 TL ile, ön ödeme 250.000 TL ile sınırlıdır.",
)
QUESTION = "1507 programında destek oranı nedir?"


def _answer(cevap: str, atiflar: list[int] | None = None, *, ok: bool = True) -> LlmOutput:
    return LlmOutput(cevaplanabilir=ok, cevap=cevap, atiflar=[1] if atiflar is None else atiflar)


def _check(
    cevap: str,
    cited: list[Chunk] | None = None,
    *,
    question: str = QUESTION,
    mode: str = "strip",
    atiflar: list[int] | None = None,
) -> NumberVerdict:
    return check_numbers(
        _answer(cevap, atiflar), question, cited if cited is not None else [RATE], mode=mode
    )


# --- the case that motivated the guardrail ----------------------------------------------


def test_fabricated_duration_sentence_is_stripped() -> None:
    verdict = _check("Destek oranı en fazla %75'tir [1]. Proje süresi en fazla 24 aydır [1].")
    assert verdict.ok
    assert verdict.cevap == "Destek oranı en fazla %75'tir [1]."
    assert verdict.eksik_kisim == STRIPPED_NOTE
    assert "24" not in verdict.cevap and "24" not in (verdict.eksik_kisim or "")
    assert verdict.ungrounded == ["24"]


def test_whole_answer_ungrounded_is_rejected() -> None:
    # The real qwen3:4b output: the fabricated number shares the only cited sentence.
    verdict = _check("Destek oranı en fazla %75'tir ve proje süresi en fazla 24 aydır [1].")
    assert not verdict.ok
    assert verdict.reason == UNGROUNDED_NUMBER


# --- equivalent spellings are grounded ---------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("Oran yüzde 75'tir [1].", "Destek oranı en fazla %75'tir."),
        ("Oran %75'tir [1].", "Destek oranı en fazla yüzde 75'tir."),
        ("Oran 75% olarak uygulanır [1].", "Destek oranı %75'tir."),
        ("Bütçe 3,5 milyon TL'dir [1].", "Bütçe 3.500.000 TL ile sınırlıdır."),
        ("Ön ödeme 250 bin TL'dir [1].", "Ön ödeme 250.000 TL'dir."),
        ("Süre 18 aydır [1].", "Proje süresi en fazla on sekiz aydır."),
        ("Süre on sekiz aydır [1].", "Proje süresi en fazla 18 aydır."),
        ("Ön ödeme iki yüz elli bin TL'dir [1].", "Ön ödeme 250.000 TL'dir."),
    ],
)
def test_equivalent_number_spellings_are_grounded(answer: str, source: str) -> None:
    verdict = _check(answer, [make_chunk("c", text=source, madde_no=None, surum_tarihi=None)])
    assert verdict.ok and verdict.ungrounded == [] and verdict.cevap == answer


def test_unspelled_amount_not_in_sources_is_stripped() -> None:
    verdict = _check(
        "Bütçe üst sınırı 1.350.000 TL'dir [2]. Destek oranı %75'tir [1].",
        [RATE, BUDGET],
        atiflar=[1, 2],
    )
    assert verdict.ok
    assert verdict.cevap == "Destek oranı %75'tir [1]."
    assert verdict.ungrounded == ["1350000"]
    assert verdict.atiflar == [1]  # recomputed from the remaining inline citations


# --- metadata and question count as sources ----------------------------------------------


def test_article_number_grounded_by_madde_no() -> None:
    assert _check("Madde 7'ye göre oran %75'tir [1].").ok
    verdict = _check("Madde 12'ye göre oran %75'tir [1]. Oran %75'tir [1].")
    assert verdict.ungrounded == ["12"]


def test_program_code_grounded_by_question_or_title() -> None:
    assert _check("1507 programında oran %75'tir [1].").ungrounded == []
    assert _check("1507 programında oran %75'tir [1].", question="Oran nedir?").ungrounded == []


@pytest.mark.parametrize("date", ["25.09.2026", "25/09/2026", "2026-09-25", "25 Eylül 2026"])
def test_date_grounded_by_version_header(date: str) -> None:
    assert _check(f"{date} tarihli esaslara göre oran %75'tir [1].").ungrounded == []


def test_year_of_a_dated_source_is_grounded() -> None:
    assert _check("2026 yılı esaslarına göre oran %75'tir [1].").ungrounded == []


# --- things that are not numbers of the answer ------------------------------------------


def test_list_markers_are_ignored() -> None:
    answer = "1. Oran %75'tir [1].\n2. Oran her dönem için uygulanır [1]."
    assert _check(answer).ungrounded == []


def test_citation_markers_are_ignored() -> None:
    assert _check("Oran %75'tir [1, 2].", [RATE, BUDGET], atiflar=[1, 2]).ungrounded == []


def test_urls_are_ignored() -> None:
    assert _check("Ayrıntı www.tubitak.gov.tr/2244 adresinde; oran %75'tir [1].").ungrounded == []


def test_bare_bir_is_not_a_number() -> None:
    assert _check("Bir proje en fazla %75 oranında desteklenir [1].").ungrounded == []
    # "bir" is also the indefinite article; even before a unit it is not checked.
    assert _check("Rapor yılda bir kez, bir ay içinde verilir; oran %75 [1].").ungrounded == []


# --- near misses are not grounded --------------------------------------------------------


def test_decimal_is_not_the_integer() -> None:
    verdict = _check("Oran %7,5'tir [1]. Oran %75'tir [1].")
    assert verdict.ungrounded == ["7.5"]


def test_number_is_not_grounded_by_a_substring() -> None:
    source = make_chunk("c", madde_no=None, surum_tarihi=None, text="Ücret 1750 TL'dir.")
    verdict = _check("Oran %75'tir [1].", [source], question="Oran nedir?")
    assert not verdict.ok and verdict.ungrounded == ["75"]


def test_range_flags_both_ends() -> None:
    verdict = _check("Oran %75'tir [1]. Süre 12-36 aydır [1].")
    assert verdict.ungrounded == ["12", "36"]


# --- modes ---------------------------------------------------------------------------------


def test_not_answerable_output_is_untouched() -> None:
    output = _answer("Kaynaklarda 24 aylık süre bilgisi yok.", [], ok=False)
    verdict = check_numbers(output, QUESTION, [RATE])
    assert verdict == NumberVerdict(
        ok=True,
        reason=None,
        cevap=output.cevap,
        eksik_kisim=None,
        atiflar=[],
        ungrounded=[],
    )


def test_off_mode_is_identity() -> None:
    answer = "Süre 24 aydır [1]."
    verdict = _check(answer, mode="off")
    assert verdict.ok and verdict.cevap == answer and verdict.ungrounded == []


def test_reject_mode_rejects_any_ungrounded_number() -> None:
    verdict = _check("Oran %75'tir [1]. Süre 24 aydır [1].", mode="reject")
    assert not verdict.ok and verdict.reason == UNGROUNDED_NUMBER
    assert verdict.ungrounded == ["24"]


def test_existing_missing_part_is_kept() -> None:
    output = LlmOutput(
        cevaplanabilir=True,
        cevap="Oran %75'tir [1]. Süre 24 aydır [1].",
        atiflar=[1],
        eksik_kisim="Başvuru tarihi kaynaklarda yok.",
    )
    verdict = check_numbers(output, QUESTION, [RATE])
    assert verdict.eksik_kisim == f"Başvuru tarihi kaynaklarda yok. {STRIPPED_NOTE}"


# --- helpers -------------------------------------------------------------------------------


def test_extract_numbers_canonicalises() -> None:
    text = "%75, 3.500.000 TL, 3,5 milyon, 25.09.2026, 12-36 ay"
    assert extract_numbers(text, answer=True) == {
        "75",
        "3500000",
        "2026-09-25",
        "2026",
        "12",
        "36",
    }


def test_answer_side_number_words_need_a_unit() -> None:
    assert extract_numbers("iki ayrı aşama vardır", answer=True) == set()
    assert extract_numbers("iki ayrı aşama vardır", answer=False) == {"2"}
    assert extract_numbers("yüzde yetmiş beş", answer=True) == {"75"}


@pytest.mark.parametrize(
    "text",
    ["yüzde doksan beşi", "yüzde doksan beştir", "yüzde doksan beşine", "yüzde doksan beşinde"],
)
def test_suffix_on_the_last_number_word_keeps_the_whole_number(text: str) -> None:
    # Before: the run stopped at "doksan" because "beşi" is not a bare number word -> 90.
    assert extract_numbers(text, answer=True) == {"95"}
    assert extract_numbers(text, answer=False) == {"95"}


def test_percent_word_and_unrelated_words_are_not_number_words() -> None:
    assert extract_numbers("yüzde", answer=False) == set()
    assert extract_numbers("onun altında ondan", answer=False) == set()


def test_suffixed_number_word_quantity_keeps_its_unit() -> None:
    assert extract_quantities("yüzde doksan beşi") == {"95|yüzde"}


def test_softened_dort_before_a_suffix() -> None:
    assert extract_numbers("yüzde yirmi dördü", answer=True) == {"24"}


def test_grounding_tokens_use_question_and_cited_metadata() -> None:
    tokens = grounding_tokens("2209 nedir?", [RATE])
    assert {"2209", "1507", "7", "75", "2026-09-25", "2026"} <= tokens


# --- citation repair: a number found in an uncited *prompt* source -------------------------
# Smoke test with a real qwen3:4b: the answer said "… en fazla 18 aydır [5]." while "18 ay"
# was in prompt source [1]; small models under-cite, so the citation is repaired instead of
# the (correct) sentence being stripped.

DURATION = make_chunk(
    "sure",
    doc_title="TÜBİTAK 1507 Uygulama Esasları",
    madde_no="6",
    hiyerarsi="İkinci Bölüm › Madde 6 – Destek süresi",
    surum_tarihi="2026-09-25",
    text="Projelerin destek süresi en fazla 18 aydır.",
)
FILLER = make_chunk("dolgu", madde_no="9", surum_tarihi="2026-09-25", text="Amaç ve kapsam.")
FAQ = make_chunk(
    "sss-oran",
    doc_title="1507 Program Sayfası",
    madde_no=None,
    hiyerarsi="Sıkça Sorulan Sorular",
    surum_tarihi=None,
    text="Soru: Destek oranı nedir? Cevap: KOBİ'ler için destek oranı %75'tir.",
)
G2_SOURCES = [DURATION, FILLER, FILLER, FILLER, FAQ]
G2_QUESTION = "1507 programında KOBİ'lere destek oranı yüzde kaç ve proje süresi en fazla kaç ay?"
G2_ANSWER = "1507 programında KOBİ'lere destek oranı %75'tir ve proje süresi en fazla 18 aydır [5]."


def _repair(
    cevap: str, atiflar: list[int], sources: list[Chunk], *, mode: str = "strip"
) -> NumberVerdict:
    cited = [sources[n - 1] for n in atiflar]
    return check_numbers(
        _answer(cevap, atiflar),
        G2_QUESTION,
        cited,
        mode=mode,  # type: ignore[arg-type]
        sources=sources,
    )


def test_number_in_uncited_prompt_source_repairs_the_citation() -> None:
    verdict = _repair(G2_ANSWER, [5], G2_SOURCES)

    assert verdict.ok and verdict.reason is None
    # Nothing stripped; the repaired source is shown inline where the number is.
    assert verdict.cevap == G2_ANSWER.replace("[5]", "[1, 5]")
    assert verdict.atiflar == [1, 5]
    assert verdict.repaired_citations == [1]
    assert verdict.ungrounded == []
    assert verdict.eksik_kisim is None


def test_repair_picks_the_smallest_source_number() -> None:
    sources = [FAQ, DURATION, FILLER, DURATION]

    verdict = _repair("Oran %75'tir [1]. Süre en fazla 18 aydır [1].", [1], sources)

    assert verdict.atiflar == [1, 2] and verdict.repaired_citations == [2]


def test_reject_mode_repairs_instead_of_rejecting() -> None:
    verdict = _repair(G2_ANSWER, [5], G2_SOURCES, mode="reject")

    assert verdict.ok and verdict.atiflar == [1, 5] and verdict.repaired_citations == [1]


def test_fabricated_number_is_still_stripped_when_sources_are_given() -> None:
    cevap = "Destek oranı en fazla %75'tir [5]. Proje süresi en fazla 24 aydır [5]."

    verdict = _repair(cevap, [5], G2_SOURCES)

    assert verdict.ok
    assert verdict.cevap == "Destek oranı en fazla %75'tir [5]."
    assert verdict.ungrounded == ["24"] and verdict.repaired_citations == []
    assert verdict.atiflar == [5]


def test_repaired_and_fabricated_numbers_in_one_answer() -> None:
    cevap = "Oran %75'tir [5]. Süre en fazla 18 aydır [5]. Bütçe 9 milyon TL'dir [5]."

    verdict = _repair(cevap, [5], G2_SOURCES)

    assert verdict.ok
    assert verdict.cevap == "Oran %75'tir [5]. Süre en fazla 18 aydır [1, 5]."
    assert verdict.atiflar == [1, 5] and verdict.repaired_citations == [1]
    assert verdict.ungrounded == ["9000000"]


def test_repair_is_dropped_with_the_stripped_sentence() -> None:
    # "18" (repairable) and "24" (fabricated) share a sentence: the sentence goes, and the
    # repair that only served it goes too.
    cevap = "Oran %75'tir [5]. Süre 18 ay, uzatma 24 aydır [5]."

    verdict = _repair(cevap, [5], G2_SOURCES)

    assert verdict.cevap == "Oran %75'tir [5]."
    assert verdict.atiflar == [5] and verdict.repaired_citations == []


def test_reject_mode_rejects_when_a_number_is_in_no_prompt_source() -> None:
    cevap = "Süre en fazla 18 aydır, uzatma 24 aydır [5]."

    verdict = _repair(cevap, [5], G2_SOURCES, mode="reject")

    assert not verdict.ok and verdict.reason == UNGROUNDED_NUMBER
    assert verdict.ungrounded == ["24"]


def test_off_mode_does_not_repair() -> None:
    verdict = _repair(G2_ANSWER, [5], G2_SOURCES, mode="off")

    assert verdict.atiflar == [5] and verdict.repaired_citations == []


def test_without_prompt_sources_there_is_no_repair() -> None:
    verdict = check_numbers(_answer(G2_ANSWER, [5]), G2_QUESTION, [FAQ])

    assert not verdict.ok and verdict.repaired_citations == []


def test_inline_markers_and_atiflar_cannot_diverge_after_repair() -> None:
    verdict = _repair(G2_ANSWER, [5], G2_SOURCES)

    final = LlmOutput(cevaplanabilir=True, cevap=verdict.cevap, atiflar=verdict.atiflar)
    assert cited_numbers(final) == set(verdict.atiflar) == {1, 5}


def test_repair_marker_is_inserted_when_the_sentence_has_none() -> None:
    verdict = _repair("Oran %75'tir. Süre en fazla 18 aydır.", [5], G2_SOURCES)

    assert verdict.cevap == "Oran %75'tir. Süre en fazla 18 aydır [1]."
    assert verdict.atiflar == [1, 5] and verdict.repaired_citations == [1]


# --- a repair must not legitimise an invented number -----------------------------------------

_MADDE_3 = make_chunk("m3", madde_no="3", hiyerarsi="Madde 3 – Tanımlar", text="Tanımlar.")
_DATED = make_chunk("tarihli", surum_tarihi="2026-09-25", text="Başvurular yıl boyunca alınır.")
_STAFF = make_chunk("personel", text="Projede en fazla 15 kişi çalışabilir.")


@pytest.mark.parametrize("mode", ["strip", "reject"])
@pytest.mark.parametrize(
    ("sentence", "other"),
    [
        ("Destek süresi en fazla 3 yıldır [1].", _MADDE_3),  # madde_no is not a quantity
        ("Program 2026 yılında başladı [1].", _DATED),  # surum_tarihi is metadata
        ("Proje süresi en fazla 15 aydır [1].", _STAFF),  # "15 kişi" is not "15 ay"
    ],
)
def test_repair_ignores_metadata_and_needs_the_same_unit(
    mode: str, sentence: str, other: Chunk
) -> None:
    verdict = _repair(f"Oran %75'tir [1]. {sentence}", [1], [FAQ, other], mode=mode)

    assert verdict.repaired_citations == []
    if mode == "reject":
        assert not verdict.ok and verdict.reason == UNGROUNDED_NUMBER
    else:
        assert verdict.ok and verdict.cevap == "Oran %75'tir [1]." and verdict.atiflar == [1]


def test_repair_needs_a_unit_in_the_answer() -> None:
    # A bare "18" (no unit) is not repaired even though "18 ay" is in a source.
    verdict = _repair("Oran %75'tir [5]. Bu oran 18 puanla değişir [5].", [5], G2_SOURCES)

    assert verdict.repaired_citations == [] and verdict.cevap == "Oran %75'tir [5]."


def test_strip_keeps_answers_that_cite_only_through_atiflar() -> None:
    verdict = _check("Oran %75'tir. Uzatma 99 aydır.", atiflar=[1])

    assert verdict.ok
    assert verdict.cevap == "Oran %75'tir." and verdict.atiflar == [1]


def test_extract_quantities_pairs_numbers_with_units() -> None:
    assert extract_quantities("Süre en fazla 18 aydır, %75 oranında, 2026 yılında.") == {
        "18|ay",
        "75|yüzde",
        "2026|yıl",
    }
    assert extract_quantities("on sekiz ay ve 3.500.000 TL") == {"18|ay", "3500000|tl"}
    assert extract_quantities("Madde 18 ve 25.09.2026") == set()


def test_outline_numbers_do_not_crash_and_match_consistently() -> None:
    answer = extract_numbers("Bu husus 5.2.2 maddesinde düzenlenir.", answer=True)
    source = extract_numbers("5.2.2. Fatura ve harcama belgeleri", answer=False)
    assert answer
    assert answer <= source
