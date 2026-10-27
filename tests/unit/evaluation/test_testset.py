"""Tests for the test-set schema, loading, validation and review queue (FR-E1 … FR-E3)."""

import json
from pathlib import Path
from typing import Any

import pytest

from tubitak_rag.domain.document import Document, Section
from tubitak_rag.domain.enums import DocType, QuestionType, SectionKind, SourceKind
from tubitak_rag.domain.evaluation import EvidenceQuote, TestQuestion
from tubitak_rag.evaluation.testset import (
    TestsetError,
    contains_number,
    load_testset,
    locate_quote,
    normalize_for_match,
    quote_warnings,
    save_testset,
    type_distribution,
    validate_testset,
    verify_testset,
    write_review_queue,
)

REPO_TESTSET = Path(__file__).resolve().parents[3] / "eval" / "testset_v1.jsonl"


def _question(qid: str = "q001", **overrides: Any) -> TestQuestion:
    fields: dict[str, Any] = {
        "id": qid,
        "soru": "1507 programında destek oranı en fazla yüzde kaçtır?",
        "tur": QuestionType.TEK_BELGE,
        "zorluk": "kolay",
        "beklenen_cevap": "Her dönem için en fazla %75.",
        "kanit_alintilari": [
            EvidenceQuote(doc_id="mevzuat-031", alinti="destek oranı her dönem için en fazla %75")
        ],
        "beklenen_belgeler": ["mevzuat-031"],
        "beklenen_sayilar": ["75"],
        "program_kodu": ["1507"],
    }
    fields.update(overrides)
    return TestQuestion.model_validate(fields)


def _unanswerable(qid: str, tur: QuestionType) -> TestQuestion:
    answer = "bilgi bulunamadı" if tur == QuestionType.CEVAPSIZ else "kapsam dışı"
    return _question(
        qid,
        tur=tur,
        beklenen_cevap=answer,
        kanit_alintilari=[],
        beklenen_belgeler=[],
        beklenen_sayilar=[],
        program_kodu=[],
    )


def _section(order: int, text: str, **overrides: Any) -> Section:
    fields: dict[str, Any] = {
        "section_id": f"mevzuat-031:{order:04d}",
        "kind": SectionKind.MADDE,
        "title": "Destek oranı ve sınırı",
        "hierarchy": [],
        "madde_no": "7",
        "page_start": 6,
        "page_end": 6,
        "text": text,
        "paragraphs": [],
        "amendment_notes": [],
        "repealed": False,
    }
    fields.update(overrides)
    return Section.model_validate(fields)


def _doc(doc_id: str = "mevzuat-031", sections: list[Section] | None = None) -> Document:
    if sections is None:
        sections = [
            _section(1, "(1) Destek oranı her dönem için\nen fazla %75 olarak belirlenmiştir.")
        ]
    return Document(
        doc_id=doc_id,
        title="TÜBİTAK 1507 Uygulama Esasları",
        program_kodu="1507",
        doc_turu=DocType.UYGULAMA_ESASLARI,
        kind=SourceKind.PDF,
        kaynak_url="https://tubitak.gov.tr/1507.pdf",
        surum_tarihi=None,
        dil="tr",
        sha256="a" * 64,
        kaynak_grubu="tubitak_mevzuat",
        scanned=False,
        structured=True,
        page_count=10,
        sections=sections,
    )


# --- loading and saving --------------------------------------------------------------------


def test_load_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    questions = [_question("q001"), _unanswerable("q002", QuestionType.KAPSAM_DISI)]
    save_testset(path, questions)
    assert load_testset(path) == questions
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert "oranı" in first_line  # Turkish text is written as-is, not \u-escaped
    assert b"\r\n" not in path.read_bytes()  # LF on every platform (stable git diffs)


def test_load_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text("\n" + _question().model_dump_json() + "\n\n", encoding="utf-8")
    assert [q.id for q in load_testset(path)] == ["q001"]


def test_invalid_line_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    bad = _question("q002").model_dump(mode="json")
    bad["tur"] = "bilinmeyen"
    path.write_text(_question().model_dump_json() + "\n" + json.dumps(bad) + "\n", "utf-8")
    with pytest.raises(TestsetError, match=r"satır 2.*tur"):
        load_testset(path)


def test_broken_json_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text('{"id": "q001",\n', encoding="utf-8")
    with pytest.raises(TestsetError, match="satır 1"):
        load_testset(path)


# --- normalisation -------------------------------------------------------------------------


def test_normalize_for_match_case_and_whitespace() -> None:
    assert normalize_for_match("Destek  oranı\n%75") == normalize_for_match("destek oranı %75")
    assert normalize_for_match("DESTEK ORANI") == "destek oranı"  # Turkish I -> ı
    assert normalize_for_match("İLK") == "ilk"


def test_normalize_for_match_punctuation_and_hyphenation() -> None:
    assert normalize_for_match("Türkiye’de “proje” – süre") == normalize_for_match(
        'Türkiye\'de "proje" - süre'
    )
    assert normalize_for_match("madde 7 , fıkra ( 2 )") == normalize_for_match("madde 7, fıkra (2)")
    assert normalize_for_match("proje-\nlerin") == normalize_for_match("projelerin")
    assert normalize_for_match("Ar-\nGe") == normalize_for_match("Ar-Ge")  # real hyphen kept


def test_normalize_for_match_joins_upper_case_hyphenation() -> None:
    assert normalize_for_match("TÜBİ-\nTAK desteği") == normalize_for_match("TÜBİTAK desteği")


def test_normalize_for_match_keeps_documented_non_equivalences() -> None:
    assert normalize_for_match("% 60") != normalize_for_match("%60")
    assert normalize_for_match("dâhil") != normalize_for_match("dahil")
    assert normalize_for_match("1.350.000") != normalize_for_match("1350000")


# --- validation ----------------------------------------------------------------------------


def test_valid_question_passes() -> None:
    assert validate_testset([_question()], {"mevzuat-031": _doc()}) == []


def test_validate_requires_quotes_for_answerable() -> None:
    errors = validate_testset([_question(kanit_alintilari=[])])
    assert any("q001" in e and "alıntı" in e for e in errors)
    errors = validate_testset([_question(beklenen_belgeler=[])])
    assert any("q001" in e and "beklenen_belgeler" in e for e in errors)


def test_validate_limits_quote_count() -> None:
    quote = EvidenceQuote(doc_id="mevzuat-031", alinti="destek oranı")
    errors = validate_testset([_question(kanit_alintilari=[quote] * 4)])
    assert any("q001" in e and "en fazla 3" in e for e in errors)


def test_validate_rejects_quotes_for_unanswerable() -> None:
    q = _unanswerable("q005", QuestionType.CEVAPSIZ)
    with_quote = q.model_copy(
        update={"kanit_alintilari": [EvidenceQuote(doc_id="mevzuat-031", alinti="x")]}
    )
    assert validate_testset([q]) == []
    errors = validate_testset([with_quote])
    assert any("q005" in e and "alıntı" in e for e in errors)


def test_validate_unanswerable_expected_answer_text() -> None:
    q = _unanswerable("q006", QuestionType.KAPSAM_DISI)
    wrong = q.model_copy(update={"beklenen_cevap": "bilgi bulunamadı"})
    errors = validate_testset([wrong])
    assert any("q006" in e and "kapsam dışı" in e for e in errors)


def test_validate_duplicate_ids() -> None:
    errors = validate_testset([_question("q001"), _question("q001")])
    assert any("q001" in e and "tekrar" in e for e in errors)


def test_validate_quote_doc_must_be_expected() -> None:
    errors = validate_testset([_question(beklenen_belgeler=["mevzuat-030"])])
    assert any("q001" in e and "mevzuat-031" in e for e in errors)


def test_validate_multi_doc_types_need_two_documents() -> None:
    for tur in (QuestionType.COK_BELGE, QuestionType.BELIRSIZ):
        errors = validate_testset([_question(tur=tur)])
        assert any("q001" in e and "iki farklı belge" in e for e in errors), tur


def test_validate_expected_numbers_appear_in_answer() -> None:
    errors = validate_testset([_question(beklenen_sayilar=["60"])])
    assert any("q001" in e and "60" in e for e in errors)


def test_validate_expected_number_needs_number_boundaries() -> None:
    q = _question(beklenen_cevap="Destek süresi en fazla 24 aydır.", beklenen_sayilar=["2"])
    assert any("q001" in e and "'2'" in e for e in validate_testset([q]))


@pytest.mark.parametrize(
    ("number", "text", "found"),
    [
        ("2", "aylık %2 oranı", True),
        ("2", "en fazla 24 ay", False),
        ("30", "en fazla 30.", True),
        ("75", "oran %75'tir", True),
        ("1.350.000", "1.350.000 TL yatırım", True),
        ("350", "1.350.000 TL", False),
        ("12", "1,12 katı", False),
        ("beş", "ilk beş projesi", True),
    ],
)
def test_contains_number_is_boundary_aware(number: str, text: str, found: bool) -> None:
    assert contains_number(text, number) is found


def test_validate_checks_quote_in_document_text() -> None:
    fabricated = EvidenceQuote(doc_id="mevzuat-031", alinti="destek oranı yüzde doksandır")
    errors = validate_testset(
        [_question("q042", kanit_alintilari=[fabricated])], {"mevzuat-031": _doc()}
    )
    assert len(errors) == 1 and "q042" in errors[0] and "bulunamadı" in errors[0]


def test_validate_quote_matches_across_line_breaks() -> None:
    quote = EvidenceQuote(doc_id="mevzuat-031", alinti="her dönem için en fazla %75 olarak")
    assert validate_testset([_question(kanit_alintilari=[quote])], {"mevzuat-031": _doc()}) == []


def test_validate_unknown_document() -> None:
    errors = validate_testset([_question()], {"mevzuat-030": _doc("mevzuat-030")})
    assert any("q001" in e and "mevzuat-031" in e and "yok" in e for e in errors)


def test_type_distribution() -> None:
    questions = [
        _question("q1"),
        _question("q2"),
        _unanswerable("q3", QuestionType.CEVAPSIZ),
    ]
    dist = type_distribution(questions)
    assert list(dist) == [t.value for t in QuestionType]  # every type, enum order
    assert dist["tek_belge"] == 2 and dist["cevapsiz"] == 1 and dist["surum"] == 0


# --- review queue --------------------------------------------------------------------------


def test_write_review_queue_groups_and_batches(tmp_path: Path) -> None:
    questions = [_question(f"q{i:03d}") for i in range(1, 16)]
    questions += [_unanswerable(f"q{i:03d}", QuestionType.CEVAPSIZ) for i in range(16, 24)]
    out = tmp_path / "review_queue.md"
    write_review_queue(list(reversed(questions)), {"mevzuat-031": _doc()}, out)
    text = out.read_text(encoding="utf-8")
    assert text.count("## Parti ") == 3  # 23 questions -> 10 + 10 + 3
    assert text.count("- [ ] **q") == 23
    assert text.index("**q001**") < text.index("**q015**") < text.index("**q016**")
    assert "TÜBİTAK 1507 Uygulama Esasları" in text and "Madde 7" in text and "s. 6" in text
    assert "destek oranı her dönem için en fazla %75" in text


def test_review_queue_marks_unlocated_quote(tmp_path: Path) -> None:
    fabricated = EvidenceQuote(doc_id="mevzuat-031", alinti="hiç geçmeyen bir ifade")
    out = tmp_path / "queue.md"
    write_review_queue([_question(kanit_alintilari=[fabricated])], {"mevzuat-031": _doc()}, out)
    assert "konum bulunamadı" in out.read_text(encoding="utf-8")


def test_review_queue_falls_back_to_file_name_for_blank_title(tmp_path: Path) -> None:
    doc = _doc().model_copy(update={"title": " ", "kaynak_url": "https://x.gov.tr/a/cagri.pdf"})
    out = tmp_path / "queue.md"
    write_review_queue([_question()], {"mevzuat-031": doc}, out)
    assert "(cagri.pdf [mevzuat-031]" in out.read_text(encoding="utf-8")


def test_review_queue_starts_new_batch_at_each_type(tmp_path: Path) -> None:
    questions = [_question(f"q{i:03d}") for i in range(1, 13)]
    questions += [_unanswerable(f"q{i:03d}", QuestionType.CEVAPSIZ) for i in range(13, 16)]
    out = tmp_path / "queue.md"
    write_review_queue(questions, {"mevzuat-031": _doc()}, out)
    text = out.read_text(encoding="utf-8")
    assert text.count("## Parti ") == 3  # 10 + 2 tek_belge, then 3 cevapsiz
    assert "## Parti 2 — tek_belge (q011 … q012)" in text
    assert "## Parti 3 — cevapsiz (q013 … q015)" in text


# --- quote location, warnings, file verification ---------------------------------------


def test_locate_quote_reports_quote_spanning_sections() -> None:
    doc = _doc(
        sections=[
            _section(1, "(1) Destek oranı en fazla %75 olarak belirlenmiştir."),
            _section(2, "(2) Ek destek verilmez.", madde_no="8"),
        ]
    )
    assert locate_quote(doc, "belirlenmiştir. (2) Ek destek") == "birden fazla bölüme yayılıyor"


def test_locate_quote_names_faq_question() -> None:
    faq = _section(
        1,
        "Cevap: Türkiye'de yerleşik KOBİ ölçeğindeki sermaye şirketleri başvurabilir.",
        kind=SectionKind.SSS,
        title="Kimler başvurabilir?",
        madde_no=None,
        page_start=None,
        page_end=None,
    )
    assert locate_quote(_doc(sections=[faq]), "sermaye şirketleri") == "SSS: Kimler başvurabilir?"


def test_quote_warnings_flags_very_short_quotes() -> None:
    short = EvidenceQuote(doc_id="mevzuat-031", alinti="en fazla %75")
    warnings = quote_warnings([_question("q007", kanit_alintilari=[short]), _question("q008")])
    assert len(warnings) == 1 and "q007" in warnings[0]


def test_verify_testset_reads_file_and_checks_corpus(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    fabricated = EvidenceQuote(doc_id="mevzuat-031", alinti="hiç geçmeyen bir ifade burada")
    save_testset(path, [_question("q001"), _question("q002", kanit_alintilari=[fabricated])])
    errors = verify_testset(path, {"mevzuat-031": _doc()})
    assert len(errors) == 1 and errors[0].startswith("q002")


# --- the committed test set ----------------------------------------------------------------


def test_repo_testset_is_valid_and_distributed() -> None:
    questions = load_testset(REPO_TESTSET)
    assert validate_testset(questions) == []  # FR-E2 (schema + rules, no corpus needed)
    assert len(questions) >= 90  # FR-E1
    dist = type_distribution(questions)
    assert all(count >= 5 for count in dist.values()), dist  # FR-E3


@pytest.mark.integration
def test_repo_testset_quotes_match_parsed_corpus() -> None:
    """Every evidence quote is found verbatim in the parsed text (needs ``data/raw``)."""
    root = REPO_TESTSET.parents[1]
    if not (root / "data" / "raw").is_dir():
        pytest.skip("data/raw yok; alıntı doğrulaması atlandı")
    from tubitak_rag.ingestion.manifest import load_sources
    from tubitak_rag.ingestion.parsed_cache import ParsedCache, parse_all

    entries = load_sources(root / "data" / "manifests")
    cache = ParsedCache(root / "artifacts" / "cache" / "parsed")
    docs, _ = parse_all(entries, root / "data" / "raw", cache)
    errors = validate_testset(load_testset(REPO_TESTSET), {d.doc_id: d for d in docs})
    assert errors == []
