"""Smoke test: parse the real corpus in ``data/raw`` (FR-P1 … FR-P10 on real files).

Run with ``pytest tests/integration/test_parse_real_corpus.py -m integration -s``; the
statistics printed here go into the development log. Skipped when ``data/raw`` is absent
(e.g. in CI). Parsed documents are cached in ``artifacts/cache/parsed`` (FR-X9), so only the
first run is slow.
"""

import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from tubitak_rag.domain.document import Document
from tubitak_rag.domain.enums import SectionKind, SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.manifest import MEVZUAT, load_sources
from tubitak_rag.ingestion.normalize import tr_lower
from tubitak_rag.ingestion.parsed_cache import ParsedCache, ParseOutcome, parse_all

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PAGE_1501 = (
    "sayfa-sanayi__ulusal-destek-programlari__"
    "1501-tubitak-sanayi-ar-ge-projeleri-destekleme-programi"
)

Corpus = tuple[list[SourceEntry], dict[str, Document], list[ParseOutcome]]
FOOTNOTE_RE = re.compile(r"^\d{1,2}\s+[“\"]")
HISTORY_LEAK_RE = re.compile(
    r"kabul edil(?:diği|en) (?:bilim|yönetim|başkanlık)|yürürlüğe giriş tarihi"
    r"|(?:ilişkin|ait) cetvel|değişiklik yapılan (?:bilim|yönetim|başkanlık)|tıklayınız"
)
STALE_GIB_RE = re.compile(r"\([^()]*önceki (?:ibare|bent|madde|fıkra)\)|Değişmeden önceki şekli")


def _history_leaks(docs: dict[str, Document]) -> list[str]:
    """Sections after the last article of a legislation PDF that still hold table text."""
    leaks = []
    for doc in docs.values():
        if doc.kaynak_grubu != MEVZUAT or not doc.structured:
            continue
        madde = [i for i, s in enumerate(doc.sections) if s.kind == SectionKind.MADDE]
        for section in doc.sections[madde[-1] :]:
            if HISTORY_LEAK_RE.search(tr_lower(section.text)):
                leaks.append(section.section_id)
    return leaks


def _stale_gib_chars(docs: dict[str, Document]) -> int:
    gib = [d for d in docs.values() if d.kind == SourceKind.GIB_JSON]
    return sum(len(m.group()) for d in gib for m in STALE_GIB_RE.finditer(d.full_text()))


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    if not RAW.is_dir():
        pytest.skip("data/raw yok; gerçek corpus testi atlandı")
    entries = load_sources(ROOT / "data" / "manifests")
    docs, outcomes = parse_all(entries, RAW, ParsedCache(ROOT / "artifacts" / "cache" / "parsed"))
    _print_report(entries, {d.doc_id: d for d in docs}, outcomes)
    return entries, {d.doc_id: d for d in docs}, outcomes


def _print_report(
    entries: list[SourceEntry], docs: dict[str, Document], outcomes: list[ParseOutcome]
) -> None:
    by_id = {e.doc_id: e for e in entries}
    print("\n== Ayrıştırma özeti ==")
    print("Durum:", dict(Counter(o.status for o in outcomes)))
    groups = Counter((e.kaynak_grubu, e.kind.value) for e in entries)
    for (group, kind), count in sorted(groups.items()):
        group_docs = [d for d in docs.values() if (d.kaynak_grubu, d.kind.value) == (group, kind)]
        sections = sum(len(d.sections) for d in group_docs)
        print(f"  {group:28} {kind:9} belge={count:3} bölüm={sections:5}")
    mevzuat = [d for d in docs.values() if d.kaynak_grubu == MEVZUAT]
    articles = [sum(s.kind == SectionKind.MADDE for s in d.sections) for d in mevzuat]
    structured = [d for d in mevzuat if d.structured]
    print(
        f"Mevzuat PDF: {len(mevzuat)} belge, yapılı {len(structured)}, "
        f"madde toplam {sum(articles)}, min/medyan/maks "
        f"{min(articles)}/{statistics.median(articles)}/{max(articles)}"
    )
    print("  Kılavuz ayrıştırıcısına düşenler:", [d.doc_id for d in mevzuat if not d.structured])
    repealed = sum(s.repealed for d in mevzuat for s in d.sections)
    notes = sum(len(s.amendment_notes) for d in docs.values() for s in d.sections)
    print(f"  Mülga madde: {repealed}, değişiklik notu (tüm corpus): {notes}")
    scanned = [d.doc_id for d in docs.values() if d.scanned]
    print("Taranmış:", [(i, by_id[i].local_path.rsplit("/", 1)[-1]) for i in scanned])
    faq = sum(s.kind == SectionKind.SSS for d in docs.values() for s in d.sections)
    print(f"SSS çifti: {faq}")
    empty = [d.doc_id for d in docs.values() if not d.sections and not d.scanned]
    print(f"Bölümsüz (taranmamış) belge: {len(empty)}")
    madde = [s for d in mevzuat for s in d.sections if s.kind == SectionKind.MADDE]
    titled = sum(1 for s in madde if s.title) / len(madde)
    footnotes = sum(1 for s in madde for n in s.amendment_notes if FOOTNOTE_RE.match(n))
    annexes = [
        s.title for d in mevzuat if d.structured for s in d.sections if s.kind == SectionKind.BOLUM
    ]
    print(f"  Başlıklı madde: %{100 * titled:.1f}; sayfa dipnotu taşındı: {footnotes}")
    print(f"  EK bölümü (yapılı mevzuat): {len(annexes)} {annexes}")
    print(f"  Geçmiş tablosu sızıntısı: {len(_history_leaks(docs))}")
    print(f"  GİB eski ibare karakteri (metinde): {_stale_gib_chars(docs)}")
    print(
        "Hata/eksik:",
        [(o.doc_id, o.status, o.message) for o in outcomes if o.status in ("error", "missing")],
    )


def test_every_manifest_entry_is_parsed(corpus: Corpus) -> None:
    entries, docs, outcomes = corpus
    assert len(entries) >= 282  # 60 mevzuat + 105 sayfa + 115 dosya + 2 kanun (+ added)
    assert [o for o in outcomes if o.status in ("error", "missing")] == []
    assert len(docs) == len(entries)


def test_scanned_pdfs_detected(corpus: Corpus) -> None:
    _, docs, _ = corpus
    scanned = [d for d in docs.values() if d.scanned]
    assert 3 <= len(scanned) <= 4  # FR-P9
    assert all(d.kind == SourceKind.PDF for d in scanned)


def test_most_legislation_pdfs_are_structured(corpus: Corpus) -> None:
    _, docs, _ = corpus
    mevzuat = [d for d in docs.values() if d.kaynak_grubu == MEVZUAT]
    assert sum(d.structured for d in mevzuat) / len(mevzuat) >= 0.70


def test_yenilik_yonergesi_has_no_combining_marks(corpus: Corpus) -> None:
    _, docs, _ = corpus
    text = docs["mevzuat-118"].full_text()
    assert sum(1 for ch in text if unicodedata.combining(ch)) == 0  # FR-P2


@pytest.mark.parametrize("doc_id", ["mevzuat-030", "mevzuat-031"])
def test_article_7_title(corpus: Corpus, doc_id: str) -> None:
    _, docs, _ = corpus
    article = next(s for s in docs[doc_id].sections if s.madde_no == "7")
    assert article.title == "Destek oranı ve sınırı"  # FR-P4


def test_real_1507_pdf(corpus: Corpus) -> None:
    _, docs, _ = corpus
    doc = docs["mevzuat-031"]
    assert sum(s.kind == SectionKind.MADDE for s in doc.sections) >= 60
    article = next(s for s in doc.sections if s.madde_no == "7")
    assert "destek oranı her dönem için en fazla %75 olarak belirlenmiştir" in article.text
    assert article.amendment_notes and "Değişik" not in article.text  # FR-P5
    assert article.page_start == 6


def test_1501_page_faq_pairs(corpus: Corpus) -> None:
    _, docs, _ = corpus
    faq = [s for s in docs[PAGE_1501].sections if s.kind == SectionKind.SSS]
    assert len(faq) == 16  # FR-P7
    assert faq[0].title == "Kimler başvurabilir?"


def test_no_page_furniture_in_pdf_text(corpus: Corpus) -> None:
    _, docs, _ = corpus
    for doc in docs.values():
        if doc.kind != SourceKind.PDF:
            continue
        text = doc.full_text()
        assert "tıklayınız" not in text, doc.doc_id  # FR-P3
        assert "KANUN VE YÖNETMELİKLER III-" not in text, doc.doc_id


def test_every_section_has_page_or_name(corpus: Corpus) -> None:
    _, docs, _ = corpus
    for doc in docs.values():
        for section in doc.sections:
            located = (
                section.page_start is not None
                or section.madde_no
                or section.title
                or section.hierarchy
            )
            assert located, section.section_id  # FR-P8


def test_second_run_uses_cache(corpus: Corpus) -> None:
    entries, _, _ = corpus
    sample = [e for e in entries if e.kind == SourceKind.PDF][:5]
    _, outcomes = parse_all(sample, RAW, ParsedCache(ROOT / "artifacts" / "cache" / "parsed"))
    assert {o.status for o in outcomes} <= {"cached"}  # FR-X9


def test_no_amendment_history_table_after_the_last_article(corpus: Corpus) -> None:
    _, docs, _ = corpus
    assert _history_leaks(docs) == []  # review I-2


def test_gib_text_has_no_superseded_wording(corpus: Corpus) -> None:
    _, docs, _ = corpus
    assert _stale_gib_chars(docs) == 0  # review I-1


def test_page_footnotes_are_not_in_article_text(corpus: Corpus) -> None:
    _, docs, _ = corpus
    pattern = re.compile(r"(?:^|\s)\d{1,2} “[^”]{1,200}” şeklindeki başlık")
    for doc in docs.values():
        assert not pattern.search(doc.full_text()), doc.doc_id  # review I-4


@pytest.mark.parametrize(
    "doc_id,madde_no,title",
    [
        (
            "mevzuat-031",
            "40",
            "Tamamlanan projeler için destek sürecinin sonlandırılması ve "
            "ticarileşme izleme süreci",
        ),
        (
            "mevzuat-112",
            "6",
            "Çalışanların geliştirdiği veya destek programları kapsamında "
            "ortaya çıkan fikri ürün üzerinde kurumun hak talebi",
        ),
    ],
)
def test_two_line_titles(corpus: Corpus, doc_id: str, madde_no: str, title: str) -> None:
    _, docs, _ = corpus
    article = next(s for s in docs[doc_id].sections if s.madde_no == madde_no)
    assert article.title == title  # review I-3


def test_programme_codes_from_file_names(corpus: Corpus) -> None:
    _, docs, _ = corpus
    assert docs["mevzuat-034"].program_kodu == "1503"  # review I-6
    assert docs["mevzuat-065"].program_kodu is None
