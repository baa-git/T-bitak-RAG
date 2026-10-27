"""Hand-built ``Document`` objects for chunking tests (no parser needed).

Section texts follow the parsers' conventions: article (madde) paragraphs are rendered as
``"<label> <text>"`` and joined by ``"\\n"``; other sections join paragraphs by ``"\\n\\n"``.
"""

from collections.abc import Sequence
from datetime import date
from typing import Any

from tubitak_rag.domain.document import Document, Paragraph, Section
from tubitak_rag.domain.enums import DocType, SectionKind, SourceKind

LEGISLATION_TITLE = "TÜBİTAK 1507- KOBİ Ar-Ge Başlangıç Destek Programı Uygulama Esasları"
PART = "İkinci Kısım – Başvuru"
CHAPTER = "Birinci Bölüm – Çağrı"


def make_document(sections: Sequence[Section], **overrides: Any) -> Document:
    """Return a valid legislation-like Document holding ``sections``."""
    fields: dict[str, Any] = {
        "doc_id": "mevzuat-031",
        "title": LEGISLATION_TITLE,
        "program_kodu": "1507",
        "doc_turu": DocType.UYGULAMA_ESASLARI,
        "kind": SourceKind.PDF,
        "kaynak_url": "https://tubitak.gov.tr/1507_ue.pdf",
        "surum_tarihi": date(2026, 9, 25),
        "dil": "tr",
        "sha256": "a" * 64,
        "kaynak_grubu": "tubitak_mevzuat",
        "scanned": False,
        "structured": True,
        "page_count": 10,
        "sections": list(sections),
    }
    fields.update(overrides)
    return Document(**fields)


def make_article(
    order: int,
    madde_no: str,
    title: str | None,
    paragraphs: Sequence[tuple[str | None, str]],
    *,
    hierarchy: Sequence[str] = (PART, CHAPTER),
    page: int = 1,
    notes: Sequence[str] = (),
    repealed: bool = False,
    doc_id: str = "mevzuat-031",
) -> Section:
    """An article whose paragraphs are ``(label, text)`` pairs, all on ``page``."""
    paras = [Paragraph(label=label, text=text, page=page) for label, text in paragraphs]
    return Section(
        section_id=f"{doc_id}:{order:04d}",
        kind=SectionKind.MADDE,
        title=title,
        hierarchy=list(hierarchy),
        madde_no=madde_no,
        page_start=page,
        page_end=page,
        text="\n".join(f"{p.label} {p.text}" if p.label else p.text for p in paras),
        paragraphs=paras,
        amendment_notes=list(notes),
        repealed=repealed,
    )


def make_section(
    order: int,
    kind: SectionKind,
    title: str | None,
    paragraphs: Sequence[str],
    *,
    hierarchy: Sequence[str] = (),
    page: int | None = None,
    text: str | None = None,
    doc_id: str = "mevzuat-031",
) -> Section:
    """A non-article section (heading, FAQ pair, paragraph or table) with plain paragraphs."""
    paras = [Paragraph(label=None, text=p, page=page) for p in paragraphs]
    return Section(
        section_id=f"{doc_id}:{order:04d}",
        kind=kind,
        title=title,
        hierarchy=list(hierarchy),
        madde_no=None,
        page_start=page,
        page_end=page,
        text=text if text is not None else "\n\n".join(paragraphs),
        paragraphs=paras,
        amendment_notes=[],
        repealed=False,
    )


def make_legislation_doc() -> Document:
    """Five short articles (madde 5–8 and Geçici 1) on pages 1–4 under one part/chapter."""
    articles = [
        make_article(1, "5", "Kapsam", [("(1)", "Bu esaslar KOBİ projelerini kapsar.")], page=2),
        make_article(2, "6", "Başvuru", [("(1)", "Başvurular çağrı ile alınır.")], page=2),
        make_article(
            3,
            "7",
            "Destek oranı ve sınırı",
            [
                ("(1)", "Destek oranı her dönem için en fazla %75 olarak belirlenmiştir."),
                ("(2)", "Bütçe sınırı Yönetim Kurulu kararı ile uygulanır."),
            ],
            page=3,
            notes=["(Değişik:29/12/2025-77YK)"],
        ),
        make_article(4, "8", "Süre", [("(1)", "Proje süresi en fazla 18 aydır.")], page=4),
        make_article(5, "Geçici 1", None, [(None, "Bu esaslar yayımı tarihinde yürürlüğe girer.")]),
    ]
    return make_document(articles)


def make_long_article_doc(n_paragraphs: int) -> Document:
    """One article (madde 7) with ``n_paragraphs`` numbered fıkras of 15 regex tokens each."""
    paragraphs = [
        (f"({i})", f"Fıkra {i} metni destek oranı ve bütçe sınırı hakkında kural koyar.")
        for i in range(1, n_paragraphs + 1)
    ]
    return make_document([make_article(1, "7", "Destek oranı ve sınırı", paragraphs, page=6)])


def make_program_page_doc() -> Document:
    """A Drupal program page: two text blocks and two FAQ pairs (no pages)."""
    doc_id = "sayfa-1501"
    faq = "Sıkça Sorulan Sorular"
    sections = [
        make_section(
            1,
            SectionKind.BOLUM,
            "Program Hakkında",
            ["1501 programı sanayi Ar-Ge projelerini destekler.", "Başvurular süreklidir."],
            doc_id=doc_id,
        ),
        make_section(
            2,
            SectionKind.BOLUM,
            "Destek Oranı",
            ["Destek oranı en fazla %60'tır."],
            doc_id=doc_id,
        ),
        make_section(
            3,
            SectionKind.SSS,
            "Kimler başvurabilir?",
            ["Türkiye'de yerleşik sermaye şirketleri başvurabilir."],
            hierarchy=[faq],
            text="Soru: Kimler başvurabilir?\n"
            "Cevap: Türkiye'de yerleşik sermaye şirketleri başvurabilir.",
            doc_id=doc_id,
        ),
        make_section(
            4,
            SectionKind.SSS,
            "Başvuru ne zaman yapılır?",
            ["Başvurular yıl boyunca yapılabilir."],
            hierarchy=[faq],
            text="Soru: Başvuru ne zaman yapılır?\nCevap: Başvurular yıl boyunca yapılabilir.",
            doc_id=doc_id,
        ),
    ]
    return make_document(
        sections,
        doc_id=doc_id,
        title="1501 - TÜBİTAK Sanayi Ar-Ge Projeleri Destekleme Programı",
        program_kodu="1501",
        doc_turu=DocType.PROGRAM_SAYFASI,
        kind=SourceKind.HTML,
        kaynak_url="https://tubitak.gov.tr/tr/destekler/sanayi/1501",
        surum_tarihi=None,
        kaynak_grubu="tubitak_program_sayfalari",
        structured=False,
        page_count=None,
    )


def make_guide_doc(n_paragraphs: int = 10) -> Document:
    """A guide without headings: ``n_paragraphs`` PARAGRAF sections of 12 tokens each."""
    doc_id = "dosya-abc"
    sentence = "başvuru sürecinde dikkat edilecek önemli bir kuralı kısaca açıklar ."
    sections = [
        make_section(
            i,
            SectionKind.PARAGRAF,
            None,
            [f"Paragraf {i} {sentence}"],
            page=1 + (i - 1) // 4,
            doc_id=doc_id,
        )
        for i in range(1, n_paragraphs + 1)
    ]
    return make_document(
        sections,
        doc_id=doc_id,
        title="1501 Başvuru Kılavuzu",
        program_kodu="1501",
        doc_turu=DocType.KILAVUZ,
        kaynak_grubu="tubitak_program_sayfalari",
        structured=False,
    )
