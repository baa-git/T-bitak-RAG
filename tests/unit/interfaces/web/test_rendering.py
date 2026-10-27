"""Turkish formatting, citation links and safe Markdown rendering."""

from datetime import date

import pytest

from tests.eval_builders import make_answer
from tubitak_rag.domain.answer import SourceView
from tubitak_rag.interfaces.web.rendering import (
    cited_sources,
    date_tr,
    link_citations,
    location_parts,
    number_tr,
    plain_paragraphs,
    render_markdown,
    seconds_tr,
    signed_tr,
)
from tubitak_rag.interfaces.web.schemas import SourceContext


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [(0.7143, 3, "0,714"), (7108.0, 0, "7.108"), (1234.5, 2, "1.234,50"), (None, 3, "—")],
)
def test_number_tr(value: float | None, digits: int, expected: str) -> None:
    assert number_tr(value, digits) == expected


def test_signed_tr_uses_explicit_sign_and_typographic_minus() -> None:
    assert signed_tr(0.086) == "+0,086"
    assert signed_tr(-312.0, 0) == "−312"
    assert signed_tr(0.0) == "0,000"
    assert signed_tr(None) == "—"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-09-25", "25.09.2026"),
        ("2026-10-15T09:05:18", "15.10.2026 09:05"),
        (date(2020, 7, 18), "18.07.2020"),
        (None, "—"),
        ("", "—"),
        ("bilinmiyor", "bilinmiyor"),
    ],
)
def test_date_tr(value: str | date | None, expected: str) -> None:
    assert date_tr(value) == expected


def test_link_citations_links_only_known_sources() -> None:
    html = str(link_citations("Oran %75'tir [1]. Ayrıca [7].", [1, 2]))
    assert 'href="#kaynak-1"' in html and 'aria-label="Kaynak 1"' in html
    assert "kaynak-7" not in html and "[7]" in html


def test_link_citations_prefix_keeps_two_answers_apart() -> None:
    html = str(link_citations("Oran [1].", [1], prefix="b-"))
    assert 'href="#kaynak-b-1"' in html and 'data-kaynak="1"' in html


def test_link_citations_escapes_and_splits_paragraphs() -> None:
    html = str(link_citations("<b>x</b> [1]\n\nİkinci paragraf", {1}))
    assert "<b>" not in html and "&lt;b&gt;" in html
    assert html.count("<p>") == 2


def test_link_citations_links_urls() -> None:
    html = str(link_citations("İlgili duyuru: https://tubitak.gov.tr/a?b=1&c=2.", []))
    assert '<a href="https://tubitak.gov.tr/a?b=1&amp;c=2" rel="noopener">' in html
    assert html.endswith(".</p>")  # trailing period is not part of the link


def test_plain_paragraphs_keeps_line_breaks() -> None:
    assert str(plain_paragraphs("a\nb\n\nc")) == "<p>a<br>b</p><p>c</p>"


def test_render_markdown_tables_code_and_wrapped_tables() -> None:
    html = str(render_markdown("| a | b |\n|---|---:|\n| 1 | 2 |\n\n```yaml\nx: 1\n```\n"))
    assert '<div class="tablo-kaydir"><table>' in html and "</table></div>" in html
    assert '<code class="language-yaml">' in html


def test_render_markdown_shows_raw_html_as_text() -> None:
    html = str(render_markdown("Soru: <script>alert(1)</script>\n\n<div>blok</div>\n"))
    assert "<script>" not in html and "<div>blok" not in html
    assert "&lt;script&gt;" in html


def test_render_markdown_can_shift_headings() -> None:
    html = str(render_markdown("# Başlık\n\n## Alt", shift_headings=1))
    assert "<h2>Başlık</h2>" in html and "<h3>Alt</h3>" in html


def _source(no: int) -> SourceView:
    return SourceView(
        no=no,
        chunk_id=f"mevzuat-031:000{no}:00",
        doc_title="Belge",
        konum=f"Madde {no}",
        kaynak_url="https://tubitak.gov.tr/x.pdf",
        text="metin",
        score=1.0,
    )


def test_cited_sources_keep_only_numbers_used_in_the_answer() -> None:
    answer = make_answer(kaynaklar=[_source(1), _source(2), _source(3)])  # cites [1]
    assert [s.no for s in cited_sources(answer)] == [1]


def test_cited_sources_fall_back_to_all_sources_without_citations() -> None:
    answer = make_answer(atiflar=[], kaynaklar=[_source(1), _source(2)])
    assert [s.no for s in cited_sources(answer)] == [1, 2]


def test_location_parts_combine_article_page_and_section() -> None:
    context = SourceContext(
        hiyerarsi="İkinci Bölüm – Destek Kapsamı",
        madde_baslik="Madde 7 – Destek oranı ve sınırı",
        sayfa=6,
    )
    assert location_parts("Madde 7 (1)", context) == [
        "Madde 7 – Destek oranı ve sınırı",
        "sayfa 6",
        "İkinci Bölüm – Destek Kapsamı",
    ]


def test_location_parts_without_context_or_with_a_page_location() -> None:
    assert location_parts("SSS: Başvuru", None) == ["SSS: Başvuru"]
    assert location_parts("Sayfa 6", SourceContext(sayfa=6)) == ["Sayfa 6"]


@pytest.mark.parametrize(("ms", "expected"), [(7131.9, "7,1 sn"), (412.6, "0,4 sn"), (0, "0,0 sn")])
def test_seconds_tr(ms: float, expected: str) -> None:
    assert seconds_tr(ms) == expected


def test_link_citations_splits_grouped_citations_into_links() -> None:
    html = str(link_citations("Oran %75'tir [1, 2, 9].", [1, 2]))
    assert 'href="#kaynak-1"' in html and 'href="#kaynak-2"' in html
    assert "kaynak-9" not in html and "[9]" in html  # unknown number stays text


def test_location_parts_skip_a_section_already_named_in_the_location() -> None:
    faq = SourceContext(hiyerarsi="Sıkça Sorulan Sorular › Destek oranı nedir?")
    assert location_parts("SSS: Destek oranı nedir?", faq) == ["SSS: Destek oranı nedir?"]
    section = SourceContext(hiyerarsi="Destek Kapsamı")
    assert location_parts("Bölüm: Destek Kapsamı", section) == ["Bölüm: Destek Kapsamı"]
