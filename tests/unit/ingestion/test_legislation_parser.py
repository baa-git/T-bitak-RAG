"""Tests for the legislation parser: hierarchy, articles, clauses, amendments (FR-P4/5/6/8)."""

import json
from pathlib import Path

import pytest

from tubitak_rag.domain.document import Section
from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.ingestion.legislation_parser import (
    ARTICLE_RE,
    HIER_RE,
    PARA_RE,
    build_paragraphs,
    extract_amendment_notes,
    parse_sections,
    strip_footnote_digit,
)
from tubitak_rag.ingestion.pdf_text import PageText, strip_headers_footers


@pytest.fixture
def sections(fixtures: Path) -> list[Section]:
    data = json.loads((fixtures / "mevzuat_pages.json").read_text(encoding="utf-8"))
    pages = strip_headers_footers([PageText(**page) for page in data])
    return parse_sections(pages, "mevzuat-999")


def _pages(*lines: str) -> list[PageText]:
    return [PageText(number=1, lines=list(lines))]


def test_finds_all_articles(sections: list[Section]) -> None:
    assert [s.madde_no for s in sections] == ["1", "2", "3", "4", "Geçici 1", "5"]
    assert all(s.kind == SectionKind.MADDE for s in sections)
    assert [s.section_id for s in sections[:2]] == ["mevzuat-999:0001", "mevzuat-999:0002"]


def test_marker_without_space(sections: list[Section]) -> None:
    # "MADDE2- …" (no space) is how the real 1507 PDF writes article 1.
    assert sections[1].madde_no == "2"
    assert sections[1].text.startswith("Bu Uygulama Esaslarında geçen;")


@pytest.mark.parametrize(
    "line,number,rest",
    [
        ("MADDE 1 - (1) Bu", "1", "(1) Bu"),
        ("MADDE1- (1) Bu", "1", "(1) Bu"),
        ("MADDE 1 – (1) Bu", "1", "(1) Bu"),
        ("Madde 1- (1) Bu", "1", "(1) Bu"),
        ("Madde 1. Bu Esaslar", "1", "Bu Esaslar"),
        ("MADDE 5/A - (1) Koordinatörlük", "5/A", "(1) Koordinatörlük"),
        ("MADDE 17/A- (1) Programın", "17/A", "(1) Programın"),
    ],
)
def test_article_marker_variants(line: str, number: str, rest: str) -> None:
    match = ARTICLE_RE.match(line)
    assert match is not None
    assert (match["no"], match["rest"]) == (number, rest)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("GEÇİCİ MADDE 16- (1) Eski", "Geçici 16"),
        ("Geçici Madde 1. 3/11/2007", "Geçici 1"),
        ("EK MADDE 1 – (1) Bu", "Ek 1"),
        ("Ek Madde 1- Stratejik", "Ek 1"),
    ],
)
def test_temporary_and_additional_article_numbers(line: str, expected: str) -> None:
    (section,) = parse_sections(_pages(line, "metin devam eder."), "d")
    assert section.madde_no == expected


@pytest.mark.parametrize(
    "line",
    ["madde 16.1’ de belirtilen", "Maddede belirtilen", "MADDE 1", "Madde 30.1 kapsamında"],
)
def test_non_article_lines(line: str) -> None:
    assert ARTICLE_RE.match(line) is None


def test_title_is_previous_line_and_footnote_digit_stripped(sections: list[Section]) -> None:
    assert sections[2].title == "Projelerin sunulması"
    assert sections[0].title == "Amaç"
    assert sections[5].title == "Yürütme"


def test_title_line_is_removed_from_previous_article(sections: list[Section]) -> None:
    assert "Tanımlar" not in sections[0].text
    assert "Geçiş hükmü" not in sections[3].text


def test_sentence_line_is_not_taken_as_title() -> None:
    parsed = parse_sections(
        _pages("MADDE 1- (1) Birinci madde", "metni burada biter.", "MADDE 2- (1) İkinci."), "d"
    )
    assert parsed[1].title is None
    assert parsed[0].text.endswith("biter.")


def test_strip_footnote_digit() -> None:
    assert strip_footnote_digit("Projelerin sunulması2") == "Projelerin sunulması"
    assert strip_footnote_digit("Destek oranı12") == "Destek oranı"
    assert strip_footnote_digit("1507 programı") == "1507 programı"


def test_hierarchy_labels(sections: list[Section]) -> None:
    assert sections[0].hierarchy == ["Birinci Kısım – Genel Hükümler"]
    assert sections[2].hierarchy == ["İkinci Kısım – Başvuru", "Birinci Bölüm – Çağrı"]
    assert sections[3].hierarchy == ["İkinci Kısım – Başvuru", "İkinci Bölüm – Değerlendirme"]


def test_bolum_resets_on_new_kisim(sections: list[Section]) -> None:
    assert sections[4].hierarchy == ["Üçüncü Kısım – Son Hükümler"]


@pytest.mark.parametrize(
    "line",
    ["ONBİRİNCİ BÖLÜM", "ON ALTINCI BÖLÜM", "YİRMİNCİ BÖLÜM", "BÖLÜM I", "BİRİNCİ KISIM"],
)
def test_hierarchy_line_variants(line: str) -> None:
    assert HIER_RE.match(line) is not None


def test_hierarchy_without_title_line() -> None:
    parsed = parse_sections(_pages("ON ALTINCI BÖLÜM", "MADDE 70- (1) Metin."), "d")
    assert parsed[0].hierarchy == ["On Altıncı Bölüm"]


def test_amendment_notes_moved_to_metadata(sections: list[Section]) -> None:
    note = "(Değişik:29/12/2025-77YK-05/03/2026 tarihli ve 908456 sayılı Başkanlık Olur’u)"
    assert note in sections[2].amendment_notes
    assert "Değişik" not in sections[2].text
    assert "PRODİS üzerinden (Ek-1’deki form ile) sunulur." in sections[2].text  # FR-P5


def test_note_split_over_lines_is_joined(sections: list[Section]) -> None:
    assert (
        "(Değişik:11/05/2023-51YK-15/06/2023 tarihli ve 437675 sayılı Başkanlık Olur’u)"
        in sections[1].amendment_notes
    )
    assert "(Ek:24/07/2024-67YK)" in sections[1].amendment_notes


def test_nested_parentheses_in_note() -> None:
    text, notes = extract_amendment_notes(
        "(1) (Değişik:01/01/2025-70YK (Başkanlık Olur’u ile)) Proje (bkz. Ek-1) sunulur."
    )
    assert notes == ["(Değişik:01/01/2025-70YK (Başkanlık Olur’u ile))"]
    assert text == "(1) Proje (bkz. Ek-1) sunulur."


@pytest.mark.parametrize(
    "note",
    [
        "(Mülga: 28/01/2020-16YK)",
        "(EK:11/05/2023-51YK)",
        "(Ek iki cümle: 16/2/2016-6676/28 md.)",
        "(Değişik ibare: 24/07/2024-67YK)",
        "(Yeniden düzenleme: 2/3/2020-20YK)",
        "(6676 sayılı Kanunun 26 ncı maddesiyle değişen ibare Yürürlük; 01.03.2016)",
    ],
)
def test_amendment_note_forms(note: str) -> None:
    text, notes = extract_amendment_notes(f"Başı {note} sonu.")
    assert notes == [note] and text == "Başı sonu."


def test_annex_reference_is_not_a_note() -> None:
    text, notes = extract_amendment_notes("Form (Ek-1: Başvuru formu) doldurulur.")
    assert notes == [] and text == "Form (Ek-1: Başvuru formu) doldurulur."


def test_repealed_article_flagged(sections: list[Section]) -> None:
    assert sections[3].repealed is True and sections[3].text == ""  # FR-P6
    assert sections[3].amendment_notes == ["(Mülga: 28/01/2020-16YK)"]
    assert not any(s.repealed for s in sections if s is not sections[3])


def test_paragraphs_split_on_fikra_and_bent(sections: list[Section]) -> None:
    assert [p.label for p in sections[1].paragraphs] == [None, "(1)", "a)", "b)", "(2)"]
    bent_a = sections[1].paragraphs[2]
    assert bent_a.text == "Başkanlık: Teknoloji ve Yenilik Destek Programları Başkanlığını,"


def test_section_text_keeps_labels_one_per_line(sections: list[Section]) -> None:
    lines = sections[1].text.split("\n")
    assert lines[1] == "(1) Aşağıdaki tanımlar geçerlidir:"
    assert lines[2].startswith("a) Başkanlık:")


def test_alt_bent_label() -> None:
    paragraphs, _ = build_paragraphs(
        [(1, "a) Bent:"), (1, "1) alt bent metni,"), (2, "ikinci satır")]
    )
    assert [(p.label, p.text, p.page) for p in paragraphs] == [
        ("a)", "Bent:", 1),
        ("1)", "alt bent metni, ikinci satır", 1),
    ]


def test_page_numbers_tracked(sections: list[Section]) -> None:
    assert (sections[1].page_start, sections[1].page_end) == (1, 2)  # FR-P8
    assert sections[1].paragraphs[-1].page == 2
    assert (sections[2].page_start, sections[2].page_end) == (2, 2)


def test_history_table_dropped(sections: list[Section]) -> None:
    full = "\n".join(s.text for s in sections)
    assert "tıklayınız" not in full and "Kabul edildiği" not in full
    assert sections[-1].title == "Yürütme"
    assert sections[-1].text == "(1) Bu Uygulama Esaslarını TÜBİTAK Başkanı yürütür."
    assert "CETVEL" not in sections[-1].text


def test_no_articles_gives_empty_list() -> None:
    assert parse_sections(_pages("1. Giriş", "Bu kılavuz başvuru içindir."), "d") == []


def test_programme_code_title_line() -> None:
    # BİDEB regulations title articles with the programme name: "2213 - Yurt Dışı … Programı".
    parsed = parse_sections(
        _pages(
            "MADDE 1- (1) Önceki.",
            "2251 - Bilimsel Ağ Geliştirme Programı2, 3",
            "MADDE 2- (1) Metin.",
        ),
        "d",
    )
    assert parsed[1].title == "2251 - Bilimsel Ağ Geliştirme Programı"
    assert parsed[0].text == "(1) Önceki."


# --- review fixes: amendment-history tables, annexes, ordinals, clauses, notes ------------

YURUTME = ["Yürütme", "MADDE 9- (1) Bu Esasları TÜBİTAK Başkanı yürütür."]
TABLE_ROWS = [
    "07/04/2018 tarihli ve 276 sayılı Bilim Kurulu",
    "07/04/2018",
    "Karar için tıklayınız",
]


@pytest.mark.parametrize(
    "header",
    [
        ["Kabul Edilen Bilim Kurulu Kararı Yürürlük Tarihi"],  # 088
        ["Kabul Edilen Bilim Kurulu Kararı", "Yürürlük Tarihi"],  # 118
        ["Kabul Edilen Başkanlık Olur’u Yürürlüğe giriş tarihi"],  # 129
        ["Kabul Edilen Yönetim Kurulu Yürürlük Tarihi"],  # 135
        ["Kabul edildiği Bilim", "Yürürlüğe Giriş", "Kurulu Metin"],  # 092
        ["Yürürlüğe Giriş", "Kabul edildiği Bilim Kurulu Kararı", "Tarihi"],  # 011
        ["Yürürlüğe", "Kabul edildiği Bilim Kurulu Kararı", "Giriş Tarihi"],  # 070
        ["UYGULAMA ESASLARINDA DEĞİŞİKLİK YAPILAN BİLİM KURULU", "KARARLARI"],  # 097
        ["Değişiklik yapılan Başkanlık Olur’u Yürürlüğe giriş tarihi"],
        ["Değişikliğin Kabul Edildiği Yönetim Yürürlüğe Giriş Tarihi"],  # 037
        ["Yayımlandığı Resmi Gazetenin Tarih ve Sayısı Yürürlüğe Giriş Tarihi"],  # 009
        ["Yönetmeliğin Yayınlandığı Resmi Gazete", "Yürürlüğe Giriş Tarihi"],  # 013
        ["Yönetmeliğin Yayımlandığı Resmi Yürürlüğe Giriş Tarihi"],  # 119
        ["ÖRNEK ESASLARI’NA İLİŞKİN CETVEL", "Bilim Kurulu Kararı Yürürlüğe METİN"],  # 069
    ],
)
def test_history_table_variants_are_dropped(header: list[str]) -> None:
    parsed = parse_sections(_pages(*YURUTME, *header, *TABLE_ROWS), "d")
    assert parsed[-1].text == "(1) Bu Esasları TÜBİTAK Başkanı yürütür."


def test_history_table_caps_title_lines_are_dropped() -> None:
    lines = [
        *YURUTME,
        "TÜBİTAK",
        "ÖRNEK DESTEK PROGRAMI UYGULAMA",
        "ESASLARINA İLİŞKİN CETVEL",
        "Yürürlüğe Giriş",
        "Kabul edildiği Bilim Kurulu Kararı",
    ]
    assert parse_sections(_pages(*lines), "d")[-1].text == YURUTME[1].split("- ", 1)[1]


def test_history_words_inside_the_body_do_not_cut_the_document() -> None:
    lines = [
        "MADDE 1- (1) Değişiklikler, Yönetim Kurulunca",
        "kabul edilen bilim kurulu kararlarının yürürlük tarihinden itibaren uygulanır.",
        "Yürürlük",
        "MADDE 2- (1) Bu Esaslar kabul edildiği tarihte yürürlüğe girer ve",
        "yürürlüğe giriş",
        "tarihinden sonraki başvurulara uygulanır.",
    ]
    parsed = parse_sections(_pages(*lines), "d")
    assert [s.madde_no for s in parsed] == ["1", "2"]
    assert parsed[1].text.endswith("tarihinden sonraki başvurulara uygulanır.")


def test_annexes_after_last_article_are_separate_sections() -> None:
    lines = [
        *YURUTME,
        "EK-I: (Değişik: 09/05/2019-9YK)",
        "UBYT PROGRAMI TEŞVİK HESAPLAMA YÖNTEMİ",
        "1) (Değişik:26/03/2024-63YK) UBYT Dergi Listesi oluşturulur.",
        "EK - 2 Performans Göstergeleri (Değişik:11/05/2023-51YK-15/06/2023 tarihli ve",
        "438032 sayılı Başkanlık Olur’u)",
        "Kategori 1: Yükseköğretim kurumu hisseye sahip değildir.",
        "Kabul edildiği Bilim Kurulu Kararı Yürürlüğe Giriş Tarihi",
        *TABLE_ROWS,
    ]
    parsed = parse_sections(_pages(*lines), "d")
    assert parsed[0].text == "(1) Bu Esasları TÜBİTAK Başkanı yürütür."
    first, second = parsed[1], parsed[2]
    assert first.kind == SectionKind.BOLUM and first.madde_no is None
    assert first.title == "EK-I: UBYT PROGRAMI TEŞVİK HESAPLAMA YÖNTEMİ"
    assert first.amendment_notes == ["(Değişik: 09/05/2019-9YK)", "(Değişik:26/03/2024-63YK)"]
    assert first.text == "1) UBYT Dergi Listesi oluşturulur."
    assert second.title == "EK - 2 Performans Göstergeleri"
    assert second.amendment_notes == [
        "(Değişik:11/05/2023-51YK-15/06/2023 tarihli ve 438032 sayılı Başkanlık Olur’u)"
    ]
    assert second.text == "Kategori 1: Yükseköğretim kurumu hisseye sahip değildir."
    assert [s.section_id for s in parsed] == ["d:0001", "d:0002", "d:0003"]


@pytest.mark.parametrize(
    "line,label",
    [
        ("ONUNCU BÖLÜM", "Onuncu Bölüm"),
        ("YİRMİNCİ BÖLÜM", "Yirminci Bölüm"),
        ("ONBİRİNCİ BÖLÜM", "On Birinci Bölüm"),
        ("YİRMİ İKİNCİ BÖLÜM", "Yirmi İkinci Bölüm"),
    ],
)
def test_ordinal_labels(line: str, label: str) -> None:
    parsed = parse_sections(_pages(line, "MADDE 70- (1) Metin."), "d")
    assert parsed[0].hierarchy == [label]


def test_wrapped_number_in_parentheses_is_not_a_fikra() -> None:
    paragraphs, _ = build_paragraphs([(1, "(2) Süre en az bir"), (1, "(1) ay olarak belirlenir.")])
    assert [(p.label, p.text) for p in paragraphs] == [
        ("(2)", "Süre en az bir (1) ay olarak belirlenir.")
    ]


@pytest.mark.parametrize(
    "start", ["(3) Bu", "(4) (Ek: 1/1/2020-1YK) Metin", "(5) “Proje”", "(6) 5018"]
)
def test_fikra_starts(start: str) -> None:
    assert PARA_RE.match(start) is not None


def test_note_with_five_words_before_colon() -> None:
    note = "(Değişik altıncı ve yedinci cümleler: 01/01/2020-10YK)"
    assert extract_amendment_notes(f"Başı {note} sonu.") == ("Başı sonu.", [note])


@pytest.mark.parametrize(
    "first,second,title",
    [
        (  # 1507 madde 40
            "Tamamlanan projeler için destek sürecinin sonlandırılması ve ticarileşme",
            "izleme süreci3",
            "Tamamlanan projeler için destek sürecinin sonlandırılması ve ticarileşme "
            "izleme süreci",
        ),
        (  # 1515 madde 14
            "Kararın oluşturulması, bildirimi, sözleşme imzalanması ve proje destek",
            "tarihi",
            "Kararın oluşturulması, bildirimi, sözleşme imzalanması ve proje destek tarihi",
        ),
    ],
)
def test_two_line_title(first: str, second: str, title: str) -> None:
    lines = [
        "MADDE 39- (1) İzleyiciden ek görüş alınarak",
        "değerlendirme sonuçlandırılır.",
        first,
        second,
        "MADDE 40- (1) Metin.",
    ]
    parsed = parse_sections(_pages(*lines), "d")
    assert parsed[1].title == title
    assert parsed[0].text == "(1) İzleyiciden ek görüş alınarak değerlendirme sonuçlandırılır."


def test_lowercase_line_after_sentence_end_is_not_a_title() -> None:
    lines = ["MADDE 1- (1) Birinci madde.", "devam eden satır", "MADDE 2- (1) Metin."]
    parsed = parse_sections(_pages(*lines), "d")
    assert parsed[1].title is None
    assert parsed[0].text.endswith("devam eden satır")


def test_page_footnote_goes_to_the_marked_article() -> None:
    # 1507 page 29: the footnote of madde 40's title used to be glued into fıkra (4).
    footnote = "3 “Tamamlanan projeler” şeklindeki başlık 32 sayılı YK'da değiştirilmiştir."
    pages = [
        PageText(
            number=29,
            lines=[
                "MADDE 39- (1) Metin.",
                "Tamamlanan projeler için destek sürecinin sonlandırılması ve ticarileşme",
                "izleme süreci3",
                "MADDE 40- (1) Birinci fıkra.",
                "(4) Proje sonuç raporu, kuruluş tarafından",
            ],
            footnotes=[footnote],
        ),
        PageText(number=30, lines=["sunulur.", "MADDE 41- (1) Metin."]),
    ]
    parsed = parse_sections(pages, "d")
    assert (
        parsed[1].text == "(1) Birinci fıkra.\n(4) Proje sonuç raporu, kuruluş tarafından sunulur."
    )
    assert parsed[1].amendment_notes == [footnote]
    assert parsed[0].amendment_notes == []


def test_unmarked_footnote_goes_to_the_article_running_at_page_end() -> None:
    note = "7 “Eski başlık” şeklindeki başlık değiştirilmiştir."
    pages = [
        PageText(number=1, lines=["MADDE 1- (1) Bir.", "MADDE 2- (1) İki."], footnotes=[note]),
        PageText(number=2, lines=["MADDE 3- (1) Üç."]),
    ]
    assert parse_sections(pages, "d")[1].amendment_notes == [note]
