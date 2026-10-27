"""Tests for Unicode normalisation of extracted text (FR-P2)."""

import unicodedata

import pytest

from tubitak_rag.ingestion.normalize import (
    collapse_ws,
    normalize_text,
    repair_legacy_glyphs,
    tidy_punctuation,
    tr_lower,
    tr_title,
)


def test_nfc_composes_decomposed_chars() -> None:
    decomposed = "u\u0308cret"  # "ü" as u + COMBINING DIAERESIS
    assert normalize_text(decomposed) == "ücret"
    assert len(normalize_text(decomposed)) == 5


def test_strips_leftover_combining_marks() -> None:
    # Real artefacts from the Yenilik Yönergesi PDF: a mark repeated after an already
    # composed letter survives NFC and must be removed.
    assert normalize_text("gördüğü\u0308") == "gördüğü"
    assert normalize_text("kuruluş \u0327yetkilisi") == "kuruluş yetkilisi"
    assert not any(unicodedata.combining(ch) for ch in normalize_text("beş \u0327 gün"))


def test_nbsp_and_multispace() -> None:
    assert normalize_text("Destek\u00a0oranı   %75\t ve") == "Destek oranı %75 ve"


def test_removes_soft_hyphen_and_zero_width_chars() -> None:
    assert normalize_text("Araş\u00adtırma\u200b Ge\ufeffliştirme") == "Araştırma Geliştirme"


def test_keeps_newlines_and_strips_line_edges() -> None:
    assert normalize_text("  MADDE 1 \r\n  (1) Bu  esaslar \n") == "MADDE 1\n(1) Bu esaslar"


def test_keeps_typographic_apostrophe() -> None:
    assert normalize_text("TÜBİTAK’a") == "TÜBİTAK’a"


def test_collapse_ws() -> None:
    assert collapse_ws("  a\n b\t\tc \u00a0 ") == "a b c"


@pytest.mark.parametrize(
    "text,expected",
    [("İKİNCİ", "ikinci"), ("KISIM", "kısım"), ("Iğdır", "ığdır"), ("ÖZEL", "özel")],
)
def test_tr_lower(text: str, expected: str) -> None:
    assert tr_lower(text) == expected


def test_tr_title() -> None:
    assert tr_title("ON İKİNCİ BÖLÜM") == "On İkinci Bölüm"
    assert tr_title("BİRİNCİ KISIM") == "Birinci Kısım"


def test_repair_mac_turkish_glyphs() -> None:
    # 1990s Mac fonts: ı -> ›, İ -> ‹, ğ -> ¤, Ğ -> ⁄, ş -> fl, Ş -> fi (next to capitals)
    broken = "TÜB‹TAK taraf›ndan de¤erlendirilmesine iliflkin ARAfiTIRMA DE⁄ER"
    assert repair_legacy_glyphs(broken) == (
        "TÜBİTAK tarafından değerlendirilmesine ilişkin ARAŞTIRMA DEĞER"
    )


def test_repair_mac_turkish_keeps_lowercase_fi() -> None:
    assert repair_legacy_glyphs("TÜB‹TAK finansman› sa¤lar") == "TÜBİTAK finansmanı sağlar"


def test_repair_mac_turkish_sharp_s_variant() -> None:
    assert repair_legacy_glyphs("TÜB‹TAK olußturabilecek TEÞV‹K") == (
        "TÜBİTAK oluşturabilecek TEŞVİK"
    )


def test_repair_g_dot_family() -> None:
    assert repair_legacy_glyphs("BĠDEB BAġVURULARI DanıĢman ĠLĠġKĠN") == (
        "BİDEB BAŞVURULARI Danışman İLİŞKİN"
    )


def test_repair_leaves_normal_text_untouched() -> None:
    # "›" is a legitimate breadcrumb separator elsewhere; without enough signal nothing changes.
    text = "Belge › Bölüm › Madde 7 – reflü finansman"
    assert repair_legacy_glyphs(text) == text


def test_tidy_punctuation() -> None:
    assert tidy_punctuation("Başı , ortası ; sonu .") == "Başı, ortası; sonu."
    assert tidy_punctuation("a\n. b") == "a. b"
