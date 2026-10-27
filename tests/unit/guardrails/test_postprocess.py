"""Tests for answer post-processing (FR-GR5, FR-GR6)."""

import pytest

from tests.builders import make_chunk
from tubitak_rag.guardrails.postprocess import (
    append_missing_part,
    attach_announcement_link,
)

DEFERRING = make_chunk(
    "ue",
    kaynak_url="https://tubitak.gov.tr/ue.pdf",
    text="Başvuru tarihleri ve bütçe üst sınırı çağrı duyurusunda belirtilir.",
)
PROGRAM_PAGE = make_chunk(
    "sayfa",
    doc_turu="program_sayfasi",
    kaynak_url="https://tubitak.gov.tr/1507",
    text="1507 programı yılda iki kez çağrıya çıkar.",
)


def test_announcement_link_added_when_source_defers_to_call() -> None:
    text = attach_announcement_link("Tarihler çağrıda belirtilir [1].", [DEFERRING])
    assert text == (
        "Tarihler çağrıda belirtilir [1].\n\n"
        "İlgili duyuru/çağrı sayfası: https://tubitak.gov.tr/ue.pdf"
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "Çağrı Duyurusunda Belirtilir",
        "çağrı duyurularında yer alır",
        "çağrı metninde açıklanır",
        "çağrı dokümanında belirtilen",
    ],
)
def test_announcement_phrases_are_recognised(phrase: str) -> None:
    chunk = make_chunk("c", kaynak_url="https://x", text=f"Destek üst limiti {phrase}.")
    assert "https://x" in attach_announcement_link("Cevap.", [chunk])


def test_prefers_program_page_url() -> None:
    text = attach_announcement_link("Cevap [1][2].", [DEFERRING, PROGRAM_PAGE])
    assert text.endswith("İlgili duyuru/çağrı sayfası: https://tubitak.gov.tr/1507")
    assert "ue.pdf" not in text


def test_no_link_when_not_deferred() -> None:
    assert attach_announcement_link("Cevap [1].", [PROGRAM_PAGE]) == "Cevap [1]."


def test_no_link_when_url_already_in_text() -> None:
    text = "Ayrıntı: https://tubitak.gov.tr/ue.pdf [1]."
    assert attach_announcement_link(text, [DEFERRING]) == text


def test_missing_part_appended() -> None:
    text = append_missing_part("Destek oranı %75'tir [1].", "Proje süresi kaynaklarda yok.")
    assert text == (
        "Destek oranı %75'tir [1].\n\nCevaplanamayan kısım: Proje süresi kaynaklarda yok."
    )


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_no_missing_part_leaves_text(missing: str | None) -> None:
    assert append_missing_part("Cevap.", missing) == "Cevap."
