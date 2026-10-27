"""Tests for the scope checker (FR-GR4)."""

import pytest

from tests.fakes.llm import FakeLLM
from tubitak_rag.guardrails.scope import SCOPE_SYSTEM_PROMPT, ScopeChecker, ScopeResult
from tubitak_rag.ports.llm import LlmError


@pytest.mark.parametrize(
    "question",
    [
        "1507'de destek oranı nedir?",
        "TÜBİTAK projelerinde personel gideri desteklenir mi?",
        "Teknoloji geliştirme bölgesinde çalışan personelin gelir vergisi istisnası var mı?",
        "ARGE merkezi kurmak için kaç personel gerekir?",
        "Bursiyer ödemeleri nasıl yapılır?",
        # in-scope questions an earlier lexicon missed
        "Yazılım firması gelir vergisi stopajı istisnasından nasıl yararlanır?",
        "Doktora bursu aylık ne kadar?",
        "Patent ve faydalı model masrafları karşılanır mı?",
        "Personel gideri hesabında SGK işveren payı dahil mi?",
        "Transfer ödemesi ön ödeme oranı kaçtır?",
        "Yeminli mali müşavir raporu ne zaman verilir?",
        "Mentor desteği ücretsiz mi?",
        "YMM raporu zorunlu mu?",
        "Proje bütçesindeki harcama kalemleri nelerdir?",
    ],
)
def test_lexicon_accepts_program_code_question(question: str) -> None:
    assert ScopeChecker("lexicon").check(question) == ScopeResult(in_scope=True, method="lexicon")


@pytest.mark.parametrize(
    "question",
    [
        "İstanbul'da hava nasıl?",
        "Bursa'da hava nasıl?",
        "En iyi kek tarifi hangisi?",
        "Galatasaray dün kaç gol attı?",
        # false positives of an earlier lexicon
        "1923'te cumhuriyet nasıl kuruldu?",
        "1999 depremi ne zaman oldu?",
        "3500 TL maaşla geçinilir mi?",
        "4000 lira kira çok mu?",
        "Projektör lambası nasıl değiştirilir?",
        "Çağrı merkezi numaranız nedir?",
    ],
)
def test_lexicon_rejects_weather_question_in_lexicon_mode(question: str) -> None:
    assert ScopeChecker("lexicon").check(question) == ScopeResult(in_scope=False, method="lexicon")


def test_lexicon_terms_match_at_word_start_with_suffixes() -> None:
    checker = ScopeChecker("lexicon")
    assert checker.check("KOBİ'lere verilen hibeler nelerdir?").in_scope
    assert not checker.check("Bu bir süper deste kart mı?").in_scope  # "deste" is not "destek"
    assert not checker.check("Kaynakça nasıl yazılır?").in_scope


def test_years_are_not_program_codes() -> None:
    assert not ScopeChecker("lexicon").check("2024 yılında kaç gün var?").in_scope
    assert ScopeChecker("lexicon").check("2209-A başvurusu ne zaman?").in_scope


def test_both_mode_asks_llm_only_when_lexicon_misses() -> None:
    llm = FakeLLM(['{"kapsam_ici": false}'])
    checker = ScopeChecker("both", llm)
    assert checker.check("1507 destek oranı?") == ScopeResult(in_scope=True, method="lexicon")
    assert llm.calls == []
    assert checker.check("Hava nasıl?") == ScopeResult(in_scope=False, method="llm")
    assert len(llm.calls) == 1
    assert llm.calls[0]["system"] == SCOPE_SYSTEM_PROMPT
    assert llm.calls[0]["json_schema"]["required"] == ["kapsam_ici"]
    assert "Hava nasıl?" in llm.calls[0]["user"]


def test_llm_mode_always_asks_llm() -> None:
    llm = FakeLLM(['{"kapsam_ici": true}'])
    assert ScopeChecker("llm", llm).check("1507 destek oranı?") == ScopeResult(
        in_scope=True, method="llm"
    )
    assert len(llm.calls) == 1


@pytest.mark.parametrize("reply", ["bozuk", '{"baska": 1}'])
def test_llm_failure_fails_open(reply: str) -> None:
    result = ScopeChecker("llm", FakeLLM([reply])).check("Hava nasıl?")
    assert result == ScopeResult(in_scope=True, method="llm")


def test_llm_error_fails_open() -> None:
    class BrokenLLM(FakeLLM):
        def generate(self, *args: object, **kwargs: object) -> str:
            raise LlmError("Ollama'ya ulaşılamadı")

    assert ScopeChecker("llm", BrokenLLM([])).check("Hava nasıl?").in_scope


def test_off_mode() -> None:
    assert ScopeChecker("off").check("Hava nasıl?") == ScopeResult(in_scope=True, method="off")


def test_llm_modes_require_an_llm() -> None:
    with pytest.raises(ValueError, match="LLM"):
        ScopeChecker("both")
