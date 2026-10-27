"""Scope check: is the question about TÜBİTAK support programmes / R&D legislation? (FR-GR4).

The lexicon is cheap and transparent but misses paraphrases; the LLM classifier catches those.
In ``both`` mode the LLM is asked only when the lexicon finds no term. A failing classifier
lets the question through (fail-open): the retrieval gate is the second line of defence.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from tubitak_rag.generation.schema import LlmOutputError, parse_json_reply
from tubitak_rag.ports import LLM
from tubitak_rag.ports.llm import LlmError
from tubitak_rag.retrieval.turkish import turkish_lower

ScopeMode = Literal["lexicon", "llm", "both", "off"]

# Lower-cased stems; a term matches at the start of a word so Turkish suffixes are allowed
# ("projelerinde", "KOBİ'lere").
IN_SCOPE_TERMS = (
    "tübitak",
    "ar-ge",
    "arge",
    "ar ge",
    "destek",
    "desteğ",
    "hibe",
    "teydeb",
    "ardeb",
    "bideb",
    "teknopark",
    "teknoloji geliştirme bölge",
    "tgb",
    "kobi",
    "başvuru",
    "bursiyer",
    "doktora",
    "pti",
    "mali rapor",
    "mali müşavir",
    "ymm",
    "yönetmelik",
    "esaslar",
    "mevzuat",
    "yenilik",
    "inovasyon",
    "tasarım merkez",
    "girişim",
    "stopaj",
    "istisna",
    "patent",
    "personel gider",
    "sgk",
    "ön ödeme",
    "transfer ödeme",
    "bütçe",
    "harcama",
)
# Stems that need an exception: "projektör", "çağrı merkezi", "Bursa".
_GUARDED_STEMS = (r"proje(?!ktör)", r"çağrı(?! merkez)", r"burs(?!a\b|a')")
# Programme and law numbers (1001, 1507, 2209, 4691, 5746 …). 1900–2099 are left out so that
# years do not count, and an amount ("3500 TL", "4000 lira") is not a programme code.
_PROGRAM_CODE = r"\b(?:1[0-8]\d{2}|2[1-9]\d{2}|[3-5]\d{3})\b(?!\s*(?:tl\b|lira))"
_LEXICON_RE = re.compile(
    "|".join(
        [
            _PROGRAM_CODE,
            *(rf"\b{stem}" for stem in _GUARDED_STEMS),
            *(rf"\b{re.escape(term)}" for term in IN_SCOPE_TERMS),
        ]
    )
)

SCOPE_SYSTEM_PROMPT = """\
Bir sorunun TÜBİTAK destek programları, Ar-Ge ve yenilik destekleri, burslar, teknoloji \
geliştirme bölgeleri, Ar-Ge/tasarım merkezleri veya bunlarla ilgili mevzuat kapsamında olup \
olmadığını sınıflandırıyorsun. Kapsam içindeyse "kapsam_ici": true, değilse false döndür. \
Yalnızca {"kapsam_ici": true} ya da {"kapsam_ici": false} biçiminde JSON yaz."""


class ScopeResult(BaseModel):
    """Outcome of the scope check and which method decided it."""

    model_config = ConfigDict(frozen=True)

    in_scope: bool
    method: str  # "lexicon" | "llm" | "off"


class _ScopeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kapsam_ici: bool


class ScopeChecker:
    """Decides whether a question is in scope using a term lexicon, an LLM, or both."""

    def __init__(self, mode: ScopeMode, llm: LLM | None = None) -> None:
        if mode in ("llm", "both") and llm is None:
            raise ValueError(f"Kapsam kontrolü '{mode}' modunda bir LLM gerektirir.")
        self.mode = mode
        self.llm = llm

    def check(self, question: str) -> ScopeResult:
        """Return whether ``question`` is in scope."""
        if self.mode == "off":
            return ScopeResult(in_scope=True, method="off")
        if self.mode in ("lexicon", "both") and _LEXICON_RE.search(turkish_lower(question)):
            return ScopeResult(in_scope=True, method="lexicon")
        if self.mode == "lexicon":
            return ScopeResult(in_scope=False, method="lexicon")
        return ScopeResult(in_scope=self._ask_llm(question), method="llm")

    def _ask_llm(self, question: str) -> bool:
        assert self.llm is not None  # guaranteed by __init__
        try:
            reply = self.llm.generate(
                SCOPE_SYSTEM_PROMPT,
                f"Soru: {question}",
                json_schema=_ScopeVerdict.model_json_schema(),
            )
            return parse_json_reply(reply, _ScopeVerdict).kapsam_ici
        except (LlmError, LlmOutputError):
            return True  # fail-open: the retrieval gate still guards the answer
