"""Answer judging: an LLM judge for correctness/faithfulness plus deterministic rule checks.

The judge (FR-E7, FR-E8) compares the generated answer with the expected answer and with the
texts of the sources the answer cites. A judge is itself a model that can be wrong, so its
name is printed in every report and two cheap rules run next to it:

* ``numbers_present`` — every expected number (rate, amount, duration) occurs in the answer;
* ``citation_precision`` — the share of cited chunks that are actually relevant.

A judge failure never aborts a run: the verdict becomes ``(None, None, "hakem_hatasi")`` and
the metric falls back to the rules.
"""

import logging
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from tubitak_rag.domain.answer import Answer
from tubitak_rag.domain.evaluation import TestQuestion
from tubitak_rag.domain.retrieval import ScoredChunk
from tubitak_rag.evaluation.matching import quote_in_chunk
from tubitak_rag.evaluation.testset import contains_number
from tubitak_rag.generation.schema import parse_json_reply
from tubitak_rag.ports import LLM

logger = logging.getLogger(__name__)

JUDGE_ERROR = "hakem_hatasi"

JUDGE_SYSTEM_PROMPT = """\
Sen TÜBİTAK destek programları ve Ar-Ge mevzuatı konusunda titiz bir değerlendiricisin.
Sana bir soru, uzmanın yazdığı beklenen cevap, bir asistanın ürettiği cevap ve asistanın atıf
yaptığı kaynak metinleri verilecek. İki ayrı karar ver:

1. "dogru": Üretilen cevap, beklenen cevabın özünü (oranlar, tutarlar, süreler, koşullar,
   program adları) doğru veriyor mu? Ek ama doğru bilgi cevabı yanlış yapmaz; eksik ya da
   çelişen bir temel bilgi yanlış yapar. Sayılar birebir aynı olmalıdır ("%75" ile
   "yüzde 75" aynıdır).
2. "sadik": Üretilen cevaptaki her iddia, verilen kaynak metinlerinde dayanağı olan bir
   bilgi mi? Kaynaklarda olmayan bir iddia (doğru olsa bile) sadakati bozar. Kaynak metni
   verilmemişse "sadik" false olur.

Kararını tek cümlelik Türkçe bir gerekçeyle açıkla. Yalnızca şu biçimde bir JSON nesnesi
yaz, başka hiçbir şey yazma: {"dogru": true/false, "sadik": true/false, "gerekce": "..."}
"""

_THOUSANDS_RE = re.compile(r"(?<=\d)\.(?=\d{3}(?!\d))")


class _JudgeReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dogru: bool
    sadik: bool
    gerekce: str


class JudgeVerdict(BaseModel):
    """Judge decision; ``None`` fields mean the judge could not decide."""

    model_config = ConfigDict(frozen=True)

    dogru: bool | None
    sadik: bool | None
    gerekce: str


def build_judge_prompt(question: TestQuestion, answer: Answer, cited_texts: Sequence[str]) -> str:
    """User prompt for the judge: question, expected answer, generated answer and sources."""
    sources = "\n\n".join(f"<kaynak>\n{text}\n</kaynak>" for text in cited_texts)
    return (
        f"<soru>\n{question.soru}\n</soru>\n\n"
        f"<beklenen_cevap>\n{question.beklenen_cevap}\n</beklenen_cevap>\n\n"
        f"<uretilen_cevap>\n{answer.cevap}\n</uretilen_cevap>\n\n"
        f"<atif_yapilan_kaynaklar>\n{sources or '(kaynak yok)'}\n</atif_yapilan_kaynaklar>"
    )


class LlmJudge:
    """LLM-as-a-judge with a JSON verdict (temperature 0 for repeatable runs)."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    @property
    def model_name(self) -> str:
        """Name of the judging model, printed in every report (FR-E8)."""
        return self._llm.model_name

    def judge(
        self, question: TestQuestion, answer: Answer, cited_texts: Sequence[str]
    ) -> JudgeVerdict:
        """Judge one answer; any LLM or parsing failure yields an undecided verdict."""
        try:
            raw = self._llm.generate(
                JUDGE_SYSTEM_PROMPT,
                build_judge_prompt(question, answer, cited_texts),
                json_schema=_JudgeReply.model_json_schema(),
                temperature=0.0,
            )
            reply = parse_json_reply(raw, _JudgeReply)
        except Exception as exc:  # one bad verdict must not abort a long run
            logger.warning("Hakem %s sorusunda karar veremedi: %s", question.id, exc)
            return JudgeVerdict(dogru=None, sadik=None, gerekce=JUDGE_ERROR)
        return JudgeVerdict(dogru=reply.dogru, sadik=reply.sadik, gerekce=reply.gerekce)


def _canonical_numbers(text: str) -> str:
    """Drop percent signs and thousands dots so "%75" ~ "75" and "2.300.000" ~ "2300000"."""
    return _THOUSANDS_RE.sub("", text.replace("%", " "))


def numbers_present(expected: Sequence[str], answer_text: str) -> bool | None:
    """Whether every expected number occurs in the answer; None when none is expected."""
    if not expected:
        return None
    text = _canonical_numbers(answer_text)
    return all(contains_number(text, _canonical_numbers(number).strip()) for number in expected)


def citation_precision(
    answer: Answer, question: TestQuestion, retrieved: Sequence[ScoredChunk]
) -> float | None:
    """Share of cited chunks from an expected document or containing an evidence quote."""
    if not answer.atiflar:
        return None
    texts = {r.chunk.chunk_id: r.chunk.text for r in retrieved}
    expected = set(question.beklenen_belgeler)

    def relevant(chunk_id: str, doc_id: str) -> bool:
        text = texts.get(chunk_id, "")
        return doc_id in expected or any(
            quote.doc_id == doc_id and quote_in_chunk(quote.alinti, text)
            for quote in question.kanit_alintilari
        )

    hits = sum(relevant(c.chunk_id, c.doc_id) for c in answer.atiflar)
    return hits / len(answer.atiflar)


def programs_split(question: TestQuestion, answer_text: str) -> bool | None:
    """For ambiguous questions: does the answer name at least two of the programs? (FR-GR8)"""
    if len(question.program_kodu) < 2:
        return None
    named = sum(contains_number(answer_text, code) for code in question.program_kodu)
    return named >= 2
