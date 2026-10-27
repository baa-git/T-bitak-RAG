"""Question answering pipeline: scope -> retrieval -> rerank -> gate -> LLM -> guardrails.

Every guardrail is a separate step that can only *narrow* what reaches the user, and the
cheap deterministic checks run before the expensive model call:

1. scope check (lexicon / LLM)          -> KAPSAM_DISI, no retrieval
2. retrieval (``k_candidates``)
3. reranking of all candidates (optional), the first ``top_n`` go on
4. retrieval score gate                 -> BILGI_BULUNAMADI, the LLM is not called (FR-GR1)
5. generation (JSON schema)             -> LlmError / invalid JSON: BILGI_BULUNAMADI
6. model said "not answerable"          -> BILGI_BULUNAMADI
7. citation validation                  -> ATIF_REDDI with the "bilgi bulunamadı" message
8. number grounding (off|strip|reject)  -> a number found only in an uncited prompt source
                                           adds that citation (repair); a number in no
                                           source: strip / BILGI_BULUNAMADI
9. post-processing: announcement link (cited chunks), missing part

Every refusal returns no sources and no citations (FR-GR2). The disclaimer is the ``uyari``
field of every answer, never repeated inside ``cevap`` (FR-G5). ``AskTrace`` records what each
stage saw and decided, for the evaluation runner and for ``rag ask --trace``.
"""

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from pydantic import BaseModel

from tubitak_rag.domain.answer import Answer
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.enums import AnswerStatus
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.generation.generator import (
    INVALID_OUTPUT,
    AnswerGenerator,
    GenerationResult,
    citations_from,
    source_views,
)
from tubitak_rag.generation.prompts import BILGI_BULUNAMADI_MESAJI, KAPSAM_DISI_MESAJI
from tubitak_rag.generation.schema import cited_numbers
from tubitak_rag.guardrails.citations import (
    INVALID_CITATION,
    UNCITED,
    CitationVerdict,
    validate,
)
from tubitak_rag.guardrails.gate import GateResult, RetrievalGate
from tubitak_rag.guardrails.numbers import (
    UNGROUNDED_NUMBER,
    NumberCheckMode,
    NumberVerdict,
    check_numbers,
)
from tubitak_rag.guardrails.postprocess import append_missing_part, attach_announcement_link
from tubitak_rag.guardrails.scope import ScopeChecker, ScopeResult
from tubitak_rag.ports import Reranker, Retriever, VectorStore
from tubitak_rag.ports.llm import LlmError
from tubitak_rag.reranking.cross_encoder import rerank_with

LLM_HATASI_MESAJI = (
    "Dil modelinden şu anda cevap alınamadı (bağlantı hatası ya da yarım kalan çıktı). "
    "Lütfen biraz sonra yeniden deneyin."
)

# Refusal reason codes (``AskTrace.rejection_reason``, the evaluation runner's ``red_nedeni``)
# and their Turkish explanations.
OUT_OF_SCOPE = "kapsam_disi"
NO_RESULTS = "sonuc_yok"
BELOW_THRESHOLD = "esik_alti"
LLM_FAILURE = "llm_hatasi"
NOT_ANSWERABLE = "cevaplanamaz"
REFUSAL_REASONS: dict[str, str] = {
    OUT_OF_SCOPE: "soru kapsam dışı bulundu",
    NO_RESULTS: "arama hiçbir sonuç döndürmedi; dil modeli çağrılmadı",
    BELOW_THRESHOLD: "en iyi arama skoru eşiğin altında; dil modeli çağrılmadı",
    LLM_FAILURE: "dil modeli hata verdi",
    INVALID_OUTPUT: "dil modelinin çıktısı beklenen JSON şemasına uymadı",
    NOT_ANSWERABLE: "dil modeli kaynakları soruyu cevaplamaya yetersiz buldu",
    UNCITED: "cevapta hiç atıf yoktu",
    INVALID_CITATION: "cevap getirilmemiş bir kaynağa atıf yaptı",
    UNGROUNDED_NUMBER: "cevaptaki bir sayı alıntılanan kaynaklarda bulunamadı",
}


class AskTrace(BaseModel):
    """What each stage of one ``ask`` saw and decided (filled in place when passed).

    ``reranked`` holds *all* candidates in reranker order (empty without a reranker), so the
    evaluation can compute recall@10 even when only ``top_n`` sources reach the prompt.
    """

    candidates: list[ScoredChunk] = []
    reranked: list[ScoredChunk] = []
    gate: GateResult | None = None
    scope: ScopeResult | None = None
    generation: GenerationResult | None = None  # raw parsed output and prompt-order sources
    citation: CitationVerdict | None = None
    number_check: NumberVerdict | None = None
    atif_onarimi: list[int] = []  # source numbers added by citation repair (number check)
    rejection_reason: str | None = None  # code from REFUSAL_REASONS; None when answered
    error: str | None = None  # LlmError message when the generation call failed


class _Timer:
    """Wall-clock milliseconds per stage; only stages that ran get a key."""

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self.ms: dict[str, float] = {}

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.ms[stage] = (time.perf_counter() - started) * 1000

    def total(self) -> dict[str, float]:
        return {**self.ms, "total": (time.perf_counter() - self._started) * 1000}


class RagPipeline:
    """Answers one question from the indexed corpus with every guardrail applied."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        reranker: Reranker | None,
        generator: AnswerGenerator,
        scope: ScopeChecker,
        gate: RetrievalGate,
        k_candidates: int,
        top_n: int,
        disclaimer: str,
        number_check: NumberCheckMode = "strip",
        store: VectorStore | None = None,
        collection: str | None = None,
    ) -> None:
        if top_n <= 0:
            raise ValueError(f"top_n pozitif olmalıdır, verilen: {top_n}")
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.scope = scope
        self.gate = gate
        self.k_candidates = k_candidates
        self.top_n = top_n
        self.disclaimer = disclaimer
        self.number_check: NumberCheckMode = number_check
        self.store = store
        self.collection = collection

    def ask(
        self,
        question: str,
        filters: MetadataFilter | None = None,
        *,
        trace: AskTrace | None = None,
    ) -> Answer:
        """Answer ``question``; ``trace`` (if given) is filled with every stage's result."""
        trace = AskTrace() if trace is None else trace
        timer = _Timer()
        llm_calls = 0

        def refuse(durum: AnswerStatus, message: str, reason: str) -> Answer:
            trace.rejection_reason = reason
            return self._answer(question, durum, message, timer, llm_calls, trace.gate)

        with timer("scope"):
            trace.scope = self.scope.check(question)
        llm_calls += trace.scope.method == "llm"
        if not trace.scope.in_scope:
            return refuse(AnswerStatus.KAPSAM_DISI, KAPSAM_DISI_MESAJI, OUT_OF_SCOPE)

        with timer("retrieval"):
            trace.candidates = self.retriever.retrieve(
                question, k=self.k_candidates, filters=filters
            )
        results = trace.candidates[: self.top_n]
        if self.reranker is not None:
            with timer("rerank"):
                size = max(self.top_n, len(trace.candidates))
                trace.reranked = rerank_with(self.reranker, question, trace.candidates, size)
            results = trace.reranked[: self.top_n]

        trace.gate = self.gate.evaluate(results)
        if not trace.gate.passed:
            reason = BELOW_THRESHOLD if trace.gate.key else NO_RESULTS
            return refuse(AnswerStatus.BILGI_BULUNAMADI, BILGI_BULUNAMADI_MESAJI, reason)

        llm_calls += 1
        try:
            with timer("llm"):
                generation = self.generator.generate(question, results)
        except LlmError as exc:
            trace.error = str(exc)
            return refuse(AnswerStatus.BILGI_BULUNAMADI, LLM_HATASI_MESAJI, LLM_FAILURE)
        trace.generation = generation
        output = generation.output
        if output is None:
            reason = generation.error or INVALID_OUTPUT
            return refuse(AnswerStatus.BILGI_BULUNAMADI, BILGI_BULUNAMADI_MESAJI, reason)
        if not output.cevaplanabilir:
            return refuse(AnswerStatus.BILGI_BULUNAMADI, BILGI_BULUNAMADI_MESAJI, NOT_ANSWERABLE)

        sources = generation.sources
        trace.citation = validate(output, len(sources))
        if not trace.citation.ok:
            reason = trace.citation.reason or INVALID_CITATION
            return refuse(AnswerStatus.ATIF_REDDI, BILGI_BULUNAMADI_MESAJI, reason)

        cited = [sources[number - 1].chunk for number in sorted(cited_numbers(output))]
        trace.number_check = check_numbers(
            output,
            question,
            cited,
            mode=self.number_check,
            sources=[source.chunk for source in sources],
        )
        if not trace.number_check.ok:
            reason = trace.number_check.reason or UNGROUNDED_NUMBER
            return refuse(AnswerStatus.BILGI_BULUNAMADI, BILGI_BULUNAMADI_MESAJI, reason)
        trace.atif_onarimi = trace.number_check.repaired_citations
        final = output.model_copy(
            update={
                "cevap": trace.number_check.cevap,
                "eksik_kisim": trace.number_check.eksik_kisim,
                "atiflar": trace.number_check.atiflar,
            }
        )

        citations = citations_from(final, sources)
        cited = [sources[citation.no - 1].chunk for citation in citations]
        text = attach_announcement_link(final.cevap, cited)
        text = append_missing_part(text, final.eksik_kisim)
        return Answer(
            soru=question,
            durum=AnswerStatus.CEVAPLANDI,
            cevaplanabilir=True,
            cevap=text,
            atiflar=citations,
            kaynaklar=source_views(sources),
            uyari=self.disclaimer,
            gecikme_ms=timer.total(),
            llm_cagri_sayisi=llm_calls,
            gate_key=trace.gate.key,
            gate_score=trace.gate.score,
            atif_onarimi=trace.atif_onarimi,
        )

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Stored chunks of the pipeline's collection (web source panel context)."""
        if self.store is None or self.collection is None:
            return []
        return self.store.get_chunks(self.collection, chunk_ids)

    def _answer(
        self,
        question: str,
        durum: AnswerStatus,
        message: str,
        timer: _Timer,
        llm_calls: int,
        gate: GateResult | None,
    ) -> Answer:
        """A refusal: fixed message, no citations and no sources (FR-GR2)."""
        return Answer(
            soru=question,
            durum=durum,
            cevaplanabilir=False,
            cevap=message,
            atiflar=[],
            kaynaklar=[],
            uyari=self.disclaimer,
            gecikme_ms=timer.total(),
            llm_cagri_sayisi=llm_calls,
            gate_key=gate.key if gate else None,
            gate_score=gate.score if gate else None,
        )
