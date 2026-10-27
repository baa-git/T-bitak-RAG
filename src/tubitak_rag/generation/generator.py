"""Answer generator: numbered sources -> LLM (JSON schema) -> parsed output (FR-G1..G3)."""

import time
from collections.abc import Sequence

from pydantic import BaseModel

from tubitak_rag.domain.answer import Citation, SourceView
from tubitak_rag.domain.retrieval import ScoredChunk
from tubitak_rag.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_sources,
    order_sources,
    source_location,
)
from tubitak_rag.generation.schema import (
    LlmOutput,
    LlmOutputError,
    cited_numbers,
    output_json_schema,
    parse_llm_output,
)
from tubitak_rag.ports import LLM

INVALID_OUTPUT = "gecersiz_cikti"


class GenerationResult(BaseModel):
    """One generation: the parsed output (or an error code) and the sources in prompt order."""

    output: LlmOutput | None
    error: str | None
    sources: list[ScoredChunk]  # numbered [1..n] in this order
    prompt_chars: int
    latency_ms: float


class AnswerGenerator:
    """Builds the grounded prompt, calls the LLM with the output schema and parses the reply."""

    def __init__(self, llm: LLM, temperature: float = 0.0) -> None:
        self.llm = llm
        self.temperature = temperature

    def generate(self, question: str, results: Sequence[ScoredChunk]) -> GenerationResult:
        """Answer ``question`` from ``results``; an unparsable reply becomes ``error``."""
        sources = order_sources(results)
        user = build_user_prompt(question, format_sources(sources))
        started = time.perf_counter()
        reply = self.llm.generate(
            SYSTEM_PROMPT, user, json_schema=output_json_schema(), temperature=self.temperature
        )
        latency_ms = (time.perf_counter() - started) * 1000
        output: LlmOutput | None = None
        error: str | None = None
        try:
            output = parse_llm_output(reply)
        except LlmOutputError:
            error = INVALID_OUTPUT
        return GenerationResult(
            output=output,
            error=error,
            sources=sources,
            prompt_chars=len(SYSTEM_PROMPT) + len(user),
            latency_ms=latency_ms,
        )


def citations_from(output: LlmOutput, sources: Sequence[ScoredChunk]) -> list[Citation]:
    """Map cited numbers (``atiflar`` and inline ``[k]``) to source metadata (FR-G3).

    Numbers are sorted and de-duplicated; numbers outside ``1..len(sources)`` are dropped.
    """
    citations: list[Citation] = []
    for number in sorted(cited_numbers(output)):
        if not 1 <= number <= len(sources):
            continue
        chunk = sources[number - 1].chunk
        citations.append(
            Citation(
                no=number,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                konum=source_location(chunk),
                kaynak_url=chunk.kaynak_url,
            )
        )
    return citations


def source_views(sources: Sequence[ScoredChunk]) -> list[SourceView]:
    """The numbered source panel, numbered exactly as in the prompt."""
    return [
        SourceView(
            no=number,
            chunk_id=result.chunk.chunk_id,
            doc_title=result.chunk.doc_title,
            konum=source_location(result.chunk),
            kaynak_url=result.chunk.kaynak_url,
            text=result.chunk.text,
            score=result.score,
        )
        for number, result in enumerate(sources, start=1)
    ]
