"""Scripted pipeline for evaluation-runner tests: canned answers and traces per question."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tubitak_rag.domain.answer import Answer
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.guardrails.gate import GateResult


@dataclass
class FakeTrace:
    """Same readable fields as ``AskTrace`` (the runner's ``PipelineTrace``)."""

    candidates: list[ScoredChunk] = field(default_factory=list)
    reranked: list[ScoredChunk] = field(default_factory=list)
    gate: GateResult | None = None
    rejection_reason: str | None = None
    number_check: Any = None  # NumberVerdict (or a stand-in with newer fields)


@dataclass
class Scripted:
    """What the fake pipeline returns and records in the trace for one question."""

    answer: Answer
    candidates: list[ScoredChunk] = field(default_factory=list)
    reranked: list[ScoredChunk] = field(default_factory=list)
    gate: GateResult | None = None
    rejection_reason: str | None = None
    number_check: Any = None  # NumberVerdict (or a stand-in with newer fields)


class FakePipeline:
    """Answers each question text from ``script``; :attr:`calls` lists the questions asked."""

    def __init__(self, script: Mapping[str, Scripted]) -> None:
        self.script = dict(script)
        self.calls: list[str] = []

    def ask(
        self,
        question: str,
        filters: MetadataFilter | None = None,
        *,
        trace: FakeTrace | None = None,
    ) -> Answer:
        self.calls.append(question)
        scripted = self.script[question]
        if trace is not None:
            trace.candidates = list(scripted.candidates)
            trace.reranked = list(scripted.reranked)
            trace.gate = scripted.gate
            trace.rejection_reason = scripted.rejection_reason
            trace.number_check = scripted.number_check
        return scripted.answer
