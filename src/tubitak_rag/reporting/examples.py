"""Pick the good and bad questions shown in a report (FR-X6)."""

from collections.abc import Sequence
from itertools import chain, zip_longest

from tubitak_rag.domain.enums import AnswerStatus
from tubitak_rag.domain.evaluation import QuestionResult
from tubitak_rag.evaluation.metrics import UNANSWERABLE, status
from tubitak_rag.evaluation.testset import ANSWERABLE


def _score(result: QuestionResult) -> float:
    return result.gate_score if result.gate_score is not None else float("-inf")


def _is_good(result: QuestionResult) -> bool:
    return (
        result.tur in ANSWERABLE
        and result.first_hit_rank == 1
        and result.judge.get("dogru") is True
    )


def _answered_unanswerable(result: QuestionResult) -> bool:
    return result.tur in UNANSWERABLE and status(result) == AnswerStatus.CEVAPLANDI


def _missed_answerable(result: QuestionResult) -> bool:
    answered = status(result) == AnswerStatus.CEVAPLANDI
    return result.tur in ANSWERABLE and (result.first_hit_rank is None or not answered)


def _is_bad(result: QuestionResult) -> bool:
    return _answered_unanswerable(result) or _missed_answerable(result)


def pick_examples(
    results: Sequence[QuestionResult], n_good: int = 1, n_bad: int = 2
) -> tuple[list[QuestionResult], list[QuestionResult]]:
    """Good: evidence at rank 1 and judged correct. Bad: alternately an answer to a question
    that has no answer in the corpus and an answerable question whose evidence was missed or
    that was refused, so the report shows both failure kinds.

    Missing examples are filled with the highest (good) / lowest (bad) gate scores; a
    question is never shown in both lists.
    """
    ordered = sorted(results, key=lambda r: r.id)
    good = [r for r in ordered if _is_good(r)][:n_good]
    taken = {r.id for r in good}
    if len(good) < n_good:
        fill = [r for r in ordered if r.id not in taken and r.tur in ANSWERABLE and not _is_bad(r)]
        good += sorted(fill, key=_score, reverse=True)[: n_good - len(good)]
        taken = {r.id for r in good}
    invented = [r for r in ordered if _answered_unanswerable(r) and r.id not in taken]
    # A wrong answer or a refusal after a search says most about the configuration; a right
    # answer from another source and a refusal before any search (scope check) come last.
    missed = sorted(
        (r for r in ordered if _missed_answerable(r) and r.id not in taken),
        key=lambda r: (r.judge.get("dogru") is True, not r.retrieved),
    )
    alternating = chain.from_iterable(zip_longest(invented, missed))
    bad = [r for r in alternating if r is not None][:n_bad]
    if len(bad) < n_bad:
        taken |= {r.id for r in bad}
        rest = sorted((r for r in ordered if r.id not in taken), key=_score)
        bad += rest[: n_bad - len(bad)]
    return good, bad
