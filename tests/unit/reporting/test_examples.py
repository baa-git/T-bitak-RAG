"""Tests for picking report examples (FR-X6)."""

from pathlib import Path

from tests.eval_builders import make_result, refused
from tubitak_rag.domain.enums import QuestionType
from tubitak_rag.reporting.examples import pick_examples
from tubitak_rag.reporting.report import load_run


def test_examples_one_good_two_bad(fixtures: Path) -> None:
    _, results = load_run(fixtures / "run_example")
    good, bad = pick_examples(results)
    assert [r.id for r in good] == ["q001"]
    # one answer to an unanswerable question, then one missed / refused answerable question
    assert [r.id for r in bad] == ["q031", "q010"]


def test_bad_examples_alternate_failure_kinds() -> None:
    invented = [
        make_result(qid, tur=QuestionType.CEVAPSIZ, first_hit_rank=None) for qid in ("a", "b")
    ]
    missed = [make_result(qid, first_hit_rank=None) for qid in ("c", "d")]
    _, bad = pick_examples([*invented, *missed], n_good=0, n_bad=3)
    assert [r.id for r in bad] == ["a", "c", "b"]


def test_missed_evidence_with_a_wrong_answer_comes_before_a_right_one() -> None:
    wrong = {"dogru": False, "sadik": True, "atif_dogru": None, "gerekce": ""}
    right = make_result("a", first_hit_rank=None)  # judged correct: answer from another source
    wrong_answer = make_result("b", first_hit_rank=None, judge=wrong)
    _, bad = pick_examples([right, wrong_answer], n_good=0, n_bad=1)
    assert [r.id for r in bad] == ["b"]


def test_examples_fill_from_gate_scores_and_never_repeat() -> None:
    results = [
        make_result(
            "a",
            judge={"dogru": False, "sadik": True, "atif_dogru": None, "gerekce": ""},
            gate_score=0.9,
        ),
        make_result(
            "b",
            judge={"dogru": None, "sadik": None, "atif_dogru": None, "gerekce": ""},
            gate_score=0.6,
        ),
        refused("c", QuestionType.CEVAPSIZ, "esik_alti", gate_score=0.2),
    ]
    good, bad = pick_examples(results, n_good=2)
    assert [r.id for r in good] == ["a", "b"]  # no judged-correct row: highest gate scores
    assert [r.id for r in bad] == ["c"]  # only one row left
    assert not {r.id for r in good} & {r.id for r in bad}


def test_examples_empty() -> None:
    assert pick_examples([]) == ([], [])
