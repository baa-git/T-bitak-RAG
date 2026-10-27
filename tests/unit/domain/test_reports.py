"""Tests for report comparison rows."""

from tubitak_rag.domain.reports import ComparisonRow


def test_comparison_row_allows_missing_previous_run() -> None:
    row = ComparisonRow(metric="recall_at_5", current=0.6, previous=None, delta=None)
    assert row.previous is None and row.delta is None


def test_comparison_row_round_trips() -> None:
    row = ComparisonRow(metric="mrr", current=0.7, previous=0.5, delta=0.2)
    assert ComparisonRow.model_validate_json(row.model_dump_json()) == row
