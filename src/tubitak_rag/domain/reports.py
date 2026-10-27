"""Report models used when comparing a run with the previous rung of the ladder."""

from tubitak_rag.domain.base import DomainModel


class ComparisonRow(DomainModel):
    """One metric of the current run next to the previous run and their difference."""

    metric: str
    current: float
    previous: float | None
    delta: float | None
