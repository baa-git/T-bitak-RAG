"""Tests for the shared immutable domain base model."""

import pytest
from pydantic import ValidationError

from tubitak_rag.domain.base import DomainModel


class _Point(DomainModel):
    x: int


def test_domain_models_are_frozen() -> None:
    point = _Point(x=1)
    with pytest.raises(ValidationError):
        point.x = 2  # type: ignore[misc]


def test_frozen_domain_models_are_hashable_values() -> None:
    assert _Point(x=1) == _Point(x=1)
    assert hash(_Point(x=1)) == hash(_Point(x=1))
