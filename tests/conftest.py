"""Shared pytest fixtures for the whole test suite."""

from pathlib import Path

import pytest

from tests.fakes import register_fakes


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the "fake" providers in the factory registries for one test only."""
    register_fakes(monkeypatch.setitem)


@pytest.fixture
def fixtures(repo_root: Path) -> Path:
    """Return the directory of small committed test fixtures."""
    return repo_root / "tests" / "fixtures"
