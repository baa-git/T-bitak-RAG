"""Tests for the dependency-free regex token counter."""

from tests.fakes.tokens import RegexTokenCounter as FakeExport
from tubitak_rag.chunking.tokens import RegexTokenCounter
from tubitak_rag.ports import TokenCounter


def test_regex_counter_counts_words_and_punct() -> None:
    assert RegexTokenCounter().count("Madde 7 - (1)") == 6


def test_regex_counter_round_trips_tokens() -> None:
    counter = RegexTokenCounter()
    ids = counter.encode("destek oranı %75.")
    assert len(ids) == counter.count("destek oranı %75.") == 5
    assert counter.decode(ids) == "destek oranı % 75 ."
    assert counter.decode(counter.encode("oranı destek")) == "oranı destek"


def test_regex_counter_satisfies_port_and_is_the_test_fake() -> None:
    assert isinstance(RegexTokenCounter(), TokenCounter)
    assert FakeExport is RegexTokenCounter
