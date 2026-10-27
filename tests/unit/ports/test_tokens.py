"""Tests for the tokens port."""

from tests.fakes.tokens import RegexTokenCounter
from tubitak_rag.ports.tokens import TokenCounter


def test_fake_token_counter_satisfies_protocol() -> None:
    assert isinstance(RegexTokenCounter(), TokenCounter)


def test_plain_object_is_not_a_token_counter() -> None:
    assert not isinstance(object(), TokenCounter)
