"""Tests for the Hugging Face tokenizer adapter with a fake ``tokenizers`` module."""

import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from tubitak_rag.ports import TokenCounter


@dataclass
class _Encoding:
    ids: list[int]


class _FakeTokenizer:
    loaded: list[str] = []

    def __init__(self) -> None:
        self.truncation = True  # a tokenizer.json may ship with truncation/padding enabled
        self.padding = True

    def no_truncation(self) -> None:
        self.truncation = False

    def no_padding(self) -> None:
        self.padding = False

    @classmethod
    def from_pretrained(cls, model_id: str) -> "_FakeTokenizer":
        cls.loaded.append(model_id)
        return cls()

    def encode(self, text: str, add_special_tokens: bool = True) -> _Encoding:
        ids = [len(word) for word in text.split()]
        return _Encoding(ids=[2, *ids] if add_special_tokens else ids)

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join("x" * i for i in ids)


@pytest.fixture
def fake_tokenizers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("tokenizers")
    module.Tokenizer = _FakeTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tokenizers", module)
    _FakeTokenizer.loaded = []


def test_module_imports_without_tokenizers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tokenizers", None)  # any import of it would fail
    monkeypatch.delitem(sys.modules, "tubitak_rag.adapters.hf_tokenizer", raising=False)
    import tubitak_rag.adapters.hf_tokenizer  # noqa: F401


@pytest.mark.usefixtures("fake_tokenizers")
def test_hf_counter_loads_model_lazily_and_counts_without_special_tokens() -> None:
    from tubitak_rag.adapters.hf_tokenizer import HfTokenCounter

    counter = HfTokenCounter("google/embeddinggemma-2")
    assert _FakeTokenizer.loaded == ["google/embeddinggemma-2"]
    assert isinstance(counter, TokenCounter)
    assert counter.count("a bb") == 2
    assert counter.encode("a bb") == [1, 2]
    assert counter.decode([1, 2]) == "x xx"


@pytest.mark.usefixtures("fake_tokenizers")
def test_hf_counter_disables_truncation_and_padding() -> None:
    from tubitak_rag.adapters.hf_tokenizer import HfTokenCounter

    tokenizer = HfTokenCounter("x")._tokenizer
    assert (tokenizer.truncation, tokenizer.padding) == (False, False)
