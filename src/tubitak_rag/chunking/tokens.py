"""Dependency-free token counter for tests (``tokenizer: regex``).

Test-only: real chunk sizes must be measured with the embedding model's own tokenizer
(``tokenizer: hf:<model>``, ``adapters/hf_tokenizer.py``). This counter only approximates
it (a token is a word or one punctuation mark; the EmbeddingGemma tokenizer counts ~1.8x as
many tokens on this corpus) and decoding does not keep the original spacing.
"""

import re
from collections.abc import Sequence

_TOKEN = re.compile(r"\w+|[^\w\s]")


class RegexTokenCounter:
    """A token is a word or one punctuation mark; ids index a vocabulary grown on demand."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._tokens: list[str] = []

    def encode(self, text: str) -> list[int]:
        """Return the ids of the tokens of ``text`` (new tokens extend the vocabulary)."""
        ids = []
        for token in _TOKEN.findall(text):
            if token not in self._vocab:
                self._vocab[token] = len(self._tokens)
                self._tokens.append(token)
            ids.append(self._vocab[token])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Return the tokens joined by single spaces (original spacing is not kept)."""
        return " ".join(self._tokens[i] for i in ids)

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        return len(_TOKEN.findall(text))
