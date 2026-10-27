"""Token counter backed by a Hugging Face ``tokenizers`` tokenizer (e.g. EmbeddingGemma 2).

Chunk sizes are measured in the embedding model's own tokens, so "512 tokens" means what the
model will actually see. ``tokenizers`` is imported lazily (NFR-TEST2).
"""

from collections.abc import Sequence
from typing import Any


class HfTokenCounter:
    """``TokenCounter`` for the tokenizer of a Hub model id; special tokens are not counted."""

    def __init__(self, model_id: str) -> None:
        from tokenizers import Tokenizer

        self.model_id = model_id
        self._tokenizer: Any = Tokenizer.from_pretrained(model_id)
        # a tokenizer.json may enable truncation/padding; counts must see the whole text
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def encode(self, text: str) -> list[int]:
        """Return the token ids of ``text`` without BOS/EOS tokens."""
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Sequence[int]) -> str:
        """Return the text of ``ids``."""
        return str(self._tokenizer.decode(list(ids)))

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        return len(self.encode(text))
