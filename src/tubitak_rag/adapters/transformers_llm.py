"""LLM run in-process with Hugging Face ``transformers`` (``llm.provider: transformers``).

The Hugging Face Space runs on ZeroGPU: there is no Ollama server and no paid API, so the same
Qwen3 4B model that Ollama serves locally is loaded here from the Hub (``Qwen/Qwen3-4B``,
bfloat16) and generates the answer itself. Generation follows the local setup: the chat
template with the thinking mode off (``enable_thinking=False``, as ``think: false`` in Ollama)
and greedy decoding at temperature 0.

There is no constrained decoding: when a JSON schema is requested it is described in the
system prompt (``json_prompt.with_schema_instruction``) and the outermost JSON object of the
reply is returned; ``generation.schema.parse_json_reply`` validates it afterwards.

``torch`` and ``transformers`` are imported only when a model is loaded. Tests inject a fake
model and tokenizer through ``loader``, so no weights are ever downloaded in unit tests.
"""

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from tubitak_rag.adapters.json_prompt import (
    extract_json_object,
    strip_thinking,
    with_schema_instruction,
)
from tubitak_rag.adapters.st_embedder import resolve_device
from tubitak_rag.ports.llm import LlmError

logger = logging.getLogger(__name__)

DTYPES = ("bfloat16", "float16", "float32", "auto")
DEFAULT_MODEL = "Qwen/Qwen3-4B"

# (model id, concrete device, dtype name) -> (tokenizer, causal LM on that device)
Loader = Callable[[str, str, str], tuple[Any, Any]]


def load_model(model: str, device: str, dtype: str) -> tuple[Any, Any]:
    """Load the tokenizer and the causal LM of ``model`` and move the LM to ``device``.

    On a ZeroGPU Space this runs at import time: ``.to("cuda")`` is recorded by the CUDA
    emulation and the weights reach the GPU when a ``@spaces.GPU`` function starts.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model)
    lm: Any = AutoModelForCausalLM.from_pretrained(model, dtype=torch_dtype)
    lm.to(device)
    lm.eval()
    return tokenizer, lm


@contextmanager
def _inference_mode() -> Iterator[None]:
    """``torch.inference_mode()`` when torch is installed (fake models in tests need none)."""
    try:
        import torch
    except ImportError:
        yield
        return
    with torch.inference_mode():
        yield


def _as_list(ids: Any) -> list[int]:
    return [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]


def _token_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(v) for v in value}


class TransformersLLM:
    """Chat generation with a local ``transformers`` causal LM (Qwen3 by default)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
        think: bool = False,
        loader: Loader | None = None,
    ) -> None:
        if dtype not in DTYPES:
            raise LlmError(f"Geçersiz dtype: {dtype!r} (seçenekler: {DTYPES})")
        self.model_name = model
        self._max_new_tokens = max_new_tokens
        self._think = think
        self._tokenizer, self._model = (loader or load_model)(model, resolve_device(device), dtype)
        self._stop_ids = self._eos_ids()

    def _eos_ids(self) -> set[int]:
        config = getattr(self._model, "generation_config", None)
        ids = _token_ids(getattr(config, "eos_token_id", None))
        return ids | _token_ids(getattr(self._tokenizer, "eos_token_id", None))

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return the reply; with ``json_schema`` the JSON object the model wrote."""
        if json_schema is not None:
            system = with_schema_instruction(system, json_schema)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self._think
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_tokens = len(inputs["input_ids"][0])
        try:
            with _inference_mode():
                output = self._model.generate(
                    **inputs, max_new_tokens=self._max_new_tokens, **self._sampling(temperature)
                )
        except Exception as exc:  # CUDA out of memory, a ZeroGPU time-out, a broken model
            raise LlmError(f"Dil modeli cevap üretemedi ({self.model_name}): {exc}") from exc
        new_ids = _as_list(output[0][prompt_tokens:])
        if self._truncated(new_ids):
            raise LlmError(
                f"LLM yanıtında çıktı kesildi ({self.model_name}, {self._max_new_tokens} token "
                "sınırı); llm.transformers.max_new_tokens değerini artırın."
            )
        text = strip_thinking(self._tokenizer.decode(new_ids, skip_special_tokens=True))
        if not text:
            raise LlmError(f"LLM ({self.model_name}) boş yanıt döndürdü.")
        logger.debug("%s: %d giriş, %d çıkış token", self.model_name, prompt_tokens, len(new_ids))
        return extract_json_object(text) if json_schema is not None else text

    def _sampling(self, temperature: float) -> dict[str, Any]:
        """Greedy decoding at temperature 0; the model's own top-p/top-k are switched off."""
        pad = getattr(self._tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self._tokenizer, "eos_token_id", None)
        if pad is None and self._stop_ids:
            pad = min(self._stop_ids)
        options: dict[str, Any] = {} if pad is None else {"pad_token_id": pad}
        if temperature > 0:
            return {**options, "do_sample": True, "temperature": temperature}
        return {**options, "do_sample": False, "temperature": None, "top_p": None, "top_k": None}

    def _truncated(self, new_ids: Sequence[int]) -> bool:
        return len(new_ids) >= self._max_new_tokens and new_ids[-1] not in self._stop_ids
