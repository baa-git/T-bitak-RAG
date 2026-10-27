"""Tests for the in-process ``transformers`` LLM with a fake tokenizer and model (no weights)."""

import json
import sys
import types
from typing import Any

import pytest

from tubitak_rag.adapters.json_prompt import SCHEMA_INSTRUCTION
from tubitak_rag.adapters.transformers_llm import TransformersLLM
from tubitak_rag.config import LlmConfig, TransformersConfig
from tubitak_rag.factory import build_llm
from tubitak_rag.ports import LLM
from tubitak_rag.ports.llm import LlmError

SCHEMA = {"type": "object", "properties": {"cevap": {"type": "string"}}}
EOS = 99
PROMPT_IDS = [1, 2, 3]


class _Batch(dict[str, Any]):
    """Stands in for ``BatchEncoding``: a mapping with ``.to(device)``."""

    def to(self, device: str) -> "_Batch":
        self["device"] = device
        return self


class FakeTokenizer:
    """Chat template -> recorded text; reply ids decode through ``vocab``."""

    eos_token_id = EOS
    pad_token_id: int | None = None

    def __init__(self, vocab: dict[int, str]) -> None:
        self.vocab = vocab
        self.template_calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.template_calls.append({"messages": messages, **kwargs})
        return "PROMPT"

    def __call__(self, text: str, return_tensors: str) -> _Batch:
        return _Batch(input_ids=[list(PROMPT_IDS)], attention_mask=[[1, 1, 1]])

    def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "".join(self.vocab.get(i, "") for i in ids if i != EOS)


class FakeModel:
    """Returns the prompt followed by ``reply``; records every ``generate`` call."""

    device = "cuda"

    def __init__(self, reply: list[int], error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.generation_config = types.SimpleNamespace(eos_token_id=[EOS, 98])
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return [kwargs["input_ids"][0] + self.reply]


def _llm(
    vocab: dict[int, str], reply: list[int], *, max_new_tokens: int = 16, **kwargs: Any
) -> tuple[TransformersLLM, FakeTokenizer, FakeModel]:
    tokenizer, model = FakeTokenizer(vocab), FakeModel(reply, kwargs.pop("error", None))
    seen: dict[str, str] = {}

    def loader(name: str, device: str, dtype: str) -> tuple[Any, Any]:
        seen.update(name=name, device=device, dtype=dtype)
        return tokenizer, model

    llm = TransformersLLM(
        "Qwen/Qwen3-4B", device="cpu", max_new_tokens=max_new_tokens, loader=loader, **kwargs
    )
    assert seen == {"name": "Qwen/Qwen3-4B", "device": "cpu", "dtype": "bfloat16"}
    return llm, tokenizer, model


def test_satisfies_the_llm_port() -> None:
    llm, _, _ = _llm({10: "x"}, [10, EOS])
    assert isinstance(llm, LLM) and llm.model_name == "Qwen/Qwen3-4B"


def test_plain_generation_uses_chat_template_without_thinking_and_greedy_decoding() -> None:
    llm, tokenizer, model = _llm({10: "Merhaba", 11: " dünya"}, [10, 11, EOS])

    assert llm.generate("sistem", "soru") == "Merhaba dünya"

    (call,) = tokenizer.template_calls
    assert call["messages"] == [
        {"role": "system", "content": "sistem"},
        {"role": "user", "content": "soru"},
    ]
    assert call["enable_thinking"] is False and call["add_generation_prompt"] is True
    assert call["tokenize"] is False
    (generate,) = model.calls
    assert generate["do_sample"] is False and generate["temperature"] is None
    assert generate["top_p"] is None and generate["top_k"] is None
    assert generate["max_new_tokens"] == 16 and generate["pad_token_id"] == EOS
    assert generate["device"] == "cuda"  # inputs moved to the model's device


def test_positive_temperature_samples() -> None:
    llm, _, model = _llm({10: "x"}, [10, EOS])
    llm.generate("s", "u", temperature=0.7)
    assert model.calls[0]["do_sample"] is True and model.calls[0]["temperature"] == 0.7


def test_think_flag_reaches_the_chat_template() -> None:
    llm, tokenizer, _ = _llm({10: "x"}, [10, EOS], think=True)
    llm.generate("s", "u")
    assert tokenizer.template_calls[0]["enable_thinking"] is True


def test_schema_goes_into_the_system_prompt_and_the_json_object_is_returned() -> None:
    reply = 'Elbette: ```json\n{"cevap": "x", "ic": {"a": 1}}\n``` umarım yardımcı olur'
    llm, tokenizer, _ = _llm({10: reply}, [10, EOS])

    text = llm.generate("s", "u", json_schema=SCHEMA)

    assert json.loads(text) == {"cevap": "x", "ic": {"a": 1}}
    system = tokenizer.template_calls[0]["messages"][0]["content"]
    assert system == "s" + SCHEMA_INSTRUCTION.format(schema=json.dumps(SCHEMA, ensure_ascii=False))


def test_think_block_is_removed() -> None:
    llm, _, _ = _llm({10: "<think>\nplan\n</think>\n\n", 11: '{"cevap": "x"}'}, [10, 11, EOS])
    assert llm.generate("s", "u", json_schema=SCHEMA) == '{"cevap": "x"}'


def test_reply_cut_at_the_token_limit_is_an_error() -> None:
    llm, _, _ = _llm({10: "x"}, [10, 10, 10], max_new_tokens=3)
    with pytest.raises(LlmError, match="max_new_tokens"):
        llm.generate("s", "u")


def test_reply_ending_with_eos_at_the_limit_is_complete() -> None:
    llm, _, _ = _llm({10: "x"}, [10, 10, 98], max_new_tokens=3)  # 98: model's second EOS id
    assert llm.generate("s", "u") == "xx"


def test_empty_reply_is_an_error() -> None:
    llm, _, _ = _llm({}, [EOS])
    with pytest.raises(LlmError, match="boş yanıt"):
        llm.generate("s", "u")


def test_generation_failure_becomes_llm_error() -> None:
    llm, _, _ = _llm({10: "x"}, [10], error=RuntimeError("CUDA out of memory"))
    with pytest.raises(LlmError, match="CUDA out of memory"):
        llm.generate("s", "u")


def test_invalid_dtype_is_rejected() -> None:
    with pytest.raises(LlmError, match="dtype"):
        TransformersLLM("m", dtype="int3", loader=lambda *_: (None, None))


def test_factory_builds_it_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """``llm.model`` is the Hub id; device, dtype and token limit come from ``llm.transformers``."""
    seen: dict[str, Any] = {}

    class _Auto:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> Any:
            seen.setdefault("loaded", []).append((cls.__name__, name, kwargs))
            return FakeModel([EOS]) if cls.__name__ == "AutoModelForCausalLM" else FakeTokenizer({})

    module = types.ModuleType("transformers")
    module.AutoTokenizer = type("AutoTokenizer", (_Auto,), {})  # type: ignore[attr-defined]
    module.AutoModelForCausalLM = type("AutoModelForCausalLM", (_Auto,), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(
        FakeModel, "to", lambda self, device: seen.update(device=device), raising=False
    )
    monkeypatch.setattr(FakeModel, "eval", lambda self: None, raising=False)
    torch = types.ModuleType("torch")
    torch.float16 = "fp16"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delenv("TUBITAK_RAG_DEVICE", raising=False)
    cfg = LlmConfig(
        provider="transformers",
        model="Qwen/Qwen3-0.6B",
        transformers=TransformersConfig(device="cpu", dtype="float16", max_new_tokens=64),
    )

    llm = build_llm(cfg)

    assert isinstance(llm, TransformersLLM) and llm.model_name == "Qwen/Qwen3-0.6B"
    assert seen["device"] == "cpu"
    assert ("AutoModelForCausalLM", "Qwen/Qwen3-0.6B", {"dtype": "fp16"}) in seen["loaded"]
