"""Typed application configuration loaded from YAML.

Every pipeline choice (chunker, embedder, retriever, reranker, LLM, guardrails, evaluation)
lives in one validated ``AppConfig``. YAML files may ``extends:`` a base file and use
``${VAR:default}`` environment interpolation; secrets never live in YAML (NFR-CONF2).

Interpolation happens exactly once: on the merged YAML data inside :func:`load_config`, and
on URL defaults declared in code when they are created. A value read from the environment is
never interpolated again. ``$${`` is an escape that yields a literal ``${``.
"""

import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, ValidationError
from pydantic_core import ErrorDetails

# "$${" is an escape for a literal "${"; otherwise ${VAR} or ${VAR:default}.
_ENV_PATTERN = re.compile(r"\$(?:(\$)\{|\{(\w+)(?::([^}]*))?\})")
_EXTENDS_KEY = "extends"


class ConfigError(Exception):
    """Raised when a configuration file cannot be read or is invalid (message in Turkish)."""


def interpolate_env(value: str) -> str:
    """Resolve ``${VAR}`` / ``${VAR:default}`` from the environment in a single pass.

    Unset variables without a default become ``""``. ``$${`` yields a literal ``${``.
    Substituted values are inserted verbatim (never interpolated again).
    """

    def _replace(match: re.Match[str]) -> str:
        if match.group(1):
            return "${"
        return os.environ.get(match.group(2), match.group(3) or "")

    return _ENV_PATTERN.sub(_replace, value)


def _escape_placeholders(value: str) -> str:
    """Inverse of :func:`interpolate_env` for literal text: ``${`` -> ``$${``."""
    return value.replace("${", "$${")


def _env_default(template: str) -> Any:
    """Default factory that resolves ``template`` when the config object is created."""
    return Field(default_factory=lambda: interpolate_env(template))


# A path that is written with forward slashes in YAML/JSON on every OS.
PortablePath = Annotated[Path, PlainSerializer(lambda p: p.as_posix(), when_used="json")]


class _Section(BaseModel):
    """Base for all config sections: unknown keys are errors (NFR-CONF1)."""

    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(_Section):
    """Identity of an experiment run in the ladder (v0 … v3b)."""

    name: str = "default"
    description: str = ""
    concept: str = "full_pipeline"
    previous: str | None = None


class PathsConfig(_Section):
    """Filesystem locations, relative to the working directory."""

    data_dir: PortablePath = Path("data")
    raw_dir: PortablePath = Path("data/raw")
    manifests_dir: PortablePath = Path("data/manifests")
    artifacts_dir: PortablePath = Path("artifacts")
    testset: PortablePath = Path("eval/testset_v1.jsonl")


class EmbedderConfig(_Section):
    """Embedding model selection."""

    provider: Literal["sentence_transformers", "ollama", "fake"] = "sentence_transformers"
    model: str = "google/embeddinggemma-2"
    dimension: int = 768
    device: str = "auto"
    batch_size: int = 32
    text_only: bool = True
    cache: bool = True
    url: str = _env_default("${OLLAMA_URL:http://localhost:11434}")


class FixedChunkConfig(_Section):
    """Window parameters of the fixed-size chunker (in tokens)."""

    size: int = 512
    overlap: int = 64


class ChunkerConfig(_Section):
    """Chunking strategy."""

    name: Literal["fixed", "structural"] = "structural"
    max_tokens: int = 450
    context_header: bool = True
    tokenizer: str = "hf:google/embeddinggemma-2"
    fixed: FixedChunkConfig = Field(default_factory=FixedChunkConfig)


class VectorStoreConfig(_Section):
    """Vector database selection."""

    provider: Literal["qdrant", "numpy", "fake"] = "qdrant"
    url: str = _env_default("${QDRANT_URL:http://localhost:6333}")
    export_dir: PortablePath = Path("artifacts/export")


class Bm25Config(_Section):
    """Lexical (BM25) index parameters."""

    enabled: bool = True
    provider: Literal["bm25s", "fake"] = "bm25s"
    stemming: bool = True
    k1: float = 1.5
    b: float = 0.75


class RetrieverConfig(_Section):
    """Retrieval mode and candidate counts."""

    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    k_candidates: int = 30
    top_n: int = 5
    rrf_k: int = 60


class RerankerConfig(_Section):
    """Cross-encoder reranker selection."""

    enabled: bool = True
    provider: Literal["cross_encoder", "fake"] = "cross_encoder"
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"
    max_length: int = 512
    # Upper bound on the candidates reranked when the reranker runs on the CPU (NFR-PERF1):
    # ~0.6-0.8 s per pair on CPU; 6 candidates measured 4.7 s in Docker (8: 5.7-6.4 s).
    # 0 turns the reranker off on CPU and
    # null removes the bound. Overridden by RERANKER_CPU_CANDIDATES (device_policy.py).
    cpu_max_candidates: int | None = Field(default=6, ge=0)


class AnthropicConfig(_Section):
    """Options used only when ``llm.provider`` is ``anthropic`` (never the default)."""

    model: str = "claude-opus-5"
    effort: str = "low"


LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "ollama"})


class OpenAICompatConfig(_Section):
    """Options used only when ``llm.provider`` is ``openai_compat`` (``/chat/completions``).

    Any OpenAI-compatible endpoint works: a hosted provider (Groq, OpenRouter, Hugging Face
    Inference Providers, Gemini) or a local server (Ollama's ``/v1``). The model is
    ``llm.model``. The API key is never written here: ``api_key_env`` names the environment
    variable that holds it (NFR-CONF2); an unset variable sends no key (local servers).
    """

    base_url: str = _env_default("${LLM_BASE_URL:http://localhost:11434/v1}")
    api_key_env: str = "LLM_API_KEY"
    # How the JSON schema is requested; providers differ, so a rejected mode falls back to
    # the next one (json_schema -> json_object -> prompt) at run time.
    json_mode: Literal["json_schema", "json_object", "prompt"] = "json_schema"
    max_tokens: int = Field(default=2048, gt=0)


class TransformersConfig(_Section):
    """Options used only when ``llm.provider`` is ``transformers`` (generation in-process).

    The Hugging Face Space (ZeroGPU) runs the answer model itself, without Ollama or an API.
    The model is ``llm.model`` (a Hugging Face id such as ``Qwen/Qwen3-4B``); the thinking
    mode follows ``llm.think`` and sampling ``llm.temperature`` (0 = greedy decoding).
    """

    device: str = "auto"  # auto | cpu | cuda (TUBITAK_RAG_DEVICE overrides it)
    dtype: Literal["bfloat16", "float16", "float32", "auto"] = "bfloat16"
    max_new_tokens: int = Field(default=1024, gt=0)


class LlmConfig(_Section):
    """Answer-generating language model; local Ollama by default (NFR-PRIV1, NFR-HW3)."""

    provider: Literal["ollama", "anthropic", "openai_compat", "transformers", "fake"] = "ollama"
    model: str = "qwen3:4b"
    url: str = _env_default("${OLLAMA_URL:http://localhost:11434}")
    temperature: float = 0.0
    num_ctx: int = 8192
    think: bool = False
    # LLM_TIMEOUT_S lets slow CPU-only hosts (Docker CPU profile) wait longer for an answer.
    timeout_s: float = Field(default_factory=lambda: float(interpolate_env("${LLM_TIMEOUT_S:120}")))
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)
    transformers: TransformersConfig = Field(default_factory=TransformersConfig)

    @property
    def model_name(self) -> str:
        """Model id sent to the provider (the Anthropic adapter keeps its own model field)."""
        return self.anthropic.model if self.provider == "anthropic" else self.model

    def is_external(self, provider: str | None = None) -> bool:
        """True when ``provider`` (default: this one) sends requests to a third-party service.

        An OpenAI-compatible endpoint on this machine or in the compose network (Ollama's
        ``/v1``) stays local.
        """
        provider = provider or self.provider
        if provider == "anthropic":
            return True
        if provider != "openai_compat":
            return False
        host = urlparse(self.openai_compat.base_url).hostname or ""
        return host not in LOCAL_HOSTS


class GuardrailConfig(_Section):
    """Scope check mode, retrieval-gate thresholds per score key and the disclaimer text."""

    scope_mode: Literal["lexicon", "llm", "both", "off"] = "both"
    thresholds: dict[str, float] = {"dense": 0.45, "rerank": -2.0, "bm25": 0.0}
    # Numbers in an answer must appear in the question or a cited source (guardrails/numbers.py).
    number_check: Literal["off", "strip", "reject"] = "strip"
    disclaimer: str = (
        "Bu cevap bilgilendirme amaçlıdır; resmî görüş değildir. "
        "Kesin bilgi için ilgili mevzuata ve TÜBİTAK'a başvurunuz."
    )


class JudgeConfig(_Section):
    """LLM-as-judge used during evaluation (connection settings come from ``llm``)."""

    provider: Literal["ollama", "anthropic", "openai_compat", "transformers", "fake"] = "ollama"
    model: str = "qwen3:4b"


class EvalConfig(_Section):
    """Evaluation settings."""

    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    k_values: list[int] = [5, 10]
    limit: int | None = None


class AppConfig(_Section):
    """Root configuration object passed to the factory."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    bm25: Bm25Config = Field(default_factory=Bm25Config)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    def to_yaml(self) -> str:
        """Serialise the resolved configuration as YAML that :func:`load_config` reloads as-is.

        Literal ``${`` in values is written as ``$${`` so reloading does not interpolate it.
        """
        data = _map_strings(self.model_dump(mode="json"), _escape_placeholders)
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def load_config(path: Path, overrides: Mapping[str, Any] | None = None) -> AppConfig:
    """Load, merge (``extends`` → file → ``overrides``), interpolate and validate a config.

    Raises:
        ConfigError: if a file is missing or malformed, or validation fails. The message is
            Turkish and names the offending field path, e.g. ``llm.modelx``.
    """
    _load_dotenv(Path.cwd() / ".env")
    data = _load_with_extends(Path(path), seen=())
    if overrides:
        data = _deep_merge(data, overrides)
    try:
        return AppConfig.model_validate(_map_strings(data, interpolate_env))
    except ValidationError as exc:
        details = "; ".join(_describe_error(err) for err in exc.errors())
        raise ConfigError(f"Geçersiz yapılandırma ({path}): {details}") from exc


def _load_dotenv(dotenv_path: Path) -> None:
    """Export non-empty ``.env`` values that are not already set in the environment.

    Empty lines such as ``HF_HOME=`` (copied from ``.env.example``) are skipped: an empty
    variable is not the same as an unset one for many libraries.
    """
    if not dotenv_path.is_file():
        return
    for key, value in dotenv_values(dotenv_path).items():
        if value and key not in os.environ:
            os.environ[key] = value


def _load_with_extends(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(p.name for p in (*seen, resolved))
        raise ConfigError(f"extends döngüsü: {chain}")
    data = _read_yaml(path)
    base_ref = data.pop(_EXTENDS_KEY, None)
    if base_ref is None:
        return data
    base = _load_with_extends(path.parent / str(base_ref), seen=(*seen, resolved))
    return _deep_merge(base, data)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Yapılandırma dosyası bulunamadı: {path}") from exc
    except OSError as exc:  # e.g. a directory or an unreadable file
        raise ConfigError(f"Yapılandırma dosyası okunamadı ({path}): {exc.strerror}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Yapılandırma dosyası YAML olarak okunamadı ({path}): {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Yapılandırma dosyası 'anahtar: değer' eşlemesi olmalıdır: {path}")
    return data


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated recursively with ``override`` (nested mappings are merged)."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _map_strings(obj: Any, func: Callable[[str], str]) -> Any:
    """Apply ``func`` to every string value inside nested dicts/lists (keys untouched)."""
    if isinstance(obj, str):
        return func(obj)
    if isinstance(obj, Mapping):
        return {key: _map_strings(value, func) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_map_strings(item, func) for item in obj]
    return obj


def _describe_error(err: ErrorDetails) -> str:
    location = ".".join(str(part) for part in err["loc"])
    match err["type"]:
        case "extra_forbidden":
            reason = "bilinmeyen alan"
        case "literal_error":
            expected = (err.get("ctx") or {}).get("expected", "")
            reason = f"geçersiz değer {err['input']!r}; izin verilenler: {expected}"
        case _:
            reason = err["msg"]
    return f"{location}: {reason}"
