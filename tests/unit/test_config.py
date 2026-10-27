"""Tests for typed configuration loading (NFR-CONF1, NFR-CONF2, NFR-REPRO2, NFR-PRIV1)."""

import os
from pathlib import Path

import pytest
import yaml

from tubitak_rag.config import AppConfig, ConfigError, load_config


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _isolated_cwd(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run every test from an empty directory so a developer's real .env never leaks in."""
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


def test_default_yaml_loads(repo_root: Path) -> None:
    cfg = load_config(repo_root / "configs/default.yaml")
    assert cfg.llm.model == "qwen3:4b" and cfg.llm.temperature == 0.0
    assert cfg.llm.provider == "ollama"  # NFR-PRIV1: no API provider selected
    assert cfg.guardrails.thresholds["dense"] > 0  # FR-GR9


def test_default_yaml_matches_model_defaults(repo_root: Path, tmp_path: Path) -> None:
    """default.yaml spells out every field explicitly; it must equal the code defaults."""
    cfg = load_config(repo_root / "configs/default.yaml")
    defaults = load_config(_write(tmp_path / "empty.yaml", ""))
    assert cfg.model_dump(exclude={"experiment"}) == defaults.model_dump(exclude={"experiment"})


def test_default_yaml_lists_every_field(repo_root: Path) -> None:
    """Teaching aid: every config field appears explicitly in default.yaml."""
    raw = yaml.safe_load((repo_root / "configs/default.yaml").read_text(encoding="utf-8"))
    for section, model in AppConfig.model_fields.items():
        assert section in raw, section
        sub = model.annotation
        assert sub is not None
        assert set(raw[section]) == set(sub.model_fields), section


def test_unknown_field_raises_config_error(tmp_path: Path) -> None:
    p = _write(tmp_path / "bad.yaml", "llm:\n  modelx: qwen3\n")
    with pytest.raises(ConfigError) as e:
        load_config(p)
    assert "llm.modelx" in str(e.value)  # NFR-CONF1: field path in message
    assert "Geçersiz yapılandırma" in str(e.value)
    assert "bilinmeyen alan" in str(e.value)  # Turkish reason, not pydantic's English text


def test_invalid_literal_raises_config_error(tmp_path: Path) -> None:
    p = _write(tmp_path / "bad.yaml", "retriever:\n  mode: semantic\n")
    with pytest.raises(ConfigError) as e:
        load_config(p)
    assert "retriever.mode" in str(e.value) and "'semantic'" in str(e.value)


def test_wrong_type_raises_config_error(tmp_path: Path) -> None:
    p = _write(tmp_path / "bad.yaml", "retriever:\n  top_n: many\n")
    with pytest.raises(ConfigError) as e:
        load_config(p)
    assert "retriever.top_n" in str(e.value)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as e:
        load_config(tmp_path / "yok.yaml")
    assert "yok.yaml" in str(e.value)


def test_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    p = _write(tmp_path / "bad.yaml", "llm: [unclosed\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_non_mapping_yaml_raises_config_error(tmp_path: Path) -> None:
    p = _write(tmp_path / "bad.yaml", "- a\n- b\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_empty_yaml_gives_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    cfg = load_config(_write(tmp_path / "empty.yaml", ""))
    assert cfg.llm.url == "http://localhost:11434"
    assert cfg.vector_store.url == "http://localhost:6333"
    assert cfg.retriever.mode == "hybrid" and cfg.chunker.name == "structural"


def test_env_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://q:1")
    p = _write(tmp_path / "c.yaml", "vector_store:\n  url: ${QDRANT_URL:http://localhost:6333}\n")
    assert load_config(p).vector_store.url == "http://q:1"


def test_env_default_when_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    p = _write(tmp_path / "c.yaml", "vector_store:\n  url: ${QDRANT_URL:http://localhost:6333}\n")
    assert load_config(p).vector_store.url == "http://localhost:6333"


def test_code_defaults_are_interpolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults declared in code use the same ${VAR:default} syntax as YAML."""
    monkeypatch.setenv("OLLAMA_URL", "http://o:2")
    cfg = load_config(_write(tmp_path / "empty.yaml", ""))
    assert cfg.llm.url == "http://o:2"
    assert cfg.embedder.url == "http://o:2"


def test_env_without_default_and_unset_becomes_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TUBITAK_RAG_UNSET_VAR", raising=False)
    p = _write(tmp_path / "c.yaml", "experiment:\n  description: x${TUBITAK_RAG_UNSET_VAR}y\n")
    assert load_config(p).experiment.description == "xy"


def test_dotenv_is_loaded_without_overriding_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # setenv-then-delenv makes monkeypatch restore the variable that load_dotenv sets.
    monkeypatch.setenv("OLLAMA_URL", "placeholder")
    monkeypatch.delenv("OLLAMA_URL")
    monkeypatch.setenv("QDRANT_URL", "http://from-env:1")
    _write(tmp_path / ".env", "OLLAMA_URL=http://from-dotenv:2\nQDRANT_URL=http://dotenv:3\n")
    cfg = load_config(_write(tmp_path / "c.yaml", ""))
    assert cfg.llm.url == "http://from-dotenv:2"
    assert cfg.vector_store.url == "http://from-env:1"


def test_empty_dotenv_values_do_not_pollute_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`KEY=` lines copied from .env.example must not become empty env vars (e.g. HF_HOME='')."""
    monkeypatch.chdir(tmp_path)
    for name in ("HF_HOME", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "placeholder")  # registers restore even if code leaks it
        monkeypatch.delenv(name)
    _write(tmp_path / ".env", "ANTHROPIC_API_KEY=\nHF_HOME=\n")
    load_config(_write(tmp_path / "c.yaml", ""))
    assert "HF_HOME" not in os.environ
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_placeholders_are_resolved_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An env value that itself contains ${...} is used verbatim, in every kind of field."""
    monkeypatch.setenv("QDRANT_URL", "http://h/${TUBITAK_RAG_OTHER}")
    monkeypatch.setenv("TUBITAK_RAG_OTHER", "wrong")
    p = _write(
        tmp_path / "c.yaml",
        "vector_store:\n  url: ${QDRANT_URL}\nexperiment:\n  description: ${QDRANT_URL}\n",
    )
    cfg = load_config(p)
    assert cfg.vector_store.url == "http://h/${TUBITAK_RAG_OTHER}"
    assert cfg.experiment.description == "http://h/${TUBITAK_RAG_OTHER}"


def test_double_dollar_escapes_a_literal_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TUBITAK_RAG_X", "wrong")
    p = _write(
        tmp_path / "c.yaml",
        "vector_store:\n  url: a$${TUBITAK_RAG_X}b\n"
        "experiment:\n  description: $${TUBITAK_RAG_X}\n",
    )
    cfg = load_config(p)
    assert cfg.vector_store.url == "a${TUBITAK_RAG_X}b"
    assert cfg.experiment.description == "${TUBITAK_RAG_X}"


def test_to_yaml_keeps_literal_placeholders_on_reload(tmp_path: Path) -> None:
    p = _write(tmp_path / "c.yaml", "experiment:\n  description: maliyet $${HOME}\n")
    cfg = load_config(p)
    assert load_config(_write(tmp_path / "dump.yaml", cfg.to_yaml())) == cfg


def test_directory_path_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as e:
        load_config(tmp_path)
    assert "okunamadı" in str(e.value)


def test_extends_cycle_raises_config_error(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "extends: b.yaml\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ConfigError) as e:
        load_config(tmp_path / "a.yaml")
    assert "extends döngüsü" in str(e.value)


def test_extends_self_reference_raises_config_error(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "extends: ./a.yaml\n")
    with pytest.raises(ConfigError) as e:
        load_config(tmp_path / "a.yaml")
    assert "extends döngüsü" in str(e.value)


def test_extends_deep_merges(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "llm:\n  model: a\n  temperature: 0.0\n")
    _write(tmp_path / "x.yaml", "extends: base.yaml\nllm:\n  model: b\n")
    cfg = load_config(tmp_path / "x.yaml")
    assert cfg.llm.model == "b" and cfg.llm.temperature == 0.0


def test_extends_is_relative_to_file(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    _write(tmp_path / "base.yaml", "llm:\n  model: base\n")
    _write(tmp_path / "sub" / "x.yaml", "extends: ../base.yaml\n")
    assert load_config(tmp_path / "sub" / "x.yaml").llm.model == "base"


def test_overrides_are_deep_merged_last(tmp_path: Path) -> None:
    p = _write(tmp_path / "c.yaml", "llm:\n  model: a\n  num_ctx: 4096\n")
    cfg = load_config(p, overrides={"llm": {"model": "z"}})
    assert cfg.llm.model == "z" and cfg.llm.num_ctx == 4096


def test_to_yaml_round_trips(tmp_path: Path, repo_root: Path) -> None:
    cfg = load_config(repo_root / "configs/experiments/v3b.yaml")
    dumped = cfg.to_yaml()
    assert "TÜBİTAK" in dumped  # allow_unicode: disclaimer stays readable
    assert isinstance(yaml.safe_load(dumped), dict)
    assert load_config(_write(tmp_path / "dump.yaml", dumped)) == cfg


@pytest.mark.parametrize(
    "name,chunker,header,mode,rerank,embed",
    [
        ("v0", "fixed", False, "dense", False, "google/embeddinggemma-2"),
        ("v0r", "fixed", False, "dense", True, "google/embeddinggemma-2"),
        ("v1", "structural", True, "dense", False, "google/embeddinggemma-2"),
        ("v2", "structural", True, "hybrid", False, "google/embeddinggemma-2"),
        ("v3", "structural", True, "hybrid", True, "google/embeddinggemma-2"),
        ("v3b", "structural", True, "hybrid", True, "BAAI/bge-m3"),
    ],
)
def test_experiment_configs_match_ladder(
    repo_root: Path,
    name: str,
    chunker: str,
    header: bool,
    mode: str,
    rerank: bool,
    embed: str,
) -> None:
    cfg = load_config(repo_root / f"configs/experiments/{name}.yaml")
    assert (
        cfg.experiment.name,
        cfg.chunker.name,
        cfg.chunker.context_header,
        cfg.retriever.mode,
        cfg.reranker.enabled,
        cfg.embedder.model,
    ) == (name, chunker, header, mode, rerank, embed)
    assert cfg.llm.temperature == 0.0  # NFR-REPRO2
    assert cfg.experiment.description  # Turkish "lesson taught" text


@pytest.mark.parametrize(
    "name,concept,previous",
    [
        ("v0", "fixed_chunking", None),
        ("v0r", "reranker", "v0"),
        ("v1", "structural_context", "v0"),
        ("v2", "hybrid_rrf", "v1"),
        ("v3", "full_pipeline", "v2"),
        ("v3b", "embedding_comparison", "v3"),
    ],
)
def test_experiment_concepts_and_previous(
    repo_root: Path, name: str, concept: str, previous: str | None
) -> None:
    cfg = load_config(repo_root / f"configs/experiments/{name}.yaml")
    assert (cfg.experiment.concept, cfg.experiment.previous) == (concept, previous)


def test_v3b_uses_bge_m3_dimension(repo_root: Path) -> None:
    assert load_config(repo_root / "configs/experiments/v3b.yaml").embedder.dimension == 1024


def test_default_llm_is_under_8b(repo_root: Path) -> None:
    """NFR-HW3: the default LLM tag must be a small (< 8B) model."""
    assert load_config(repo_root / "configs/default.yaml").llm.model in {"qwen3:4b", "qwen3:0.6b"}


def test_bm25_provider_is_bm25s_by_default_and_validated(tmp_path: Path) -> None:
    # Phase 4 contract change: a provider key lets tests inject the fake lexical index.
    assert AppConfig().bm25.provider == "bm25s"
    with pytest.raises(ConfigError, match=r"bm25\.provider"):
        load_config(_write(tmp_path / "c.yaml", "bm25:\n  provider: elastic\n"))


def test_number_check_defaults_to_strip_and_is_validated(tmp_path: Path) -> None:
    assert load_config(_write(tmp_path / "empty.yaml", "")).guardrails.number_check == "strip"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path / "bad.yaml", "guardrails:\n  number_check: belki\n"))


def test_openai_compat_settings_never_hold_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NFR-CONF2: only the *name* of the key variable is configured; the URL comes from env."""
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.test/v1")
    text = (
        "llm:\n  provider: openai_compat\n  model: m\n  openai_compat:\n    json_mode: prompt\n"
        "eval:\n  judge:\n    provider: openai_compat\n"
    )
    cfg = load_config(_write(tmp_path / "c.yaml", text))
    assert cfg.llm.provider == "openai_compat" and cfg.eval.judge.provider == "openai_compat"
    settings = cfg.llm.openai_compat
    assert settings.base_url == "https://api.example.test/v1"
    assert settings.api_key_env == "LLM_API_KEY" and settings.json_mode == "prompt"
    assert "api_key" not in settings.model_dump()
    bad = _write(tmp_path / "bad.yaml", "llm:\n  openai_compat:\n    json_mode: xml\n")
    with pytest.raises(ConfigError, match=r"llm\.openai_compat\.json_mode"):
        load_config(bad)
