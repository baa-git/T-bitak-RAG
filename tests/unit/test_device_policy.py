"""Device selection and the CPU reranker policy (NFR-HW1, NFR-HW2, NFR-PERF1); no torch."""

import logging

import pytest

from tubitak_rag.config import AppConfig, ConfigError, EmbedderConfig, RerankerConfig
from tubitak_rag.device_policy import (
    CPU_CANDIDATES_ENV,
    DEVICE_ENV,
    apply_device_policy,
    choose_device,
    cpu_candidates,
    device_setting,
    plan_reranker,
)


class Probe:
    """Fake GPU probe that counts how often it is asked."""

    def __init__(self, available: bool) -> None:
        self.available = available
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.available


def test_env_overrides_configured_device() -> None:
    assert device_setting("auto", {DEVICE_ENV: "CPU"}) == "cpu"
    assert device_setting("cuda", {}) == "cuda"
    assert device_setting("auto", {DEVICE_ENV: "  "}) == "auto"


@pytest.mark.parametrize(
    ("setting", "gpu", "expected"),
    [("auto", True, "cuda"), ("auto", False, "cpu"), ("cuda", True, "cuda"), ("cpu", True, "cpu")],
)
def test_choose_device(setting: str, gpu: bool, expected: str) -> None:
    assert choose_device(setting, Probe(gpu)) == expected


def test_cpu_setting_never_probes_the_gpu() -> None:
    probe = Probe(True)
    choose_device("cpu", probe)
    assert probe.calls == 0


def test_cuda_without_gpu_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="GPU yok"):
        choose_device("cuda", Probe(False))


def test_unknown_device_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="Geçersiz cihaz"):
        choose_device("tpu", Probe(True))


def test_cpu_candidates_env_override() -> None:
    assert cpu_candidates(8, {}) == 8
    assert cpu_candidates(8, {CPU_CANDIDATES_ENV: "12"}) == 12
    assert cpu_candidates(8, {CPU_CANDIDATES_ENV: "0"}) == 0
    assert cpu_candidates(8, {CPU_CANDIDATES_ENV: "none"}) is None
    with pytest.raises(ConfigError, match=CPU_CANDIDATES_ENV):
        cpu_candidates(8, {CPU_CANDIDATES_ENV: "-3"})


def test_plan_caps_candidates_on_cpu() -> None:
    plan = plan_reranker(enabled=True, device="cpu", k_candidates=30, cpu_max_candidates=8)
    assert (plan.enabled, plan.k_candidates) == (True, 8)
    assert plan.note and "30 → 8" in plan.note


def test_plan_leaves_gpu_untouched() -> None:
    plan = plan_reranker(enabled=True, device="cuda", k_candidates=30, cpu_max_candidates=8)
    assert (plan.enabled, plan.k_candidates, plan.note) == (True, 30, None)


def test_plan_zero_disables_reranker_on_cpu() -> None:
    plan = plan_reranker(enabled=True, device="cpu", k_candidates=30, cpu_max_candidates=0)
    assert (plan.enabled, plan.k_candidates) == (False, 30)


def test_plan_without_bound_or_below_bound_changes_nothing() -> None:
    unbounded = plan_reranker(enabled=True, device="cpu", k_candidates=30, cpu_max_candidates=None)
    small = plan_reranker(enabled=True, device="cpu", k_candidates=5, cpu_max_candidates=8)
    assert (unbounded.k_candidates, unbounded.note) == (30, None)
    assert (small.k_candidates, small.note) == (5, None)


def test_apply_policy_on_cpu_only_machine(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    cfg = apply_device_policy(AppConfig(), cuda_available=Probe(False), environ={})
    assert (cfg.embedder.device, cfg.reranker.device) == ("cpu", "cpu")
    assert cfg.reranker.enabled and cfg.retriever.k_candidates == 6
    assert "Reranker CPU'da" in caplog.text


def test_apply_policy_on_gpu_machine_keeps_config() -> None:
    cfg = apply_device_policy(AppConfig(), cuda_available=Probe(True), environ={})
    assert (cfg.embedder.device, cfg.reranker.device) == ("cuda", "cuda")
    assert cfg.retriever.k_candidates == 30


def test_apply_policy_env_forces_cpu_and_disables_reranker() -> None:
    env = {DEVICE_ENV: "cpu", CPU_CANDIDATES_ENV: "0"}
    cfg = apply_device_policy(AppConfig(), cuda_available=Probe(True), environ=env)
    assert cfg.embedder.device == "cpu" and not cfg.reranker.enabled
    assert cfg.retriever.k_candidates == 30


def test_apply_policy_is_idempotent() -> None:
    probe = Probe(False)
    once = apply_device_policy(AppConfig(), cuda_available=probe, environ={})
    assert apply_device_policy(once, cuda_available=probe, environ={}) == once


def test_fake_components_are_not_probed() -> None:
    probe = Probe(True)
    cfg = AppConfig(
        embedder=EmbedderConfig(provider="fake"), reranker=RerankerConfig(provider="fake")
    )
    assert apply_device_policy(cfg, cuda_available=probe, environ={}) == cfg
    assert probe.calls == 0


def test_cpu_max_candidates_must_not_be_negative() -> None:
    with pytest.raises(ValueError):
        RerankerConfig(cpu_max_candidates=-1)
