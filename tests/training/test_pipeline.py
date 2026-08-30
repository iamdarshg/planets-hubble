from __future__ import annotations

from types import SimpleNamespace

import pytest

import training.pipeline as pipeline
from synthetic import SyntheticConfig
from training import TinyAstroAdapter
from training.pipeline import (
    DEFAULT_SYNTHETIC_CACHE_SIZE_MIB,
    DEFAULT_SYNTHETIC_MIN_EXAMPLES,
    PhaseReport,
    train_synthetic_then_real,
)


def _phase_with_examples(phase: str, examples_seen: int) -> PhaseReport:
    return PhaseReport(
        phase=phase,
        reports=(SimpleNamespace(samples_seen=examples_seen, last_loss=0.5),),
        stopped_reason="step_budget_exhausted",
    )


def test_real_parent_phase_is_skipped_until_default_synthetic_warmup_is_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: admitting real parents after fewer than 4,096 examples."""

    def fake_train_phase(_trainer, phase, *_args, **_kwargs):
        if phase == "synthetic_pretraining":
            return _phase_with_examples(phase, DEFAULT_SYNTHETIC_MIN_EXAMPLES - 1)
        raise AssertionError("real training must not begin before synthetic warm-up completes")

    monkeypatch.setattr(pipeline, "_train_phase", fake_train_phase)

    result = train_synthetic_then_real(
        model=TinyAstroAdapter(),
        synthetic_config=SyntheticConfig(seed=7),
        real_parents=(object(),),
        device="cpu",
        synthetic_max_steps=1,
        real_max_steps=1,
    )

    assert result["synthetic_examples_seen"] == 4095
    assert result["real_phase_gate"].is_open is False
    assert result["real"].stopped_reason == (
        "synthetic warm-up incomplete: 4095/4096 synthetic examples seen"
    )
    assert result["real_checkpoint"] is None


def test_real_parent_phase_is_admitted_at_default_synthetic_warmup_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: blocking real parents after the configured warm-up completed."""
    phases_run: list[str] = []

    def fake_train_phase(_trainer, phase, *_args, **_kwargs):
        phases_run.append(phase)
        if phase == "synthetic_pretraining":
            return _phase_with_examples(phase, DEFAULT_SYNTHETIC_MIN_EXAMPLES)
        return _phase_with_examples(phase, 1)

    monkeypatch.setattr(pipeline, "_train_phase", fake_train_phase)

    result = train_synthetic_then_real(
        model=TinyAstroAdapter(),
        synthetic_config=SyntheticConfig(seed=8),
        real_parents=(object(),),
        device="cpu",
        synthetic_max_steps=1,
        real_max_steps=1,
    )

    assert phases_run == ["synthetic_pretraining", "real_parent_finetuning"]
    assert result["synthetic_examples_seen"] == 4096
    assert result["real_phase_gate"].is_open is True
    assert result["real"].stopped_reason == "step_budget_exhausted"


def test_lower_synthetic_warmup_requires_explicit_bounded_smoke_mode() -> None:
    """Break caught: a production invocation silently weakens the curriculum gate."""

    with pytest.raises(ValueError, match="bounded_smoke_test=True"):
        train_synthetic_then_real(
            model=TinyAstroAdapter(),
            synthetic_config=SyntheticConfig(seed=9),
            device="cpu",
            synthetic_max_steps=1,
            real_max_steps=0,
            synthetic_min_examples=2,
        )


def test_procedural_stream_receives_ssd_cache_contract_and_reports_disk_usage(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: cache settings are dropped or cached bytes are not auditable."""
    cache_dir = tmp_path / "ssd-synthetic-cache"
    cache_contracts: list[tuple[object, int]] = []

    def fake_train_phase(_trainer, phase, *_args, **kwargs):
        if phase == "synthetic_pretraining":
            cache_contracts.append(
                (kwargs["synthetic_cache_dir"], kwargs["synthetic_cache_size"])
            )
            cache_dir.mkdir()
            (cache_dir / "stream-state.bin").write_bytes(b"cache")
            return _phase_with_examples(phase, DEFAULT_SYNTHETIC_MIN_EXAMPLES)
        return _phase_with_examples(phase, 1)

    monkeypatch.setattr(pipeline, "_train_phase", fake_train_phase)

    result = train_synthetic_then_real(
        model=TinyAstroAdapter(),
        synthetic_config=SyntheticConfig(seed=10),
        device="cpu",
        synthetic_max_steps=1,
        real_max_steps=0,
        synthetic_cache_dir=cache_dir,
    )

    assert cache_contracts == [(cache_dir, DEFAULT_SYNTHETIC_CACHE_SIZE_MIB)]
    assert result["synthetic_cache_path"] == str(cache_dir)
    assert result["synthetic_cache_bytes"] == 5
    assert result["synthetic_cache_integration_note"] is None


def test_early_stop_cannot_close_real_gate_before_synthetic_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: a low loss must not admit real parents before 4,096 views."""

    def fake_train_phase(_trainer, phase, *_args, **_kwargs):
        if phase == "synthetic_pretraining":
            return _phase_with_examples(phase, DEFAULT_SYNTHETIC_MIN_EXAMPLES - 1)
        raise AssertionError("real training must not begin before synthetic warm-up completes")

    monkeypatch.setattr(pipeline, "_train_phase", fake_train_phase)

    result = train_synthetic_then_real(
        model=TinyAstroAdapter(),
        synthetic_config=SyntheticConfig(seed=11),
        real_parents=(object(),),
        device="cpu",
        synthetic_max_steps=1,
        real_max_steps=1,
        target_loss=1e9,
    )

    assert result["real_phase_gate"].is_open is False
    assert result["real"].stopped_reason.startswith("synthetic warm-up incomplete")
    assert result["real_checkpoint"] is None
