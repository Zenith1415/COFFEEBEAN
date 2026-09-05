"""Tests for Phase 5 — Evaluation metrics."""
import pytest
import numpy as np
import torch
import yaml
from pathlib import Path


@pytest.fixture
def cfg():
    return yaml.safe_load(Path("configs/config.yaml").read_text())


@pytest.fixture
def dummy_clean():
    np.random.seed(42)
    return np.random.randn(16000).astype(np.float64) * 0.5


@pytest.fixture
def dummy_noise():
    np.random.seed(99)
    return np.random.randn(16000).astype(np.float64) * 0.1


# ── SNR ──────────────────────────────────────────────────────────────────────

def test_snr_perfect_reconstruction(dummy_clean):
    from src.evaluation.evaluate import compute_snr
    snr = compute_snr(dummy_clean, dummy_clean)
    assert snr == float("inf") or snr > 60.0


def test_snr_with_noise(dummy_clean, dummy_noise):
    from src.evaluation.evaluate import compute_snr
    noisy = dummy_clean + dummy_noise
    snr = compute_snr(dummy_clean, noisy)
    assert isinstance(snr, float)
    assert snr < 60.0    # not perfect


def test_snr_improvement_structure(dummy_clean, dummy_noise):
    from src.evaluation.evaluate import compute_snr_improvement
    noisy    = dummy_clean + dummy_noise
    enhanced = dummy_clean + dummy_noise * 0.1    # partially denoised
    metrics  = compute_snr_improvement(dummy_clean, noisy, enhanced)
    assert "snr_before"      in metrics
    assert "snr_after"       in metrics
    assert "snr_improvement" in metrics
    assert metrics["snr_improvement"] > 0    # improvement should be positive


# ── STOI ─────────────────────────────────────────────────────────────────────

def test_stoi_range(dummy_clean):
    from src.evaluation.evaluate import compute_stoi
    noisy = dummy_clean + np.random.randn(16000).astype(np.float64) * 0.3
    try:
        score = compute_stoi(dummy_clean, noisy, sample_rate=16000)
        assert 0.0 <= score <= 1.0
    except ImportError:
        pytest.skip("pystoi / torchmetrics not installed")


def test_stoi_perfect_is_one(dummy_clean):
    from src.evaluation.evaluate import compute_stoi
    try:
        score = compute_stoi(dummy_clean, dummy_clean, sample_rate=16000)
        assert score >= 0.99
    except ImportError:
        pytest.skip("torchmetrics not installed")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def test_evaluate_anc_model_returns_dict(cfg):
    from src.evaluation.evaluate import evaluate_anc_model
    from src.training.model import ANCAudioModel

    model = ANCAudioModel(cfg)
    clean = torch.randn(1, 16000)
    noise = torch.randn(1, 16000) * 0.3

    metrics = evaluate_anc_model(model, clean, noise, sample_rate=cfg["audio"]["sample_rate"])
    assert isinstance(metrics, dict)
    assert "snr_before"      in metrics
    assert "snr_after"       in metrics
    assert "snr_improvement" in metrics


def test_evaluate_anc_model_snr_finite(cfg):
    from src.evaluation.evaluate import evaluate_anc_model
    from src.training.model import ANCAudioModel

    model   = ANCAudioModel(cfg)
    clean   = torch.randn(1, 16000)
    noise   = torch.randn(1, 16000) * 0.3
    metrics = evaluate_anc_model(model, clean, noise, sample_rate=16000)
    assert np.isfinite(metrics["snr_before"])
    assert np.isfinite(metrics["snr_after"])
