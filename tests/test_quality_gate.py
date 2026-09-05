"""Tests for Phase 6 — Quality Gate."""
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_quality_gate_passes():
    from src.deployment.quality_gate import evaluate_quality_gate
    metrics = {
        "avg_snr_improvement": 5.0,
        "avg_stoi":            0.80,
        "avg_pesq":            3.0,
        "model_size_mb":       10.0,
        "final_train_loss":    0.01,
    }
    thresholds = {
        "min_snr_improvement": 3.0,
        "min_avg_stoi":        0.65,
        "min_avg_pesq":        2.0,
        "max_model_size_mb":   50.0,
        "max_final_train_loss": 0.1,
    }
    passed, failures, warnings = evaluate_quality_gate(metrics, thresholds)
    assert passed
    assert len(failures) == 0


def test_quality_gate_fails_low_snr():
    from src.deployment.quality_gate import evaluate_quality_gate
    metrics = {
        "avg_snr_improvement": 1.0,    # below threshold of 3.0
        "avg_stoi":            0.80,
        "avg_pesq":            3.0,
        "model_size_mb":       10.0,
        "final_train_loss":    0.01,
    }
    thresholds = {
        "min_snr_improvement": 3.0,
        "min_avg_stoi":        0.65,
        "min_avg_pesq":        2.0,
        "max_model_size_mb":   50.0,
        "max_final_train_loss": 0.1,
    }
    passed, failures, _ = evaluate_quality_gate(metrics, thresholds)
    assert not passed
    assert any("SNR" in f for f in failures)


def test_quality_gate_missing_metric_warns():
    from src.deployment.quality_gate import evaluate_quality_gate
    metrics    = {"avg_stoi": 0.80}     # missing most metrics
    thresholds = {"min_snr_improvement": 3.0, "min_avg_stoi": 0.65}
    _, _, warnings = evaluate_quality_gate(metrics, thresholds)
    assert len(warnings) > 0


def test_quality_gate_model_too_large():
    from src.deployment.quality_gate import evaluate_quality_gate
    metrics = {
        "avg_snr_improvement": 5.0,
        "avg_stoi":            0.80,
        "avg_pesq":            3.0,
        "model_size_mb":       100.0,  # too large
        "final_train_loss":    0.01,
    }
    thresholds = {
        "min_snr_improvement": 3.0,
        "min_avg_stoi":        0.65,
        "min_avg_pesq":        2.0,
        "max_model_size_mb":   50.0,
        "max_final_train_loss": 0.1,
    }
    passed, failures, _ = evaluate_quality_gate(metrics, thresholds)
    assert not passed


def test_load_thresholds_returns_dict():
    from src.deployment.quality_gate import load_thresholds
    thresholds = load_thresholds()
    assert isinstance(thresholds, dict)
    assert len(thresholds) > 0
