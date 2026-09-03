"""
Tests for MLflow configuration and training module structure.
Note: These tests validate structure and config, not live MLflow connectivity.
"""

import pytest
from pathlib import Path


def test_train_module_importable():
    """train.py must be importable without errors."""
    import src.training.train as train_module
    assert hasattr(train_module, "train")
    assert hasattr(train_module, "load_config")


def test_evaluate_module_importable():
    """evaluate.py must be importable without errors."""
    import src.evaluation.evaluate as eval_module
    assert hasattr(eval_module, "compute_snr")
    assert hasattr(eval_module, "compute_stoi")
    assert hasattr(eval_module, "compute_pesq")


def test_preprocess_module_importable():
    """preprocess.py must be importable without errors."""
    import src.preprocessing.preprocess as pre_module
    assert hasattr(pre_module, "load_dataset")
    assert hasattr(pre_module, "preprocess_audio")


def test_load_config_reads_yaml():
    """load_config() must read and return valid project config."""
    from src.training.train import load_config
    cfg = load_config()
    assert cfg["project"]["name"] == "COFFEEBEAN"
    assert cfg["training"]["batch_size"] > 0
    assert cfg["training"]["learning_rate"] > 0
    assert cfg["audio"]["sample_rate"] == 16000


def test_mlflow_tracking_uri_env(monkeypatch):
    """MLFLOW_TRACKING_URI env var must override default."""
    import importlib
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://custom-host:5000")
    import src.training.train as train_module
    importlib.reload(train_module)
    assert train_module.MLFLOW_TRACKING_URI == "http://custom-host:5000"


def test_evaluation_stubs_raise_not_implemented():
    """Evaluation stubs must raise NotImplementedError (not silently pass)."""
    from src.evaluation.evaluate import compute_snr, compute_stoi, compute_pesq
    with pytest.raises(NotImplementedError):
        compute_snr(None, None)
    with pytest.raises(NotImplementedError):
        compute_stoi(None, None, 16000)
    with pytest.raises(NotImplementedError):
        compute_pesq(None, None, 16000)


def test_dataset_directories_exist():
    """DVC-managed dataset directories must exist with correct structure."""
    base = Path("data/raw")
    expected = ["speech", "stationary_noise", "nonstationary_noise", "impulsive_noise"]
    for category in expected:
        assert (base / category).exists(), f"Missing: data/raw/{category}/"
