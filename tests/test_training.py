"""
Tests for Phase 4 — PyTorch training pipeline.
"""

import pytest
import torch
import yaml
from pathlib import Path


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return yaml.safe_load(Path("configs/config.yaml").read_text())


@pytest.fixture
def dummy_dataset():
    """Minimal in-memory dataset with one speech and one noise file."""
    speech  = torch.randn(1, 32000)   # 2 sec @ 16kHz
    noise   = torch.randn(1, 32000)
    return {
        "speech":               [speech],
        "stationary_noise":     [noise],
        "nonstationary_noise":  [],
        "impulsive_noise":      [],
    }


# ── Model tests ────────────────────────────────────────────────────────────────

def test_model_importable():
    from src.training.model import ANCAudioModel
    assert ANCAudioModel is not None


def test_model_forward_pass(cfg):
    """Model must accept [B, 1, T] and return same shape."""
    from src.training.model import ANCAudioModel
    model = ANCAudioModel(cfg)
    model.eval()
    x = torch.randn(2, 1, 16000)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"


def test_model_counts_parameters(cfg):
    """Model must have a non-zero parameter count."""
    from src.training.model import ANCAudioModel
    model = ANCAudioModel(cfg)
    assert model.count_parameters() > 0


def test_model_size_mb(cfg):
    """Model size must be > 0 MB."""
    from src.training.model import ANCAudioModel
    model = ANCAudioModel(cfg)
    assert model.model_size_mb() > 0.0


# ── Preprocessing tests ────────────────────────────────────────────────────────

def test_mix_signals_output_shape():
    """mix_signals must return tensor matching input length."""
    from src.preprocessing.preprocess import mix_signals
    speech = torch.randn(1, 16000)
    noise  = torch.randn(1, 16000)
    noisy, scale = mix_signals(speech, noise, target_snr_db=0.0)
    assert noisy.shape == speech.shape


def test_mix_signals_no_clipping():
    """mix_signals output must stay within [-1, 1]."""
    from src.preprocessing.preprocess import mix_signals
    speech = torch.randn(1, 16000)
    noise  = torch.randn(1, 16000) * 10   # loud noise
    noisy, _ = mix_signals(speech, noise, target_snr_db=0.0)
    assert noisy.abs().max().item() <= 1.0 + 1e-5


def test_anc_dataset_len(dummy_dataset):
    """ANCDataset length must match num_samples."""
    from src.preprocessing.preprocess import ANCDataset
    ds = ANCDataset(dummy_dataset, chunk_samples=16000, num_samples=50)
    assert len(ds) == 50


def test_anc_dataset_item_shapes(dummy_dataset):
    """Each dataset item must be a (noisy, clean) pair of shape [1, chunk]."""
    from src.preprocessing.preprocess import ANCDataset
    ds = ANCDataset(dummy_dataset, chunk_samples=16000, num_samples=10)
    noisy, clean = ds[0]
    assert noisy.shape == (1, 16000)
    assert clean.shape == (1, 16000)


def test_anc_dataset_no_speech_raises():
    """ANCDataset must raise ValueError if no speech files provided."""
    from src.preprocessing.preprocess import ANCDataset
    with pytest.raises(ValueError, match="No speech files"):
        ANCDataset({"speech": [], "stationary_noise": [], "nonstationary_noise": [], "impulsive_noise": []})


# ── Training loop smoke test ───────────────────────────────────────────────────

def test_one_training_step(cfg, dummy_dataset):
    """A single training step must reduce or produce a finite loss."""
    from src.training.model import ANCAudioModel
    from src.preprocessing.preprocess import ANCDataset
    from torch.utils.data import DataLoader

    model = ANCAudioModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    ds = ANCDataset(dummy_dataset, chunk_samples=16000, num_samples=4)
    loader = DataLoader(ds, batch_size=2, drop_last=True)

    model.train()
    noisy, clean = next(iter(loader))
    optimizer.zero_grad()
    output = model(noisy)
    loss = loss_fn(output, clean)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


# ── DVC pipeline test ──────────────────────────────────────────────────────────

def test_dvc_yaml_exists():
    """dvc.yaml must exist and define train stage."""
    import yaml
    dvc = yaml.safe_load(Path("dvc.yaml").read_text())
    assert "stages" in dvc
    assert "train" in dvc["stages"]


def test_metrics_structure():
    """configs/config.yaml must have all required training keys."""
    cfg_data = yaml.safe_load(Path("configs/config.yaml").read_text())
    assert "training" in cfg_data
    assert "batch_size" in cfg_data["training"]
    assert "epochs" in cfg_data["training"]
    assert "learning_rate" in cfg_data["training"]
