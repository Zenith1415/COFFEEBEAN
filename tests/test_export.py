"""Tests for Phase 7 — ONNX Export."""
import pytest
import torch
import yaml
import numpy as np
from pathlib import Path


@pytest.fixture
def cfg():
    return yaml.safe_load(Path("configs/config.yaml").read_text())


@pytest.fixture
def tmp_onnx(tmp_path, cfg):
    """Export a fresh ONNX model to a temp path."""
    from src.training.model import ANCAudioModel
    from src.deployment.export import export_to_onnx

    model     = ANCAudioModel(cfg)
    onnx_path = str(tmp_path / "test_model.onnx")
    export_to_onnx(model, onnx_path, sample_rate=cfg["audio"]["sample_rate"])
    return onnx_path


def test_export_creates_file(tmp_onnx):
    assert Path(tmp_onnx).exists()
    assert Path(tmp_onnx).stat().st_size > 0


def test_export_valid_onnx(tmp_onnx):
    import onnx
    model = onnx.load(tmp_onnx)
    onnx.checker.check_model(model)    # Raises if invalid


def test_onnx_runtime_inference(tmp_onnx, cfg):
    import onnxruntime as ort
    sr            = cfg["audio"]["sample_rate"]
    dummy         = np.random.randn(1, 1, sr).astype(np.float32)
    session       = ort.InferenceSession(tmp_onnx, providers=["CPUExecutionProvider"])
    input_name    = session.get_inputs()[0].name
    output        = session.run(None, {input_name: dummy})
    assert output[0].shape == dummy.shape


def test_onnx_output_shape_matches_input(tmp_onnx, cfg):
    import onnxruntime as ort
    sr         = cfg["audio"]["sample_rate"]
    for length in [sr // 2, sr, sr * 2]:    # test multiple input lengths
        dummy   = np.random.randn(1, 1, length).astype(np.float32)
        session = ort.InferenceSession(tmp_onnx, providers=["CPUExecutionProvider"])
        out     = session.run(None, {session.get_inputs()[0].name: dummy})
        assert out[0].shape == dummy.shape, f"Shape mismatch at length {length}"


def test_inference_engine_loads(tmp_onnx, cfg):
    from src.deployment.inference import ANCInferenceEngine
    engine = ANCInferenceEngine(tmp_onnx, sample_rate=cfg["audio"]["sample_rate"])
    assert engine.session is not None


def test_inference_engine_enhance_shape(tmp_onnx, cfg):
    from src.deployment.inference import ANCInferenceEngine
    sr     = cfg["audio"]["sample_rate"]
    engine = ANCInferenceEngine(tmp_onnx, sample_rate=sr)
    audio  = np.random.randn(sr).astype(np.float32)
    out    = engine.enhance(audio)
    assert out.shape == audio.shape
