"""
COFFEEBEAN — Edge Inference Engine (ONNX Runtime)
Phase 9: Real-time ANC inference on edge devices using ONNX Runtime.

Designed for:
  - Raspberry Pi 4 / Jetson Nano
  - No internet required after deployment
  - Handles audio in 1-second chunks

Usage:
    python src/deployment/inference.py --model models/anc_model.onnx
    python src/deployment/inference.py --model models/anc_model.onnx --benchmark
"""

import os
import sys
import io
import time
import json
import logging
import argparse
import numpy as np
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ANCInferenceEngine:
    """
    Real-time ANC inference engine using ONNX Runtime.

    Runs on edge device (CPU), processes 1-second audio chunks,
    no internet required after deployment.
    """

    def __init__(self, model_path: str, sample_rate: int = 16000):
        """
        Initialize inference engine.

        Args:
            model_path:  Path to .onnx model file.
            sample_rate: Audio sample rate in Hz.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Run: pip install onnxruntime")

        self.model_path  = model_path
        self.sample_rate = sample_rate
        self._load_time_ms = 0.0

        t0 = time.perf_counter()
        self.session = ort.InferenceSession(
            model_path,
            providers=self._get_providers(),
        )
        self._load_time_ms = (time.perf_counter() - t0) * 1000

        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"ANC engine loaded: {model_path}")
        logger.info(f"  Load time:   {self._load_time_ms:.1f} ms")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Provider:    {self.session.get_providers()[0]}")

    def _get_providers(self) -> list:
        """Select best available ONNX Runtime execution provider."""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except Exception:
            return ["CPUExecutionProvider"]

        # Priority: CUDA > DirectML > CPU
        for provider in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]:
            if provider in available:
                return [provider]
        return ["CPUExecutionProvider"]

    def enhance(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Enhance a single audio chunk (remove noise).

        Args:
            audio_chunk: Noisy audio [samples] or [1, samples] (float32, normalized).

        Returns:
            Enhanced audio [samples] (float32, normalized).
        """
        # Ensure shape [1, 1, samples]
        chunk = np.asarray(audio_chunk, dtype=np.float32).flatten()
        inp   = chunk.reshape(1, 1, -1)

        output = self.session.run([self.output_name], {self.input_name: inp})
        return output[0].flatten()

    def benchmark(self, chunk_seconds: float = 1.0, num_runs: int = 100) -> dict:
        """
        Benchmark inference latency.

        Returns dict with mean/p95/p99 latency and real-time factor.
        A real_time_factor < 1.0 means the model runs faster than real-time.
        """
        chunk_samples = int(self.sample_rate * chunk_seconds)
        dummy = np.random.randn(chunk_samples).astype(np.float32)

        # Warmup
        for _ in range(5):
            self.enhance(dummy)

        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.enhance(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)

        arr = np.array(latencies)
        audio_ms = chunk_seconds * 1000.0

        return {
            "latency_mean_ms":   round(float(np.mean(arr)),           2),
            "latency_p50_ms":    round(float(np.percentile(arr, 50)), 2),
            "latency_p95_ms":    round(float(np.percentile(arr, 95)), 2),
            "latency_p99_ms":    round(float(np.percentile(arr, 99)), 2),
            "real_time_factor":  round(float(np.mean(arr)) / audio_ms, 4),
            "real_time_capable": float(np.mean(arr)) < audio_ms,
            "model_path":        self.model_path,
            "chunk_seconds":     chunk_seconds,
            "num_runs":          num_runs,
        }

    def get_model_info(self) -> dict:
        """Return model metadata."""
        size_mb = Path(self.model_path).stat().st_size / (1024 ** 2)
        return {
            "model_path":    self.model_path,
            "size_mb":       round(size_mb, 4),
            "input_name":    self.input_name,
            "output_name":   self.output_name,
            "load_time_ms":  round(self._load_time_ms, 2),
            "providers":     self.session.get_providers(),
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="COFFEEBEAN Edge Inference")
    p.add_argument("--model",    default="models/anc_model.onnx",
                   help="Path to ONNX model")
    p.add_argument("--benchmark", action="store_true",
                   help="Run latency benchmark")
    p.add_argument("--input-wav", default=None,
                   help="Run on a specific WAV file and save output")
    p.add_argument("--output-wav", default="output_enhanced.wav")
    args = p.parse_args()

    engine = ANCInferenceEngine(args.model)
    print(json.dumps(engine.get_model_info(), indent=2))

    if args.benchmark:
        results = engine.benchmark()
        print("\nBenchmark Results:")
        print(json.dumps(results, indent=2))

    if args.input_wav:
        try:
            import soundfile as sf
            audio, sr = sf.read(args.input_wav, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # convert to mono
            enhanced = engine.enhance(audio)
            sf.write(args.output_wav, enhanced, sr)
            print(f"Enhanced audio saved: {args.output_wav}")
        except Exception as e:
            try:
                import torchaudio
                waveform, sr = torchaudio.load(args.input_wav)
                audio = waveform.squeeze().numpy()
                enhanced = engine.enhance(audio)
                import torch
                torchaudio.save(args.output_wav, torch.tensor(enhanced).unsqueeze(0), sr)
                print(f"Enhanced audio saved: {args.output_wav}")
            except Exception as e2:
                logger.error(f"Failed to process WAV: {e} / {e2}")
