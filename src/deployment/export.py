"""
COFFEEBEAN — ONNX Export + Optimization
Phase 7: Convert trained PyTorch model to ONNX for edge deployment.

Usage:
    python src/deployment/export.py --checkpoint models/anc_checkpoint.pt
    python src/deployment/export.py --checkpoint models/anc_checkpoint.pt --quantize
"""

import os
import sys
import io
import time
import json
import argparse
import logging
import yaml
import torch
import numpy as np
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.model import ANCAudioModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/config.yaml")
MODELS_DIR  = Path("models")


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    sample_rate: int = 16000,
    chunk_seconds: float = 1.0,
    opset_version: int = 17,
) -> dict:
    """
    Export PyTorch model to ONNX format.

    Args:
        model:          Trained ANCAudioModel (in eval mode).
        output_path:    Output .onnx file path.
        sample_rate:    Audio sample rate in Hz.
        chunk_seconds:  Input chunk length in seconds.
        opset_version:  ONNX opset version.

    Returns:
        Dict with export metadata (path, size_mb, opset).
    """
    import onnx

    model.eval()
    chunk_samples = int(sample_rate * chunk_seconds)
    dummy_input   = torch.randn(1, 1, chunk_samples)  # [batch, channels, samples]

    output_path = str(output_path)
    logger.info(f"Exporting to ONNX: {output_path}")
    logger.info(f"  Input shape: {list(dummy_input.shape)}")
    logger.info(f"  ONNX opset: {opset_version}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["noisy_audio"],
        output_names=["enhanced_audio"],
        dynamic_axes={
            "noisy_audio":    {0: "batch_size", 2: "samples"},
            "enhanced_audio": {0: "batch_size", 2: "samples"},
        },
    )

    # Validate the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model validation: PASSED")

    size_mb = Path(output_path).stat().st_size / (1024 ** 2)
    logger.info(f"ONNX model size: {size_mb:.2f} MB")

    return {
        "path":          output_path,
        "size_mb":       round(size_mb, 4),
        "opset_version": opset_version,
        "input_shape":   list(dummy_input.shape),
    }


def benchmark_onnx(
    onnx_path: str,
    sample_rate: int = 16000,
    chunk_seconds: float = 1.0,
    num_runs: int = 100,
) -> dict:
    """
    Benchmark ONNX model inference latency.

    Args:
        onnx_path:     Path to .onnx model.
        sample_rate:   Audio sample rate.
        chunk_seconds: Input chunk size in seconds.
        num_runs:      Number of inference runs to average.

    Returns:
        Dict with latency stats (mean, p50, p95, p99, real_time_factor).
    """
    import onnxruntime as ort

    chunk_samples = int(sample_rate * chunk_seconds)
    dummy_input   = np.random.randn(1, 1, chunk_samples).astype(np.float32)

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    audio_duration_ms = chunk_seconds * 1000.0
    real_time_factor  = np.mean(latencies) / audio_duration_ms

    results = {
        "latency_mean_ms":  round(float(np.mean(latencies)),   2),
        "latency_p50_ms":   round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms":   round(float(np.percentile(latencies, 95)), 2),
        "latency_p99_ms":   round(float(np.percentile(latencies, 99)), 2),
        "real_time_factor": round(float(real_time_factor), 4),
        "real_time_capable": real_time_factor < 1.0,
    }

    logger.info(f"Latency (mean): {results['latency_mean_ms']} ms")
    logger.info(f"Latency (p95):  {results['latency_p95_ms']} ms")
    logger.info(f"Real-time factor: {real_time_factor:.3f}x "
                f"({'OK' if results['real_time_capable'] else 'TOO SLOW'})")

    return results


def quantize_onnx(input_path: str, output_path: str, mode: str = "dynamic") -> dict:
    """
    Quantize ONNX model for faster edge inference.

    Args:
        input_path:  Path to FP32 .onnx model.
        output_path: Path for quantized .onnx model.
        mode:        'dynamic' (INT8 weights) or 'static' (requires calibration data).

    Returns:
        Dict with size reduction stats.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    logger.info(f"Quantizing ONNX model: {mode}")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    original_mb   = Path(input_path).stat().st_size  / (1024 ** 2)
    quantized_mb  = Path(output_path).stat().st_size / (1024 ** 2)
    reduction_pct = (1 - quantized_mb / original_mb) * 100

    logger.info(f"Original:  {original_mb:.2f} MB")
    logger.info(f"Quantized: {quantized_mb:.2f} MB  ({reduction_pct:.1f}% reduction)")

    return {
        "original_size_mb":  round(original_mb,  4),
        "quantized_size_mb": round(quantized_mb,  4),
        "size_reduction_pct": round(reduction_pct, 2),
        "quantized_path":    output_path,
    }


def main():
    p = argparse.ArgumentParser(description="COFFEEBEAN ONNX Export")
    p.add_argument("--checkpoint", default="models/anc_checkpoint.pt")
    p.add_argument("--output",     default="models/anc_model.onnx")
    p.add_argument("--quantize",   action="store_true", help="Quantize to INT8")
    p.add_argument("--benchmark",  action="store_true", help="Run latency benchmark")
    p.add_argument("--log-mlflow", action="store_true", help="Log results to MLflow")
    p.add_argument("--run-id",     default=None,        help="MLflow run ID to log to")
    args = p.parse_args()

    cfg    = yaml.safe_load(CONFIG_PATH.read_text())
    sr     = cfg["audio"]["sample_rate"]
    MODELS_DIR.mkdir(exist_ok=True)

    # Load model
    model = ANCAudioModel(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Export FP32 ONNX
    export_info = export_to_onnx(model, args.output, sample_rate=sr)
    results = {"export": export_info}

    # Quantize
    if args.quantize:
        q_path = args.output.replace(".onnx", "_int8.onnx")
        quant_info = quantize_onnx(args.output, q_path)
        results["quantization"] = quant_info

    # Benchmark
    if args.benchmark:
        bench = benchmark_onnx(args.output, sample_rate=sr)
        results["benchmark_fp32"] = bench

        if args.quantize:
            bench_q = benchmark_onnx(q_path, sample_rate=sr)
            results["benchmark_int8"] = bench_q

    # Save results
    results_path = MODELS_DIR / "export_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved: {results_path}")

    # Log to MLflow
    if args.log_mlflow:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        ctx = mlflow.start_run(run_id=args.run_id) if args.run_id else mlflow.start_run()
        with ctx:
            mlflow.log_artifact(args.output)
            mlflow.log_artifact(str(results_path))
            if "benchmark_fp32" in results:
                mlflow.log_metrics(results["benchmark_fp32"])
            if "quantization" in results:
                mlflow.log_metrics({
                    "onnx_size_mb":         results["quantization"]["quantized_size_mb"],
                    "size_reduction_pct":   results["quantization"]["size_reduction_pct"],
                })

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
