"""
COFFEEBEAN — Training Entry Point
Phase 3: MLflow experiment tracking stub.
Phase 4: Replace placeholder metrics with real PyTorch training loop.
"""

import sys
import io
# Fix Windows cp1252 terminal — MLflow prints emoji in run summary
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import yaml
import mlflow
import mlflow.pytorch
from pathlib import Path


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "COFFEEBEAN_ANC"
CONFIG_PATH = Path("configs/config.yaml")


def load_config() -> dict:
    """Load project configuration from YAML."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text())


def train():
    """
    Main training function.

    Phase 3: Logs config params and placeholder metrics to MLflow.
    Phase 4: Replace the placeholder section with real model training.
    """
    cfg = load_config()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Experiment  : {EXPERIMENT_NAME}")
        print(f"Tracking URI: {MLFLOW_TRACKING_URI}")

        # ── Log parameters from config ─────────────────────────────────────
        mlflow.log_param("batch_size",     cfg["training"]["batch_size"])
        mlflow.log_param("epochs",         cfg["training"]["epochs"])
        mlflow.log_param("learning_rate",  cfg["training"]["learning_rate"])
        mlflow.log_param("sample_rate",    cfg["audio"]["sample_rate"])
        mlflow.log_param("channels",       cfg["audio"]["channels"])
        mlflow.log_param("model_arch",     "placeholder")
        mlflow.log_param("dataset_version","v1")

        # ── Log config as artifact ─────────────────────────────────────────
        mlflow.log_artifact(str(CONFIG_PATH))

        # ── PHASE 4: Replace this block with real training loop ────────────
        # TODO: load dataset via DVC
        # TODO: instantiate your ANC model
        # TODO: run training loop, log metrics per epoch
        # ──────────────────────────────────────────────────────────────────

        # Placeholder metrics (stand-ins until real model exists)
        placeholder_metrics = {
            "snr_before":       5.0,
            "snr_after":        15.0,
            "snr_improvement":  10.0,
            "stoi":             0.75,
            "pesq":             2.5,
            "val_loss":         0.05,
            "inference_latency_ms": 0.0,
            "model_size_mb":    0.0,
        }

        for metric, value in placeholder_metrics.items():
            mlflow.log_metric(metric, value)

        print("\nMetrics logged:")
        for k, v in placeholder_metrics.items():
            print(f"  {k}: {v}")

        print(f"\nRun complete. View at: {MLFLOW_TRACKING_URI}")

    return run.info.run_id


if __name__ == "__main__":
    train()
