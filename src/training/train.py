"""
COFFEEBEAN — Training Entry Point
Phase 4: Real PyTorch training loop with DVC dataset + MLflow logging.

Usage:
    python src/training/train.py

Environment variables (set before running):
    MLFLOW_TRACKING_URI      default: http://localhost:5000
    MLFLOW_S3_ENDPOINT_URL   default: http://localhost:9000
    AWS_ACCESS_KEY_ID        default: minioadmin
    AWS_SECRET_ACCESS_KEY    default: minioadmin
    PYTHONUTF8=1             recommended on Windows
"""

import sys
import io

# Fix Windows cp1252 terminal — MLflow prints emoji in run summary
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import json
import logging
import time
import yaml
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.preprocess import load_dataset, ANCDataset
from src.training.model import ANCAudioModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Auto-load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/Zenith1415/COFFEEBEAN.mlflow")
EXPERIMENT_NAME = "COFFEEBEAN_ANC"
CONFIG_PATH = Path("configs/config.yaml")
MODELS_DIR = Path("models")
METRICS_FILE = Path("metrics.json")


def load_config() -> dict:
    """Load project configuration from YAML."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one full training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        enhanced = model(noisy)
        loss = loss_fn(enhanced, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train():
    """
    Main training function.
    Reads config → loads DVC dataset → trains model → logs to MLflow.
    """
    cfg = load_config()
    MODELS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load dataset from DVC-tracked data/raw/ ────────────────────────────
    logger.info("Loading dataset from data/raw/ ...")
    raw_dataset = load_dataset("data/raw", target_sr=cfg["audio"]["sample_rate"])

    total_files = sum(len(v) for v in raw_dataset.values())
    logger.info(f"Total audio files loaded: {total_files}")

    torch_dataset = ANCDataset(
        dataset=raw_dataset,
        chunk_samples=cfg["audio"]["sample_rate"],   # 1 second chunks
        snr_range_db=(-5.0, 20.0),
        num_samples=max(total_files * 10, 100),      # scale with dataset size
    )

    loader = DataLoader(
        torch_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,        # 0 for Windows compatibility
        drop_last=True,
    )

    # ── Build model ────────────────────────────────────────────────────────
    model = ANCAudioModel(cfg).to(device)
    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"Model size: {model.model_size_mb():.2f} MB")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
    )
    loss_fn = nn.MSELoss()

    # ── MLflow run ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Log parameters
        mlflow.log_params({
            "batch_size":       cfg["training"]["batch_size"],
            "epochs":           cfg["training"]["epochs"],
            "learning_rate":    cfg["training"]["learning_rate"],
            "sample_rate":      cfg["audio"]["sample_rate"],
            "model_arch":       model.__class__.__name__,
            "model_params":     model.count_parameters(),
            "model_size_mb":    round(model.model_size_mb(), 4),
            "device":           str(device),
            "dataset_files":    total_files,
            "dataset_version":  "v1",
        })

        # ── Training loop ──────────────────────────────────────────────────
        logger.info(f"Starting training for {cfg['training']['epochs']} epochs ...")
        best_loss = float("inf")

        for epoch in range(cfg["training"]["epochs"]):
            t0 = time.time()
            loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
            elapsed = time.time() - t0

            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("epoch_time_s", elapsed, step=epoch)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}/{cfg['training']['epochs']} "
                    f"| loss={loss:.6f} | time={elapsed:.2f}s"
                )

            if loss < best_loss:
                best_loss = loss

        # ── Save model ─────────────────────────────────────────────────────
        checkpoint_path = MODELS_DIR / "anc_checkpoint.pt"
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # ── Log artifacts ──────────────────────────────────────────────────
        mlflow.log_artifact(str(checkpoint_path))
        mlflow.log_artifact(str(CONFIG_PATH))
        mlflow.pytorch.log_model(model, "anc_model")

        # Final metrics
        final_metrics = {
            "final_train_loss":  best_loss,
            "model_size_mb":     model.model_size_mb(),
            "total_epochs":      cfg["training"]["epochs"],
            # Phase 5: replace these with real SNR/STOI/PESQ
            "snr_before":        None,
            "snr_after":         None,
            "snr_improvement":   None,
            "stoi":              None,
            "pesq":              None,
        }

        for k, v in final_metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)

        # Write DVC metrics file
        dvc_metrics = {
            "final_train_loss": best_loss,
            "model_size_mb":    model.model_size_mb(),
        }
        METRICS_FILE.write_text(json.dumps(dvc_metrics, indent=2))
        mlflow.log_artifact(str(METRICS_FILE))

        logger.info(f"Training complete. Best loss: {best_loss:.6f}")
        logger.info(f"View run at: {MLFLOW_TRACKING_URI}/#/experiments/1/runs/{run.info.run_id}")

    return run.info.run_id


if __name__ == "__main__":
    train()
