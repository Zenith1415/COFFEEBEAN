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
from src.training.model import get_model, ANCAudioModel

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
    """Run one full training epoch. Returns mean training loss."""
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


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model on validation loader. Returns mean eval loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            enhanced = model(noisy)
            loss = loss_fn(enhanced, clean)
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train():
    """
    Main training function.
    Reads config → loads DVC dataset → trains model → logs to MLflow.
    """
    print("=" * 60, flush=True)
    print("  COFFEEBEAN ANC Model Training Pipeline", flush=True)
    print("=" * 60, flush=True)
    cfg = load_config()
    MODELS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Compute Device: {device}", flush=True)

    # ── Load dataset from DVC-tracked data/raw/ ────────────────────────────
    print("📂 Loading dataset from data/raw/ ...", flush=True)
    raw_dataset = load_dataset("data/raw", target_sr=cfg["audio"]["sample_rate"])

    total_files = sum(len(v) for v in raw_dataset.values())
    print(f"📦 Total audio files loaded: {total_files}", flush=True)

    full_dataset = ANCDataset(
        dataset=raw_dataset,
        chunk_samples=cfg["audio"]["sample_rate"],   # 1 second chunks
        snr_range_db=(-5.0, 20.0),
        num_samples=max(total_files * 10, 100),      # scale with dataset size
    )

    val_size = max(int(len(full_dataset) * 0.2), 1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,        # 0 for Windows compatibility
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ── Build model ────────────────────────────────────────────────────────
    model = get_model(cfg).to(device)
    logger.info(f"Model ({cfg.get('model', {}).get('type', 'default')}): {model.count_parameters():,} parameters")
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
            "train_samples":    train_size,
            "val_samples":      val_size,
            "dataset_version":  "v1",
        })

        # ── Training loop ──────────────────────────────────────────────────
        print(f"\n🚀 Starting training for {cfg['training']['epochs']} epochs on {device}...", flush=True)
        print(f"📊 Tracking live at: {MLFLOW_TRACKING_URI}\n", flush=True)
        best_train_loss = float("inf")
        best_val_loss   = float("inf")

        for epoch in range(cfg["training"]["epochs"]):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss   = evaluate_loss(model, val_loader, loss_fn, device)
            elapsed = time.time() - t0

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss",   val_loss,   step=epoch)
            mlflow.log_metric("eval_loss",  val_loss,   step=epoch)
            mlflow.log_metric("epoch_time_s", elapsed,  step=epoch)

            print(
                f"  Epoch [{epoch+1:2d}/{cfg['training']['epochs']:2d}] "
                f"-> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {elapsed:.2f}s",
                flush=True,
            )

            if train_loss < best_train_loss:
                best_train_loss = train_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

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
            "final_train_loss":  best_train_loss,
            "final_val_loss":    best_val_loss,
            "final_eval_loss":   best_val_loss,
            "model_size_mb":     model.model_size_mb(),
            "total_epochs":      cfg["training"]["epochs"],
        }

        for k, v in final_metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)

        # Write DVC metrics file
        dvc_metrics = {
            "final_train_loss": best_train_loss,
            "final_val_loss":   best_val_loss,
            "final_eval_loss":  best_val_loss,
            "model_size_mb":    model.model_size_mb(),
        }
        METRICS_FILE.write_text(json.dumps(dvc_metrics, indent=2))
        mlflow.log_artifact(str(METRICS_FILE))

        logger.info(f"Training complete. Best train loss: {best_train_loss:.6f} | Best val/eval loss: {best_val_loss:.6f}")
        logger.info(f"View run at: {MLFLOW_TRACKING_URI}/#/experiments/1/runs/{run.info.run_id}")

    return run.info.run_id


if __name__ == "__main__":
    train()
