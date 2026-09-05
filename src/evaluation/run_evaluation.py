"""
COFFEEBEAN — Standalone Evaluation Script
Phase 5: Run evaluation on a trained model checkpoint and log to MLflow.

Usage:
    python src/evaluation/run_evaluation.py --checkpoint models/anc_checkpoint.pt
    python src/evaluation/run_evaluation.py --run-id <mlflow_run_id>
"""

import os
import sys
import io
import argparse
import logging
import yaml
import torch
import mlflow
import numpy as np
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.model import get_model, ANCAudioModel
from src.preprocessing.preprocess import load_dataset, ANCDataset, mix_signals
from src.evaluation.evaluate import evaluate_anc_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Auto-load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/Zenith1415/COFFEEBEAN.mlflow")
CONFIG_PATH = Path("configs/config.yaml")


def parse_args():
    p = argparse.ArgumentParser(description="COFFEEBEAN ANC Evaluation")
    p.add_argument("--checkpoint", type=str, default="models/anc_checkpoint.pt",
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--run-id",     type=str, default=None,
                   help="Existing MLflow run ID to log metrics to")
    p.add_argument("--snr-levels", type=float, nargs="+", default=[-5.0, 0.0, 5.0, 10.0],
                   help="Input SNR levels (dB) to evaluate at")
    p.add_argument("--num-pairs",  type=int, default=10,
                   help="Number of speech/noise pairs to average over")
    return p.parse_args()


def run_evaluation(checkpoint: str, run_id: str, snr_levels: list, num_pairs: int):
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    sr  = cfg["audio"]["sample_rate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(cfg).to(device)
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # Load dataset
    raw = load_dataset("data/raw", target_sr=sr)
    speech_files = raw.get("speech", [])
    noise_files  = (raw.get("stationary_noise", []) +
                    raw.get("nonstationary_noise", []) +
                    raw.get("impulsive_noise", []))

    if not speech_files or not noise_files:
        logger.error("Need both speech and noise files in data/raw/ to evaluate.")
        sys.exit(1)

    # Evaluate across SNR levels
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("COFFEEBEAN_ANC")

    all_results = {}
    for snr in snr_levels:
        batch_metrics = []
        for i in range(min(num_pairs, len(speech_files))):
            clean = speech_files[i % len(speech_files)]
            noise = noise_files[i % len(noise_files)]
            try:
                m = evaluate_anc_model(model, clean, noise, sr,
                                       target_snr_db=snr, device=device)
                batch_metrics.append(m)
            except Exception as e:
                logger.warning(f"Eval failed for pair {i} at SNR={snr}: {e}")

        if batch_metrics:
            avg = {}
            for key in batch_metrics[0]:
                vals = [m[key] for m in batch_metrics if m[key] is not None]
                avg[key] = round(float(np.mean(vals)), 4) if vals else None
            snr_name = f"minus_{abs(int(snr))}" if snr < 0 else f"plus_{int(snr)}"
            all_results[f"snr_{snr_name}dB"] = avg
            logger.info(f"SNR={snr:+.0f}dB -> loss={avg.get('eval_loss')} | "
                        f"improvement={avg.get('snr_improvement')} dB | "
                        f"STOI={avg.get('stoi_enhanced')} | PESQ={avg.get('pesq_enhanced')}")

    # Log to MLflow
    ctx = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run()
    with ctx as run:
        mlflow.log_param("eval_checkpoint", str(ckpt_path))
        mlflow.log_param("eval_snr_levels", str(snr_levels))
        mlflow.log_param("eval_num_pairs",  num_pairs)

        for snr_tag, metrics in all_results.items():
            for k, v in metrics.items():
                if v is not None:
                    mlflow.log_metric(f"{snr_tag}_{k}", float(v))

        # Log overall averages across all SNR levels
        all_eval_losses  = [r.get("eval_loss") for r in all_results.values()
                            if r.get("eval_loss") is not None]
        all_improvements = [r.get("snr_improvement") for r in all_results.values()
                            if r.get("snr_improvement") is not None]
        all_stoi = [r.get("stoi_enhanced") for r in all_results.values()
                    if r.get("stoi_enhanced") is not None]
        all_pesq = [r.get("pesq_enhanced") for r in all_results.values()
                    if r.get("pesq_enhanced") is not None]

        if all_eval_losses:
            mlflow.log_metric("avg_eval_loss", round(float(np.mean(all_eval_losses)), 6))
        if all_improvements:
            mlflow.log_metric("avg_snr_improvement", round(np.mean(all_improvements), 4))
        if all_stoi:
            mlflow.log_metric("avg_stoi", round(np.mean(all_stoi), 4))
        if all_pesq:
            mlflow.log_metric("avg_pesq", round(np.mean(all_pesq), 4))

        logger.info(f"Evaluation logged to MLflow run: {run.info.run_id}")

    return all_results


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.checkpoint, args.run_id, args.snr_levels, args.num_pairs)
