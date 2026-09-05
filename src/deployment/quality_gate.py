"""
COFFEEBEAN — Quality Gate
Phase 6: Automated model acceptance/rejection based on evaluation metrics.

Usage:
    python src/deployment/quality_gate.py --run-id <mlflow_run_id>
"""

import os
import sys
import io
import json
import argparse
import logging
import mlflow
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/Zenith1415/COFFEEBEAN.mlflow")

# ── Default thresholds (team sets these based on experiments + hardware) ────────
# These are NOT official DRDO/SIH thresholds — set by team after experimentation.
DEFAULT_THRESHOLDS = {
    "min_snr_improvement": 3.0,    # dB
    "min_avg_stoi":        0.65,   # 0-1
    "min_avg_pesq":        2.0,    # -0.5 to 4.5
    "max_model_size_mb":   50.0,   # MB
    "max_final_train_loss": 0.1,   # training convergence check
}


def load_thresholds(path: str = "configs/config.yaml") -> dict:
    """Load quality gate thresholds from config if defined, else use defaults."""
    try:
        import yaml
        cfg = yaml.safe_load(Path(path).read_text())
        return cfg.get("quality_gate", DEFAULT_THRESHOLDS)
    except Exception:
        return DEFAULT_THRESHOLDS


def get_run_metrics(run_id: str) -> dict:
    """Fetch all metrics from an MLflow run."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    run    = client.get_run(run_id)
    return dict(run.data.metrics)


def evaluate_quality_gate(metrics: dict, thresholds: dict) -> tuple[bool, list, list]:
    """
    Check metrics against thresholds.

    Returns:
        (passed: bool, failures: list[str], warnings: list[str])
    """
    failures = []
    warnings = []

    def check(metric_key, threshold_key, operator=">=", label=None):
        label = label or metric_key
        val   = metrics.get(metric_key)
        thr   = thresholds.get(threshold_key)
        if val is None:
            warnings.append(f"MISSING metric '{metric_key}' — skipping check")
            return
        if operator == ">=" and val < thr:
            failures.append(f"FAIL: {label} = {val:.4f} < {thr} (required >= {thr})")
        elif operator == "<=" and val > thr:
            failures.append(f"FAIL: {label} = {val:.4f} > {thr} (required <= {thr})")
        else:
            logger.info(f"  PASS: {label} = {val:.4f} ({operator} {thr})")

    check("avg_snr_improvement",  "min_snr_improvement",  ">=", "SNR improvement")
    check("avg_stoi",             "min_avg_stoi",          ">=", "STOI")
    check("avg_pesq",             "min_avg_pesq",          ">=", "PESQ")
    check("model_size_mb",        "max_model_size_mb",     "<=", "Model size MB")
    check("final_train_loss",     "max_final_train_loss",  "<=", "Final train loss")

    return len(failures) == 0, failures, warnings


def run_quality_gate(run_id: str, auto_register: bool = False) -> bool:
    """
    Run quality gate for an MLflow run.

    Args:
        run_id:        MLflow run ID to evaluate.
        auto_register: If True and passed, auto-register to Model Registry.

    Returns:
        True if passed, False if failed.
    """
    thresholds = load_thresholds()
    metrics    = get_run_metrics(run_id)

    logger.info(f"\n{'='*50}")
    logger.info(f"Quality Gate — Run: {run_id}")
    logger.info(f"{'='*50}")
    logger.info(f"Thresholds: {json.dumps(thresholds, indent=2)}")
    logger.info(f"\nMetrics:")
    for k, v in sorted(metrics.items()):
        logger.info(f"  {k}: {v}")
    logger.info("")

    passed, failures, warnings = evaluate_quality_gate(metrics, thresholds)

    # Tag the run with gate result
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "quality_gate",        "PASSED" if passed else "FAILED")
    client.set_tag(run_id, "quality_gate_details", str(failures) if failures else "all checks passed")

    for w in warnings:
        logger.warning(w)

    if passed:
        logger.info("QUALITY GATE: PASSED")
        if auto_register:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from src.deployment.registry import register_model, promote_to_staging
            version = register_model(run_id)
            promote_to_staging(version)
            logger.info(f"Model auto-registered as v{version} (Staging)")
    else:
        logger.error("QUALITY GATE: FAILED")
        for f in failures:
            logger.error(f"  {f}")

    return passed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="COFFEEBEAN Quality Gate")
    p.add_argument("--run-id",       required=True, help="MLflow run ID")
    p.add_argument("--auto-register", action="store_true",
                   help="Auto-register to Model Registry if gate passes")
    args = p.parse_args()

    passed = run_quality_gate(args.run_id, args.auto_register)
    sys.exit(0 if passed else 1)
