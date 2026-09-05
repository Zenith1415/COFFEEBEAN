"""
COFFEEBEAN Airflow DAG — Full MLOps Pipeline
Orchestrates DVC pull → preprocess → train → evaluate → quality gate

Usage (after airflow init):
    airflow dags trigger coffeebean_anc_pipeline

Environment variables needed:
    MLFLOW_TRACKING_URI       = https://dagshub.com/Zenith1415/COFFEEBEAN.mlflow
    MLFLOW_TRACKING_USERNAME  = Zenith1415
    MLFLOW_TRACKING_PASSWORD  = <your_dagshub_token>
    PYTHONUTF8                = 1
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# ── Default args ───────────────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner":            "coffeebean-mlops",
    "depends_on_past":  False,
    "start_date":       datetime(2026, 9, 1),
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

PROJECT_DIR = os.getenv("COFFEEBEAN_DIR", "/app")

# Quality gate thresholds — set by team based on experiments
QUALITY_GATE = {
    "min_snr_improvement": 3.0,    # dB — minimum acceptable improvement
    "min_stoi_enhanced":   0.65,   # minimum intelligibility
    "min_pesq_enhanced":   2.0,    # minimum perceived quality
    "max_model_size_mb":   50.0,   # MB — max model size for edge
}

# ── Task functions ─────────────────────────────────────────────────────────────

def task_dvc_pull(**context):
    """Pull latest dataset version from DVC remote (MinIO/Google Drive)."""
    result = subprocess.run(
        ["dvc", "pull"],
        cwd=PROJECT_DIR,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"dvc pull failed:\n{result.stderr}")
    print(result.stdout)
    return "dvc pull successful"


def task_preprocess(**context):
    """Validate and scan dataset structure."""
    import sys
    sys.path.insert(0, PROJECT_DIR)
    from src.preprocessing.preprocess import load_dataset
    import yaml

    cfg = yaml.safe_load(open(f"{PROJECT_DIR}/configs/config.yaml"))
    dataset = load_dataset(f"{PROJECT_DIR}/data/raw", cfg["audio"]["sample_rate"])

    total = sum(len(v) for v in dataset.values())
    print(f"Dataset validated: {total} audio files across {len(dataset)} categories")

    if total == 0:
        raise ValueError("No audio files found — add data to data/raw/ and dvc push")

    context["ti"].xcom_push(key="dataset_file_count", value=total)
    return total


def task_train(**context):
    """Run training and return MLflow run_id via XCom."""
    import sys
    sys.path.insert(0, PROJECT_DIR)
    os.environ.setdefault("PYTHONUTF8", "1")

    from src.training.train import train
    run_id = train()

    context["ti"].xcom_push(key="mlflow_run_id", value=run_id)
    print(f"Training complete. MLflow run_id: {run_id}")
    return run_id


def task_evaluate(**context):
    """Run evaluation on the trained checkpoint and log to MLflow."""
    import sys
    sys.path.insert(0, PROJECT_DIR)

    run_id = context["ti"].xcom_pull(key="mlflow_run_id", task_ids="train")

    from src.evaluation.run_evaluation import run_evaluation
    results = run_evaluation(
        checkpoint=f"{PROJECT_DIR}/models/anc_checkpoint.pt",
        run_id=run_id,
        snr_levels=[-5.0, 0.0, 5.0, 10.0],
        num_pairs=5,
    )

    # Flatten for XCom
    flat = {}
    for snr_tag, metrics in results.items():
        for k, v in metrics.items():
            flat[f"{snr_tag}__{k}"] = v

    context["ti"].xcom_push(key="eval_metrics", value=flat)
    return flat


def task_quality_gate(**context):
    """Check evaluation metrics against thresholds. Return branch name."""
    import numpy as np
    flat_metrics = context["ti"].xcom_pull(key="eval_metrics", task_ids="evaluate")

    # Collect averages across SNR levels
    snr_improvements = [v for k, v in flat_metrics.items()
                        if k.endswith("__snr_improvement") and v is not None]
    stoi_scores      = [v for k, v in flat_metrics.items()
                        if k.endswith("__stoi_enhanced") and v is not None]
    pesq_scores      = [v for k, v in flat_metrics.items()
                        if k.endswith("__pesq_enhanced") and v is not None]

    avg_snr  = float(np.mean(snr_improvements)) if snr_improvements else 0.0
    avg_stoi = float(np.mean(stoi_scores))      if stoi_scores      else 0.0
    avg_pesq = float(np.mean(pesq_scores))      if pesq_scores      else 0.0

    print(f"Quality Gate Check:")
    print(f"  SNR improvement: {avg_snr:.2f} dB  (min: {QUALITY_GATE['min_snr_improvement']})")
    print(f"  STOI enhanced:   {avg_stoi:.4f}    (min: {QUALITY_GATE['min_stoi_enhanced']})")
    print(f"  PESQ enhanced:   {avg_pesq:.4f}    (min: {QUALITY_GATE['min_pesq_enhanced']})")

    passed = (
        avg_snr  >= QUALITY_GATE["min_snr_improvement"] and
        avg_stoi >= QUALITY_GATE["min_stoi_enhanced"]   and
        avg_pesq >= QUALITY_GATE["min_pesq_enhanced"]
    )

    if passed:
        print("QUALITY GATE: PASSED")
        return "register_model"
    else:
        print("QUALITY GATE: FAILED — model rejected")
        return "reject_model"


def task_register_model(**context):
    """Register approved model in MLflow Model Registry."""
    import sys, mlflow
    sys.path.insert(0, PROJECT_DIR)

    run_id = context["ti"].xcom_pull(key="mlflow_run_id", task_ids="train")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    model_uri = f"runs:/{run_id}/anc_model"
    result    = mlflow.register_model(model_uri, "COFFEEBEAN_ANC")
    print(f"Model registered: version {result.version}")

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="COFFEEBEAN_ANC",
        version=result.version,
        stage="Staging",
    )
    print(f"Model v{result.version} moved to Staging")
    return result.version


def task_reject_model(**context):
    """Log rejection reason."""
    print("Model did not meet quality thresholds. Marking run as FAILED.")
    run_id = context["ti"].xcom_pull(key="mlflow_run_id", task_ids="train")
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "quality_gate", "FAILED")


# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="coffeebean_anc_pipeline",
    description="COFFEEBEAN full ANC MLOps pipeline: DVC → Train → Evaluate → Quality Gate",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,      # Manual trigger only (no scheduled runs yet)
    catchup=False,
    tags=["coffeebean", "anc", "mlops"],
) as dag:

    start = EmptyOperator(task_id="start")

    pull = PythonOperator(
        task_id="dvc_pull",
        python_callable=task_dvc_pull,
    )

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=task_train,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=task_evaluate,
    )

    quality_gate = BranchPythonOperator(
        task_id="quality_gate",
        python_callable=task_quality_gate,
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=task_register_model,
    )

    reject = PythonOperator(
        task_id="reject_model",
        python_callable=task_reject_model,
    )

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    # ── DAG edges ──────────────────────────────────────────────────────────────
    start >> pull >> preprocess >> train >> evaluate >> quality_gate
    quality_gate >> [register, reject]
    [register, reject] >> end
