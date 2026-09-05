"""Airflow entry point for the containerized RNNoise training job."""

from __future__ import annotations

import subprocess
from datetime import timedelta

from airflow.sdk import Param, dag, get_current_context, task


@dag(
    dag_id="rnnoise_retraining",
    description="Build, train, validate, and export an RNNoise v0.2 candidate",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0},
    params={
        "train_speech": Param("/data/input/train-speech.pcm", type="string"),
        "train_noise": Param("/data/input/train-noise.pcm", type="string"),
        "eval_speech": Param("/data/input/eval-speech.pcm", type="string"),
        "eval_noise": Param("/data/input/eval-noise.pcm", type="string"),
        "data_card": Param("/data/input/DATA_CARD.md", type="string"),
        "train_sequences": Param(200_000, type="integer", minimum=1),
        "eval_sequences": Param(2_000, type="integer", minimum=1),
        "batch_size": Param(128, type="integer", minimum=1),
        "epochs": Param(0, type="integer", minimum=0),
        "seed": Param(1_969_587, type="integer"),
        "max_validation_loss": Param(None, type=["null", "number"]),
        "baseline_validation_loss": Param(None, type=["null", "number"]),
    },
    tags=["mlops", "rnnoise"],
)
def rnnoise_retraining():
    @task(execution_timeout=timedelta(days=3))
    def run() -> None:
        context = get_current_context()
        params = context["params"]
        output = f"/data/runs/{context['run_id'].replace(':', '_').replace('/', '_')}"
        command = [
            "python", "/opt/coffeebean/mlops/rnnoise_pipeline.py", "run",
            "--train-speech", params["train_speech"],
            "--train-noise", params["train_noise"],
            "--eval-speech", params["eval_speech"],
            "--eval-noise", params["eval_noise"],
            "--data-card", params["data_card"],
            "--output", output,
            "--train-sequences", str(params["train_sequences"]),
            "--eval-sequences", str(params["eval_sequences"]),
            "--batch-size", str(params["batch_size"]),
            "--epochs", str(params["epochs"]),
            "--seed", str(params["seed"]),
            "--run-name", context["run_id"],
        ]
        for name in ("max_validation_loss", "baseline_validation_loss"):
            if params[name] is not None:
                command += ["--" + name.replace("_", "-"), str(params[name])]
        subprocess.run(command, check=True)

    run()


rnnoise_retraining()
