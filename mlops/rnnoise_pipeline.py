#!/usr/bin/env python3
"""Reproducible RNNoise v0.2 retraining with MLflow lineage and quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RNNOISE_COMMIT = "904a876dce1f9ab8860c0a5000ed151f9f6eef58"
RNNOISE_ARCHIVE_URL = (
    "https://codeload.github.com/xiph/rnnoise/tar.gz/" + RNNOISE_COMMIT
)
RNNOISE_ARCHIVE_SHA256 = (
    "975488a6b6ed404b176f9e7a72b418470e50b1e7596f875eb7799f5b785e3ebf"
)
RNNOISE_MODEL_VERSION = "0b50c45"
RNNOISE_MODEL_URL = (
    "https://media.xiph.org/rnnoise/models/"
    f"rnnoise_data-{RNNOISE_MODEL_VERSION}.tar.gz"
)
RNNOISE_MODEL_SHA256 = (
    "4ac81c5c0884ec4bd5907026aaae16209b7b76cd9d7f71af582094a2f98f4b43"
)
SOURCE_CACHE_FINGERPRINT = f"{RNNOISE_ARCHIVE_SHA256}:{RNNOISE_MODEL_SHA256}"
FEATURE_WIDTH = 98
SEQUENCE_LENGTH = 2_000
FRAME_SIZE = 480
SAMPLE_RATE = 48_000


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pcm(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    size = path.stat().st_size
    if size == 0 or size % 2:
        raise ValueError(f"{label} must be non-empty, headerless int16 PCM: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "duration_seconds": size / 2 / SAMPLE_RATE,
        "sha256": sha256_file(path),
    }


def recommended_epochs(sequence_count: int, batch_size: int) -> int:
    batches = sequence_count // batch_size
    if batches < 1:
        raise ValueError("train sequences must be at least one full batch")
    return max(1, math.ceil(75_000 / batches))


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def prepare_source(cache_dir: Path) -> Path:
    source = cache_dir / f"rnnoise-{RNNOISE_COMMIT}"
    marker = source / ".coffeebean-source-sha256"
    if (
        marker.is_file()
        and marker.read_text().strip() == SOURCE_CACHE_FINGERPRINT
        and (source / "dump_features").is_file()
    ):
        return source

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as temp_name:
        temp = Path(temp_name)
        archive = temp / "rnnoise.tar.gz"
        print(f"Downloading pinned RNNoise source {RNNOISE_COMMIT}", flush=True)
        urllib.request.urlretrieve(RNNOISE_ARCHIVE_URL, archive)
        actual_hash = sha256_file(archive)
        if actual_hash != RNNOISE_ARCHIVE_SHA256:
            raise RuntimeError(
                f"RNNoise source checksum mismatch: {actual_hash}"
            )
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(temp, filter="data")
        extracted = next(path for path in temp.iterdir() if path.is_dir())
        if source.exists():
            shutil.rmtree(source)
        shutil.move(str(extracted), source)

    feature_source = source / "src" / "dump_features.c"
    contents = feature_source.read_text(encoding="utf-8")
    needle = "seed = getpid();"
    replacement = (
        'seed = getenv("RNNOISE_FEATURE_SEED") '
        '? (unsigned)strtoul(getenv("RNNOISE_FEATURE_SEED"), NULL, 10) : getpid();'
    )
    if contents.count(needle) != 1:
        raise RuntimeError("Pinned RNNoise feature seed hook changed upstream")
    feature_source.write_text(contents.replace(needle, replacement), encoding="utf-8")

    model_archive = source / f"rnnoise_data-{RNNOISE_MODEL_VERSION}.tar.gz"
    print(f"Downloading pinned RNNoise model data {RNNOISE_MODEL_VERSION}", flush=True)
    urllib.request.urlretrieve(RNNOISE_MODEL_URL, model_archive)
    actual_model_hash = sha256_file(model_archive)
    if actual_model_hash != RNNOISE_MODEL_SHA256:
        raise RuntimeError(f"RNNoise model checksum mismatch: {actual_model_hash}")
    with tarfile.open(model_archive, "r:gz") as bundle:
        bundle.extractall(source, filter="data")

    _run(["autoreconf", "-i"], cwd=source)
    _run(["./configure", "--disable-shared"], cwd=source)
    _run(["make", "-j2", "dump_features"], cwd=source)
    marker.write_text(SOURCE_CACHE_FINGERPRINT + "\n", encoding="utf-8")
    return source


def generate_features(
    source: Path,
    speech: Path,
    noise: Path,
    output: Path,
    count: int,
    seed: int,
    rir_list: Path | None = None,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(source / "dump_features")]
    if rir_list:
        command += ["-rir_list", str(rir_list)]
    command += [str(speech), str(noise), str(output), str(count)]
    env = os.environ.copy()
    env["RNNOISE_FEATURE_SEED"] = str(seed)
    _run(command, cwd=source, env=env)
    expected = count * SEQUENCE_LENGTH * FEATURE_WIDTH * 4
    actual = output.stat().st_size
    if actual != expected:
        raise RuntimeError(f"Feature file is {actual} bytes; expected {expected}")
    return {"path": str(output), "bytes": actual, "sha256": sha256_file(output)}


def train(source: Path, features: Path, output: Path, args: argparse.Namespace) -> Path:
    epochs = args.epochs or recommended_epochs(args.train_sequences, args.batch_size)
    command = [
        sys.executable,
        str(source / "torch" / "rnnoise" / "train_rnnoise.py"),
        str(features),
        str(output),
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.learning_rate),
        "--lr-decay", str(args.learning_rate_decay),
        "--cond-size", str(args.cond_size),
        "--gru-size", str(args.gru_size),
        "--gamma", str(args.gamma),
    ]
    if args.initial_checkpoint:
        command += ["--initial-checkpoint", str(args.initial_checkpoint)]
    wrapper = (
        "import os,random,runpy,sys,numpy as np,torch;"
        "seed=int(os.environ['COFFEEBEAN_SEED']);"
        "random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);"
        "torch.cuda.manual_seed_all(seed);"
        "torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;"
        "sys.argv=" + repr(command[1:]) + ";"
        "sys.path.insert(0,os.path.dirname(sys.argv[0]));"
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    env = os.environ.copy()
    env.update({
        "COFFEEBEAN_SEED": str(args.seed),
        "PYTHONHASHSEED": str(args.seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    _run([sys.executable, "-c", wrapper], cwd=source / "torch", env=env)
    checkpoints = sorted(
        (output / "checkpoints").glob("rnnoise_*.pth"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if not checkpoints:
        raise RuntimeError("RNNoise training produced no checkpoint")
    return checkpoints[-1]


def evaluate(
    source: Path,
    checkpoint: Path,
    features: Path,
    batch_size: int,
    gamma: float,
) -> dict[str, float]:
    import numpy as np
    import torch

    sys.path.insert(0, str(source / "torch" / "rnnoise"))
    import rnnoise  # type: ignore[import-not-found]

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = rnnoise.RNNoise(*saved.get("model_args", ()), **saved["model_kwargs"])
    model.load_state_dict(saved["state_dict"], strict=False)
    model.eval()
    raw = np.memmap(features, dtype="float32", mode="r")
    sequences = raw.reshape(-1, SEQUENCE_LENGTH, FEATURE_WIDTH)
    totals = {"gain_loss": 0.0, "vad_loss": 0.0, "loss": 0.0}
    evaluated_sequences = 0
    with torch.inference_mode():
        for first in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[first : first + batch_size].copy())
            predicted_gain, predicted_vad, _states = model(batch[:, :, :65])
            gain = batch[:, 3:-1, 65:-1]
            vad = batch[:, 3:-1, -1:]
            target_gain = torch.clamp(gain, min=0)
            target_gain *= torch.tanh(5 * target_gain) ** 2
            mask = torch.clamp(gain + 1, max=1)
            gain_loss = torch.mean(mask * (predicted_gain**gamma - target_gain**gamma) ** 2)
            vad_loss = torch.mean(
                torch.abs(2 * vad - 1)
                * (-vad * torch.log(0.01 + predicted_vad)
                   - (1 - vad) * torch.log(1.01 - predicted_vad))
            )
            loss = gain_loss + 0.0005 * vad_loss
            for name, value in (("gain_loss", gain_loss), ("vad_loss", vad_loss), ("loss", loss)):
                totals[name] += float(value) * len(batch)
            evaluated_sequences += len(batch)
    if not evaluated_sequences:
        raise RuntimeError("Evaluation feature set is empty")
    return {name: value / evaluated_sequences for name, value in totals.items()}


def export(source: Path, checkpoint: Path, output: Path) -> list[Path]:
    _run(
        [
            sys.executable,
            str(source / "torch" / "rnnoise" / "dump_rnnoise_weights.py"),
            "--quantize",
            str(checkpoint),
            str(output),
        ],
        cwd=source / "torch",
    )
    artifacts = [output / "rnnoise_data.c", output / "rnnoise_data.h"]
    if not all(path.is_file() and path.stat().st_size for path in artifacts):
        raise RuntimeError("RNNoise quantized C export is incomplete")
    return artifacts


def _validate_rir_list(path: Path | None) -> None:
    if path is None:
        return
    if not path.is_file():
        raise ValueError(f"RIR list does not exist: {path}")
    entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError("RIR list is empty")
    missing = [entry for entry in entries if not (path.parent / entry).is_file() and not Path(entry).is_file()]
    if missing:
        raise ValueError(f"RIR list contains missing file: {missing[0]}")


def code_lineage() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "mlops", "pyproject.toml"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "working_tree_dirty": dirty.returncode != 0 or bool(dirty.stdout.strip()),
        "pipeline_sha256": sha256_file(Path(__file__)),
    }


def dataset_fingerprint(inputs: dict[str, dict[str, Any]], split: str) -> str:
    joined = ":".join(
        inputs[f"{split}_{kind}"]["sha256"] for kind in ("speech", "noise")
    )
    return hashlib.sha256(joined.encode()).hexdigest()


def make_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if not args.data_card.is_file():
        raise ValueError(f"Data card does not exist: {args.data_card}")
    _validate_rir_list(args.rir_list)
    inputs = {
        "train_speech": validate_pcm(args.train_speech, "train speech"),
        "train_noise": validate_pcm(args.train_noise, "train noise"),
        "eval_speech": validate_pcm(args.eval_speech, "eval speech"),
        "eval_noise": validate_pcm(args.eval_noise, "eval noise"),
    }
    for kind in ("speech", "noise"):
        if inputs[f"train_{kind}"]["sha256"] == inputs[f"eval_{kind}"]["sha256"]:
            raise ValueError(f"train and evaluation {kind} must be different corpora")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "code": code_lineage(),
        "rnnoise_commit": RNNOISE_COMMIT,
        "source_archive_sha256": RNNOISE_ARCHIVE_SHA256,
        "default_model_version": RNNOISE_MODEL_VERSION,
        "default_model_archive_sha256": RNNOISE_MODEL_SHA256,
        "data_card": {
            "path": str(args.data_card.resolve()),
            "sha256": sha256_file(args.data_card),
        },
        "inputs": inputs,
        "dataset_fingerprints": {
            split: dataset_fingerprint(inputs, split) for split in ("train", "eval")
        },
        "feature_generation": {
            "train_sequences": args.train_sequences,
            "eval_sequences": args.eval_sequences,
            "sequence_length": SEQUENCE_LENGTH,
            "feature_width": FEATURE_WIDTH,
            "frame_size": FRAME_SIZE,
            "sample_rate": SAMPLE_RATE,
            "seed": args.seed,
            "rir_list": str(args.rir_list.resolve()) if args.rir_list else None,
        },
    }


def _mlflow_context(args: argparse.Namespace):
    if args.no_mlflow:
        return nullcontext(None)
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("MLflow is unavailable; run this in the MLOps image") from error
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.enable_system_metrics_logging()
    return mlflow.start_run(run_name=args.run_name)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    manifest = make_manifest(args)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with _mlflow_context(args) as active_run:
        mlflow = None
        if active_run is not None:
            import mlflow
            mlflow.log_params({
                "rnnoise_commit": RNNOISE_COMMIT,
                "train_sequences": args.train_sequences,
                "eval_sequences": args.eval_sequences,
                "batch_size": args.batch_size,
                "epochs": args.epochs or recommended_epochs(args.train_sequences, args.batch_size),
                "learning_rate": args.learning_rate,
                "learning_rate_decay": args.learning_rate_decay,
                "cond_size": args.cond_size,
                "gru_size": args.gru_size,
                "gamma": args.gamma,
                "seed": args.seed,
                "pipeline_sha256": manifest["code"]["pipeline_sha256"],
            })
            mlflow.set_tags({
                "code.git_commit": manifest["code"]["git_commit"] or "unknown",
                "code.dirty": str(manifest["code"]["working_tree_dirty"]).lower(),
                "dataset.train.sha256": manifest["dataset_fingerprints"]["train"],
                "dataset.eval.sha256": manifest["dataset_fingerprints"]["eval"],
                "validation_status": "running",
            })
            mlflow.log_artifact(str(manifest_path), "lineage")
            mlflow.log_artifact(str(args.data_card), "lineage")

        source = prepare_source(args.source_cache)
        train_features = generate_features(
            source, args.train_speech, args.train_noise,
            args.output / "features" / "train.f32",
            args.train_sequences, args.seed, args.rir_list,
        )
        eval_features = generate_features(
            source, args.eval_speech, args.eval_noise,
            args.output / "features" / "eval.f32",
            args.eval_sequences, args.seed + 1, args.rir_list,
        )
        manifest["features"] = {"train": train_features, "eval": eval_features}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        checkpoint = train(source, Path(train_features["path"]), args.output / "training", args)
        metrics = evaluate(
            source,
            checkpoint,
            Path(eval_features["path"]),
            args.eval_batch_size,
            args.gamma,
        )
        metrics["training_loss"] = float(
            __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)["loss"]
        )
        passed = all(math.isfinite(value) for value in metrics.values())
        if args.max_validation_loss is not None:
            passed &= metrics["loss"] <= args.max_validation_loss
        if args.baseline_validation_loss is not None:
            passed &= metrics["loss"] < args.baseline_validation_loss
        status = "candidate" if passed else "rejected"
        report = {"status": status, "metrics": metrics, "checkpoint": str(checkpoint)}
        report_path = args.output / "validation.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        exported: list[Path] = []
        if passed:
            exported = export(source, checkpoint, args.output / "export")
        if mlflow:
            mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
            mlflow.set_tag("validation_status", status)
            mlflow.log_artifact(str(checkpoint), "checkpoints")
            mlflow.log_artifact(str(report_path), "validation")
            mlflow.log_artifact(str(manifest_path), "lineage")
            for artifact in exported:
                mlflow.log_artifact(str(artifact), "candidate-c-source")
        if not passed:
            raise RuntimeError(f"Candidate failed validation: {metrics}")
        return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("validate", "run"))
    result.add_argument("--train-speech", type=Path, required=True)
    result.add_argument("--train-noise", type=Path, required=True)
    result.add_argument("--eval-speech", type=Path, required=True)
    result.add_argument("--eval-noise", type=Path, required=True)
    result.add_argument("--data-card", type=Path, required=True)
    result.add_argument("--rir-list", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--source-cache", type=Path, default=Path("/cache/rnnoise"))
    result.add_argument("--train-sequences", type=int, default=200_000)
    result.add_argument("--eval-sequences", type=int, default=2_000)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--eval-batch-size", type=int, default=16)
    result.add_argument("--epochs", type=int, default=0, help="0 targets about 75,000 updates")
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--learning-rate-decay", type=float, default=5e-5)
    result.add_argument("--cond-size", type=int, default=128)
    result.add_argument("--gru-size", type=int, default=384)
    result.add_argument("--gamma", type=float, default=1 / 6)
    result.add_argument("--seed", type=int, default=1_969_587)
    result.add_argument("--initial-checkpoint", type=Path)
    result.add_argument("--max-validation-loss", type=float)
    result.add_argument("--baseline-validation-loss", type=float)
    result.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    result.add_argument("--experiment", default="coffeebean-rnnoise")
    result.add_argument("--run-name")
    result.add_argument("--no-mlflow", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    for name in (
        "train_speech", "train_noise", "eval_speech", "eval_noise",
        "data_card", "rir_list", "output", "source_cache", "initial_checkpoint",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    for name in (
        "train_sequences", "eval_sequences", "batch_size", "eval_batch_size",
        "cond_size", "gru_size",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.train_sequences < args.batch_size:
        raise SystemExit("--train-sequences must be at least --batch-size")
    if args.epochs < 0:
        raise SystemExit("--epochs cannot be negative")
    for name in ("learning_rate", "learning_rate_decay", "gamma"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("max_validation_loss", "baseline_validation_loss"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise SystemExit(f"--{name.replace('_', '-')} must be positive and finite")
    if args.initial_checkpoint is not None and not args.initial_checkpoint.is_file():
        raise SystemExit(f"initial checkpoint does not exist: {args.initial_checkpoint}")
    if args.command == "validate":
        manifest = make_manifest(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(json.dumps(run_pipeline(args), indent=2))


if __name__ == "__main__":
    main()
