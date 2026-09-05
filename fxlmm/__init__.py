"""Standalone normalized FxLMS/FxLMM/modified-FxLMM offline benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SAMPLE_RATE = 8_000
CONTROLLER_TAPS = 64
STEP_SIZE = 0.001
EPSILON = 1e-6
TARGET_RMS = 0.1
REALITY_SEED = 1_969_587
SECONDARY_ESTIMATION_NOISE_STD = 0.1
ADC_NOISE_STD = 0.02
PROCESSING_LATENCY_MS = 2.0
THRESHOLD_UPDATE_SAMPLES = 256
THRESHOLD_WARMUP_SAMPLES = 128
ALGORITHMS = ("fxlms", "fxlmm", "modified-fxlmm")
REPO_ROOT = Path(__file__).resolve().parent.parent

PRIMARY_PATH = np.zeros(64, dtype=np.float64)
PRIMARY_PATH[10:13] = (1.0, 0.5, 0.2)
SECONDARY_PATH = np.zeros(32, dtype=np.float64)
SECONDARY_PATH[5:8] = (1.0, 0.4, 0.1)


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def hampel_score(error: float, thresholds: tuple[float, float, float]) -> float:
    """Return Hampel's bounded, redescending influence score."""
    lower, middle, upper = thresholds
    magnitude = abs(error)
    sign = math.copysign(1.0, error)
    if magnitude <= lower:
        return float(error)
    if magnitude <= middle:
        return lower * sign
    if magnitude <= upper:
        return lower * (upper - magnitude) / (upper - middle) * sign
    return 0.0


def _thresholds(errors: deque[float]) -> tuple[float, float, float]:
    values = np.abs(np.asarray(errors, dtype=np.float64))
    quantiles = np.quantile(values, (0.95, 0.975, 0.99))
    gap = max(float(quantiles[-1]) * 1e-6, 1e-12)
    lower = max(float(quantiles[0]), gap)
    middle = max(float(quantiles[1]), lower + gap)
    upper = max(float(quantiles[2]), middle + gap)
    return lower, middle, upper


def run_anc(
    reference: np.ndarray,
    *,
    algorithm: str,
    taps: int = CONTROLLER_TAPS,
    step_size: float = STEP_SIZE,
    sample_rate: int = SAMPLE_RATE,
    epsilon: float = EPSILON,
    seed: int = REALITY_SEED,
    secondary_estimation_noise_std: float = SECONDARY_ESTIMATION_NOISE_STD,
    adc_noise_std: float = ADC_NOISE_STD,
    processing_latency_ms: float = PROCESSING_LATENCY_MS,
) -> dict[str, np.ndarray]:
    """Run one offline feed-forward ANC simulation."""
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("Reference must be a non-empty mono signal")
    if not np.all(np.isfinite(reference)):
        raise ValueError("Reference contains non-finite samples")
    if rms(reference) <= np.finfo(np.float64).eps:
        raise ValueError("Reference must contain non-silent audio")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if taps <= 0 or sample_rate <= 0 or step_size <= 0 or epsilon <= 0:
        raise ValueError("taps, sample_rate, step_size, and epsilon must be positive")
    if secondary_estimation_noise_std < 0 or adc_noise_std < 0:
        raise ValueError("noise standard deviations must be non-negative")
    if processing_latency_ms < 0:
        raise ValueError("processing_latency_ms must be non-negative")

    generator = np.random.default_rng(seed)
    secondary_path_estimate = SECONDARY_PATH + (
        secondary_estimation_noise_std * generator.standard_normal(len(SECONDARY_PATH))
    )
    processing_latency_samples = int(
        round(processing_latency_ms * sample_rate / 1_000)
    )
    processing_delay: deque[float] = deque(
        (0.0 for _ in range(processing_latency_samples)),
        maxlen=processing_latency_samples or None,
    )
    disturbance = signal.lfilter(PRIMARY_PATH, (1.0,), reference)
    weights = np.zeros(taps, dtype=np.float64)
    reference_delay = np.zeros(taps, dtype=np.float64)
    filtered_delay = np.zeros(taps, dtype=np.float64)
    secondary_model_delay = np.zeros(len(secondary_path_estimate), dtype=np.float64)
    secondary_delay = np.zeros(len(SECONDARY_PATH), dtype=np.float64)
    residual = np.zeros_like(reference)
    controller_output = np.zeros_like(reference)
    coefficient_norm = np.zeros_like(reference)

    errors: deque[float] = deque(maxlen=sample_rate)
    references: deque[float] = deque(maxlen=sample_rate)
    error_thresholds = (math.inf, math.inf, math.inf)
    reference_thresholds = (math.inf, math.inf, math.inf)

    for index, sample in enumerate(reference):
        detected_sample = sample + generator.normal(0.0, adc_noise_std)
        if processing_latency_samples:
            processed_sample = processing_delay.popleft()
            processing_delay.append(detected_sample)
        else:
            processed_sample = detected_sample

        if (
            algorithm == "modified-fxlmm"
            and len(references) >= THRESHOLD_WARMUP_SAMPLES
            and index % THRESHOLD_UPDATE_SAMPLES == 0
        ):
            reference_thresholds = _thresholds(references)
        references.append(processed_sample)
        adaptation_sample = (
            hampel_score(processed_sample, reference_thresholds)
            if algorithm == "modified-fxlmm"
            else processed_sample
        )
        secondary_model_delay[1:] = secondary_model_delay[:-1]
        secondary_model_delay[0] = adaptation_sample
        filtered_sample = float(
            np.dot(secondary_path_estimate, secondary_model_delay)
        )

        reference_delay[1:] = reference_delay[:-1]
        reference_delay[0] = processed_sample
        filtered_delay[1:] = filtered_delay[:-1]
        filtered_delay[0] = filtered_sample

        controller_sample = float(np.dot(weights, reference_delay))
        controller_output[index] = controller_sample
        secondary_delay[1:] = secondary_delay[:-1]
        secondary_delay[0] = controller_sample
        error = float(
            disturbance[index]
            - np.dot(SECONDARY_PATH, secondary_delay)
            + generator.normal(0.0, adc_noise_std)
        )

        if (
            algorithm != "fxlms"
            and len(errors) >= THRESHOLD_WARMUP_SAMPLES
            and index % THRESHOLD_UPDATE_SAMPLES == 0
        ):
            error_thresholds = _thresholds(errors)
        errors.append(error)
        update_error = (
            error
            if algorithm == "fxlms"
            else hampel_score(error, error_thresholds)
        )
        weights += (
            step_size
            * update_error
            * filtered_delay
            / (epsilon + float(np.dot(filtered_delay, filtered_delay)))
        )

        residual[index] = error
        coefficient_norm[index] = float(np.linalg.norm(weights))

    if not all(
        np.all(np.isfinite(values))
        for values in (residual, controller_output, coefficient_norm, weights)
    ):
        raise FloatingPointError("ANC adaptation diverged")
    return {
        "disturbance": disturbance,
        "residual": residual,
        "controller_output": controller_output,
        "coefficient_norm": coefficient_norm,
        "weights": weights,
        "secondary_path_estimate": secondary_path_estimate,
    }


def generate_drifting_engine(
    *, duration: float = 10.0, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Generate 50-to-60 Hz engine drift plus a fixed 120 Hz harmonic."""
    if duration <= 0 or sample_rate <= 0:
        raise ValueError("duration and sample_rate must be positive")
    count = int(round(duration * sample_rate))
    time = np.arange(count, dtype=np.float64) / sample_rate
    frequency = np.linspace(50.0, 60.0, count)
    phase = (
        2
        * np.pi
        * np.concatenate(([0.0], np.cumsum(frequency[:-1])))
        / sample_rate
    )
    return 0.5 * np.sin(phase) + 0.3 * np.sin(2 * np.pi * 120 * time)


def load_noise(
    path: Path, *, sample_rate: int = SAMPLE_RATE, target_rms: float = TARGET_RMS
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Read, center, resample, and consistently scale one recording."""
    if sample_rate <= 0 or target_rms <= 0:
        raise ValueError("sample_rate and target_rms must be positive")
    samples, source_rate = sf.read(path, always_2d=True, dtype="float64")
    if samples.shape[0] == 0:
        raise ValueError(f"Noise input is empty: {path}")
    mono = np.mean(samples, axis=1)
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"Noise input contains non-finite samples: {path}")
    dc_offset = float(np.mean(mono))
    mono -= dc_offset
    divisor = math.gcd(int(source_rate), sample_rate)
    if source_rate != sample_rate:
        mono = signal.resample_poly(
            mono, sample_rate // divisor, int(source_rate) // divisor
        )
    input_rms = rms(mono)
    if input_rms <= np.finfo(np.float64).eps:
        raise ValueError(f"Noise input must contain non-silent audio: {path}")
    gain = target_rms / input_rms
    mono *= gain
    return mono, {
        "source": str(path.resolve()),
        "source_sample_rate": int(source_rate),
        "source_channels": int(samples.shape[1]),
        "source_frames": int(samples.shape[0]),
        "processed_frames": int(len(mono)),
        "duration_seconds": len(mono) / sample_rate,
        "dc_offset": dc_offset,
        "input_rms_after_dc": input_rms,
        "normalization_gain": gain,
    }


def _metrics(result: dict[str, np.ndarray]) -> dict[str, float]:
    disturbance_rms = rms(result["disturbance"])
    residual_rms = rms(result["residual"])
    return {
        "disturbance_rms": disturbance_rms,
        "residual_rms": residual_rms,
        "attenuation_db": 20 * math.log10(disturbance_rms / residual_rms),
        "residual_peak": float(np.max(np.abs(result["residual"]))),
        "coefficient_norm_peak": float(np.max(result["coefficient_norm"])),
        "coefficient_norm_final": float(result["coefficient_norm"][-1]),
    }


def _plot_recording(
    output: Path,
    name: str,
    reference: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    sample_rate: int,
) -> None:
    from matplotlib import pyplot as plt

    time_axis = np.arange(len(reference)) / sample_rate
    window = max(1, sample_rate // 20)
    kernel = np.ones(window) / window
    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time_axis, results["fxlms"]["disturbance"], color="tab:red", linewidth=0.7)
    axes[0].set_ylabel("Disturbance")
    for algorithm, color in (
        ("fxlms", "tab:blue"),
        ("fxlmm", "tab:green"),
        ("modified-fxlmm", "tab:orange"),
    ):
        residual = results[algorithm]["residual"]
        learning_curve = np.sqrt(signal.lfilter(kernel, (1.0,), residual**2))
        axes[1].plot(time_axis, residual, label=algorithm, color=color, linewidth=0.6)
        axes[2].semilogy(
            time_axis,
            np.maximum(learning_curve, 1e-12),
            label=algorithm,
            color=color,
            linewidth=0.9,
        )
    axes[1].set_ylabel("Residual")
    axes[2].set_ylabel("50 ms RMS")
    axes[2].set_xlabel("Time (s)")
    axes[1].legend()
    axes[2].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle(f"{name}: normalized FxLMS, FxLMM, and modified FxLMM")
    figure.tight_layout()
    figure.savefig(output / f"{name}-comparison.png", dpi=150)
    plt.close(figure)


def _plot_summary(output: Path, recordings: list[dict[str, object]]) -> None:
    from matplotlib import pyplot as plt

    names = [str(item["name"]) for item in recordings]
    positions = np.arange(len(names))
    width = 0.26
    figure, axis = plt.subplots(figsize=(10, 5))
    for offset, algorithm, color in (
        (-width, "fxlms", "tab:blue"),
        (0.0, "fxlmm", "tab:green"),
        (width, "modified-fxlmm", "tab:orange"),
    ):
        values = [
            float(item["algorithms"][algorithm]["attenuation_db"])
            for item in recordings
        ]
        axis.bar(positions + offset, values, width, label=algorithm, color=color)
    axis.set_xticks(positions, names)
    axis.set_ylabel("RMS attenuation (dB)")
    axis.set_title("Real recordings through simulated acoustic paths")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "summary.png", dpi=150)
    plt.close(figure)


def benchmark(
    noise_dir: Path,
    output: Path,
    *,
    step_size: float = STEP_SIZE,
    taps: int = CONTROLLER_TAPS,
    sample_rate: int = SAMPLE_RATE,
    seed: int = REALITY_SEED,
) -> dict[str, object]:
    """Benchmark every WAV recording in a directory and write inspectable artifacts."""
    if not noise_dir.is_dir():
        raise ValueError(f"Noise directory does not exist: {noise_dir}")
    paths = sorted(noise_dir.glob("*.wav"))
    if not paths:
        raise ValueError(f"Noise directory contains no WAV files: {noise_dir}")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / ".plot-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")

    recordings: list[dict[str, object]] = []
    for path in paths:
        reference, source = load_noise(path, sample_rate=sample_rate)
        results = {
            algorithm: run_anc(
                reference,
                algorithm=algorithm,
                taps=taps,
                step_size=step_size,
                sample_rate=sample_rate,
                seed=seed,
            )
            for algorithm in ALGORITHMS
        }
        name = path.stem
        sf.write(
            output / f"{name}-disturbance.wav",
            results["fxlms"]["disturbance"],
            sample_rate,
            subtype="FLOAT",
        )
        for algorithm, result in results.items():
            sf.write(
                output / f"{name}-{algorithm}-residual.wav",
                result["residual"],
                sample_rate,
                subtype="FLOAT",
            )
        _plot_recording(output, name, reference, results, sample_rate)
        recordings.append(
            {
                "name": name,
                "source": source,
                "algorithms": {
                    algorithm: _metrics(result)
                    for algorithm, result in results.items()
                },
            }
        )

    report: dict[str, object] = {
        "configuration": {
            "sample_rate": sample_rate,
            "seed": seed,
            "controller_taps": taps,
            "step_size": step_size,
            "epsilon": EPSILON,
            "target_rms": TARGET_RMS,
            "threshold_window_samples": sample_rate,
            "threshold_update_samples": THRESHOLD_UPDATE_SAMPLES,
            "threshold_percentiles": [0.95, 0.975, 0.99],
            "modified_fxlmm_reference_score": "Hampel three-part redescending",
            "primary_path": PRIMARY_PATH.tolist(),
            "secondary_path": SECONDARY_PATH.tolist(),
            "secondary_estimation_noise_std": SECONDARY_ESTIMATION_NOISE_STD,
            "adc_noise_std": ADC_NOISE_STD,
            "processing_latency_ms": PROCESSING_LATENCY_MS,
            "physical_test": False,
        },
        "recordings": recordings,
    }
    _plot_summary(output, recordings)
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def stress_test(
    output: Path,
    *,
    duration: float = 10.0,
    seed: int = REALITY_SEED,
    step_size: float = STEP_SIZE,
    taps: int = CONTROLLER_TAPS,
) -> dict[str, object]:
    """Run the seeded drifting-engine hardware-reality simulation."""
    output.mkdir(parents=True, exist_ok=True)
    cache = output / ".plot-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")

    reference = generate_drifting_engine(duration=duration)
    results = {
        algorithm: run_anc(
            reference,
            algorithm=algorithm,
            taps=taps,
            step_size=step_size,
            seed=seed,
        )
        for algorithm in ALGORITHMS
    }
    name = "drifting-engine"
    sf.write(
        output / f"{name}-reference.wav", reference, SAMPLE_RATE, subtype="FLOAT"
    )
    sf.write(
        output / f"{name}-disturbance.wav",
        results["fxlms"]["disturbance"],
        SAMPLE_RATE,
        subtype="FLOAT",
    )
    for algorithm, result in results.items():
        sf.write(
            output / f"{name}-{algorithm}-residual.wav",
            result["residual"],
            SAMPLE_RATE,
            subtype="FLOAT",
        )
    _plot_recording(output, name, reference, results, SAMPLE_RATE)
    report: dict[str, object] = {
        "configuration": {
            "seed": seed,
            "sample_rate": SAMPLE_RATE,
            "duration_seconds": duration,
            "controller_taps": taps,
            "step_size": step_size,
            "drifting_frequency_hz": [50.0, 60.0],
            "fixed_frequency_hz": 120.0,
            "secondary_estimation_noise_std": SECONDARY_ESTIMATION_NOISE_STD,
            "adc_noise_std": ADC_NOISE_STD,
            "processing_latency_ms": PROCESSING_LATENCY_MS,
            "processing_latency_samples": round(
                PROCESSING_LATENCY_MS * SAMPLE_RATE / 1_000
            ),
            "secondary_path": SECONDARY_PATH.tolist(),
            "secondary_path_estimate": results["fxlms"][
                "secondary_path_estimate"
            ].tolist(),
            "physical_test": False,
        },
        "algorithms": {
            algorithm: _metrics(result) for algorithm, result in results.items()
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fxlmm",
        description="Benchmark normalized FxLMS, FxLMM, and modified FxLMM offline.",
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=REPO_ROOT / "samples" / "real-noise" / "wav",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "runs" / "fxlmm"
    )
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--taps", type=int, default=CONTROLLER_TAPS)
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="run the drifting-engine, path-error, ADC-noise, and 2 ms latency test",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=REALITY_SEED)
    arguments = parser.parse_args(argv)
    if arguments.stress_test:
        report = stress_test(
            arguments.output.resolve(),
            duration=arguments.duration,
            seed=arguments.seed,
            taps=arguments.taps,
            step_size=(
                arguments.step_size
                if arguments.step_size is not None
                else STEP_SIZE
            ),
        )
        recording_count = 1
    else:
        report = benchmark(
            arguments.noise_dir.resolve(),
            arguments.output.resolve(),
            step_size=(
                arguments.step_size
                if arguments.step_size is not None
                else STEP_SIZE
            ),
            taps=arguments.taps,
            seed=arguments.seed,
        )
        recording_count = len(report["recordings"])
    print(
        json.dumps(
            {
                "metrics": str((arguments.output.resolve() / "metrics.json")),
                "recordings": recording_count,
                "physical_test": False,
            },
            indent=2,
        )
    )
    return 0


__all__ = [
    "ALGORITHMS",
    "ADC_NOISE_STD",
    "CONTROLLER_TAPS",
    "PROCESSING_LATENCY_MS",
    "PRIMARY_PATH",
    "REALITY_SEED",
    "SAMPLE_RATE",
    "SECONDARY_PATH",
    "SECONDARY_ESTIMATION_NOISE_STD",
    "STEP_SIZE",
    "benchmark",
    "generate_drifting_engine",
    "hampel_score",
    "load_noise",
    "main",
    "rms",
    "run_anc",
    "stress_test",
]
