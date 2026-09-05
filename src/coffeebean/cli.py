from __future__ import annotations

import argparse
import ctypes
import csv
import json
import math
import os
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SPEECH_SAMPLE_RATE = 48_000
PAPER_SAMPLE_RATE = 5_000
EPSILON = np.finfo(np.float64).eps
RNNOISE_MODEL = "RNNoise-v0.2-little"
RNNOISE_ALIASES = {"rnnoise", "rnnoise-v0.2", RNNOISE_MODEL.casefold()}
RNNOISE_DELAY_FRAMES = 2


def fxlms_score(error: float, _thresholds: tuple[float, float, float]) -> float:
    return float(error)


def fxlmm_score(error: float, thresholds: tuple[float, float, float]) -> float:
    """Hampel three-part redescending score used by conventional FXLMM."""
    xi, delta1, delta2 = thresholds
    magnitude = abs(error)
    sign = math.copysign(1.0, error)
    if magnitude < xi:
        return float(error)
    if magnitude < delta1:
        return xi * sign
    if magnitude < delta2:
        return xi * (delta2 - magnitude) / (delta2 - delta1) * sign
    return 0.0


def modified_fxlmm_score(
    error: float, thresholds: tuple[float, float, float]
) -> float:
    """Continuous triangle implied by Li and Yu's prose and Figure 1.

    The manuscript's typeset equation is inconsistent with its surrounding prose.
    This normalized form follows the stated line from (xi, xi) to (delta2, 0).
    """
    xi, _delta1, delta2 = thresholds
    magnitude = abs(error)
    sign = math.copysign(1.0, error)
    if magnitude < xi:
        return float(error)
    if magnitude < delta2:
        return xi * (delta2 - magnitude) / (delta2 - xi) * sign
    return 0.0


SCORE_FUNCTIONS = {
    "fxlms": fxlms_score,
    "fxlmm": fxlmm_score,
    "modified-fxlmm": modified_fxlmm_score,
}


def _ordered_thresholds(values: deque[float]) -> tuple[float, float, float]:
    if len(values) < 64:
        return (0.08, 0.12, 0.18)
    xi, delta1, delta2 = np.quantile(np.asarray(values), (0.95, 0.975, 0.99))
    xi = max(float(xi), 1e-6)
    delta1 = max(float(delta1), xi + 1e-6)
    delta2 = max(float(delta2), delta1 + 1e-6)
    return xi, delta1, delta2


def generate_paper_signal(
    *,
    duration: float = 20.0,
    sample_rate: int = PAPER_SAMPLE_RATE,
    impulse_times: tuple[float, ...] = (9.5, 14.8),
    impulse_width: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(round(duration * sample_rate))
    t = np.arange(count, dtype=np.float64) / sample_rate
    reference = (
        0.05 * np.sin(2 * np.pi * 300 * t)
        + 0.10 * np.sin(2 * np.pi * 500 * t)
        + 0.05 * np.sin(2 * np.pi * 700 * t)
    )
    for start in impulse_times:
        first = max(0, int(round(start * sample_rate)))
        last = min(count, first + int(round(impulse_width * sample_rate)))
        burst_t = np.arange(last - first, dtype=np.float64) / sample_rate
        envelope = np.exp(-35.0 * burst_t)
        reference[first:last] += 0.30 * envelope * np.sin(2 * np.pi * 600 * burst_t)
    return t, reference


def run_anc(
    reference: np.ndarray,
    *,
    algorithm: str,
    taps: int = 32,
    step_size: float = 1e-5,
    sample_rate: int = PAPER_SAMPLE_RATE,
) -> dict[str, np.ndarray]:
    if algorithm not in SCORE_FUNCTIONS:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # ponytail: fixed lab paths; replace with measured paths when hardware exists.
    primary_path = np.asarray((0.78, 0.28, -0.09), dtype=np.float64)
    secondary_path = np.asarray((0.0, 0.0, 0.62, 0.24, -0.07), dtype=np.float64)
    disturbance = signal.lfilter(primary_path, (1.0,), reference)
    filtered_reference = signal.lfilter(secondary_path, (1.0,), reference)

    weights = np.zeros(taps, dtype=np.float64)
    reference_delay = np.zeros(taps, dtype=np.float64)
    filtered_delay = np.zeros(taps, dtype=np.float64)
    secondary_delay = np.zeros(len(secondary_path), dtype=np.float64)
    residual = np.zeros_like(reference)
    coefficient_norm = np.zeros_like(reference)
    controller_output = np.zeros_like(reference)

    recent_errors: deque[float] = deque(maxlen=sample_rate)
    thresholds = (0.08, 0.12, 0.18)
    score = SCORE_FUNCTIONS[algorithm]

    for index, (sample, filtered_sample) in enumerate(
        zip(reference, filtered_reference, strict=True)
    ):
        reference_delay[1:] = reference_delay[:-1]
        reference_delay[0] = sample
        filtered_delay[1:] = filtered_delay[:-1]
        filtered_delay[0] = filtered_sample

        controller_sample = float(np.dot(weights, reference_delay))
        controller_output[index] = controller_sample
        secondary_delay[1:] = secondary_delay[:-1]
        secondary_delay[0] = controller_sample
        anti_noise = float(np.dot(secondary_path, secondary_delay))
        error = float(disturbance[index] - anti_noise)

        if index % 250 == 0:
            thresholds = _ordered_thresholds(recent_errors)
        recent_errors.append(abs(error))
        weights += step_size * score(error, thresholds) * filtered_delay

        residual[index] = error
        coefficient_norm[index] = float(np.linalg.norm(weights))

    return {
        "disturbance": disturbance,
        "residual": residual,
        "coefficient_norm": coefficient_norm,
        "controller_output": controller_output,
    }


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _paper_metrics(
    result: dict[str, np.ndarray],
    *,
    sample_rate: int,
    impulse_times: tuple[float, ...],
) -> dict[str, float | int]:
    residual = result["residual"]
    norm = result["coefficient_norm"]
    windows: list[np.ndarray] = []
    recoveries: list[int] = []
    for impulse_time in impulse_times:
        center = int(round(impulse_time * sample_rate))
        first = max(0, center - sample_rate // 10)
        last = min(len(residual), center + sample_rate // 2)
        windows.append(residual[first:last])

        before = residual[max(0, first - sample_rate) : first]
        baseline = _rms(before) if len(before) else _rms(residual[: max(1, first)])
        block = max(1, sample_rate // 100)
        recovery = sample_rate
        for cursor in range(center, min(len(residual) - block, center + sample_rate), block):
            if _rms(residual[cursor : cursor + block]) <= max(1.25 * baseline, 1e-6):
                recovery = cursor - center
                break
        recoveries.append(recovery)

    impulse_samples = np.concatenate(windows) if windows else residual[:0]
    return {
        "residual_rms": _rms(residual),
        "residual_peak": float(np.max(np.abs(residual))),
        "impulse_window_rms": _rms(impulse_samples) if len(impulse_samples) else 0.0,
        "coefficient_norm_peak": float(np.max(norm)),
        "recovery_samples_max": max(recoveries, default=0),
    }


def simulate_paper(
    output: Path,
    *,
    seed: int = 1_969_587,
    duration: float = 20.0,
    impulse_times: tuple[float, ...] = (9.5, 14.8),
    impulse_width: float = 0.10,
) -> dict[str, object]:
    np.random.default_rng(seed)  # Reserve the public seed for future path perturbations.
    output.mkdir(parents=True, exist_ok=True)
    t, reference = generate_paper_signal(
        duration=duration,
        impulse_times=impulse_times,
        impulse_width=impulse_width,
    )
    results = {
        name: run_anc(reference, algorithm=name) for name in SCORE_FUNCTIONS
    }
    metrics: dict[str, object] = {
        "configuration": {
            "seed": seed,
            "sample_rate": PAPER_SAMPLE_RATE,
            "duration_seconds": duration,
            "controller_taps": 32,
            "step_size": 1e-5,
            "impulse_times_seconds": list(impulse_times),
            "impulse_width_seconds": impulse_width,
        },
        "algorithms": {
            name: _paper_metrics(
                result,
                sample_rate=PAPER_SAMPLE_RATE,
                impulse_times=impulse_times,
            )
            for name, result in results.items()
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    with (output / "residuals.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "reference", "disturbance", *results))
        disturbance = results["fxlms"]["disturbance"]
        writer.writerows(
            zip(
                t,
                reference,
                disturbance,
                *(results[name]["residual"] for name in results),
                strict=True,
            )
        )

    sf.write(output / "reference.wav", reference, PAPER_SAMPLE_RATE, subtype="FLOAT")
    for name, result in results.items():
        sf.write(
            output / f"residual-{name}.wav",
            result["residual"],
            PAPER_SAMPLE_RATE,
            subtype="FLOAT",
        )

    plot_cache = output / ".plot-cache"
    plot_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(11, 9))
    rms_window = max(1, PAPER_SAMPLE_RATE // 50)
    rms_filter = np.ones(rms_window) / rms_window
    for name, result in results.items():
        residual_envelope = np.sqrt(
            signal.lfilter(rms_filter, (1.0,), np.square(result["residual"]))
        )
        axes[0].plot(t, residual_envelope, label=name, linewidth=1.0)
        axes[1].plot(t, result["residual"], label=name, linewidth=0.9, alpha=0.85)
        axes[2].plot(t, result["coefficient_norm"], label=name, linewidth=1.0)
    first_impulse = impulse_times[0] if impulse_times else duration / 2
    axes[0].set_ylabel("20 ms residual RMS")
    axes[1].set_ylabel("Residual error")
    axes[1].set_xlim(max(0.0, first_impulse - 0.15), min(duration, first_impulse + 0.45))
    axes[1].set_title("First impulse detail")
    axes[2].set_ylabel("Controller coefficient norm")
    axes[2].set_xlabel("Time (s)")
    axes[0].legend()
    axes[2].legend()
    figure.suptitle("Impulsive ANC algorithm comparison")
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)
    return metrics


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    if samples.shape[1] != 1:
        samples = np.mean(samples, axis=1, keepdims=True)
    mono = samples[:, 0]
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"Audio contains non-finite samples: {path}")
    return mono, int(sample_rate)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.copy()
    divisor = math.gcd(source_rate, target_rate)
    return signal.resample_poly(samples, target_rate // divisor, source_rate // divisor)


def mix_audio(
    clean_path: Path,
    output_path: Path,
    *,
    profile: str | None = None,
    noise_path: Path | None = None,
    snr_db: float,
    seed: int = 1_969_587,
) -> dict[str, float | int | str]:
    if (profile is None) == (noise_path is None):
        raise ValueError("Choose exactly one of profile or noise_path")

    clean, source_rate = _read_audio(clean_path)
    clean = _resample(clean, source_rate, SPEECH_SAMPLE_RATE)
    if len(clean) == 0 or _rms(clean) <= EPSILON:
        raise ValueError("Clean input must contain non-silent audio")

    if noise_path is not None:
        noise, noise_rate = _read_audio(noise_path)
        noise = _resample(noise, noise_rate, SPEECH_SAMPLE_RATE)
        if len(noise) == 0 or _rms(noise) <= EPSILON:
            raise ValueError("Noise input must contain non-silent audio")
        noise = np.tile(noise, math.ceil(len(clean) / len(noise)))[: len(clean)]
        profile_name = "recording"
    else:
        generator = np.random.default_rng(seed)
        t = np.arange(len(clean), dtype=np.float64) / SPEECH_SAMPLE_RATE
        white = generator.standard_normal(len(clean))
        low_noise = signal.lfilter((1.0,), (1.0, -0.985), white)
        tonal = np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 240 * t)
        noise = 0.7 * low_noise / max(_rms(low_noise), EPSILON) + 0.3 * tonal

        if profile == "impulsive":
            for center in (len(clean) // 3, 2 * len(clean) // 3):
                width = min(int(0.12 * SPEECH_SAMPLE_RATE), len(clean) - center)
                burst_t = np.arange(width, dtype=np.float64) / SPEECH_SAMPLE_RATE
                burst = np.exp(-35 * burst_t) * np.sin(2 * np.pi * 600 * burst_t)
                noise[center : center + width] += 8.0 * burst
        elif profile != "continuous":
            raise ValueError(f"Unknown noise profile: {profile}")
        profile_name = profile

    target_noise_rms = _rms(clean) / (10 ** (snr_db / 20.0))
    noise *= target_noise_rms / max(_rms(noise), EPSILON)
    noise_crest_factor_db = 20 * math.log10(
        float(np.max(np.abs(noise))) / max(_rms(noise), EPSILON)
    )
    mixed = clean + noise
    peak = float(np.max(np.abs(mixed)))
    applied_gain = min(1.0, 0.98 / peak) if peak else 1.0
    mixed *= applied_gain

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, mixed, SPEECH_SAMPLE_RATE, subtype="FLOAT")
    actual_snr = 20 * math.log10(_rms(clean) / max(_rms(noise), EPSILON))
    metadata: dict[str, float | int | str] = {
        "clean": str(clean_path),
        "output": str(output_path),
        "profile": profile_name,
        "noise": str(noise_path) if noise_path is not None else "generated",
        "seed": seed,
        "sample_rate": SPEECH_SAMPLE_RATE,
        "requested_snr_db": snr_db,
        "actual_snr_db": actual_snr,
        "noise_crest_factor_db": noise_crest_factor_db,
        "output_gain": applied_gain,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def enhance_audio(
    input_path: Path,
    output_path: Path,
    *,
    model: str = "DeepFilterNet3",
) -> dict[str, float | int | str]:
    samples, sample_rate = _read_audio(input_path)
    if sample_rate != SPEECH_SAMPLE_RATE:
        raise ValueError("Enhancement input must be 48 kHz")
    if len(samples) == 0:
        raise ValueError("Enhancement input must not be empty")

    if model.casefold() in RNNOISE_ALIASES:
        started = time.perf_counter()
        enhanced, latency_samples = _enhance_rnnoise(samples)
        elapsed = time.perf_counter() - started
        model = RNNOISE_MODEL
    else:
        enhanced, elapsed, latency_samples = _enhance_deepfilternet(samples, model)

    if not np.all(np.isfinite(enhanced)):
        raise RuntimeError(f"{model} produced non-finite audio")
    peak = float(np.max(np.abs(enhanced))) if len(enhanced) else 0.0
    output_gain = min(1.0, 0.999 / peak) if peak else 1.0
    enhanced *= output_gain
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, enhanced, sample_rate, subtype="FLOAT")

    duration = len(samples) / sample_rate
    benchmark: dict[str, float | int | str] = {
        "input": str(input_path),
        "output": str(output_path),
        "model": model,
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "elapsed_seconds": elapsed,
        "real_time_factor": elapsed / duration if duration else math.inf,
        "algorithmic_latency_samples": latency_samples,
        "algorithmic_latency_ms": latency_samples / sample_rate * 1_000,
        "output_gain": output_gain,
    }
    output_path.with_suffix(output_path.suffix + ".benchmark.json").write_text(
        json.dumps(benchmark, indent=2) + "\n", encoding="utf-8"
    )
    return benchmark


def _enhance_deepfilternet(
    samples: np.ndarray, model: str
) -> tuple[np.ndarray, float, int]:

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"`torchaudio\.backend\.common\.AudioMetaData`.*"
            )
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch\.load` with `weights_only=False`.*",
                category=FutureWarning,
            )
            import torch
            from df.enhance import enhance, init_df

            network, state, _suffix = init_df(
                model, log_file=None, log_level="ERROR"
            )
    except ImportError as error:
        raise RuntimeError(
            "DeepFilterNet is unavailable; run: uv sync --extra enhance"
        ) from error

    tensor = torch.from_numpy(samples.astype(np.float32, copy=False)).unsqueeze(0)
    started = time.perf_counter()
    with torch.inference_mode():
        enhanced = enhance(network, state, tensor, pad=True).squeeze(0).numpy()
    elapsed = time.perf_counter() - started
    return enhanced, elapsed, 1_920


def _enhance_rnnoise(samples: np.ndarray) -> tuple[np.ndarray, int]:
    library_suffix = ".dylib" if os.uname().sysname == "Darwin" else ".so"
    default_library = (
        Path(__file__).resolve().parents[2]
        / "build"
        / "rnnoise-v0.2"
        / f"librnnoise{library_suffix}"
    )
    library_path = Path(os.environ.get("COFFEEBEAN_RNNOISE_LIB", default_library))
    if not library_path.exists():
        raise RuntimeError(
            "RNNoise v0.2 is unavailable; run: ./scripts/build-rnnoise-v0.2.sh"
        )

    library = ctypes.CDLL(str(library_path))
    float_pointer = ctypes.POINTER(ctypes.c_float)
    library.rnnoise_create.argtypes = [ctypes.c_void_p]
    library.rnnoise_create.restype = ctypes.c_void_p
    library.rnnoise_destroy.argtypes = [ctypes.c_void_p]
    library.rnnoise_get_frame_size.restype = ctypes.c_int
    library.rnnoise_process_frame.argtypes = [
        ctypes.c_void_p,
        float_pointer,
        float_pointer,
    ]
    library.rnnoise_process_frame.restype = ctypes.c_float

    frame_size = int(library.rnnoise_get_frame_size())
    if frame_size != 480:
        raise RuntimeError(f"Unsupported RNNoise frame size: {frame_size}")
    padded_length = math.ceil(len(samples) / frame_size) * frame_size
    source = np.zeros(
        padded_length + RNNOISE_DELAY_FRAMES * frame_size, dtype=np.float32
    )
    source[: len(samples)] = np.clip(samples, -1.0, 1.0) * 32_767.0
    state = library.rnnoise_create(None)
    if not state:
        raise RuntimeError("RNNoise state allocation failed")
    frames: list[np.ndarray] = []
    try:
        for offset in range(0, len(source), frame_size):
            input_frame = np.ascontiguousarray(source[offset : offset + frame_size])
            output_frame = np.empty(frame_size, dtype=np.float32)
            library.rnnoise_process_frame(
                state,
                output_frame.ctypes.data_as(float_pointer),
                input_frame.ctypes.data_as(float_pointer),
            )
            frames.append(output_frame)
    finally:
        library.rnnoise_destroy(state)

    # v0.2 has two causal convolution frames of delay. Discard their warm-up
    # output and feed two zero frames above so file duration remains unchanged.
    enhanced = np.concatenate(frames[RNNOISE_DELAY_FRAMES:])[: len(samples)]
    enhanced /= 32_768.0
    return (
        enhanced.astype(np.float64, copy=False),
        RNNOISE_DELAY_FRAMES * frame_size,
    )


def list_input_devices() -> list[dict[str, object]]:
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Microphone support is unavailable; run: uv sync --extra live"
        ) from error

    default_input = int(sd.default.device[0])
    return [
        {
            "index": index,
            "name": str(device["name"]),
            "input_channels": int(device["max_input_channels"]),
            "default_sample_rate": int(round(float(device["default_samplerate"]))),
            "default": index == default_input,
        }
        for index, device in enumerate(sd.query_devices())
        if int(device["max_input_channels"]) > 0
    ]


def live_demo(
    output: Path,
    *,
    duration: float,
    device: int | str | None = None,
    model: str = "DeepFilterNet3",
) -> dict[str, object]:
    """Record first, then enhance; this is intentionally not streaming ANC."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be a positive finite number")
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Microphone support is unavailable; run: uv sync --extra live"
        ) from error

    output.mkdir(parents=True, exist_ok=True)
    noisy_path = output / "noisy.wav"
    enhanced_path = output / "enhanced.wav"
    sample_count = int(round(duration * SPEECH_SAMPLE_RATE))
    if sample_count < 1:
        raise ValueError("Duration is too short to capture one sample")
    block_size = 480
    blocks: list[np.ndarray] = []
    overflow_blocks = 0
    capture_started = time.perf_counter()
    with sd.InputStream(
        samplerate=SPEECH_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=block_size,
        device=device,
        latency="low",
    ) as stream:
        input_latency = float(stream.latency)
        remaining = sample_count
        while remaining:
            block, overflowed = stream.read(min(block_size, remaining))
            blocks.append(np.asarray(block[:, 0], dtype=np.float32).copy())
            overflow_blocks += int(overflowed)
            remaining -= len(block)
    capture_elapsed = time.perf_counter() - capture_started
    noisy = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float32)
    if len(noisy) != sample_count or not np.all(np.isfinite(noisy)):
        raise RuntimeError("Microphone capture returned invalid audio")
    sf.write(noisy_path, noisy, SPEECH_SAMPLE_RATE, subtype="FLOAT")

    enhancement_started = time.perf_counter()
    benchmark = enhance_audio(noisy_path, enhanced_path, model=model)
    post_capture_elapsed = time.perf_counter() - enhancement_started
    enhanced, enhanced_rate = _read_audio(enhanced_path)
    if enhanced_rate != SPEECH_SAMPLE_RATE or len(enhanced) != sample_count:
        raise RuntimeError("Enhanced recording did not preserve format and duration")

    report: dict[str, object] = {
        "mode": "record_then_enhance",
        "streaming": False,
        "device": device if device is not None else "default",
        "sample_rate": SPEECH_SAMPLE_RATE,
        "duration_seconds": sample_count / SPEECH_SAMPLE_RATE,
        "block_size_samples": block_size,
        "input_latency_seconds": input_latency,
        "input_overflow_blocks": overflow_blocks,
        "capture_elapsed_seconds": capture_elapsed,
        "post_capture_elapsed_seconds": post_capture_elapsed,
        "noisy": {
            "path": str(noisy_path),
            "peak": float(np.max(np.abs(noisy))),
            "clipping_fraction": float(np.mean(np.abs(noisy) >= 1.0)),
        },
        "enhanced": {
            "path": str(enhanced_path),
            "peak": float(np.max(np.abs(enhanced))),
            "clipping_fraction": float(np.mean(np.abs(enhanced) >= 1.0)),
        },
        "benchmark": benchmark,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _load_stream_enhancer(model: str):
    if model.casefold() in RNNOISE_ALIASES:
        return lambda samples: _enhance_rnnoise(samples)[0]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"`torchaudio\..*AudioMetaData`.*")
            warnings.filterwarnings("ignore", message=r"You are using `torch\.load`.*")
            import torch
            from df.enhance import enhance, init_df

            network, state, _suffix = init_df(
                model, log_file=None, log_level="ERROR"
            )
    except Exception:
        import sys
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        try:
            from src.training.model import get_model
        except ImportError:
            from training.model import get_model
        import yaml
        try:
            cfg = yaml.safe_load(open("configs/config.yaml"))
        except Exception:
            cfg = {"audio": {"sample_rate": SPEECH_SAMPLE_RATE}}
        model_obj = get_model(cfg)
        model_obj.eval()

        def run(samples: np.ndarray) -> np.ndarray:
            t = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            with torch.inference_mode():
                out = model_obj(t)
            return out.squeeze().numpy()

        return run


def stream_demo(
    output: Path,
    *,
    duration: float,
    device: int | str | None = None,
    model: str = "DeepFilterNet3",
    chunk_ms: float = 40.0,
    context_ms: float = 500.0,
) -> dict[str, object]:
    """Enhance bounded microphone chunks during capture using overlap-save."""
    for name, value in (("duration", duration), ("chunk_ms", chunk_ms), ("context_ms", context_ms)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Microphone support is unavailable; run: uv sync --extra live"
        ) from error

    chunk_size = int(round(chunk_ms * SPEECH_SAMPLE_RATE / 1_000))
    context_size = int(round(context_ms * SPEECH_SAMPLE_RATE / 1_000))
    sample_count = int(round(duration * SPEECH_SAMPLE_RATE))
    if min(chunk_size, context_size, sample_count) < 1:
        raise ValueError("Streaming settings are shorter than one sample")

    enhance_window = _load_stream_enhancer(model)
    output.mkdir(parents=True, exist_ok=True)
    noisy_path = output / "noisy.wav"
    enhanced_path = output / "enhanced.wav"
    history = np.zeros(context_size, dtype=np.float32)
    pending: tuple[np.ndarray, int] | None = None
    processing_times: list[float] = []
    overflow_blocks = 0
    deadline_misses = 0
    limited_chunks = 0
    captured = 0
    input_latency = 0.0
    started = time.perf_counter()

    with (
        sf.SoundFile(noisy_path, "w", SPEECH_SAMPLE_RATE, 1, subtype="FLOAT") as noisy_file,
        sf.SoundFile(enhanced_path, "w", SPEECH_SAMPLE_RATE, 1, subtype="FLOAT") as enhanced_file,
        sd.InputStream(
            samplerate=SPEECH_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            device=device,
            latency="low",
        ) as stream,
    ):
        input_latency = float(stream.latency)
        while captured < sample_count:
            valid = min(chunk_size, sample_count - captured)
            block, overflowed = stream.read(valid)
            current = np.zeros(chunk_size, dtype=np.float32)
            current[:valid] = np.asarray(block[:, 0], dtype=np.float32)
            noisy_file.write(current[:valid])
            overflow_blocks += int(overflowed)
            captured += valid

            if pending is not None:
                previous, previous_valid = pending
                window = np.concatenate((history, previous, current))
                process_started = time.perf_counter()
                enhanced_window = enhance_window(window)
                processing = time.perf_counter() - process_started
                processing_times.append(processing)
                deadline_misses += int(processing > chunk_size / SPEECH_SAMPLE_RATE)
                first = len(window) - 2 * chunk_size
                enhanced_block = enhanced_window[first : first + previous_valid]
                peak = float(np.max(np.abs(enhanced_block)))
                if peak > 0.999:
                    enhanced_block *= 0.999 / peak
                    limited_chunks += 1
                enhanced_file.write(enhanced_block)
                history = np.concatenate((history, previous))[-context_size:]
            pending = current, valid

        if pending is not None:
            previous, previous_valid = pending
            window = np.concatenate((history, previous, np.zeros(chunk_size, dtype=np.float32)))
            process_started = time.perf_counter()
            enhanced_window = enhance_window(window)
            processing = time.perf_counter() - process_started
            processing_times.append(processing)
            deadline_misses += int(processing > chunk_size / SPEECH_SAMPLE_RATE)
            first = len(window) - 2 * chunk_size
            enhanced_block = enhanced_window[first : first + previous_valid]
            peak = float(np.max(np.abs(enhanced_block)))
            if peak > 0.999:
                enhanced_block *= 0.999 / peak
                limited_chunks += 1
            enhanced_file.write(enhanced_block)

    elapsed = time.perf_counter() - started
    noisy, noisy_rate = _read_audio(noisy_path)
    enhanced, enhanced_rate = _read_audio(enhanced_path)
    if noisy_rate != SPEECH_SAMPLE_RATE or enhanced_rate != SPEECH_SAMPLE_RATE:
        raise RuntimeError("Streaming output sample rate changed")
    if len(noisy) != sample_count or len(enhanced) != sample_count:
        raise RuntimeError("Streaming output duration changed")
    processing_array = np.asarray(processing_times, dtype=np.float64)
    report: dict[str, object] = {
        "mode": "rolling_window_stream",
        "streaming": True,
        "monitoring": False,
        "device": device if device is not None else "default",
        "model": model,
        "sample_rate": SPEECH_SAMPLE_RATE,
        "duration_seconds": sample_count / SPEECH_SAMPLE_RATE,
        "chunk_ms": chunk_size / SPEECH_SAMPLE_RATE * 1_000,
        "context_ms": context_size / SPEECH_SAMPLE_RATE * 1_000,
        "input_latency_seconds": input_latency,
        "estimated_output_latency_seconds": input_latency + chunk_size / SPEECH_SAMPLE_RATE + float(np.median(processing_array)),
        "input_overflow_blocks": overflow_blocks,
        "processing_deadline_misses": deadline_misses,
        "limited_chunks": limited_chunks,
        "processing_seconds_median": float(np.median(processing_array)),
        "processing_seconds_max": float(np.max(processing_array)),
        "elapsed_seconds": elapsed,
        "noisy": {
            "path": str(noisy_path),
            "peak": float(np.max(np.abs(noisy))),
            "clipping_fraction": float(np.mean(np.abs(noisy) >= 1.0)),
        },
        "enhanced": {
            "path": str(enhanced_path),
            "peak": float(np.max(np.abs(enhanced))),
            "clipping_fraction": float(np.mean(np.abs(enhanced) >= 1.0)),
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    reference_energy = float(np.dot(reference, reference)) + EPSILON
    target = np.dot(estimate, reference) / reference_energy * reference
    noise = estimate - target
    return float(
        10
        * np.log10(
            (np.dot(target, target) + EPSILON) / (np.dot(noise, noise) + EPSILON)
        )
    )


def _align(
    reference: np.ndarray, estimate: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, int]:
    probe = min(len(reference), len(estimate), 10 * sample_rate)
    max_lag = min(sample_rate // 10, probe - 1)
    if probe < 2 or max_lag < 1:
        length = min(len(reference), len(estimate))
        return reference[:length], estimate[:length], 0
    correlation = signal.correlate(
        estimate[:probe], reference[:probe], mode="full", method="fft"
    )
    lags = signal.correlation_lags(probe, probe, mode="full")
    allowed = np.abs(lags) <= max_lag
    lag = int(lags[allowed][np.argmax(correlation[allowed])])
    if lag > 0:
        estimate = estimate[lag:]
    elif lag < 0:
        reference = reference[-lag:]
    length = min(len(reference), len(estimate))
    return reference[:length], estimate[:length], lag


def _audio_metrics(
    reference: np.ndarray, estimate: np.ndarray, sample_rate: int
) -> dict[str, float]:
    from pystoi import stoi

    return {
        "si_sdr_db": _si_sdr(reference, estimate),
        "stoi": float(stoi(reference, estimate, sample_rate, extended=False)),
        "peak": float(np.max(np.abs(estimate))),
        "clipping_fraction": float(np.mean(np.abs(estimate) >= 1.0)),
    }


def evaluate_audio(
    clean_path: Path,
    noisy_path: Path,
    enhanced_path: Path,
    output_path: Path,
) -> dict[str, object]:
    clean, clean_rate = _read_audio(clean_path)
    noisy, noisy_rate = _read_audio(noisy_path)
    enhanced, enhanced_rate = _read_audio(enhanced_path)
    if len({clean_rate, noisy_rate, enhanced_rate}) != 1:
        raise ValueError("Evaluation inputs must have the same sample rate")

    clean_noisy, noisy, noisy_lag = _align(clean, noisy, clean_rate)
    clean_enhanced, enhanced, enhanced_lag = _align(clean, enhanced, clean_rate)
    noisy_metrics = _audio_metrics(clean_noisy, noisy, clean_rate)
    enhanced_metrics = _audio_metrics(clean_enhanced, enhanced, clean_rate)
    report: dict[str, object] = {
        "sample_rate": clean_rate,
        "duration_seconds": min(len(clean_noisy), len(clean_enhanced)) / clean_rate,
        "alignment_lag_samples": {"noisy": noisy_lag, "enhanced": enhanced_lag},
        "noisy": noisy_metrics,
        "enhanced": enhanced_metrics,
        "improvement": {
            "si_sdr_db": enhanced_metrics["si_sdr_db"] - noisy_metrics["si_sdr_db"],
            "stoi": enhanced_metrics["stoi"] - noisy_metrics["stoi"],
        },
    }
    benchmark_path = enhanced_path.with_suffix(enhanced_path.suffix + ".benchmark.json")
    if benchmark_path.exists():
        report["benchmark"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _device(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coffeebean")
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser(
        "simulate-paper", help="Run the impulsive ANC comparison"
    )
    simulate.add_argument("--output", type=_path, required=True)
    simulate.add_argument("--seed", type=int, default=1_969_587)
    simulate.add_argument("--duration", type=float, default=20.0)
    simulate.add_argument(
        "--impulse-times", type=float, nargs="+", default=(9.5, 14.8)
    )
    simulate.add_argument("--impulse-width", type=float, default=0.10)

    mix = commands.add_parser("mix", help="Create a deterministic noisy speech file")
    mix.add_argument("--clean", type=_path, required=True)
    mix.add_argument("--output", type=_path, required=True)
    noise_source = mix.add_mutually_exclusive_group(required=True)
    noise_source.add_argument("--profile", choices=("continuous", "impulsive"))
    noise_source.add_argument("--noise", type=_path, help="Real noise recording")
    mix.add_argument("--snr", type=float, required=True)
    mix.add_argument("--seed", type=int, default=1_969_587)

    enhance = commands.add_parser("enhance", help="Run a pretrained denoiser")
    enhance.add_argument("--input", type=_path, required=True)
    enhance.add_argument("--output", type=_path, required=True)
    enhance.add_argument("--model", default="DeepFilterNet3")

    commands.add_parser("devices", help="List microphone input devices")

    live = commands.add_parser(
        "live", help="Record a microphone, then enhance it for A/B playback"
    )
    live.add_argument("--output", type=_path, required=True)
    live.add_argument("--duration", type=float, default=10.0)
    live.add_argument("--device", type=_device)
    live.add_argument("--model", default="DeepFilterNet3")

    stream = commands.add_parser(
        "stream", help="Enhance microphone chunks while recording"
    )
    stream.add_argument("--output", type=_path, required=True)
    stream.add_argument("--duration", type=float, default=10.0)
    stream.add_argument("--device", type=_device)
    stream.add_argument("--model", default="DeepFilterNet3")
    stream.add_argument("--chunk-ms", type=float, default=40.0)
    stream.add_argument("--context-ms", type=float, default=500.0)

    evaluate = commands.add_parser("evaluate", help="Score noisy and enhanced speech")
    evaluate.add_argument("--clean", type=_path, required=True)
    evaluate.add_argument("--noisy", type=_path, required=True)
    evaluate.add_argument("--enhanced", type=_path, required=True)
    evaluate.add_argument("--output", type=_path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "simulate-paper":
        result = simulate_paper(
            arguments.output,
            seed=arguments.seed,
            duration=arguments.duration,
            impulse_times=tuple(arguments.impulse_times),
            impulse_width=arguments.impulse_width,
        )
    elif arguments.command == "mix":
        result = mix_audio(
            arguments.clean,
            arguments.output,
            profile=arguments.profile,
            noise_path=arguments.noise,
            snr_db=arguments.snr,
            seed=arguments.seed,
        )
    elif arguments.command == "enhance":
        result = enhance_audio(arguments.input, arguments.output, model=arguments.model)
    elif arguments.command == "evaluate":
        result = evaluate_audio(
            arguments.clean,
            arguments.noisy,
            arguments.enhanced,
            arguments.output,
        )
    elif arguments.command == "devices":
        result = list_input_devices()
    elif arguments.command == "live":
        result = live_demo(
            arguments.output,
            duration=arguments.duration,
            device=arguments.device,
            model=arguments.model,
        )
    else:
        result = stream_demo(
            arguments.output,
            duration=arguments.duration,
            device=arguments.device,
            model=arguments.model,
            chunk_ms=arguments.chunk_ms,
            context_ms=arguments.context_ms,
        )
    print(json.dumps(result, indent=2))
    return 0
