from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from fxlmm import (
    PROCESSING_LATENCY_MS,
    PRIMARY_PATH,
    SAMPLE_RATE,
    SECONDARY_PATH,
    benchmark,
    generate_drifting_engine,
    hampel_score,
    load_noise,
    run_anc,
    stress_test,
)

THRESHOLDS = (1.0, 2.0, 4.0)


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_hampel_boundaries_and_symmetry(sign: float) -> None:
    assert hampel_score(sign * 0.0, THRESHOLDS) == 0.0
    assert hampel_score(sign * 1.0, THRESHOLDS) == sign
    assert hampel_score(sign * 2.0, THRESHOLDS) == sign
    assert hampel_score(sign * 3.0, THRESHOLDS) == pytest.approx(sign * 0.5)
    assert hampel_score(sign * 4.0, THRESHOLDS) == 0.0
    assert hampel_score(sign * 5.0, THRESHOLDS) == 0.0


def test_simulation_is_deterministic_finite_and_preserves_primary_delay() -> None:
    reference = np.zeros(2_000)
    reference[0] = 1.0
    first = run_anc(reference, algorithm="fxlmm")
    second = run_anc(reference, algorithm="fxlmm")
    np.testing.assert_array_equal(first["residual"], second["residual"])
    np.testing.assert_array_equal(first["disturbance"][10:13], PRIMARY_PATH[10:13])
    assert first["disturbance"][:10].tolist() == [0.0] * 10
    assert np.all(np.isfinite(first["coefficient_norm"]))


@pytest.mark.parametrize(
    "reference",
    (np.array([]), np.zeros(100), np.array([0.0, np.nan])),
)
def test_invalid_or_silent_reference_is_rejected(reference: np.ndarray) -> None:
    with pytest.raises(ValueError):
        run_anc(reference, algorithm="fxlmm")


def test_tonal_noise_converges() -> None:
    time = np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE
    reference = 0.1 * (
        np.sin(2 * np.pi * 120 * time) + 0.4 * np.sin(2 * np.pi * 260 * time)
    )
    result = run_anc(reference, algorithm="fxlmm")
    disturbance_rms = np.sqrt(np.mean(result["disturbance"] ** 2))
    residual_rms = np.sqrt(np.mean(result["residual"] ** 2))
    assert 20 * np.log10(disturbance_rms / residual_rms) > 3.0


def test_reality_stress_is_reproducible_finite_and_delayed(tmp_path) -> None:
    reference = generate_drifting_engine(duration=0.5)
    assert len(reference) == SAMPLE_RATE // 2
    assert np.all(np.isfinite(reference))

    first = run_anc(reference, algorithm="fxlms", step_size=0.001)
    second = run_anc(reference, algorithm="fxlms", step_size=0.001)
    np.testing.assert_array_equal(first["residual"], second["residual"])
    assert not np.array_equal(first["secondary_path_estimate"], SECONDARY_PATH)
    assert np.any(first["residual"][:10] != 0.0)
    assert round(PROCESSING_LATENCY_MS * SAMPLE_RATE / 1_000) == 16
    assert first["coefficient_norm"][:16].tolist() == [0.0] * 16

    report = stress_test(tmp_path, duration=0.5)
    assert report["configuration"]["processing_latency_samples"] == 16
    assert np.isfinite(report["algorithms"]["fxlms"]["attenuation_db"])
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "drifting-engine-comparison.png").exists()


def test_fxlmm_limits_impulsive_coefficient_growth() -> None:
    time = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
    reference = 0.05 * np.sin(2 * np.pi * 120 * time)
    reference[2 * SAMPLE_RATE : 2 * SAMPLE_RATE + 20] += (
        100 * np.hanning(40)[:20]
    )
    exact_path = {
        "secondary_estimation_noise_std": 0.0,
        "adc_noise_std": 0.0,
        "processing_latency_ms": 0.0,
    }
    baseline = run_anc(reference, algorithm="fxlms", **exact_path)
    robust = run_anc(reference, algorithm="fxlmm", **exact_path)
    assert np.max(robust["coefficient_norm"]) < np.max(
        baseline["coefficient_norm"]
    )


def test_modified_fxlmm_limits_reference_impulse_adaptation() -> None:
    time = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
    reference = 0.05 * np.sin(2 * np.pi * 120 * time)
    reference[2 * SAMPLE_RATE : 2 * SAMPLE_RATE + 20] += (
        100 * np.hanning(40)[:20]
    )
    exact_path = {
        "secondary_estimation_noise_std": 0.0,
        "adc_noise_std": 0.0,
        "processing_latency_ms": 0.0,
    }
    conventional = run_anc(reference, algorithm="fxlmm", **exact_path)
    modified = run_anc(reference, algorithm="modified-fxlmm", **exact_path)
    assert np.max(modified["coefficient_norm"]) < np.max(
        conventional["coefficient_norm"]
    )


def test_loading_and_benchmark_outputs(tmp_path) -> None:
    noise_dir = tmp_path / "noise"
    output = tmp_path / "output"
    noise_dir.mkdir()
    source_rate = 16_000
    time = np.arange(source_rate) / source_rate
    stereo = np.column_stack(
        (0.1 * np.sin(2 * np.pi * 120 * time), 0.1 * np.sin(2 * np.pi * 180 * time))
    )
    source = noise_dir / "field.wav"
    sf.write(source, stereo, source_rate, subtype="FLOAT")

    samples, metadata = load_noise(source)
    assert len(samples) == SAMPLE_RATE
    assert metadata["source_channels"] == 2
    assert np.sqrt(np.mean(samples**2)) == pytest.approx(0.1)

    report = benchmark(noise_dir, output)
    assert [item["name"] for item in report["recordings"]] == ["field"]
    assert report["recordings"][0]["algorithms"]["fxlmm"]["attenuation_db"] > 3.0
    saved = json.loads((output / "metrics.json").read_text())
    assert saved == report
    assert (output / "summary.png").exists()
    for name in (
        "disturbance",
        "fxlms-residual",
        "fxlmm-residual",
        "modified-fxlmm-residual",
    ):
        samples, rate = sf.read(output / f"field-{name}.wav")
        assert rate == SAMPLE_RATE
        assert len(samples) == SAMPLE_RATE
