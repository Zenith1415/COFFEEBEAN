"""
COFFEEBEAN — ANC Evaluation Metrics
Phase 3: Stubs only.
Phase 5: Implement SNR, STOI, PESQ with real audio processing.
"""

import numpy as np


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio (dB).

    Args:
        clean:  Clean reference signal.
        noisy:  Noisy or enhanced signal.

    Returns:
        SNR in dB.

    Phase 5: Implement with real numpy/scipy logic.
    """
    raise NotImplementedError(
        "SNR computation will be implemented in Phase 5. "
        "Requires clean reference and noisy/enhanced signal arrays."
    )


def compute_stoi(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Short-Time Objective Intelligibility (0–1).

    Args:
        clean:       Clean reference signal.
        enhanced:    Enhanced (ANC output) signal.
        sample_rate: Audio sample rate (Hz).

    Returns:
        STOI score between 0 and 1. Higher is better.

    Phase 5: Implement using pystoi library.
    """
    raise NotImplementedError(
        "STOI computation will be implemented in Phase 5. "
        "Requires pystoi: pip install pystoi"
    )


def compute_pesq(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    mode: str = "wb",
) -> float:
    """
    Perceptual Evaluation of Speech Quality (MOS-LQO score).

    Args:
        clean:       Clean reference signal.
        enhanced:    Enhanced (ANC output) signal.
        sample_rate: Audio sample rate (Hz). Must be 8000 or 16000.
        mode:        'nb' (narrowband) or 'wb' (wideband).

    Returns:
        PESQ MOS-LQO score. Range: -0.5 to 4.5. Higher is better.

    Phase 5: Implement using pesq library.
    """
    raise NotImplementedError(
        "PESQ computation will be implemented in Phase 5. "
        "Requires pesq: pip install pesq"
    )


def evaluate_anc_model(
    clean: np.ndarray,
    noise: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
) -> dict:
    """
    Run full ANC evaluation pipeline.

    Pipeline:
        clean + noise → mix (noisy) → ANC model → enhanced
        Then compare: noisy vs enhanced vs clean reference.

    Args:
        clean:       Clean speech reference.
        noise:       Noise signal.
        enhanced:    ANC model output.
        sample_rate: Audio sample rate (Hz).

    Returns:
        Dict with all evaluation metrics.

    Phase 5: Implement when real audio processing is available.
    """
    raise NotImplementedError(
        "Full evaluation pipeline will be implemented in Phase 5."
    )
