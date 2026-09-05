"""
COFFEEBEAN — ANC Evaluation Metrics
Phase 5: Real SNR, STOI, and PESQ via torchmetrics[audio].
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── SNR ────────────────────────────────────────────────────────────────────────

def compute_snr(clean: np.ndarray, processed: np.ndarray, eps: float = 1e-8) -> float:
    """Signal-to-Noise Ratio in dB. Higher is better."""
    clean     = np.asarray(clean,     dtype=np.float64).flatten()
    processed = np.asarray(processed, dtype=np.float64).flatten()
    min_len   = min(len(clean), len(processed))
    clean, processed = clean[:min_len], processed[:min_len]
    noise        = clean - processed
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean(noise  ** 2)
    if noise_power < eps:
        return float("inf")
    return float(10.0 * np.log10((signal_power + eps) / (noise_power + eps)))


def compute_snr_improvement(
    clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray
) -> dict:
    """Returns snr_before, snr_after, snr_improvement (all dB)."""
    snr_before = compute_snr(clean, noisy)
    snr_after  = compute_snr(clean, enhanced)
    return {
        "snr_before":      round(snr_before, 4),
        "snr_after":       round(snr_after,  4),
        "snr_improvement": round(snr_after - snr_before, 4),
    }


# ── STOI ───────────────────────────────────────────────────────────────────────

def compute_stoi(
    clean: np.ndarray, enhanced: np.ndarray, sample_rate: int, extended: bool = False
) -> float:
    """
    Short-Time Objective Intelligibility (0-1). Uses torchmetrics.
    Higher is better. >0.65 is generally intelligible.
    """
    try:
        from torchmetrics.audio import ShortTimeObjectiveIntelligibility
    except ImportError:
        raise ImportError("Run: pip install torchmetrics[audio]")

    clean_t    = torch.tensor(clean,    dtype=torch.float32).flatten()
    enhanced_t = torch.tensor(enhanced, dtype=torch.float32).flatten()
    min_len    = min(len(clean_t), len(enhanced_t))
    clean_t, enhanced_t = clean_t[:min_len], enhanced_t[:min_len]

    metric = ShortTimeObjectiveIntelligibility(fs=sample_rate, extended=extended)
    score  = metric(enhanced_t.unsqueeze(0), clean_t.unsqueeze(0))
    return float(round(score.item(), 4))


# ── PESQ ───────────────────────────────────────────────────────────────────────

def compute_pesq(
    clean: np.ndarray, enhanced: np.ndarray, sample_rate: int, mode: str = "wb"
) -> float:
    """
    Perceptual Evaluation of Speech Quality via torchmetrics.
    Range: -0.5 to 4.5. Higher is better. Good quality >3.0.
    sample_rate must be 8000 or 16000.
    """
    try:
        from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
    except ImportError:
        raise ImportError("Run: pip install torchmetrics[audio]")

    if sample_rate not in (8000, 16000):
        raise ValueError(f"PESQ requires 8000 or 16000 Hz, got {sample_rate}")

    clean_t    = torch.tensor(clean,    dtype=torch.float32).flatten()
    enhanced_t = torch.tensor(enhanced, dtype=torch.float32).flatten()
    min_len    = min(len(clean_t), len(enhanced_t))
    clean_t, enhanced_t = clean_t[:min_len], enhanced_t[:min_len]

    metric = PerceptualEvaluationSpeechQuality(fs=sample_rate, mode=mode)
    score  = metric(enhanced_t.unsqueeze(0), clean_t.unsqueeze(0))
    return float(round(score.item(), 4))


# ── Full Evaluation Pipeline ───────────────────────────────────────────────────

def evaluate_anc_model(
    model: torch.nn.Module,
    clean: torch.Tensor,
    noise: torch.Tensor,
    sample_rate: int,
    target_snr_db: float = 0.0,
    device: torch.device = None,
) -> dict:
    """
    Full ANC evaluation pipeline:
      clean + noise -> mix -> ANC model -> enhanced
      -> SNR / STOI / PESQ (before vs after)

    Returns dict with all metrics ready to log to MLflow.
    """
    if device is None:
        device = torch.device("cpu")

    from src.preprocessing.preprocess import mix_signals

    noisy, _ = mix_signals(clean, noise, target_snr_db=target_snr_db)

    model.eval()
    with torch.no_grad():
        enhanced = model(noisy.unsqueeze(0).to(device)).squeeze(0).cpu()

    clean_np    = clean.squeeze().numpy().astype(np.float64)
    noisy_np    = noisy.squeeze().numpy().astype(np.float64)
    enhanced_np = enhanced.squeeze().numpy().astype(np.float64)

    metrics = {}

    # SNR
    metrics.update(compute_snr_improvement(clean_np, noisy_np, enhanced_np))

    # STOI
    try:
        metrics["stoi_noisy"]       = compute_stoi(clean_np, noisy_np,    sample_rate)
        metrics["stoi_enhanced"]    = compute_stoi(clean_np, enhanced_np, sample_rate)
        metrics["stoi_improvement"] = round(metrics["stoi_enhanced"] - metrics["stoi_noisy"], 4)
    except Exception as e:
        logger.warning(f"STOI failed: {e}")
        metrics["stoi_noisy"] = metrics["stoi_enhanced"] = metrics["stoi_improvement"] = None

    # PESQ
    try:
        metrics["pesq_noisy"]       = compute_pesq(clean_np, noisy_np,    sample_rate)
        metrics["pesq_enhanced"]    = compute_pesq(clean_np, enhanced_np, sample_rate)
        metrics["pesq_improvement"] = round(metrics["pesq_enhanced"] - metrics["pesq_noisy"], 4)
    except Exception as e:
        logger.warning(f"PESQ failed: {e}")
        metrics["pesq_noisy"] = metrics["pesq_enhanced"] = metrics["pesq_improvement"] = None

    return metrics
