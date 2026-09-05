"""
COFFEEBEAN — Audio Preprocessing
Phase 4: Real audio loading, resampling, mixing, and dataset building.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

logger = logging.getLogger(__name__)


# ── Audio I/O ──────────────────────────────────────────────────────────────────

def load_audio(
    file_path: str,
    target_sr: int = 16000,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Load a WAV file, resample to target_sr, convert to mono, normalize.

    Args:
        file_path: Path to .wav file.
        target_sr: Target sample rate in Hz.
        normalize: If True, peak-normalizes to [-1.0, 1.0].

    Returns:
        Tensor of shape [1, samples] (float32).
    """
    try:
        import soundfile as sf
        samples, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T)  # [channels, samples]
    except Exception:
        waveform, sr = torchaudio.load(str(file_path))

    # Resample if needed
    if sr != target_sr:
        waveform = F.resample(waveform, orig_freq=sr, new_freq=target_sr)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Peak normalize
    if normalize:
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

    return waveform  # [1, samples]


def load_dataset(
    data_dir: str,
    target_sr: int = 16000,
) -> Dict[str, List[torch.Tensor]]:
    """
    Load all WAV files from DVC-tracked data/raw/ subdirectories.

    Args:
        data_dir:  Path to data/raw/.
        target_sr: Target sample rate in Hz.

    Returns:
        Dict with keys: speech, stationary_noise,
        nonstationary_noise, impulsive_noise.
        Each value is a list of waveform tensors [1, samples].
    """
    root = Path(data_dir)
    categories = [
        "speech",
        "stationary_noise",
        "nonstationary_noise",
        "impulsive_noise",
    ]

    dataset: Dict[str, List[torch.Tensor]] = {}

    for cat in categories:
        cat_dir = root / cat
        files = list(cat_dir.glob("*.wav")) if cat_dir.exists() else []
        waveforms = []

        for f in files:
            try:
                w = load_audio(str(f), target_sr=target_sr)
                waveforms.append(w)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        dataset[cat] = waveforms
        logger.info(f"  {cat}: {len(waveforms)} file(s) loaded")

    return dataset


# ── Signal Mixing ──────────────────────────────────────────────────────────────

def mix_signals(
    speech: torch.Tensor,
    noise: torch.Tensor,
    target_snr_db: float = 0.0,
) -> Tuple[torch.Tensor, float]:
    """
    Mix speech and noise at a target SNR level.

    Args:
        speech:        Clean speech tensor [1, samples].
        noise:         Noise tensor [1, samples].
        target_snr_db: Desired SNR in dB.

    Returns:
        Tuple of (noisy_mixture [1, samples], actual_scale_factor).
    """
    # Match lengths
    min_len = min(speech.shape[-1], noise.shape[-1])
    speech = speech[..., :min_len]
    noise = noise[..., :min_len]

    # Compute RMS
    speech_rms = speech.pow(2).mean().sqrt().item()
    noise_rms = noise.pow(2).mean().sqrt().item()

    if noise_rms == 0 or speech_rms == 0:
        return speech.clone(), 0.0

    # Scale noise to achieve target SNR
    target_noise_rms = speech_rms / (10 ** (target_snr_db / 20.0))
    scale = target_noise_rms / noise_rms
    noisy = speech + scale * noise

    # Re-normalize to prevent clipping
    peak = noisy.abs().max().item()
    if peak > 1.0:
        noisy = noisy / peak

    return noisy, float(scale)


# ── Training Dataset ───────────────────────────────────────────────────────────

class ANCDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for ANC training.

    Each sample is a (noisy, clean) pair created by mixing
    speech + noise at a random SNR from the configured range.
    """

    def __init__(
        self,
        dataset: Dict[str, List[torch.Tensor]],
        chunk_samples: int = 16000,
        snr_range_db: Tuple[float, float] = (-5.0, 20.0),
        num_samples: int = 1000,
    ):
        """
        Args:
            dataset:       Output of load_dataset().
            chunk_samples: Number of samples per training chunk (1 sec @ 16kHz).
            snr_range_db:  (min_snr, max_snr) range for random mixing.
            num_samples:   Number of pairs to generate per epoch.
        """
        self.speech = dataset.get("speech", [])
        self.noise = (
            dataset.get("stationary_noise", [])
            + dataset.get("nonstationary_noise", [])
            + dataset.get("impulsive_noise", [])
        )
        self.chunk_samples = chunk_samples
        self.snr_range_db = snr_range_db
        self.num_samples = num_samples

        if not self.speech:
            raise ValueError(
                "No speech files found in dataset. "
                "Add WAV files to data/raw/speech/ and run dvc push."
            )
        if not self.noise:
            logger.warning(
                "No noise files found. Training on speech only (no mixing)."
            )

    def __len__(self) -> int:
        return self.num_samples

    def _random_chunk(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract a random chunk of chunk_samples length."""
        total = waveform.shape[-1]
        if total <= self.chunk_samples:
            # Pad with zeros if too short
            pad = self.chunk_samples - total
            return torch.nn.functional.pad(waveform, (0, pad))
        start = torch.randint(0, total - self.chunk_samples, (1,)).item()
        return waveform[..., start : start + self.chunk_samples]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (noisy [1, chunk_samples], clean [1, chunk_samples]).
        """
        # Random speech chunk
        speech_idx = idx % len(self.speech)
        clean = self._random_chunk(self.speech[speech_idx])

        if not self.noise:
            return clean.clone(), clean.clone()

        # Random noise chunk
        noise_idx = torch.randint(0, len(self.noise), (1,)).item()
        noise_chunk = self._random_chunk(self.noise[noise_idx])

        # Random SNR
        snr = torch.FloatTensor(1).uniform_(*self.snr_range_db).item()
        noisy, _ = mix_signals(clean, noise_chunk, target_snr_db=snr)

        return noisy, clean
