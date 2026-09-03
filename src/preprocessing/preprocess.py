"""
COFFEEBEAN — Audio Preprocessing
Phase 3: Stubs only.
Phase 4: Implement real audio loading, resampling, and feature extraction.
"""

from pathlib import Path


def load_dataset(data_dir: str) -> dict:
    """
    Load audio dataset from DVC-tracked directory.

    Args:
        data_dir: Path to data/raw/ (DVC-managed).

    Returns:
        Dict with keys: 'speech', 'stationary_noise',
        'nonstationary_noise', 'impulsive_noise'.
        Each value is a list of file paths.

    Phase 4: Implement with torchaudio or soundfile.
    """
    root = Path(data_dir)
    dataset = {}

    for category in [
        "speech",
        "stationary_noise",
        "nonstationary_noise",
        "impulsive_noise",
    ]:
        category_dir = root / category
        if category_dir.exists():
            files = list(category_dir.glob("*.wav"))
            dataset[category] = files
            print(f"  {category}: {len(files)} file(s) found")
        else:
            dataset[category] = []
            print(f"  {category}: directory not found")

    return dataset


def preprocess_audio(file_path: str, target_sample_rate: int = 16000):
    """
    Load and preprocess a single audio file.

    Steps (Phase 4):
        1. Load audio
        2. Resample to target_sample_rate
        3. Convert to mono
        4. Normalize

    Args:
        file_path:          Path to audio file.
        target_sample_rate: Target sample rate in Hz.

    Phase 4: Implement with torchaudio.
    """
    raise NotImplementedError(
        "Audio preprocessing will be implemented in Phase 4. "
        "Requires torchaudio: pip install torchaudio"
    )


def mix_signals(
    speech,
    noise,
    target_snr_db: float = 0.0,
):
    """
    Mix speech and noise at a target SNR level.

    Args:
        speech:        Clean speech signal array.
        noise:         Noise signal array.
        target_snr_db: Desired SNR in dB for the mixture.

    Returns:
        Tuple of (noisy_signal, scale_factor).

    Phase 4: Implement for evaluation dataset creation.
    """
    raise NotImplementedError(
        "Signal mixing will be implemented in Phase 4."
    )
