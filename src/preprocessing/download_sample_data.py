"""
COFFEEBEAN — Benchmark Audio Dataset Generator
Generates sample benchmark audio files for speech and noise categories
based on REAL_NOISE_SOURCES.md specifications.
"""

import math
from pathlib import Path
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 5.0  # seconds


def create_sample_speech(output_path: Path):
    """Generate harmonic speech-like formant signal."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    # Formants for vowel /a/ and /i/
    f0 = 130 + 15 * np.sin(2 * np.pi * 1.5 * t)  # Pitch variation
    formant1 = np.sin(2 * np.pi * 700 * t)
    formant2 = 0.6 * np.sin(2 * np.pi * 1200 * t)
    formant3 = 0.3 * np.sin(2 * np.pi * 2500 * t)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
    speech = (formant1 + formant2 + formant3) * envelope * 0.4
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), speech.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
    print(f"Created speech: {output_path}")


def create_stationary_noise(output_path: Path):
    """Generate engine / helicopter hum stationary noise."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    generator = np.random.default_rng(42)
    white = generator.standard_normal(len(t)) * 0.15
    # Low-frequency engine rumble + propeller harmonics
    engine = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.2 * np.sin(2 * np.pi * 240 * t) + 0.1 * np.sin(2 * np.pi * 360 * t)
    noise = engine + white
    noise = noise / np.max(np.abs(noise)) * 0.7
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), noise.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
    print(f"Created stationary noise: {output_path}")


def create_nonstationary_noise(output_path: Path):
    """Generate siren / wind non-stationary noise."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    # Siren pitch modulation (600 Hz to 1200 Hz)
    f_siren = 800 + 300 * np.sin(2 * np.pi * 0.8 * t)
    phase = 2 * np.pi * np.cumsum(f_siren) / SAMPLE_RATE
    siren = 0.4 * np.sin(phase)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), siren.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
    print(f"Created non-stationary noise: {output_path}")


def create_impulsive_noise(output_path: Path):
    """Generate gunfire / impulsive blast noise bursts."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    noise = np.random.randn(len(t)) * 0.05
    # Add 3 sharp impulse bursts
    for burst_time in [1.2, 2.5, 3.8]:
        idx = int(burst_time * SAMPLE_RATE)
        width = int(0.08 * SAMPLE_RATE)
        burst_t = np.linspace(0, 0.08, width, endpoint=False)
        envelope = np.exp(-40.0 * burst_t)
        burst = 1.2 * envelope * np.sin(2 * np.pi * 500 * burst_t)
        noise[idx : idx + width] += burst

    noise = np.clip(noise, -1.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), noise.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
    print(f"Created impulsive noise: {output_path}")


def populate_dataset(data_dir: str = "data/raw"):
    base = Path(data_dir)
    create_sample_speech(base / "speech" / "sample_speech.wav")
    create_stationary_noise(base / "stationary_noise" / "engine_noise.wav")
    create_nonstationary_noise(base / "nonstationary_noise" / "siren_noise.wav")
    create_impulsive_noise(base / "impulsive_noise" / "gunfire_noise.wav")
    print("Dataset populated successfully!")


if __name__ == "__main__":
    populate_dataset()
