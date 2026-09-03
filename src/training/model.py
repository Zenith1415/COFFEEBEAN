"""
COFFEEBEAN — ANC Model Definition
Phase 4: Placeholder convolutional encoder-decoder.

ML team: Replace ANCAudioModel with your real architecture.
The training loop, DVC pipeline, and MLflow logging are model-agnostic.

Input shape:  [batch, 1, samples]  — noisy audio waveform
Output shape: [batch, 1, samples]  — enhanced audio waveform
"""

import torch
import torch.nn as nn


class ANCAudioModel(nn.Module):
    """
    Placeholder ANC model — convolutional encoder-decoder.

    Replace this class with your real ANC architecture:
      - TCN (Temporal Convolutional Network)
      - LSTM / BiLSTM
      - U-Net
      - Conv-TasNet
      - SEGAN
      - Custom DRDO/SIH architecture

    The rest of the pipeline (preprocessing, training loop,
    MLflow logging, DVC stages, ONNX export) is model-agnostic
    and does not need to change when you swap this class.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.sample_rate = config["audio"]["sample_rate"]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32, stride=2, padding=15),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 1, kernel_size=32, stride=2, padding=15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy audio tensor [batch, 1, samples]

        Returns:
            Enhanced audio tensor [batch, 1, samples]
        """
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)

        # Trim or pad to match input length exactly
        if decoded.shape[-1] > x.shape[-1]:
            decoded = decoded[..., : x.shape[-1]]
        elif decoded.shape[-1] < x.shape[-1]:
            pad = x.shape[-1] - decoded.shape[-1]
            decoded = torch.nn.functional.pad(decoded, (0, pad))

        return decoded

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        """Return approximate model size in MB."""
        return self.count_parameters() * 4 / (1024 ** 2)
