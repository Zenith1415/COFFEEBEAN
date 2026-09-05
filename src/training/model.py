"""
COFFEEBEAN — ANC & Noise Suppression Model Definitions
Supports:
  1. ANCAudioModel: Waveform-to-waveform Convolutional Encoder-Decoder
  2. RNNoiseModel: Recurrent Neural Network with GRU for Edge Noise Suppression

Input shape:  [batch, 1, samples]  — noisy audio waveform
Output shape: [batch, 1, samples]  — enhanced audio waveform
"""

import torch
import torch.nn as nn


class ANCAudioModel(nn.Module):
    """Convolutional Encoder-Decoder for end-to-end waveform ANC."""

    def __init__(self, config: dict):
        super().__init__()
        self.sample_rate = config.get("audio", {}).get("sample_rate", 16000)

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
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)

        if decoded.shape[-1] > x.shape[-1]:
            decoded = decoded[..., : x.shape[-1]]
        elif decoded.shape[-1] < x.shape[-1]:
            pad = x.shape[-1] - decoded.shape[-1]
            decoded = torch.nn.functional.pad(decoded, (0, pad))

        return decoded

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return self.count_parameters() * 4 / (1024 ** 2)


class RNNoiseAudioModel(nn.Module):
    """
    RNNoise-inspired Recurrent Neural Network architecture for edge devices.
    Uses 1D Conv feature extraction + GRU temporal memory + Gain Estimation.
    """

    def __init__(self, config: dict, hidden_size: int = 128):
        super().__init__()
        self.sample_rate = config.get("audio", {}).get("sample_rate", 16000)
        self.frame_size = 160  # 10ms frame at 16kHz

        self.conv1 = nn.Conv1d(1, 64, kernel_size=31, stride=4, padding=15)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.gru = nn.GRU(input_size=64, hidden_size=hidden_size, batch_first=True, num_layers=2)
        
        self.deconv = nn.ConvTranspose1d(hidden_size, 1, kernel_size=31, stride=4, padding=15)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, samples]
        batch, channels, samples = x.shape
        features = self.relu(self.bn1(self.conv1(x)))  # [batch, 64, time_steps]

        # GRU expects [batch, time_steps, features]
        features_transposed = features.transpose(1, 2)
        gru_out, _ = self.gru(features_transposed)     # [batch, time_steps, hidden_size]

        # Deconv back to waveform samples
        gru_transposed = gru_out.transpose(1, 2)
        gain_mask = self.gate(self.deconv(gru_transposed))  # [batch, 1, out_samples]

        # Match length exactly
        if gain_mask.shape[-1] > samples:
            gain_mask = gain_mask[..., :samples]
        elif gain_mask.shape[-1] < samples:
            gain_mask = torch.nn.functional.pad(gain_mask, (0, samples - gain_mask.shape[-1]))

        # Apply gain mask to input waveform
        enhanced = x * gain_mask
        return enhanced

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return self.count_parameters() * 4 / (1024 ** 2)


def get_model(config: dict) -> nn.Module:
    """Model factory based on config."""
    model_type = config.get("model", {}).get("type", "conv_enc_dec").lower()
    if model_type in ("rnnoise", "gru", "rnn"):
        return RNNoiseAudioModel(config)
    return ANCAudioModel(config)
