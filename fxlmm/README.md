# Standalone FxLMM and modified FxLMM benchmark

This folder compares normalized filtered-x LMS, filtered-x least mean
M-estimate, and modified FxLMM controllers. All use the same normalized update;
FxLMM applies Hampel's bounded score to the error, while modified FxLMM also
applies it to the reference used by the adaptation path.
The three Hampel thresholds are refreshed every 256 samples from the trailing
one-second 95th, 97.5th, and 99th absolute-sample percentiles.
The formulation follows Sun, Li, and Lim's
[robust FxLMM study](https://doi.org/10.1016/j.apacoust.2014.10.012).

The controller includes seeded hardware-reality effects: a noisy secondary-path
estimate, 0.02-RMS reference/error ADC noise, and 2 ms of processing latency.
The WAV inputs are real recordings, but the primary and secondary acoustic
paths remain FIR simulations. This is an offline algorithm benchmark, not a
physical speaker/microphone ANC test.

Run the tests and the five-recording benchmark from the repository root:

```bash
uv run pytest fxlmm/test_fxlmm.py -q
uv run python -m fxlmm
```

Run the synthetic non-stationary stress test (50-to-60 Hz drift plus a fixed
120 Hz harmonic):

```bash
uv run python -m fxlmm --stress-test --output runs/fxlmm-stress
```

Override inputs, outputs, or adaptation settings when needed:

```bash
uv run python -m fxlmm \
  --noise-dir samples/real-noise/wav \
  --output runs/fxlmm \
  --step-size 0.001 \
  --taps 64
```

`metrics.json` records preprocessing, path coefficients, RMS attenuation,
residual peaks, and coefficient norms. Each recording also produces a
disturbance WAV, residual WAVs for all three algorithms, and a learning-curve plot.
`summary.png` compares attenuation across recordings.

Do not connect this simulation to a loudspeaker. Physical ANC requires a
measured secondary path, reference and error microphones, output limiting,
feedback protection, and hearing-safety controls.
