# XMC4800 integration plan for COFFEEBEAN

## Decision

Use the XMC4800 first as a deterministic audio/DSP target, not as a replacement
for the current Python application.

1. Port and validate the existing FXLMS/FXLMM algorithms with recorded samples.
2. Add real 48 kHz audio input/output through an external I2S audio codec.
3. Attempt the pinned RNNoise v0.2 little model only as a measured feasibility
   port.
4. If RNNoise misses its real-time gate, retain the XMC4800 for audio control,
   safety, push-to-talk, and telemetry while a Raspberry Pi-class Linux board
   performs speech enhancement.

Do not attempt to run DeepFilterNet3, Python, PyTorch, or the desktop dashboard
on the XMC4800.

## Assumptions to confirm before buying hardware

- The board is `KIT_XMC48_RLX_ECAT_V2.1`, or another XMC4800 variant with 2 MB
  flash and 352 KB RAM.
- The required output is enhanced microphone speech, not acoustic anti-noise
  played into an open speaker.
- Mono, 48 kHz, 16-bit PCM is acceptable for the first hardware prototype.
- A wired debug connection is acceptable during bring-up.

If the board or chip variant differs, verify its flash size, available USIC/IIS
pins, debugger, and connector voltage before fixing the pin map.

## Recommended architecture

```text
Microphone
    |
mic preamp + 48 kHz I2S audio codec
    |
USIC/IIS + DMA ping-pong buffers
    |
XMC4800
    |-- FXLMM or RNNoise candidate
    |-- input/output limiter and bypass
    |-- deadline, clipping, and overflow counters
    |
codec output or framed PCM link to Linux companion
    |
radio / recorder / existing COFFEEBEAN evaluation tools
```

An external audio codec is preferable to the on-chip ADC/DAC for the speech
path because it supplies the microphone bias/preamp, matched audio sampling,
and a practical I2S interface. Use the XMC4800 DAC only for initial waveform
tests, not as the final speech-quality path.

Keep EtherCAT out of the first prototype. It is useful later only if the project
needs synchronized industrial nodes or deterministic telemetry; it does not
improve speech enhancement.

## Minimal additional hardware

- Existing XMC4800 board and debugger/USB cable.
- One 3.3 V-compatible, 48 kHz I2S codec/module with microphone input and audio
  output.
- Electret or analog MEMS microphone suitable for that codec.
- Headphones or a line-level recorder for controlled testing.
- Logic analyzer or oscilloscope for MCLK/BCLK/LRCLK/data bring-up.
- Raspberry Pi 4 (reliable choice) or Pi Zero 2 W (provisional) only if the MCU
  cannot meet the RNNoise timing gate.

Do not connect a speaker for physical ANC testing during the initial phases.

## Firmware shape

Start with a bare-metal foreground loop plus DMA interrupts. Add an RTOS only if
measurement shows that scheduling multiple independent tasks is actually
necessary.

```text
firmware/
  app/main.c             clock, codec, DMA, processing loop
  audio/audio_io.c       I2S and ping-pong buffers
  dsp/fxlmm.c            embedded adaptive-filter implementation
  dsp/rnnoise_adapter.c  optional 480-sample frame adapter
  safety/limiter.c       gain, saturation, bypass, fault behavior
  telemetry/metrics.c    cycles, misses, overflows, clipping
  test/golden_vectors.c  short inputs and expected outputs from Python
```

Use the Infineon XMC peripheral libraries plus CMSIS-DSP. Keep audio buffers and
model state statically allocated; no allocation is allowed in the audio loop.

## Phased implementation

### Phase 0 - Establish the golden reference (half day)

- Record the exact board/MCU marking and memory size.
- Pin the firmware toolchain and the same RNNoise commit/model already used by
  `scripts/build-rnnoise-v0.2.sh`.
- Export short deterministic input/output vectors from the Python FXLMS and
  modified-FXLMM implementation.
- Save one 48 kHz noisy speech clip and its desktop RNNoise output as the
  comparison fixture.

Exit gate: the repository can regenerate byte-identifiable input fixtures and
numeric expected outputs from a fixed seed.

### Phase 1 - Board and timing bring-up (one day)

- Create an empty XMC4800 application in ModusToolbox.
- Enable the 144 MHz system clock, debugger output, one status LED, and a
  free-running cycle counter.
- Implement a periodic interrupt and measure its jitter.
- Print firmware version, clock, reset cause, and high-water memory use.

Exit gate: a 10 ms event runs for 60 seconds with 6,000 events and no missed
events.

### Phase 2 - Offline FXLMM port (two days)

- Port only the filter update and score functions required by the current
  simulation.
- Begin with `float32`, using the Cortex-M4F FPU and CMSIS-DSP FIR routines.
- Feed the generated samples from flash; do not add live audio yet.
- Return residual samples and coefficient norms over debug UART/USB for
  comparison with Python.

Exit gate: residuals agree with the Python reference within a documented
floating-point tolerance, coefficient values stay finite, and every sample is
processed before its deadline.

### Phase 3 - 48 kHz codec and DMA (two to three days)

- Configure a USIC channel in IIS mode for the selected codec.
- Use two input and two output buffers. Start with 480 samples per buffer, which
  matches RNNoise's 10 ms frame.
- Implement unity-gain pass-through, mute, bypass, saturation, and clipping
  counters before adding enhancement.
- Verify clocks and data framing with a logic analyzer; verify frequency and
  gain using recorded test tones.

Exit gate: 10 minutes of pass-through with no DMA overrun/underrun, no unexpected
clipping, correct 48 kHz rate, and bounded end-to-end latency.

### Phase 4 - Embedded DSP baseline (one to two days)

- Run the FXLMM code on the live frame path in record-only or line-level tests.
- Log processing cycles per frame, maximum stack use, overflows, and bypass
  events.
- Capture the MCU output as WAV and evaluate it with the existing `coffeebean
  evaluate` command.

Exit gate: the live output matches the offline MCU result within tolerance and
the 60-second run has zero processing deadline misses.

### Phase 5 - RNNoise feasibility spike (three to five days, time-boxed)

- Cross-compile the repository's pinned RNNoise v0.2 little C sources for
  Cortex-M4F using hardware floating point.
- Remove desktop-only allocation/file-loading paths and use the compiled-in
  model with one static state object.
- Profile the unmodified scalar build first. Optimize only measured hotspots,
  using CMSIS-DSP where it maps cleanly.
- Process exactly 480 new samples per call and preserve RNNoise's two-frame
  latency compensation used by this repository.

Pass gate:

- linked image fits flash with at least 20% headroom;
- peak static plus stack memory fits RAM with at least 25% headroom;
- median processing time is below 8 ms per 10 ms frame;
- worst observed processing time is below 10 ms;
- zero deadline misses, DMA faults, or non-finite samples in a 60-second run;
- output quality remains close to the desktop RNNoise reference.

Stop after five working days if the timing gate is not close. Do not spend the
project schedule rewriting the neural network for this MCU.

### Phase 6 - Choose the deployment path (one day)

If RNNoise passes, make the XMC4800 a standalone mono enhancer and add the final
codec/radio interface.

If it fails, use this split:

```text
XMC4800: codec control, PTT, watchdog, limiter, bypass, fault telemetry
Linux SBC: RNNoise or DeepFilterNet3, USB/network audio, dashboard
```

The system must automatically bypass enhancement or mute safely if the companion
processor stops responding.

### Phase 7 - Hardware demonstration (two days)

- Add a firmware build/readme under `edge/xmc4800/`.
- Run clean speech, continuous noise at -5/0/+5 dB, and the repository's
  impulsive recording-only case.
- Report latency, frame misses, clipping, RAM, flash, SI-SDR improvement, and
  STOI improvement beside the existing Mac/SBC results.
- Demonstrate A/B playback only after capture; do not create an acoustic feedback
  loop.

## Recommended RNNoise board and bring-up

Use a Raspberry Pi 4 with 2 GB RAM and 64-bit Raspberry Pi OS for the first
RNNoise deployment. It provides comfortable CPU and memory headroom while
remaining small enough for the intended edge demonstration. Use a Pi 5 only if
the same board must also explore DeepFilterNet or run a heavier interface. Treat
the Pi Zero 2 W as a later size/cost optimization after the Pi 4 path passes.

For the shortest hardware path, connect a class-compliant USB microphone or USB
audio interface directly to the Pi. Do not route audio through the XMC4800 yet.
The XMC4800 can later connect over UART, CAN, or GPIO for push-to-talk, status,
watchdog, and bypass control.

### Install and build on Raspberry Pi OS

```bash
sudo apt update
sudo apt install -y build-essential curl libsndfile1 portaudio19-dev

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

cd COFFEEBEAN
uv python install 3.12
uv sync --extra live
./scripts/build-rnnoise-v0.2.sh
uv run pytest -q
```

The build script downloads the repository's pinned RNNoise source and little
model, verifies the model checksum, and builds `build/rnnoise-v0.2/librnnoise.so`.

### Prove file processing first

Copy or record a mono 48 kHz WAV, then run:

```bash
uv run coffeebean enhance \
  --input samples/local/noisy.wav \
  --output runs/pi/rnnoise.wav \
  --model RNNoise

cat runs/pi/rnnoise.wav.benchmark.json
```

Pass this step when `real_time_factor` is below `0.8`, the output contains no
clipping or invalid samples, and listening confirms that speech was not removed.

### Prove microphone capture and bounded streaming

```bash
uv run coffeebean devices

uv run coffeebean stream \
  --device DEVICE_NUMBER \
  --duration 60 \
  --chunk-ms 20 \
  --context-ms 40 \
  --model RNNoise \
  --output runs/pi/edge-smoke

cat runs/pi/edge-smoke/report.json
```

Accept the board when the report shows zero input overflows, zero processing
deadline misses, no clipping, and stable results across at least five runs.

The current `stream` command processes audio while recording and writes noisy
and enhanced WAV files; it does not send enhanced audio to a headset, radio, or
virtual microphone. That behavior is deliberate for safe benchmarking.

### Production live-audio step

After the benchmark passes, implement a small native Linux audio process rather
than extending the Python dashboard:

1. Open a mono 48 kHz ALSA capture and playback device.
2. Create one persistent RNNoise state at startup.
3. Read exactly 480 samples, convert them to RNNoise's float PCM scale, and call
   `rnnoise_process_frame` once every 10 ms.
4. Apply the output limiter, convert back to the device format, and write the
   frame to ALSA or a PipeWire virtual microphone.
5. Expose counters for capture overruns, playback underruns, processing time,
   clipping, and automatic bypass.

Budget approximately 20 ms for RNNoise's algorithmic delay plus the capture and
playback buffers. Use a wired USB audio interface first; add an I2S codec only
after the software path is stable.

## Physical ANC is a separate project gate

The XMC4800 is suitable for experimenting with deterministic FxLMS/FxNLMS
control, but physical ANC requires a reference microphone, error microphone,
identified secondary path, output limiter, emergency bypass, and hearing-safe
test procedure. Complete the record-only speech-enhancement path first. A
speaker-driven ANC loop must not be inferred from a successful offline FXLMM
port.

## Definition of done

The integration is complete when one documented 60-second hardware run produces:

- a captured input WAV and processed output WAV;
- firmware commit/toolchain/model identifiers;
- actual flash and peak RAM use;
- median and maximum processing time per frame;
- zero unreported overruns, underruns, or deadline misses;
- clipping and limiter counts;
- repository-generated evaluation metrics; and
- a clear standalone-MCU or MCU-plus-Linux deployment decision.

## Official references

- [Infineon XMC4800 product page](https://www.infineon.com/part/XMC4800-F144F2048-AA)
- [Infineon XMC4700/XMC4800 reference manual](https://www.infineon.com/assets/row/public/documents/30/44/infineon-referencemanual-xmc4700-xmc4800-um-en.pdf)
- [Infineon XMC4700/XMC4800 Relax Kit manual](https://www.infineon.com/assets/row/public/documents/30/44/infineon-board-user-manual-xmc4700-xmc4800-relax-kit-series-usermanual-en.pdf)
- [Infineon XMC4000 software-tool support](https://documentation.infineon.com/xmc4000/docs/jti1706521669019)
- [Xiph RNNoise source](https://github.com/xiph/rnnoise)
- [Arm CMSIS-DSP FIR documentation](https://arm-software.github.io/CMSIS-DSP/latest/group__FIR.html)
