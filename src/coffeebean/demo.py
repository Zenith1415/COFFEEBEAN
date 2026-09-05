from __future__ import annotations

import os
import sys
import subprocess
import threading
from datetime import datetime
from pathlib import Path

# Ensure project root and src are in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from coffeebean.cli import list_input_devices, stream_demo


def _summary_text(report: dict[str, object]) -> str:
    noisy = report["noisy"]
    enhanced = report["enhanced"]
    return (
        f"Streaming: {report['duration_seconds']:.1f}s at {report['sample_rate']} Hz\n"
        f"Median inference: {report['processing_seconds_median'] * 1000:.1f} ms\n"
        f"Estimated latency: {report['estimated_output_latency_seconds'] * 1000:.0f} ms\n"
        f"Deadline misses: {report['processing_deadline_misses']}  |  "
        f"Input overflows: {report['input_overflow_blocks']}\n"
        f"Noisy peak: {noisy['peak']:.3f}  |  Enhanced peak: {enhanced['peak']:.3f}\n"
        f"Enhanced clipping: {enhanced['clipping_fraction'] * 100:.3f}%"
    )


def main() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    cache = Path.cwd() / "runs" / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("COFFEEBEAN — Defence Communications Demo")
    root.geometry("1180x780")
    root.minsize(980, 680)
    root.configure(bg="#0b1220")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TLabel", background="#111c31", foreground="#dbeafe")
    style.configure("TFrame", background="#111c31")
    style.configure("TButton", padding=(12, 8), font=("Helvetica", 12, "bold"))
    style.configure("TCombobox", padding=5)

    header = tk.Frame(root, bg="#102a43", padx=24, pady=18)
    header.pack(fill="x")
    tk.Label(
        header,
        text="COFFEEBEAN",
        bg="#102a43",
        fg="#7dd3fc",
        font=("Helvetica", 25, "bold"),
    ).pack(anchor="w")
    tk.Label(
        header,
        text="Real-time AI speech enhancement for high-noise communications",
        bg="#102a43",
        fg="#e0f2fe",
        font=("Helvetica", 13),
    ).pack(anchor="w", pady=(2, 0))

    body = tk.Frame(root, bg="#0b1220", padx=18, pady=16)
    body.pack(fill="both", expand=True)
    controls = ttk.Frame(body, padding=16)
    controls.pack(side="left", fill="y", padx=(0, 14))
    chart_frame = ttk.Frame(body, padding=8)
    chart_frame.pack(side="right", fill="both", expand=True)

    tk.Label(
        controls,
        text="LIVE DEMONSTRATION",
        bg="#111c31",
        fg="#38bdf8",
        font=("Helvetica", 12, "bold"),
    ).pack(anchor="w", pady=(0, 14))

    ttk.Label(controls, text="Microphone").pack(anchor="w")
    device_box = ttk.Combobox(controls, state="readonly", width=34)
    device_box.pack(fill="x", pady=(4, 12))
    device_indexes: dict[str, int] = {}

    ttk.Label(controls, text="Capture duration (seconds)").pack(anchor="w")
    duration_var = tk.StringVar(value="10")
    duration_entry = ttk.Entry(controls, textvariable=duration_var, width=12)
    duration_entry.pack(anchor="w", pady=(4, 14))

    status_var = tk.StringVar(value="Ready — use headphones and keep noise playback safe.")
    summary_var = tk.StringVar(value="No run yet")
    current_report: dict[str, object] | None = None
    playback: subprocess.Popen[bytes] | None = None

    figure = Figure(figsize=(8, 6), dpi=100, facecolor="#111c31")
    axes = figure.subplots(2, 2)
    for axis, title in zip(
        axes.flat,
        ("Noisy waveform", "Enhanced waveform", "Noisy spectrum", "Enhanced spectrum"),
        strict=True,
    ):
        axis.set_title(title, color="#dbeafe", fontsize=10)
        axis.set_facecolor("#07101f")
        axis.tick_params(colors="#94a3b8", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#334155")
    figure.tight_layout(pad=2.2)
    canvas = FigureCanvasTkAgg(figure, master=chart_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def refresh_devices() -> None:
        try:
            devices = list_input_devices()
            labels = [f"[{item['index']}] {item['name']}" for item in devices]
            device_indexes.clear()
            device_indexes.update(
                {label: int(item["index"]) for label, item in zip(labels, devices, strict=True)}
            )
            device_box["values"] = labels
            if labels:
                default = next(
                    (index for index, item in enumerate(devices) if item["default"]), 0
                )
                device_box.current(default)
        except Exception as error:
            status_var.set(f"Device error: {error}")

    def draw_report(report: dict[str, object]) -> None:
        paths = (Path(report["noisy"]["path"]), Path(report["enhanced"]["path"]))
        for column, path in enumerate(paths):
            samples, sample_rate = sf.read(path, dtype="float32")
            step = max(1, len(samples) // 8_000)
            times = np.arange(0, len(samples), step) / sample_rate
            waveform = axes[0, column]
            spectrum = axes[1, column]
            waveform.clear()
            spectrum.clear()
            color = "#94a3b8" if column == 0 else "#38bdf8"
            waveform.plot(times, samples[::step], color=color, linewidth=0.7)
            waveform.set_title(
                "Noisy waveform" if column == 0 else "Enhanced waveform",
                color="#dbeafe",
                fontsize=10,
            )
            waveform.set_xlabel("seconds", color="#94a3b8", fontsize=8)
            spectrum.specgram(samples, NFFT=1024, Fs=sample_rate, noverlap=512, cmap="magma")
            spectrum.set_ylim(0, 12_000)
            spectrum.set_title(
                "Noisy spectrum" if column == 0 else "Enhanced spectrum",
                color="#dbeafe",
                fontsize=10,
            )
            spectrum.set_xlabel("seconds", color="#94a3b8", fontsize=8)
            spectrum.set_ylabel("Hz", color="#94a3b8", fontsize=8)
            for axis in (waveform, spectrum):
                axis.set_facecolor("#07101f")
                axis.tick_params(colors="#94a3b8", labelsize=8)
                for spine in axis.spines.values():
                    spine.set_color("#334155")
        figure.tight_layout(pad=2.0)
        canvas.draw_idle()

    def finish(report: dict[str, object], output: Path) -> None:
        nonlocal current_report
        current_report = report
        summary_var.set(_summary_text(report) + f"\n\nSaved to:\n{output}")
        status_var.set("Complete — compare the noisy and enhanced recordings.")
        run_button.configure(state="normal")
        noisy_button.configure(state="normal")
        enhanced_button.configure(state="normal")
        draw_report(report)

    def fail(error: Exception) -> None:
        run_button.configure(state="normal")
        status_var.set(f"Failed: {error}")
        messagebox.showerror("COFFEEBEAN", str(error))

    def start_demo() -> None:
        try:
            duration = float(duration_var.get())
            if not 1 <= duration <= 60:
                raise ValueError("Duration must be between 1 and 60 seconds")
            selected = device_box.get()
            if selected not in device_indexes:
                raise ValueError("Select a microphone")
        except ValueError as error:
            fail(error)
            return
        output = Path.cwd() / "runs" / "demo" / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_button.configure(state="disabled")
        noisy_button.configure(state="disabled")
        enhanced_button.configure(state="disabled")
        status_var.set("Capturing and enhancing now — speak normally…")

        def work() -> None:
            try:
                report = stream_demo(
                    output,
                    duration=duration,
                    device=device_indexes[selected],
                )
                root.after(0, finish, report, output)
            except Exception as error:
                root.after(0, fail, error)

        threading.Thread(target=work, daemon=True).start()

    def play(which: str) -> None:
        nonlocal playback
        if current_report is None:
            return
        audio_path = str(current_report[which]["path"])
        try:
            import sounddevice as sd
            samples, sr = sf.read(audio_path, dtype="float32")
            sd.stop()
            sd.play(samples, sr)
        except Exception:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(audio_path, winsound.SND_ASYNC | winsound.SND_FILENAME)
            elif sys.platform == "darwin":
                if playback is not None and playback.poll() is None:
                    playback.terminate()
                playback = subprocess.Popen(["afplay", audio_path])

    run_button = ttk.Button(controls, text="Start AI stream", command=start_demo)
    run_button.pack(fill="x", pady=(0, 10))
    noisy_button = ttk.Button(
        controls, text="▶ Play noisy", command=lambda: play("noisy"), state="disabled"
    )
    noisy_button.pack(fill="x", pady=4)
    enhanced_button = ttk.Button(
        controls,
        text="▶ Play enhanced",
        command=lambda: play("enhanced"),
        state="disabled",
    )
    enhanced_button.pack(fill="x", pady=4)

    ttk.Separator(controls).pack(fill="x", pady=16)
    ttk.Label(
        controls,
        textvariable=summary_var,
        justify="left",
        wraplength=300,
        font=("Menlo", 10),
    ).pack(anchor="w")
    tk.Label(
        root,
        textvariable=status_var,
        bg="#020617",
        fg="#bae6fd",
        padx=18,
        pady=10,
        anchor="w",
    ).pack(fill="x", side="bottom")

    def close() -> None:
        if playback is not None and playback.poll() is None:
            playback.terminate()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    refresh_devices()
    root.mainloop()


if __name__ == "__main__":
    main()
