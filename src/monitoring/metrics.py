"""COFFEEBEAN — Edge Metrics Collector (Phase 10)"""

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Rolling metrics collector for edge device performance monitoring.
    Tracks inference latency, real-time factor, and optional SNR estimation.
    """

    def __init__(self, window: int = 100):
        self.window     = window
        self.latencies  = deque(maxlen=window)
        self.snr_inputs = deque(maxlen=window)
        self._total_chunks = 0
        self._errors       = 0
        self._start_time   = time.time()

    def record_inference(self, latency_ms: float, input_snr_db: Optional[float] = None):
        self.latencies.append(latency_ms)
        if input_snr_db is not None:
            self.snr_inputs.append(input_snr_db)
        self._total_chunks += 1

    def record_error(self):
        self._errors += 1

    def summary(self, sample_rate: int = 16000, chunk_seconds: float = 1.0) -> dict:
        """Return current metrics summary."""
        if not self.latencies:
            return {"status": "no_data"}

        arr      = np.array(self.latencies)
        audio_ms = chunk_seconds * 1000.0
        uptime_s = time.time() - self._start_time

        return {
            "uptime_s":           round(uptime_s, 1),
            "total_chunks":       self._total_chunks,
            "errors":             self._errors,
            "error_rate_pct":     round(self._errors / max(self._total_chunks, 1) * 100, 2),
            "latency_mean_ms":    round(float(np.mean(arr)),           2),
            "latency_p95_ms":     round(float(np.percentile(arr, 95)), 2),
            "real_time_factor":   round(float(np.mean(arr)) / audio_ms, 4),
            "real_time_capable":  float(np.mean(arr)) < audio_ms,
            "avg_input_snr_db":   round(float(np.mean(self.snr_inputs)), 2)
                                  if self.snr_inputs else None,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.summary(**kwargs), indent=2)
