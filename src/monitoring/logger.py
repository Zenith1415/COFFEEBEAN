"""COFFEEBEAN — Edge Monitoring Logger (Phase 10)"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_DIR  = Path(os.getenv("COFFEEBEAN_LOG_DIR", "logs/edge"))


class EdgeLogger:
    """
    Structured JSON logger for edge device monitoring.
    Logs: inference latency, model version, CPU/RAM, error events.
    Sends batched reports to cloud (when connectivity is available).
    """

    def __init__(self, device_id: str, model_version: str, log_dir: Path = LOG_DIR):
        self.device_id     = device_id
        self.model_version = model_version
        self.log_dir       = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._log_file     = self.log_dir / f"{self.session_id}_{device_id}.jsonl"
        self._buffer: list = []

    def _entry(self, event_type: str, data: dict) -> dict:
        return {
            "ts":            datetime.now(timezone.utc).isoformat(),
            "device_id":     self.device_id,
            "model_version": self.model_version,
            "session_id":    self.session_id,
            "event":         event_type,
            **data,
        }

    def log_inference(self, latency_ms: float, input_snr_db: float = None,
                      chunk_id: int = None):
        entry = self._entry("inference", {
            "latency_ms":   round(latency_ms, 3),
            "input_snr_db": input_snr_db,
            "chunk_id":     chunk_id,
        })
        self._buffer.append(entry)
        if len(self._buffer) >= 50:
            self.flush()

    def log_system(self, cpu_pct: float = None, ram_mb: float = None,
                   temp_c: float = None):
        try:
            import psutil
            cpu_pct = cpu_pct or psutil.cpu_percent()
            ram_mb  = ram_mb  or psutil.virtual_memory().used / (1024 ** 2)
        except ImportError:
            pass

        entry = self._entry("system", {
            "cpu_pct": cpu_pct,
            "ram_mb":  round(ram_mb, 1) if ram_mb else None,
            "temp_c":  temp_c,
        })
        self._buffer.append(entry)

    def log_error(self, error: str, context: dict = None):
        entry = self._entry("error", {"error": error, "context": context or {}})
        self._buffer.append(entry)
        self.flush()   # Flush immediately on errors

    def flush(self):
        if not self._buffer:
            return
        with self._log_file.open("a", encoding="utf-8") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry) + "\n")
        self._buffer.clear()

    def __del__(self):
        self.flush()
