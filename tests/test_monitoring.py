"""Tests for Phase 10 — Monitoring."""
import pytest
import time
from pathlib import Path


def test_edge_logger_creates_file(tmp_path):
    from src.monitoring.logger import EdgeLogger
    logger = EdgeLogger("test-device-01", "v1.0", log_dir=tmp_path)
    logger.log_inference(latency_ms=12.5, chunk_id=0)
    logger.flush()
    log_files = list(tmp_path.glob("*.jsonl"))
    assert len(log_files) == 1


def test_edge_logger_entry_structure(tmp_path):
    import json
    from src.monitoring.logger import EdgeLogger
    logger = EdgeLogger("test-device-01", "v1.0", log_dir=tmp_path)
    logger.log_inference(latency_ms=12.5, input_snr_db=-3.0, chunk_id=5)
    logger.flush()
    lines  = list(tmp_path.glob("*.jsonl"))[0].read_text().strip().split("\n")
    entry  = json.loads(lines[0])
    assert entry["event"]         == "inference"
    assert entry["device_id"]     == "test-device-01"
    assert entry["model_version"] == "v1.0"
    assert entry["latency_ms"]    == 12.5


def test_metrics_collector_summary():
    from src.monitoring.metrics import MetricsCollector
    mc = MetricsCollector(window=10)
    for i in range(5):
        mc.record_inference(latency_ms=10.0 + i)
    summary = mc.summary(sample_rate=16000, chunk_seconds=1.0)
    assert "latency_mean_ms"   in summary
    assert "real_time_factor"  in summary
    assert "real_time_capable" in summary
    assert summary["total_chunks"] == 5


def test_metrics_collector_real_time(tmp_path):
    from src.monitoring.metrics import MetricsCollector
    mc = MetricsCollector(window=10)
    for _ in range(10):
        mc.record_inference(latency_ms=50.0)   # 50ms << 1000ms chunk → real-time
    s = mc.summary(chunk_seconds=1.0)
    assert s["real_time_capable"] is True


def test_metrics_error_rate():
    from src.monitoring.metrics import MetricsCollector
    mc = MetricsCollector()
    for _ in range(9):
        mc.record_inference(latency_ms=10.0)
    mc.record_error()
    s = mc.summary()
    assert s["error_rate_pct"] > 0
