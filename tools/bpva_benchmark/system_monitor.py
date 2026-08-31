"""Best-effort GPU and host/cgroup memory monitoring."""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def _read_int(path: str) -> int | None:
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
        return None if text == "max" else int(text)
    except (OSError, ValueError):
        return None


def _read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.replace(":", "").split()
            if len(parts) >= 2:
                multiplier = 1024 if len(parts) > 2 and parts[2] == "kB" else 1
                values[parts[0]] = int(parts[1]) * multiplier
    except (OSError, ValueError):
        pass
    return values


def memory_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"timestamp": time.time(), "pid": os.getpid()}
    try:
        pages = Path("/proc/self/statm").read_text(encoding="utf-8").split()
        result["process_rss_bytes"] = int(pages[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        result["process_rss_bytes"] = None

    smaps = _read_key_values(Path("/proc/self/smaps_rollup"))
    result["process_pss_bytes"] = smaps.get("Pss")
    result["process_private_dirty_bytes"] = smaps.get("Private_Dirty")

    info = _read_key_values(Path("/proc/meminfo"))
    result["host_total_bytes"] = info.get("MemTotal")
    result["host_available_bytes"] = info.get("MemAvailable")

    try:
        shm = shutil.disk_usage("/dev/shm")
        result.update(
            {
                "shm_total_bytes": shm.total,
                "shm_used_bytes": shm.used,
                "shm_free_bytes": shm.free,
            }
        )
    except OSError:
        result.update(
            {"shm_total_bytes": None, "shm_used_bytes": None, "shm_free_bytes": None}
        )

    result["memory_events"] = {}
    for prefix in (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")):
        current = _read_int(str(prefix / "memory.current"))
        if current is None:
            current = _read_int(str(prefix / "memory.usage_in_bytes"))
        limit = _read_int(str(prefix / "memory.max"))
        if limit is None:
            limit = _read_int(str(prefix / "memory.limit_in_bytes"))
        events = _read_key_values(prefix / "memory.events")
        if current is not None or limit is not None or events:
            result["cgroup_current_bytes"] = current
            result["cgroup_limit_bytes"] = limit
            result["memory_events"] = {
                "oom": events.get("oom"),
                "oom_kill": events.get("oom_kill"),
            }
            break
    result.setdefault("cgroup_current_bytes", None)
    result.setdefault("cgroup_limit_bytes", None)
    return result


class SystemMonitor:
    FIELDS = (
        "index",
        "uuid",
        "timestamp",
        "utilization_gpu_pct",
        "memory_used_mib",
        "memory_total_mib",
        "temperature_c",
        "power_w",
    )

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = max(0.1, float(interval_s))
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "SystemMonitor":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="bpva-nvidia-monitor", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))
        return self.samples

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def _run(self):
        query = (
            "index,uuid,timestamp,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw"
        )
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=max(2.0, self.interval_s),
                    check=False,
                )
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.strip() or f"exit {proc.returncode}")
                for row in csv.reader(io.StringIO(proc.stdout)):
                    if len(row) >= len(self.FIELDS):
                        sample = dict(zip(self.FIELDS, (item.strip() for item in row)))
                        sample["sample_time"] = time.time()
                        self.samples.append(sample)
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                if not self.errors:
                    self.errors.append(str(exc))
                if isinstance(exc, OSError):
                    return
            self._stop.wait(self.interval_s)
