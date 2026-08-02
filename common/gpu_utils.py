"""GPU inventory / memory helpers built on nvidia-smi (no pynvml dependency)."""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass
class GpuMem:
    index: int
    used_mb: int
    total_mb: int

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb


def snapshot() -> list[GpuMem]:
    """Return current memory usage for every visible GPU via nvidia-smi."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    gpus = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx, used, total = (p.strip() for p in line.split(","))
        gpus.append(GpuMem(int(idx), int(used), int(total)))
    return gpus


def parse_gpu_ids(gpus: str) -> list[int]:
    """Parse '0', '0,1,2', or '0-3' (or a mix, e.g. '0,2-4') into a list of ints."""
    ids: list[int] = []
    for part in str(gpus).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def wait_for_free(gpus: str, threshold_mb: int = 3000, timeout_s: float = 180.0, poll_s: float = 2.0) -> None:
    """Block until the given GPUs report < threshold_mb used (post server-teardown).

    This is best-effort: it never raises, it just makes sure one lab's GPU
    memory has actually been released before the next lab tries to use it.
    """
    ids = set(parse_gpu_ids(gpus))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            snap = {g.index: g.used_mb for g in snapshot() if g.index in ids}
        except Exception:  # noqa: BLE001 - nvidia-smi hiccup shouldn't kill the suite
            return
        if not snap or all(used < threshold_mb for used in snap.values()):
            return
        time.sleep(poll_s)
    print(f"[gpu_utils] WARNING: GPU(s) {sorted(ids)} still >{threshold_mb}MiB used after {timeout_s}s")


def print_snapshot(label: str = "") -> None:
    try:
        snap = snapshot()
    except Exception as exc:  # noqa: BLE001
        print(f"[gpu_utils] could not read nvidia-smi: {exc}")
        return
    prefix = f"[{label}] " if label else ""
    print(prefix + " | ".join(f"gpu{g.index}={g.used_mb}/{g.total_mb}MiB" for g in snap))
