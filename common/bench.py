"""Wraps the `vllm bench serve` CLI and parses its JSON result file.

Every lab drives its running server with `run_benchmark()` (one workload) or
`run_concurrent()` (several workloads fired at the same server at once, used
by the chunked-prefill lab to simulate mixed long-prompt/short-prompt
traffic).
"""
from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import config

SUMMARY_KEYS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "duration",
    "completed",
)


@dataclass
class BenchResult:
    label: str
    raw: dict
    result_file: Path

    def get(self, key, default=None):
        return self.raw.get(key, default)

    @property
    def summary(self) -> dict:
        row = {"label": self.label}
        row.update({k: self.raw.get(k) for k in SUMMARY_KEYS})
        return row


def run_benchmark(
    *,
    base_url: str,
    model: str,
    label: str,
    dataset_name: str = "random",
    num_prompts: int = 100,
    request_rate: str | float = "inf",
    max_concurrency: int | None = None,
    random_input_len: int = 512,
    random_output_len: int = 128,
    random_prefix_len: int = 0,
    seed: int = config.DEFAULT_SEED,
    extra_args: list[str] | None = None,
    result_dir: Path | None = None,
) -> BenchResult:
    """Run `vllm bench serve` once against a running server and return parsed results."""
    result_dir = result_dir or config.RESULTS_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
    result_filename = f"{safe_label}_{int(time.time() * 1000)}.json"
    result_file = result_dir / result_filename

    cmd = [
        "vllm", "bench", "serve",
        "--base-url", base_url,
        "--model", model,
        "--dataset-name", dataset_name,
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
        "--seed", str(seed),
        "--save-result",
        "--result-dir", str(result_dir),
        "--result-filename", result_filename,
    ]
    if dataset_name == "random":
        cmd += [
            "--random-input-len", str(random_input_len),
            "--random-output-len", str(random_output_len),
        ]
        if random_prefix_len:
            cmd += ["--random-prefix-len", str(random_prefix_len)]
    if max_concurrency:
        cmd += ["--max-concurrency", str(max_concurrency)]
    cmd += extra_args or []

    log_path = config.LOGS_DIR / f"bench_{safe_label}.log"
    print(f"[bench] {label}: {' '.join(cmd)}")
    with open(log_path, "w") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    if proc.returncode != 0 or not result_file.exists():
        tail = _tail(log_path)
        raise RuntimeError(
            f"Benchmark '{label}' failed (exit={proc.returncode}). "
            f"Log tail:\n{tail}\nFull log: {log_path}"
        )

    raw = json.loads(result_file.read_text())
    print(
        f"[bench] {label}: "
        f"{raw.get('output_throughput', 0):.1f} out-tok/s, "
        f"mean TTFT {raw.get('mean_ttft_ms', 0):.1f}ms, "
        f"mean TPOT {raw.get('mean_tpot_ms', 0):.2f}ms"
    )
    return BenchResult(label=label, raw=raw, result_file=result_file)


def run_concurrent(jobs: list[dict]) -> list[BenchResult]:
    """Run several run_benchmark() calls concurrently against the same server.

    Each dict in `jobs` is the kwargs for one run_benchmark() call. Useful for
    simulating mixed traffic (e.g. a trickle of long prompts alongside a
    stream of short latency-sensitive ones) to see how scheduling choices
    like chunked prefill affect one workload while the other is in flight.
    """
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(run_benchmark, **job) for job in jobs]
        return [f.result() for f in futures]


def _tail(path: Path, n_lines: int = 30) -> str:
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except OSError:
        return "(log unavailable)"
