#!/usr/bin/env python
"""Lab 00: FP32 baseline.

Serves the primary model at full FP32 precision and records throughput /
latency / memory. Every later lab compares itself against these numbers.

Standalone: python labs/lab00_fp32_baseline/lab.py [--quick] [--gpus 0]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench, cli, config, gpu_utils, report, server  # noqa: E402

LAB_NAME = "lab00_fp32_baseline"

# Shared workload shape reused (as closely as possible) by every later lab
# so cross-lab comparisons are meaningful.
RANDOM_INPUT_LEN = 512
RANDOM_OUTPUT_LEN = 128


def run(quick: bool = False, gpus: str | None = None) -> list[dict]:
    gpus = gpus or config.DEFAULT_GPU
    n_prompts = config.num_prompts(quick)

    gpu_utils.print_snapshot("lab00 before")

    extra_args = [
        "--dtype", "float32",
        "--gpu-memory-utilization", str(config.DEFAULT_GPU_MEMORY_UTILIZATION),
        "--max-model-len", str(config.DEFAULT_MAX_MODEL_LEN),
    ]

    with server.VLLMServer(
        config.MODEL_PRIMARY, gpus=gpus, extra_args=extra_args, name="lab00_fp32"
    ) as base_url:
        gpu_utils.print_snapshot("lab00 after server start (fp32 weights loaded)")
        result = bench.run_benchmark(
            base_url=base_url,
            model=config.MODEL_PRIMARY,
            label="fp32",
            num_prompts=n_prompts,
            random_input_len=RANDOM_INPUT_LEN,
            random_output_len=RANDOM_OUTPUT_LEN,
        )

    rows = [result.summary]
    table = report.print_table(rows, title="Lab 00: FP32 baseline")
    report.save_lab_results(LAB_NAME, rows)
    report.append_summary(
        LAB_NAME,
        "Lab 00: FP32 baseline",
        table,
        notes=(
            "Reference point for every other lab. Model weights alone: "
            "~32GB (8B params x 4 bytes)."
        ),
    )
    return rows


if __name__ == "__main__":
    args = cli.build_arg_parser(__doc__).parse_args()
    run(quick=args.quick, gpus=args.gpus)
