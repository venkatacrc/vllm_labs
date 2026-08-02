#!/usr/bin/env python
"""Lab 04: Automatic prefix caching (APC).

Standalone: python labs/lab04_prefix_caching/lab.py [--quick] [--gpus 0]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench, cli, config, gpu_utils, report, server  # noqa: E402

LAB_NAME = "lab04_prefix_caching"

RANDOM_PREFIX_LEN = 1500
RANDOM_INPUT_LEN = 200
RANDOM_OUTPUT_LEN = 100
NUM_PROMPTS_FULL = 80
NUM_PROMPTS_QUICK = 16

VARIANTS = [
    ("prefix_caching_off", ["--no-enable-prefix-caching"]),
    ("prefix_caching_on", ["--enable-prefix-caching"]),
]


def run(quick: bool = False, gpus: str | None = None) -> list[dict]:
    gpus = gpus or config.DEFAULT_GPU
    n_prompts = NUM_PROMPTS_QUICK if quick else NUM_PROMPTS_FULL

    rows: list[dict] = []
    for label, flag in VARIANTS:
        gpu_utils.print_snapshot(f"lab04 before {label}")
        extra_args = [
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(config.DEFAULT_GPU_MEMORY_UTILIZATION),
            "--max-model-len", str(config.DEFAULT_MAX_MODEL_LEN),
            *flag,
        ]
        with server.VLLMServer(
            config.MODEL_PRIMARY, gpus=gpus, extra_args=extra_args, name=f"lab04_{label}"
        ) as base_url:
            result = bench.run_benchmark(
                base_url=base_url,
                model=config.MODEL_PRIMARY,
                label=label,
                num_prompts=n_prompts,
                random_input_len=RANDOM_INPUT_LEN,
                random_output_len=RANDOM_OUTPUT_LEN,
                random_prefix_len=RANDOM_PREFIX_LEN,
            )
        rows.append(result.summary)

    rows = report.add_speedup(rows, baseline_label="prefix_caching_off")
    columns = list(report.DEFAULT_COLUMNS) + [("speedup", "Throughput vs caching-off", "{:.2f}x")]
    table = report.print_table(rows, columns=columns, title="Lab 04: Automatic prefix caching")
    report.save_lab_results(LAB_NAME, rows)
    report.append_summary(
        LAB_NAME,
        "Lab 04: Automatic prefix caching",
        table,
        notes=(
            f"{n_prompts} requests share a {RANDOM_PREFIX_LEN}-token random prefix. "
            "Watch mean/P99 TTFT: with caching on, only the first request pays full prefill cost."
        ),
    )
    return rows


if __name__ == "__main__":
    args = cli.build_arg_parser(__doc__).parse_args()
    run(quick=args.quick, gpus=args.gpus)
