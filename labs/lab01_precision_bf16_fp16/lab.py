#!/usr/bin/env python
"""Lab 01: Precision reduction (FP32 -> FP16 / BF16).

Standalone: python labs/lab01_precision_bf16_fp16/lab.py [--quick] [--gpus 0]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench, cli, config, gpu_utils, report, server  # noqa: E402
from labs.lab00_fp32_baseline.lab import RANDOM_INPUT_LEN, RANDOM_OUTPUT_LEN  # noqa: E402

LAB_NAME = "lab01_precision_bf16_fp16"
VARIANTS = [
    ("float16", "float16"),
    ("bfloat16", "bfloat16"),
]


def run(quick: bool = False, gpus: str | None = None) -> list[dict]:
    gpus = gpus or config.DEFAULT_GPU
    n_prompts = config.num_prompts(quick)

    rows: list[dict] = []
    baseline = report.load_baseline("lab00_fp32_baseline", "fp32")
    if baseline:
        rows.append(baseline)

    for label, dtype in VARIANTS:
        gpu_utils.print_snapshot(f"lab01 before {label}")
        extra_args = [
            "--dtype", dtype,
            "--gpu-memory-utilization", str(config.DEFAULT_GPU_MEMORY_UTILIZATION),
            "--max-model-len", str(config.DEFAULT_MAX_MODEL_LEN),
        ]
        with server.VLLMServer(
            config.MODEL_PRIMARY, gpus=gpus, extra_args=extra_args, name=f"lab01_{label}"
        ) as base_url:
            gpu_utils.print_snapshot(f"lab01 after {label} server start")
            result = bench.run_benchmark(
                base_url=base_url,
                model=config.MODEL_PRIMARY,
                label=label,
                num_prompts=n_prompts,
                random_input_len=RANDOM_INPUT_LEN,
                random_output_len=RANDOM_OUTPUT_LEN,
            )
        rows.append(result.summary)

    rows = report.add_speedup(rows, baseline_label="fp32") if baseline else rows
    columns = list(report.DEFAULT_COLUMNS)
    if baseline:
        columns = columns + [("speedup", "Speedup vs fp32", "{:.2f}x")]
    table = report.print_table(rows, columns=columns, title="Lab 01: Precision (fp32 vs fp16 vs bf16)")
    report.save_lab_results(LAB_NAME, rows)
    report.append_summary(
        LAB_NAME,
        "Lab 01: Precision reduction (fp32 -> fp16 / bf16)",
        table,
        notes="bf16 is the practical default on Blackwell: same throughput as fp16, better numerical range.",
    )
    return rows


if __name__ == "__main__":
    args = cli.build_arg_parser(__doc__).parse_args()
    run(quick=args.quick, gpus=args.gpus)
