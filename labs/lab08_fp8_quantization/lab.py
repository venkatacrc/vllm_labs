#!/usr/bin/env python
"""Lab 08: FP8 quantization (Blackwell-native), weights and/or KV cache.

Standalone: python labs/lab08_fp8_quantization/lab.py [--quick] [--gpus 0]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench, cli, config, gpu_utils, report, server  # noqa: E402
from labs.lab00_fp32_baseline.lab import RANDOM_INPUT_LEN, RANDOM_OUTPUT_LEN  # noqa: E402

LAB_NAME = "lab08_fp8_quantization"

VARIANTS = [
    ("fp8_weights", ["--quantization", "fp8"]),
    ("fp8_weights_and_kv", ["--quantization", "fp8", "--kv-cache-dtype", "fp8_e4m3"]),
]


def _run_variant(label: str, extra_args: list[str], gpus: str, n_prompts: int) -> dict:
    gpu_utils.print_snapshot(f"lab08 before {label}")
    full_extra_args = [
        *extra_args,
        "--gpu-memory-utilization", str(config.DEFAULT_GPU_MEMORY_UTILIZATION),
        "--max-model-len", str(config.DEFAULT_MAX_MODEL_LEN),
    ]
    with server.VLLMServer(
        config.MODEL_PRIMARY, gpus=gpus, extra_args=full_extra_args, name=f"lab08_{label}"
    ) as base_url:
        gpu_utils.print_snapshot(f"lab08 after {label} server start")
        result = bench.run_benchmark(
            base_url=base_url,
            model=config.MODEL_PRIMARY,
            label=label,
            num_prompts=n_prompts,
            random_input_len=RANDOM_INPUT_LEN,
            random_output_len=RANDOM_OUTPUT_LEN,
        )
    return result.summary


def run(quick: bool = False, gpus: str | None = None) -> list[dict]:
    gpus = gpus or config.DEFAULT_GPU
    n_prompts = config.num_prompts(quick)

    rows: list[dict] = []
    bf16_row = report.load_baseline("lab01_precision_bf16_fp16", "bfloat16")
    if bf16_row:
        print("[lab08] reusing bf16 baseline from lab01")
        rows.append(bf16_row)
    else:
        rows.append(_run_variant("bfloat16", ["--dtype", "bfloat16"], gpus, n_prompts))

    for label, extra_args in VARIANTS:
        rows.append(_run_variant(label, extra_args, gpus, n_prompts))

    baseline_label = rows[0]["label"]
    rows = report.add_speedup(rows, baseline_label=baseline_label)
    columns = list(report.DEFAULT_COLUMNS) + [("speedup", f"Speedup vs {baseline_label}", "{:.2f}x")]
    table = report.print_table(rows, columns=columns, title="Lab 08: FP8 quantization (Blackwell-native)")
    report.save_lab_results(LAB_NAME, rows)
    report.append_summary(
        LAB_NAME,
        "Lab 08: FP8 quantization (weights and KV cache)",
        table,
        notes=(
            "FP8 weights are quantized dynamically at load time (no calibration step needed). "
            "fp8_weights_and_kv additionally halves KV-cache bytes/token."
        ),
    )
    return rows


if __name__ == "__main__":
    args = cli.build_arg_parser(__doc__).parse_args()
    run(quick=args.quick, gpus=args.gpus)
