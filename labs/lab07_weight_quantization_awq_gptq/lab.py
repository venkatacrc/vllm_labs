#!/usr/bin/env python
"""Lab 07: Weight-only quantization (AWQ / GPTQ).

Standalone: python labs/lab07_weight_quantization_awq_gptq/lab.py [--quick] [--gpus 0]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench, cli, config, gpu_utils, report, server  # noqa: E402
from labs.lab00_fp32_baseline.lab import RANDOM_INPUT_LEN, RANDOM_OUTPUT_LEN  # noqa: E402

LAB_NAME = "lab07_weight_quantization_awq_gptq"


def _run_variant(label: str, model: str, gpus: str, n_prompts: int, extra_dtype_args: list[str]) -> dict:
    gpu_utils.print_snapshot(f"lab07 before {label}")
    extra_args = [
        *extra_dtype_args,
        "--gpu-memory-utilization", str(config.DEFAULT_GPU_MEMORY_UTILIZATION),
        "--max-model-len", str(config.DEFAULT_MAX_MODEL_LEN),
    ]
    with server.VLLMServer(model, gpus=gpus, extra_args=extra_args, name=f"lab07_{label}") as base_url:
        gpu_utils.print_snapshot(f"lab07 after {label} server start")
        result = bench.run_benchmark(
            base_url=base_url,
            model=model,
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
        print("[lab07] reusing bf16 baseline from lab01")
        rows.append(bf16_row)
    else:
        rows.append(_run_variant("bfloat16", config.MODEL_PRIMARY, gpus, n_prompts, ["--dtype", "bfloat16"]))

    rows.append(_run_variant("awq_int4", config.MODEL_AWQ, gpus, n_prompts, []))
    rows.append(_run_variant("gptq_int4", config.MODEL_GPTQ, gpus, n_prompts, []))

    baseline_label = rows[0]["label"]
    rows = report.add_speedup(rows, baseline_label=baseline_label)
    columns = list(report.DEFAULT_COLUMNS) + [("speedup", f"Speedup vs {baseline_label}", "{:.2f}x")]
    table = report.print_table(rows, columns=columns, title="Lab 07: Weight-only quantization (AWQ / GPTQ)")
    report.save_lab_results(LAB_NAME, rows)
    report.append_summary(
        LAB_NAME,
        "Lab 07: Weight-only quantization (AWQ / GPTQ)",
        table,
        notes="INT4 checkpoints cut weight memory roughly 3x vs bf16; watch the GPU memory snapshots in the log.",
    )
    return rows


if __name__ == "__main__":
    args = cli.build_arg_parser(__doc__).parse_args()
    run(quick=args.quick, gpus=args.gpus)
