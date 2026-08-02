#!/usr/bin/env python
"""Lab 10: Tensor parallelism scaling on Qwen2.5-32B-Instruct.

Standalone: python labs/lab10_tensor_parallelism/lab.py [--quick] [--gpus 0,1,2,3,4,5,6,7]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench, cli, config, gpu_utils, report, server  # noqa: E402

LAB_NAME = "lab10_tensor_parallelism"

DEFAULT_POOL = "0,1,2,3,4,5,6,7"
TP_SIZES_FULL = [1, 2, 4, 8]
TP_SIZES_QUICK = [1, 2]

INPUT_LEN = 512
OUTPUT_LEN = 128
NUM_PROMPTS_FULL = 128
NUM_PROMPTS_QUICK = 24
MAX_CONCURRENCY = 64


def _gpu_str(ids: list[int]) -> str:
    return ",".join(str(i) for i in ids)


def run(quick: bool = False, gpus: str | None = None) -> list[dict]:
    pool = gpu_utils.parse_gpu_ids(gpus or DEFAULT_POOL)
    tp_sizes = TP_SIZES_QUICK if quick else TP_SIZES_FULL
    tp_sizes = [tp for tp in tp_sizes if tp <= len(pool)]
    n_prompts = NUM_PROMPTS_QUICK if quick else NUM_PROMPTS_FULL

    rows: list[dict] = []
    for tp in tp_sizes:
        gpu_str = _gpu_str(pool[:tp])
        label = f"tp{tp}"
        gpu_utils.print_snapshot(f"lab10 before {label}")
        extra_args = [
            "--dtype", "bfloat16",
            "--tensor-parallel-size", str(tp),
            "--gpu-memory-utilization", str(config.DEFAULT_GPU_MEMORY_UTILIZATION),
            "--max-model-len", str(config.DEFAULT_MAX_MODEL_LEN),
        ]
        with server.VLLMServer(
            config.MODEL_LARGE, gpus=gpu_str, extra_args=extra_args, name=f"lab10_{label}"
        ) as base_url:
            gpu_utils.print_snapshot(f"lab10 after {label} server start")
            result = bench.run_benchmark(
                base_url=base_url,
                model=config.MODEL_LARGE,
                label=label,
                num_prompts=n_prompts,
                random_input_len=INPUT_LEN,
                random_output_len=OUTPUT_LEN,
                max_concurrency=MAX_CONCURRENCY,
            )
        rows.append(result.summary)

    if not quick:
        rows.append(_run_fp32_memory_pressure_bonus(pool))

    baseline_label = rows[0]["label"] if rows and not rows[0].get("skipped") else None
    if baseline_label:
        rows = report.add_speedup(rows, baseline_label=baseline_label)
        columns = list(report.DEFAULT_COLUMNS) + [("speedup", f"Speedup vs {baseline_label}", "{:.2f}x")]
    else:
        columns = report.DEFAULT_COLUMNS
    table = report.print_table(rows, columns=columns, title="Lab 10: Tensor parallelism (Qwen2.5-32B-Instruct)")
    report.save_lab_results(LAB_NAME, rows)
    report.append_summary(
        LAB_NAME,
        "Lab 10: Tensor parallelism",
        table,
        notes="Model: Qwen2.5-32B-Instruct. Watch for diminishing returns at TP=8 from all-reduce overhead.",
    )
    return rows


def _run_fp32_memory_pressure_bonus(pool: list[int]) -> dict:
    """TP=1, fp32, on the 32B model - either a tiny usable KV cache or an OOM. Both are the lesson."""
    label = "fp32_tp1_memory_pressure"
    gpu_str = _gpu_str(pool[:1])
    extra_args = [
        "--dtype", "float32",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.95",
        "--max-model-len", "1024",
    ]
    try:
        with server.VLLMServer(
            config.MODEL_LARGE, gpus=gpu_str, extra_args=extra_args, name=f"lab10_{label}"
        ) as base_url:
            gpu_utils.print_snapshot(f"lab10 after {label} server start")
            result = bench.run_benchmark(
                base_url=base_url,
                model=config.MODEL_LARGE,
                label=label,
                num_prompts=8,
                random_input_len=256,
                random_output_len=64,
                max_concurrency=2,
            )
        row = result.summary
        row["notes"] = "succeeded despite fp32 memory pressure"
        return row
    except (server.ServerStartError, RuntimeError) as exc:
        reason = str(exc).splitlines()[0]
        print(f"[lab10] fp32/TP=1 bonus variant did not complete (expected/OK): {reason}")
        return {"label": label, "skipped": True, "reason": reason}


if __name__ == "__main__":
    args = cli.build_arg_parser(__doc__).parse_args()
    run(quick=args.quick, gpus=args.gpus)
