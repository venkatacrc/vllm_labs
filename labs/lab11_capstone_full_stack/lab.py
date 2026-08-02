#!/usr/bin/env python
"""Lab 11: Capstone - the full fp32 -> fully-optimized journey.

Reuses saved results from earlier labs where available; reruns anything
missing so this lab also works standalone.

Standalone: python labs/lab11_capstone_full_stack/lab.py [--quick] [--gpus 0]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench, cli, config, gpu_utils, report, server  # noqa: E402
from labs.lab00_fp32_baseline.lab import RANDOM_INPUT_LEN, RANDOM_OUTPUT_LEN  # noqa: E402
from labs.lab10_tensor_parallelism.lab import MAX_CONCURRENCY as TP_MAX_CONCURRENCY  # noqa: E402
from labs.lab10_tensor_parallelism.lab import INPUT_LEN as TP_INPUT_LEN  # noqa: E402
from labs.lab10_tensor_parallelism.lab import OUTPUT_LEN as TP_OUTPUT_LEN  # noqa: E402

LAB_NAME = "lab11_capstone_full_stack"

TP4_POOL = "0,1,2,3"


def _run_8b_variant(label: str, extra_args: list[str], gpus: str, n_prompts: int) -> dict:
    gpu_utils.print_snapshot(f"lab11 before {label}")
    full_extra_args = [
        *extra_args,
        "--gpu-memory-utilization", str(config.DEFAULT_GPU_MEMORY_UTILIZATION),
        "--max-model-len", str(config.DEFAULT_MAX_MODEL_LEN),
    ]
    with server.VLLMServer(
        config.MODEL_PRIMARY, gpus=gpus, extra_args=full_extra_args, name=f"lab11_{label}"
    ) as base_url:
        result = bench.run_benchmark(
            base_url=base_url,
            model=config.MODEL_PRIMARY,
            label=label,
            num_prompts=n_prompts,
            random_input_len=RANDOM_INPUT_LEN,
            random_output_len=RANDOM_OUTPUT_LEN,
        )
    return result.summary


def _run_32b_tp4_variant(label: str, extra_args: list[str], n_prompts: int) -> dict:
    gpu_utils.print_snapshot(f"lab11 before {label}")
    full_extra_args = [
        "--tensor-parallel-size", "4",
        *extra_args,
        "--gpu-memory-utilization", str(config.DEFAULT_GPU_MEMORY_UTILIZATION),
        "--max-model-len", str(config.DEFAULT_MAX_MODEL_LEN),
    ]
    with server.VLLMServer(
        config.MODEL_LARGE, gpus=TP4_POOL, extra_args=full_extra_args, name=f"lab11_{label}"
    ) as base_url:
        result = bench.run_benchmark(
            base_url=base_url,
            model=config.MODEL_LARGE,
            label=label,
            num_prompts=n_prompts,
            random_input_len=TP_INPUT_LEN,
            random_output_len=TP_OUTPUT_LEN,
            max_concurrency=TP_MAX_CONCURRENCY,
        )
    return result.summary


def run(quick: bool = False, gpus: str | None = None) -> list[dict]:
    gpus = gpus or config.DEFAULT_GPU
    n_prompts = config.num_prompts(quick)

    # ---- 8B journey: fp32 -> bf16 -> fp8 (weights + kv) ----
    fp32_row = report.load_baseline("lab00_fp32_baseline", "fp32")
    if not fp32_row:
        fp32_row = _run_8b_variant("fp32", ["--dtype", "float32"], gpus, n_prompts)

    bf16_row = report.load_baseline("lab01_precision_bf16_fp16", "bfloat16")
    if not bf16_row:
        bf16_row = _run_8b_variant("bfloat16", ["--dtype", "bfloat16"], gpus, n_prompts)

    lab08_rows = report.load_lab_results("lab08_fp8_quantization") or []
    fp8_row = next((r for r in lab08_rows if r.get("label") == "fp8_weights_and_kv"), None)
    if not fp8_row:
        fp8_row = _run_8b_variant(
            "fp8_weights_and_kv", ["--quantization", "fp8", "--kv-cache-dtype", "fp8_e4m3"], gpus, n_prompts
        )

    journey_8b = report.add_speedup([fp32_row, bf16_row, fp8_row], baseline_label=fp32_row["label"])

    # ---- 32B journey: TP=4 bf16 -> TP=4 + fp8 ----
    lab10_rows = report.load_lab_results("lab10_tensor_parallelism") or []
    tp4_bf16_row = next((r for r in lab10_rows if r.get("label") == "tp4"), None)
    if not tp4_bf16_row:
        tp4_bf16_row = _run_32b_tp4_variant("tp4_bf16", ["--dtype", "bfloat16"], n_prompts)

    tp4_fp8_row = _run_32b_tp4_variant("tp4_fp8", ["--dtype", "bfloat16", "--quantization", "fp8"], n_prompts)

    journey_32b = report.add_speedup([tp4_bf16_row, tp4_fp8_row], baseline_label=tp4_bf16_row["label"])

    all_rows = journey_8b + journey_32b
    columns = list(report.DEFAULT_COLUMNS) + [("speedup", "Speedup vs journey start", "{:.2f}x")]
    table = report.print_table(all_rows, columns=columns, title="Lab 11: Capstone - full fp32 to optimized journey")
    report.save_lab_results(LAB_NAME, all_rows)
    report.append_summary(
        LAB_NAME,
        "Lab 11: Capstone - the full fp32 -> fully-optimized journey",
        table,
        notes=(
            "8B journey: fp32 (Lab 00) -> bf16 (Lab 01) -> fp8 weights+KV (Lab 08). "
            "32B journey: TP=4 bf16 (Lab 10) -> TP=4 + fp8. "
            "Most other techniques in this suite (continuous batching, paged KV cache, "
            "prefix caching, chunked prefill, CUDA graphs) are vLLM defaults today - "
            "precision/quantization and parallelism are the levers you actually turn."
        ),
    )
    return all_rows


if __name__ == "__main__":
    args = cli.build_arg_parser(__doc__).parse_args()
    run(quick=args.quick, gpus=args.gpus)
