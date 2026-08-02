#!/usr/bin/env python
"""Lab 03: High-efficiency attention backends.

FlashInfer vs FlashAttention vs Triton vs FlexAttention, on separate
prefill-heavy and decode-heavy workloads. Any backend unavailable in this
environment (e.g. flashinfer not installed) is skipped, not fatal.

Standalone: python labs/lab03_attention_backends/lab.py [--quick] [--gpus 0]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench, cli, config, gpu_utils, report, server  # noqa: E402

LAB_NAME = "lab03_attention_backends"

BACKENDS = ["FLASHINFER", "FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION"]

PREFILL_INPUT_LEN = 4000
PREFILL_OUTPUT_LEN = 32
PREFILL_NUM_PROMPTS_FULL = 40
PREFILL_NUM_PROMPTS_QUICK = 8

DECODE_INPUT_LEN = 128
DECODE_OUTPUT_LEN = 512
DECODE_NUM_PROMPTS_FULL = 16
DECODE_NUM_PROMPTS_QUICK = 4
DECODE_MAX_CONCURRENCY = 4

MAX_MODEL_LEN = 8192


def run(quick: bool = False, gpus: str | None = None) -> list[dict]:
    gpus = gpus or config.DEFAULT_GPU
    prefill_n = PREFILL_NUM_PROMPTS_QUICK if quick else PREFILL_NUM_PROMPTS_FULL
    decode_n = DECODE_NUM_PROMPTS_QUICK if quick else DECODE_NUM_PROMPTS_FULL

    rows: list[dict] = []
    skip_notes: list[str] = []

    for backend in BACKENDS:
        gpu_utils.print_snapshot(f"lab03 before {backend}")
        extra_args = [
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(config.DEFAULT_GPU_MEMORY_UTILIZATION),
            "--max-model-len", str(MAX_MODEL_LEN),
        ]
        try:
            with server.VLLMServer(
                config.MODEL_PRIMARY,
                gpus=gpus,
                extra_args=extra_args,
                env={"VLLM_ATTENTION_BACKEND": backend},
                name=f"lab03_{backend}",
            ) as base_url:
                r_prefill = bench.run_benchmark(
                    base_url=base_url,
                    model=config.MODEL_PRIMARY,
                    label=f"{backend.lower()}_prefill",
                    num_prompts=prefill_n,
                    random_input_len=PREFILL_INPUT_LEN,
                    random_output_len=PREFILL_OUTPUT_LEN,
                )
                r_decode = bench.run_benchmark(
                    base_url=base_url,
                    model=config.MODEL_PRIMARY,
                    label=f"{backend.lower()}_decode",
                    num_prompts=decode_n,
                    random_input_len=DECODE_INPUT_LEN,
                    random_output_len=DECODE_OUTPUT_LEN,
                    max_concurrency=DECODE_MAX_CONCURRENCY,
                )
            rows.append(r_prefill.summary)
            rows.append(r_decode.summary)
        except server.ServerStartError as exc:
            reason = str(exc).splitlines()[0]
            print(f"[lab03] SKIPPING backend={backend}: {reason}")
            rows.append({"label": f"{backend.lower()}_prefill", "skipped": True})
            rows.append({"label": f"{backend.lower()}_decode", "skipped": True})
            skip_notes.append(f"{backend}: {reason}")

    table = report.print_table(rows, title="Lab 03: Attention backend shootout")
    report.save_lab_results(LAB_NAME, rows)
    notes = "Prefill workload: ~4000 in / 32 out tokens. Decode workload: ~128 in / 512 out tokens, concurrency=4."
    if skip_notes:
        notes += "\n\nSkipped backends:\n" + "\n".join(f"- {n}" for n in skip_notes)
    report.append_summary(LAB_NAME, "Lab 03: High-efficiency attention backends", table, notes=notes)
    return rows


if __name__ == "__main__":
    args = cli.build_arg_parser(__doc__).parse_args()
    run(quick=args.quick, gpus=args.gpus)
