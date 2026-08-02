#!/usr/bin/env python
"""Lab 05: Chunked prefill.

Fires a trickle of long-prompt requests concurrently with a stream of
short latency-sensitive requests, with chunked prefill on vs off.

Standalone: python labs/lab05_chunked_prefill/lab.py [--quick] [--gpus 0]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench, cli, config, gpu_utils, report, server  # noqa: E402

LAB_NAME = "lab05_chunked_prefill"

LONG_INPUT_LEN = 6000
LONG_OUTPUT_LEN = 50
LONG_NUM_PROMPTS_FULL = 6
LONG_NUM_PROMPTS_QUICK = 2
LONG_REQUEST_RATE = 0.3

SHORT_INPUT_LEN = 100
SHORT_OUTPUT_LEN = 50
SHORT_NUM_PROMPTS_FULL = 60
SHORT_NUM_PROMPTS_QUICK = 12
SHORT_REQUEST_RATE = 5.0

MAX_MODEL_LEN = 8192

VARIANTS = [
    ("chunked_prefill_off", ["--no-enable-chunked-prefill"]),
    ("chunked_prefill_on", ["--enable-chunked-prefill"]),
]


def run(quick: bool = False, gpus: str | None = None) -> list[dict]:
    gpus = gpus or config.DEFAULT_GPU
    long_n = LONG_NUM_PROMPTS_QUICK if quick else LONG_NUM_PROMPTS_FULL
    short_n = SHORT_NUM_PROMPTS_QUICK if quick else SHORT_NUM_PROMPTS_FULL

    rows: list[dict] = []
    for label, flag in VARIANTS:
        gpu_utils.print_snapshot(f"lab05 before {label}")
        extra_args = [
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(config.DEFAULT_GPU_MEMORY_UTILIZATION),
            "--max-model-len", str(MAX_MODEL_LEN),
            *flag,
        ]
        with server.VLLMServer(
            config.MODEL_PRIMARY, gpus=gpus, extra_args=extra_args, name=f"lab05_{label}"
        ) as base_url:
            long_job = dict(
                base_url=base_url,
                model=config.MODEL_PRIMARY,
                label=f"{label}_long",
                num_prompts=long_n,
                random_input_len=LONG_INPUT_LEN,
                random_output_len=LONG_OUTPUT_LEN,
                request_rate=LONG_REQUEST_RATE,
            )
            short_job = dict(
                base_url=base_url,
                model=config.MODEL_PRIMARY,
                label=f"{label}_short",
                num_prompts=short_n,
                random_input_len=SHORT_INPUT_LEN,
                random_output_len=SHORT_OUTPUT_LEN,
                request_rate=SHORT_REQUEST_RATE,
            )
            results = bench.run_concurrent([long_job, short_job])
        rows.extend(r.summary for r in results)

    table = report.print_table(rows, title="Lab 05: Chunked prefill (mixed long/short traffic)")
    report.save_lab_results(LAB_NAME, rows)
    report.append_summary(
        LAB_NAME,
        "Lab 05: Chunked prefill",
        table,
        notes=(
            "'_long' = trickle of ~6000-token prompts; '_short' = stream of ~100-token "
            "latency-sensitive requests, run concurrently against the same server. "
            "Compare '_short' p99_ttft_ms between chunked_prefill_off and chunked_prefill_on."
        ),
    )
    return rows


if __name__ == "__main__":
    args = cli.build_arg_parser(__doc__).parse_args()
    run(quick=args.quick, gpus=args.gpus)
