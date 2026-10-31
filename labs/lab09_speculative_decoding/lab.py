#!/usr/bin/env python
"""Lab 09: Speculative decoding (draft model vs n-gram/prompt-lookup).

Standalone: python labs/lab09_speculative_decoding/lab.py [--quick] [--gpus 0]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench, cli, config, gpu_utils, report, server  # noqa: E402

LAB_NAME = "lab09_speculative_decoding"

LOW_INPUT_LEN = 128
LOW_OUTPUT_LEN = 256
LOW_NUM_PROMPTS_FULL = 20
LOW_NUM_PROMPTS_QUICK = 6
LOW_MAX_CONCURRENCY = 2

HIGH_INPUT_LEN = 128
HIGH_OUTPUT_LEN = 256
HIGH_NUM_PROMPTS_FULL = 100
HIGH_NUM_PROMPTS_QUICK = 20
HIGH_MAX_CONCURRENCY = 64

DRAFT_SPEC_CONFIG = json.dumps(
    {"method": "draft_model", "model": config.MODEL_DRAFT, "num_speculative_tokens": 5}
)
EAGLE3_SPEC_CONFIG = json.dumps(
    {"method": "eagle3", "model": config.MODEL_EAGLE3, "num_speculative_tokens": 3}
)
NGRAM_SPEC_CONFIG = json.dumps(
    {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
)

VARIANTS = [
    ("no_spec", []),
    ("draft_model_spec", ["--speculative-config", DRAFT_SPEC_CONFIG]),
    ("eagle3_spec", ["--speculative-config", EAGLE3_SPEC_CONFIG]),
    ("ngram_spec", ["--speculative-config", NGRAM_SPEC_CONFIG]),
]


def run(quick: bool = False, gpus: str | None = None) -> list[dict]:
    gpus = gpus or config.DEFAULT_GPU
    low_n = LOW_NUM_PROMPTS_QUICK if quick else LOW_NUM_PROMPTS_FULL
    high_n = HIGH_NUM_PROMPTS_QUICK if quick else HIGH_NUM_PROMPTS_FULL

    rows: list[dict] = []
    skip_notes: list[str] = []

    for label, flag in VARIANTS:
        gpu_utils.print_snapshot(f"lab09 before {label}")
        extra_args = [
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(config.DEFAULT_GPU_MEMORY_UTILIZATION),
            "--max-model-len", str(config.DEFAULT_MAX_MODEL_LEN),
            *flag,
        ]
        try:
            with server.VLLMServer(
                config.MODEL_PRIMARY, gpus=gpus, extra_args=extra_args, name=f"lab09_{label}"
            ) as base_url:
                r_low = bench.run_benchmark(
                    base_url=base_url,
                    model=config.MODEL_PRIMARY,
                    label=f"{label}_low",
                    num_prompts=low_n,
                    random_input_len=LOW_INPUT_LEN,
                    random_output_len=LOW_OUTPUT_LEN,
                    max_concurrency=LOW_MAX_CONCURRENCY,
                )
                r_high = bench.run_benchmark(
                    base_url=base_url,
                    model=config.MODEL_PRIMARY,
                    label=f"{label}_high",
                    num_prompts=high_n,
                    random_input_len=HIGH_INPUT_LEN,
                    random_output_len=HIGH_OUTPUT_LEN,
                    max_concurrency=HIGH_MAX_CONCURRENCY,
                )
            rows.append(r_low.summary)
            rows.append(r_high.summary)
        except server.ServerStartError as exc:
            reason = str(exc).splitlines()[0]
            print(f"[lab09] SKIPPING variant={label}: {reason}")
            rows.append({"label": f"{label}_low", "skipped": True})
            rows.append({"label": f"{label}_high", "skipped": True})
            skip_notes.append(f"{label}: {reason}")

    table = report.print_table(rows, title="Lab 09: Speculative decoding")
    report.save_lab_results(LAB_NAME, rows)
    notes = (
        "'_low' = max-concurrency=2 (latency regime, spec decoding should help most). "
        "'_high' = max-concurrency=64 (throughput regime, benefit shrinks/reverses)."
    )
    if skip_notes:
        notes += "\n\nSkipped variants:\n" + "\n".join(f"- {n}" for n in skip_notes)
    report.append_summary(LAB_NAME, "Lab 09: Speculative decoding", table, notes=notes)
    return rows


if __name__ == "__main__":
    args = cli.build_arg_parser(__doc__).parse_args()
    run(quick=args.quick, gpus=args.gpus)
