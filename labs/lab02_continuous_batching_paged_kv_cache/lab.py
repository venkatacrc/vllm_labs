#!/usr/bin/env python
"""Lab 02: Continuous batching + paged KV cache.

Naive HF static-batch generation vs vLLM continuous batching, at matched
and then unconstrained concurrency.

Standalone: python labs/lab02_continuous_batching_paged_kv_cache/lab.py [--quick] [--gpus 0]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench, cli, config, gpu_utils, report, server  # noqa: E402

LAB_NAME = "lab02_continuous_batching_paged_kv_cache"

NUM_PROMPTS_FULL = 48
NUM_PROMPTS_QUICK = 12
HF_BATCH_SIZE = 8
MAX_NEW_TOKENS = 150
RANDOM_INPUT_LEN = 200
RANDOM_OUTPUT_LEN = 150


def _run_hf_baseline(gpu: str, num_prompts: int, max_new_tokens: int) -> dict:
    out_path = config.RESULTS_DIR / "logs" / f"hf_baseline_{int(time.time())}.json"
    cmd = [
        sys.executable, str(Path(__file__).resolve().parents[2] / "common" / "hf_baseline.py"),
        "--model", config.MODEL_PRIMARY,
        "--num-prompts", str(num_prompts),
        "--batch-size", str(HF_BATCH_SIZE),
        "--max-new-tokens", str(max_new_tokens),
        "--dtype", "bfloat16",
        "--device", "cuda:0",
        "--output", str(out_path),
    ]
    env = {"CUDA_VISIBLE_DEVICES": gpu, "HF_HOME": config.HF_HOME}
    import os

    full_env = os.environ.copy()
    full_env.update(env)

    log_path = config.LOGS_DIR / "hf_baseline.log"
    print(f"[lab02] running naive HF baseline: {' '.join(cmd)}")
    with open(log_path, "w") as log_file:
        proc = subprocess.run(cmd, env=full_env, stdout=log_file, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"HF baseline subprocess failed (code={proc.returncode}); see {log_path}")
    return json.loads(out_path.read_text())


def run(quick: bool = False, gpus: str | None = None) -> list[dict]:
    gpus = gpus or config.DEFAULT_GPU
    gpu0 = str(gpus).split(",")[0]
    n_prompts = NUM_PROMPTS_QUICK if quick else NUM_PROMPTS_FULL
    max_new_tokens = config.scale_quick(MAX_NEW_TOKENS, quick)

    gpu_utils.print_snapshot("lab02 before")

    hf_row = _run_hf_baseline(gpu0, n_prompts, max_new_tokens)
    hf_row["label"] = "hf_naive_static_batch"
    print(f"[lab02] HF naive baseline: {hf_row['output_throughput']:.1f} out-tok/s over {n_prompts} prompts")

    rows: list[dict] = [hf_row]

    vllm_variants = [
        ("vllm_max_num_seqs_8", ["--max-num-seqs", "8"]),
        ("vllm_default", []),
    ]
    for label, extra in vllm_variants:
        extra_args = [
            "--gpu-memory-utilization", str(config.DEFAULT_GPU_MEMORY_UTILIZATION),
            "--max-model-len", str(config.DEFAULT_MAX_MODEL_LEN),
            *extra,
        ]
        with server.VLLMServer(
            config.MODEL_PRIMARY, gpus=gpu0, extra_args=extra_args, name=f"lab02_{label}"
        ) as base_url:
            result = bench.run_benchmark(
                base_url=base_url,
                model=config.MODEL_PRIMARY,
                label=label,
                num_prompts=n_prompts,
                random_input_len=RANDOM_INPUT_LEN,
                random_output_len=RANDOM_OUTPUT_LEN,
                max_concurrency=n_prompts,
            )
        rows.append(result.summary)

    rows = report.add_speedup(rows, baseline_label="hf_naive_static_batch")
    columns = list(report.DEFAULT_COLUMNS) + [("speedup", "Speedup vs HF naive", "{:.2f}x")]
    table = report.print_table(rows, columns=columns, title="Lab 02: Continuous batching + paged KV cache")
    report.save_lab_results(LAB_NAME, rows)
    report.append_summary(
        LAB_NAME,
        "Lab 02: Continuous batching + paged KV cache",
        table,
        notes=(
            "hf_naive_static_batch has no TTFT concept (batch call, not streaming), hence '-'. "
            "vllm_max_num_seqs_8 uses the same concurrency budget as the HF batch size, isolating "
            "the scheduling technique itself from raw concurrency headroom."
        ),
    )
    return rows


if __name__ == "__main__":
    args = cli.build_arg_parser(__doc__).parse_args()
    run(quick=args.quick, gpus=args.gpus)
