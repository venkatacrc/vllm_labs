#!/usr/bin/env python
"""Top-level orchestrator for the vLLM optimization labs.

Runs every lab in order (00 -> 11), each of which spins up a real
`vllm serve` process, drives it with `vllm bench serve`, tears it down,
and saves its results under results/. At the end this prints + saves a
consolidated results/SUMMARY.md covering the whole fp32 -> fully-optimized
journey.

Usage:
    python run_all_labs.py                  # run everything
    python run_all_labs.py --quick           # fast smoke test of the whole suite
    python run_all_labs.py --labs 0,1,3      # run a subset
    python run_all_labs.py --list            # list labs and exit
    python labs/lab04_prefix_caching/lab.py  # any lab can also run standalone
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from common import config, gpu_utils  # noqa: E402

# (lab number, package name) in the order they should run.
LABS: list[tuple[str, str]] = [
    ("00", "lab00_fp32_baseline"),
    ("01", "lab01_precision_bf16_fp16"),
    ("02", "lab02_continuous_batching_paged_kv_cache"),
    ("03", "lab03_attention_backends"),
    ("04", "lab04_prefix_caching"),
    ("05", "lab05_chunked_prefill"),
    ("06", "lab06_cuda_graphs_compilation"),
    ("07", "lab07_weight_quantization_awq_gptq"),
    ("08", "lab08_fp8_quantization"),
    ("09", "lab09_speculative_decoding"),
    ("10", "lab10_tensor_parallelism"),
    ("11", "lab11_capstone_full_stack"),
]


def parse_lab_selection(spec: str | None) -> list[tuple[str, str]]:
    if not spec:
        return LABS
    wanted = {s.strip().zfill(2) for s in spec.split(",") if s.strip()}
    selected = [lab for lab in LABS if lab[0] in wanted]
    missing = wanted - {lab[0] for lab in selected}
    if missing:
        print(f"[run_all_labs] WARNING: unknown lab number(s) {sorted(missing)}, ignoring")
    return selected


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labs", default=None, help="Comma list of lab numbers to run, e.g. '0,1,3'. Default: all.")
    p.add_argument("--quick", action="store_true", help="Shrink workloads for a fast smoke test of the whole suite.")
    p.add_argument("--gpus", default=None, help="Override the default GPU(s) for single-GPU labs, e.g. '0'.")
    p.add_argument("--skip-download", action="store_true", help="Skip reminding you to pre-fetch models.")
    p.add_argument("--list", action="store_true", help="List labs and exit.")
    args = p.parse_args()

    if args.list:
        for num, name in LABS:
            print(f"lab{num}: {name}")
        return 0

    selection = parse_lab_selection(args.labs)
    if not selection:
        print("[run_all_labs] Nothing to run.")
        return 1

    if not args.skip_download:
        print(
            "[run_all_labs] Tip: run `bash scripts/download_models.sh` first so lab "
            "runs aren't blocked on downloads mid-benchmark.\n"
        )

    print(f"[run_all_labs] Running {len(selection)} lab(s): {[n for _, n in selection]}")
    if args.quick:
        print("[run_all_labs] --quick enabled: workloads are shrunk for a fast smoke test, "
              "not for trustworthy numbers.")
    gpu_utils.print_snapshot("before suite")

    results: dict[str, object] = {}
    failures: list[tuple[str, str]] = []
    suite_start = time.time()

    for num, name in selection:
        print(f"\n{'=' * 88}\nlab{num}: {name}\n{'=' * 88}")
        t0 = time.time()
        try:
            mod = importlib.import_module(f"labs.{name}.lab")
            rows = mod.run(quick=args.quick, gpus=args.gpus)
            results[name] = rows
        except Exception as exc:  # noqa: BLE001 - keep the suite going on a single lab failure
            print(f"[run_all_labs] lab{num} ({name}) FAILED: {exc}")
            traceback.print_exc()
            failures.append((name, str(exc)))
        finally:
            elapsed = time.time() - t0
            print(f"[run_all_labs] lab{num} ({name}) took {elapsed / 60:.1f} min")

    total_elapsed = time.time() - suite_start
    print(f"\n{'=' * 88}")
    print(f"[run_all_labs] Finished in {total_elapsed / 60:.1f} min. "
          f"{len(results)} lab(s) succeeded, {len(failures)} failed.")
    if failures:
        print("[run_all_labs] Failures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    print(f"[run_all_labs] Full report: {config.SUMMARY_PATH}")
    print(f"[run_all_labs] Raw per-lab JSON: {config.RESULTS_DIR}/<lab_name>/results.json")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
