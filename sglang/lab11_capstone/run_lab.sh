#!/usr/bin/env bash
# Lab 11 - Capstone: stack the wins and measure the total speedup.
#
# Combines the optimizations from labs 02, 03, 05, 06, 07 -- all of which
# are "free" in the sense of being broadly applicable regardless of
# workload shape -- into one server configuration, and measures it against
# the exact same naive-HF and naive-SGLang-FP32 baselines from lab01.
# (Speculative decoding from lab08 is deliberately left out of the default
# recipe since its benefit is workload-shape-dependent -- see the README.)
#
# Stacked here:
#   - FP8 quantization                          (lab03)
#   - RadixAttention prefix caching (default)    (lab05)
#   - Continuous batching + overlap sched (default) (lab01, lab06)
#   - Chunked prefill (default)                  (lab06)
#   - CUDA graphs (default)                      (lab07)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 1
trap_stop_server

LAB=lab11_capstone
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

HF_NUM_PROMPTS="${HF_NUM_PROMPTS:-12}"
HF_MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS:-128}"

section "Launching the 'optimized recipe' server: FP8, TP=1, all default scheduler/kernel optimizations on"
start_sglang_server "$OUT_DIR/server_optimized.log" \
  --model-path "$SMALL_MODEL" \
  --tp 1 \
  --dtype bfloat16 \
  --quantization fp8

section "[1/2] Optimized server, concurrency=1 (apples-to-apples vs lab01's naive HF loop)"
run_bench "$OUT_DIR/bench_conc1.jsonl" \
  --model "$SMALL_MODEL" \
  --dataset-name random \
  --random-input-len "$BENCH_RANDOM_INPUT_LEN" \
  --random-output-len "$HF_MAX_NEW_TOKENS" \
  --num-prompts "$HF_NUM_PROMPTS" \
  --max-concurrency 1
record_result "$LAB" "optimized-conc1" "$OUT_DIR/bench_conc1.jsonl"

section "[2/2] Optimized server, concurrency=$BENCH_MAX_CONCURRENCY"
run_bench "$OUT_DIR/bench_conc64.jsonl" \
  --model "$SMALL_MODEL" \
  --dataset-name random \
  --random-input-len "$BENCH_RANDOM_INPUT_LEN" \
  --random-output-len "$BENCH_RANDOM_OUTPUT_LEN" \
  --num-prompts "$BENCH_NUM_PROMPTS" \
  --max-concurrency "$BENCH_MAX_CONCURRENCY"
record_result "$LAB" "optimized-conc${BENCH_MAX_CONCURRENCY}" "$OUT_DIR/bench_conc64.jsonl"

stop_sglang_server

section "Final comparison: naive FP32 (lab01) vs fully-optimized recipe (lab11)"
python - "$RESULTS_ROOT/summary.csv" "$BENCH_MAX_CONCURRENCY" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
CONC = sys.argv[2]
def find(lab, config):
    for r in rows:
        if r["lab"] == lab and r["config"] == config:
            return r
    return None

pairs = [
    ("HF naive loop (FP32, sequential)", find("lab01_fp32_baseline", "hf-transformers-naive-fp32")),
    ("SGLang FP32, concurrency=1", find("lab01_fp32_baseline", "sglang-fp32-tp1-conc1")),
    ("SGLang optimized recipe, concurrency=1", find("lab11_capstone", "optimized-conc1")),
]
print(f"{'Configuration':45s} | {'tok/s':>10}")
print("-" * 60)
first_tput = None
for label, row in pairs:
    if row is None or not row.get("output_throughput_tok_s"):
        print(f"{label:45s} | {'(missing, run lab01 first)':>10}")
        continue
    tput = float(row["output_throughput_tok_s"])
    if first_tput is None:
        first_tput = tput
    speedup = tput / first_tput if first_tput else 1.0
    print(f"{label:45s} | {tput:10.1f}  ({speedup:.1f}x vs naive HF loop)")

print()
pairs2 = [
    (f"SGLang FP32, concurrency={CONC}", find("lab01_fp32_baseline", f"sglang-fp32-tp1-conc{CONC}")),
    (f"SGLang optimized recipe, concurrency={CONC}", find("lab11_capstone", f"optimized-conc{CONC}")),
]
base = find("lab01_fp32_baseline", f"sglang-fp32-tp1-conc{CONC}")
base_tput = float(base["output_throughput_tok_s"]) if base and base.get("output_throughput_tok_s") else None
for label, row in pairs2:
    if row is None or not row.get("output_throughput_tok_s"):
        print(f"{label:45s} | {'(missing)':>10}")
        continue
    tput = float(row["output_throughput_tok_s"])
    extra = f"  ({tput/base_tput:.2f}x vs naive-fp32-at-same-concurrency)" if base_tput else ""
    print(f"{label:45s} | {tput:10.1f}{extra}")
PY

section "Generating final plots across every lab you've run"
python "$LABS_ROOT/common/plot_results.py" --summary-csv "$RESULTS_ROOT/summary.csv" --out-dir "$RESULTS_ROOT/plots"

echo
echo "Full history: $RESULTS_ROOT/summary.csv"
echo "Charts:       $RESULTS_ROOT/plots/*.png"
echo
echo "If you have a repetitive/structured-output workload, consider adding"
echo "N-gram (lab08) or EAGLE speculative decoding on top of this recipe for"
echo "an additional, workload-dependent boost. If you're serving a MoE model"
echo "at scale, apply lab10's EP/DP-attention findings instead of plain TP."
echo "If you're memory constrained, revisit lab03's INT4 options."
