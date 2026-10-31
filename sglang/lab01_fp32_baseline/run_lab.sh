#!/usr/bin/env bash
# Lab 01 - The naive FP32 baseline.
#
# Establishes the "before" numbers that every later lab is measured against:
#   (a) a plain HuggingFace transformers generate() loop, one request at a
#       time, FP32, on a single GPU -- the naive way most people first serve
#       a model.
#   (b) the exact same FP32 model served by SGLang with concurrency=1
#       (isolates: does SGLang's scheduler/kernels help even with zero
#       concurrent requests?)
#   (c) the same FP32 SGLang server hammered with concurrency=64 (isolates:
#       how much does continuous batching alone -- with no precision or
#       quantization changes at all -- buy you?)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 1
trap_stop_server

LAB=lab01_fp32_baseline
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

HF_NUM_PROMPTS="${HF_NUM_PROMPTS:-12}"
HF_MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS:-128}"

section "[1/3] Naive HuggingFace transformers loop (FP32, sequential, batch=1)"
python "$SCRIPT_DIR/hf_naive_baseline.py" \
  --model "$SMALL_MODEL" \
  --num-prompts "$HF_NUM_PROMPTS" \
  --max-new-tokens "$HF_MAX_NEW_TOKENS" \
  --dtype float32 \
  --out "$OUT_DIR/hf_naive.json"

HF_THROUGHPUT=$(python -c "import json;print(json.load(open('$OUT_DIR/hf_naive.json'))['output_throughput'])")
HF_MEM=$(python -c "import json;print(json.load(open('$OUT_DIR/hf_naive.json'))['peak_gpu_memory_gb'])")
record_result_raw "$LAB" "hf-transformers-naive-fp32" \
  --metric "output_throughput_tok_s=$HF_THROUGHPUT" \
  --source "$OUT_DIR/hf_naive.json" \
  --extra "gpu_mem_gb=$HF_MEM" --extra "engine=transformers"

wait_for_gpu_memory_free 30

section "[2/3] SGLang, FP32, TP=1, concurrency=1 (apples-to-apples vs the HF loop)"
start_sglang_server "$OUT_DIR/server_fp32.log" \
  --model-path "$SMALL_MODEL" \
  --tp 1 \
  --dtype float32

run_bench "$OUT_DIR/bench_conc1.jsonl" \
  --model "$SMALL_MODEL" \
  --dataset-name random \
  --random-input-len "$BENCH_RANDOM_INPUT_LEN" \
  --random-output-len "$HF_MAX_NEW_TOKENS" \
  --num-prompts "$HF_NUM_PROMPTS" \
  --max-concurrency 1
record_result "$LAB" "sglang-fp32-tp1-conc1" "$OUT_DIR/bench_conc1.jsonl" --extra "engine=sglang" --extra "dtype=fp32" --extra "concurrency=1"

section "[3/3] SGLang, FP32, TP=1, concurrency=$BENCH_MAX_CONCURRENCY (continuous batching engaged)"
run_bench "$OUT_DIR/bench_conc64.jsonl" \
  --model "$SMALL_MODEL" \
  --dataset-name random \
  --random-input-len "$BENCH_RANDOM_INPUT_LEN" \
  --random-output-len "$BENCH_RANDOM_OUTPUT_LEN" \
  --num-prompts "$BENCH_NUM_PROMPTS" \
  --max-concurrency "$BENCH_MAX_CONCURRENCY"
record_result "$LAB" "sglang-fp32-tp1-conc${BENCH_MAX_CONCURRENCY}" "$OUT_DIR/bench_conc64.jsonl" --extra "engine=sglang" --extra "dtype=fp32" --extra "concurrency=$BENCH_MAX_CONCURRENCY"

stop_sglang_server

section "Lab 01 summary"
python "$LABS_ROOT/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv" | grep -E "lab|$LAB" || true

echo
echo "Look at $OUT_DIR/*.json and $RESULTS_ROOT/summary.csv."
echo "Key question to answer for yourself: how much of SGLang's speedup over the"
echo "naive HF loop comes from *just* running behind an inference server with a"
echo "scheduler (conc=1 case), vs from batching multiple concurrent requests"
echo "together (conc=$BENCH_MAX_CONCURRENCY case)? Precision hasn't changed at all yet -- that's lab02."
