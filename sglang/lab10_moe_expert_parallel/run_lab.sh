#!/usr/bin/env bash
# Lab 10 - Mixture-of-Experts: Expert Parallelism (EP) & DP Attention.
#
# Qwen3-30B-A3B is a MoE model: only ~3B of its ~30B parameters ("active
# params") are used per token, routed dynamically per-token to a handful of
# "expert" MLPs out of many. Plain tensor parallelism (TP) shards *every*
# expert's weights across all GPUs -- simple, but each GPU ends up storing
# and computing a slice of experts that particular token didn't even route
# to. Two MoE-specific parallelism strategies address this:
#
#   - Expert Parallelism (EP, `--ep`): instead of sharding every expert
#     across all GPUs, whole experts are placed on specific GPUs, and tokens
#     are routed (all-to-all) to whichever GPU holds the expert they need.
#     In SGLang, "EP subdivides TP": `--tp 8 --ep 8` means the 8-way TP
#     group is used to distribute 8 distinct expert shards.
#   - DP Attention (`--enable-dp-attention --dp-size N`): the attention
#     layers (which do NOT benefit as much from TP -- they're small
#     relative to a MoE model's total FLOPs) are run data-parallel instead
#     (each GPU handles a full, independent copy of attention for its own
#     slice of requests), avoiding the KV-cache duplication and all-reduce
#     overhead that plain TP imposes on attention layers, while the (much
#     larger) FFN/expert layers still use TP or EP.
#
# NOTE ON A KNOWN ISSUE: combining DP-attention with EP>1 simultaneously has
# open correctness bugs on some bf16 MoE checkpoints without the DeepEP
# kernel path (see sgl-project/sglang#23729). Our $MOE_MODEL is bf16, so
# this lab deliberately measures EP and DP-attention *separately* against
# the plain-TP baseline rather than combining them -- see the README for
# when the combined "production" recipe (DPA + EP + DeepEP, typically with
# an FP8 checkpoint) is appropriate.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 8
trap_stop_server

LAB=lab10_moe_expert_parallel
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

run_moe_bench() {
  local out_file="$1"
  run_bench "$out_file" \
    --model "$MOE_MODEL" \
    --dataset-name random \
    --random-input-len "$BENCH_RANDOM_INPUT_LEN" \
    --random-output-len "$BENCH_RANDOM_OUTPUT_LEN" \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --max-concurrency "$BENCH_MAX_CONCURRENCY"
}

section "[1/3] Baseline: plain TP=8 (no EP, no DP attention)"
start_sglang_server "$OUT_DIR/server_tp8.log" --model-path "$MOE_MODEL" --tp 8 --trust-remote-code
run_moe_bench "$OUT_DIR/bench_tp8.jsonl"
record_result "$LAB" "tp8-baseline" "$OUT_DIR/bench_tp8.jsonl"
stop_sglang_server

section "[2/3] Expert Parallelism: TP=8, EP=8"
start_sglang_server "$OUT_DIR/server_tp8_ep8.log" --model-path "$MOE_MODEL" --tp 8 --ep 8 --trust-remote-code
run_moe_bench "$OUT_DIR/bench_tp8_ep8.jsonl"
record_result "$LAB" "tp8-ep8" "$OUT_DIR/bench_tp8_ep8.jsonl"
stop_sglang_server

section "[3/3] DP Attention: TP=8, DP=8, --enable-dp-attention (no EP)"
start_sglang_server "$OUT_DIR/server_tp8_dpa.log" --model-path "$MOE_MODEL" --tp 8 --dp-size 8 --enable-dp-attention --trust-remote-code
run_moe_bench "$OUT_DIR/bench_tp8_dpa.jsonl"
record_result "$LAB" "tp8-dp8-attention" "$OUT_DIR/bench_tp8_dpa.jsonl"
stop_sglang_server

section "Lab 10 summary"
python "$LABS_ROOT/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv" | grep -E "lab|$LAB" || true

echo
echo "Compare output_throughput_tok_s across the three rows for '$LAB':"
echo "  tp8-baseline       : every GPU shards every expert -- simplest, often slowest for MoE"
echo "  tp8-ep8            : experts distributed whole across GPUs -- less redundant compute"
echo "  tp8-dp8-attention  : attention run data-parallel, FFN/experts still TP-sharded"
echo "Which wins depends heavily on your concurrency level -- EP's all-to-all"
echo "communication overhead is more visible at low concurrency, and pays off"
echo "more as batch sizes (and thus useful compute per all-to-all round) grow."
echo "Try re-running with BENCH_MAX_CONCURRENCY=16 vs 128 to see this shift."
