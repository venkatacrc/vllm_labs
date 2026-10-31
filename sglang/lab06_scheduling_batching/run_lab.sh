#!/usr/bin/env bash
# Lab 06 - Continuous batching & scheduler knobs.
#
# SGLang's scheduler is what turns a static "run this batch" GPU kernel into
# an online server that packs many independent, differently-timed requests
# together every iteration. Three of its knobs matter most in practice:
#
#   A) Overlap scheduling (--disable-overlap-schedule to turn OFF): by
#      default SGLang overlaps CPU-side scheduling work for the *next* batch
#      with GPU compute for the *current* one, hiding scheduler overhead.
#   B) Chunked prefill (--chunked-prefill-size): splits long prompts into
#      chunks so a single huge prefill can't monopolize a batch and starve
#      other requests' decode steps -- a throughput/tail-latency trade.
#   C) Schedule policy (--schedule-policy lpm vs fcfs): "longest prefix
#      match" reorders the queue to group requests sharing a RadixAttention
#      prefix together, improving cache locality under prefix-heavy traffic.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 1
trap_stop_server

LAB=lab06_scheduling_batching
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

run_random_bench() {
  local out_file="$1" input_len="$2"
  run_bench "$out_file" \
    --model "$SMALL_MODEL" \
    --dataset-name random \
    --random-input-len "$input_len" \
    --random-output-len "$BENCH_RANDOM_OUTPUT_LEN" \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --max-concurrency "$BENCH_MAX_CONCURRENCY"
}

# ---------------------------------------------------------------------------
section "[A] Overlap scheduler: enabled (default) vs disabled"
# ---------------------------------------------------------------------------
start_sglang_server "$OUT_DIR/server_overlap_on.log" --model-path "$SMALL_MODEL" --tp 1
run_random_bench "$OUT_DIR/bench_overlap_on.jsonl" "$BENCH_RANDOM_INPUT_LEN"
record_result "$LAB" "overlap-scheduler-on" "$OUT_DIR/bench_overlap_on.jsonl"
stop_sglang_server

start_sglang_server "$OUT_DIR/server_overlap_off.log" --model-path "$SMALL_MODEL" --tp 1 --disable-overlap-schedule
run_random_bench "$OUT_DIR/bench_overlap_off.jsonl" "$BENCH_RANDOM_INPUT_LEN"
record_result "$LAB" "overlap-scheduler-off" "$OUT_DIR/bench_overlap_off.jsonl"
stop_sglang_server

# ---------------------------------------------------------------------------
section "[B] Chunked prefill: enabled vs disabled (long-prompt workload)"
# ---------------------------------------------------------------------------
LONG_INPUT_LEN="${LONG_INPUT_LEN:-4096}"

start_sglang_server "$OUT_DIR/server_chunked_on.log" --model-path "$SMALL_MODEL" --tp 1 --chunked-prefill-size 2048
run_random_bench "$OUT_DIR/bench_chunked_on.jsonl" "$LONG_INPUT_LEN"
record_result "$LAB" "chunked-prefill-on" "$OUT_DIR/bench_chunked_on.jsonl" --extra "input_len=$LONG_INPUT_LEN"
stop_sglang_server

start_sglang_server "$OUT_DIR/server_chunked_off.log" --model-path "$SMALL_MODEL" --tp 1 --chunked-prefill-size -1
run_random_bench "$OUT_DIR/bench_chunked_off.jsonl" "$LONG_INPUT_LEN"
record_result "$LAB" "chunked-prefill-off" "$OUT_DIR/bench_chunked_off.jsonl" --extra "input_len=$LONG_INPUT_LEN"
stop_sglang_server

# ---------------------------------------------------------------------------
section "[C] Schedule policy: fcfs (default) vs lpm (longest-prefix-match), shared-prefix workload"
# ---------------------------------------------------------------------------
GSP_ARGS=(--dataset-name generated-shared-prefix --gsp-num-groups 8 --gsp-prompts-per-group 32
          --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 128)
GSP_NUM_PROMPTS=256

start_sglang_server "$OUT_DIR/server_policy_fcfs.log" --model-path "$SMALL_MODEL" --tp 1 --schedule-policy fcfs
run_bench "$OUT_DIR/bench_policy_fcfs.jsonl" --model "$SMALL_MODEL" "${GSP_ARGS[@]}" \
  --num-prompts "$GSP_NUM_PROMPTS" --max-concurrency "$BENCH_MAX_CONCURRENCY"
record_result "$LAB" "schedule-policy-fcfs" "$OUT_DIR/bench_policy_fcfs.jsonl"
stop_sglang_server

start_sglang_server "$OUT_DIR/server_policy_lpm.log" --model-path "$SMALL_MODEL" --tp 1 --schedule-policy lpm
run_bench "$OUT_DIR/bench_policy_lpm.jsonl" --model "$SMALL_MODEL" "${GSP_ARGS[@]}" \
  --num-prompts "$GSP_NUM_PROMPTS" --max-concurrency "$BENCH_MAX_CONCURRENCY"
record_result "$LAB" "schedule-policy-lpm" "$OUT_DIR/bench_policy_lpm.jsonl"
stop_sglang_server

section "Lab 06 summary"
python "$LABS_ROOT/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv" | grep -E "lab|$LAB" || true

echo
echo "A) overlap-scheduler-on vs -off: look at output_throughput_tok_s and"
echo "   median_tpot_ms -- overlap scheduling hides CPU scheduling latency"
echo "   behind GPU compute, usually a small but consistent win with no downside."
echo "B) chunked-prefill-on vs -off, with ${LONG_INPUT_LEN}-token prompts: check"
echo "   p99_ttft_ms in the .log files -- without chunking, one huge prefill can"
echo "   delay every other in-flight request's next decode step, spiking tail"
echo "   latency even though total throughput may look similar."
echo "C) schedule-policy fcfs vs lpm on the shared-prefix workload: lpm should"
echo "   show equal-or-better throughput/TTFT by scheduling requests that share"
echo "   a RadixAttention prefix back-to-back, maximizing cache reuse."
