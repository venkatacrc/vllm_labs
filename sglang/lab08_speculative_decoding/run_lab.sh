#!/usr/bin/env bash
# Lab 08 - Speculative decoding (N-gram, no draft model required).
#
# Speculative decoding breaks the "one forward pass = one new token" rule:
# a cheap mechanism proposes several candidate future tokens, then a single
# forward pass of the real (target) model verifies all of them in parallel
# and accepts however many turn out correct. Verifying K tokens costs about
# the same as generating 1 (it's a single forward pass either way; the extra
# tokens ride along almost for free since decode is memory-bandwidth bound,
# not compute bound), so every accepted extra token is a big win.
#
# EAGLE/EAGLE3 (the common high-performance methods) need a separately
# trained draft model checkpoint compatible with your target model. To keep
# this lab fully self-contained with zero extra downloads, we use SGLang's
# NGRAM algorithm instead: it proposes candidate tokens by matching recent
# n-grams from the ongoing generation itself -- no draft model at all. It
# works best when output has local redundancy (copying/reformatting input,
# repeated structure); see the README for when to reach for EAGLE instead.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 1
trap_stop_server

LAB=lab08_speculative_decoding
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

run_general_bench() {
  local out_file="$1"
  run_bench "$out_file" \
    --model "$SMALL_MODEL" \
    --dataset-name random \
    --random-input-len "$BENCH_RANDOM_INPUT_LEN" \
    --random-output-len "$BENCH_RANDOM_OUTPUT_LEN" \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --max-concurrency 8
}

section "[1/2] Baseline: no speculative decoding"
start_sglang_server "$OUT_DIR/server_baseline.log" --model-path "$SMALL_MODEL" --tp 1
run_general_bench "$OUT_DIR/bench_baseline_general.jsonl"
record_result "$LAB" "no-spec-general" "$OUT_DIR/bench_baseline_general.jsonl"
log "Repetitive-output task (best case for n-gram speculation):"
python "$SCRIPT_DIR/repetitive_task_client.py" \
  --base-url "http://$SGLANG_HOST:$SGLANG_PORT" --model "$SMALL_MODEL" \
  --out "$OUT_DIR/repetitive_baseline.json"
BASE_REP_TPS=$(python -c "import json;print(json.load(open('$OUT_DIR/repetitive_baseline.json'))['output_throughput'])")
record_result_raw "$LAB" "no-spec-repetitive" --metric "output_throughput_tok_s=$BASE_REP_TPS" --source "$OUT_DIR/repetitive_baseline.json"
stop_sglang_server

section "[2/2] N-gram speculative decoding enabled"
start_sglang_server "$OUT_DIR/server_ngram.log" --model-path "$SMALL_MODEL" --tp 1 \
  --speculative-algorithm NGRAM \
  --speculative-num-draft-tokens 16 \
  --speculative-ngram-max-bfs-breadth 10 \
  --mem-fraction-static 0.7
run_general_bench "$OUT_DIR/bench_ngram_general.jsonl"
record_result "$LAB" "ngram-spec-general" "$OUT_DIR/bench_ngram_general.jsonl"
log "Repetitive-output task (best case for n-gram speculation):"
python "$SCRIPT_DIR/repetitive_task_client.py" \
  --base-url "http://$SGLANG_HOST:$SGLANG_PORT" --model "$SMALL_MODEL" \
  --out "$OUT_DIR/repetitive_ngram.json"
NGRAM_REP_TPS=$(python -c "import json;print(json.load(open('$OUT_DIR/repetitive_ngram.json'))['output_throughput'])")
record_result_raw "$LAB" "ngram-spec-repetitive" --metric "output_throughput_tok_s=$NGRAM_REP_TPS" --source "$OUT_DIR/repetitive_ngram.json"
stop_sglang_server

section "Lab 08 summary"
python "$LABS_ROOT/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv" | grep -E "lab|$LAB" || true

echo
echo "'*-general' rows: realistic mixed-content workload -- n-gram speculation"
echo "should give a modest but real speedup here (natural language has some"
echo "local redundancy: common phrases, punctuation patterns, etc.)."
echo "'*-repetitive' rows: tasks designed to maximize output redundancy --"
echo "n-gram speculation should show a much larger win here, since nearly"
echo "every proposed token gets accepted by the target model."
