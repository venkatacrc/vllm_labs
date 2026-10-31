#!/usr/bin/env bash
# Lab 02 - Precision: FP32 vs FP16 vs BF16.
#
# Same model, same GPU, same benchmark load. Only --dtype changes. This
# isolates the effect of numeric precision on memory footprint and
# throughput, independent of quantization (that's lab03) or parallelism
# (lab04+).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 1
trap_stop_server

LAB=lab02_precision
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

PROMPT_FOR_EYEBALL="Explain, in exactly two sentences, why lower numeric precision speeds up LLM inference."

run_one_dtype() {
  local dtype="$1"
  section "SGLang FP-precision experiment: --dtype $dtype"

  start_sglang_server "$OUT_DIR/server_${dtype}.log" \
    --model-path "$SMALL_MODEL" \
    --tp 1 \
    --dtype "$dtype"

  local mem_mb
  mem_mb=$(gpu_mem_used_mb)
  log "GPU memory in use right after load (weights + KV cache pool): ${mem_mb} MiB"

  log "Qualitative sample (same prompt, same seed, dtype=$dtype):"
  curl -s "http://$SGLANG_HOST:$SGLANG_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
          "model": "'"$SMALL_MODEL"'",
          "messages": [{"role": "user", "content": "'"$PROMPT_FOR_EYEBALL"'"}],
          "max_tokens": 96,
          "temperature": 0
        }' | python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" \
    | tee "$OUT_DIR/sample_${dtype}.txt"

  run_bench "$OUT_DIR/bench_${dtype}.jsonl" \
    --model "$SMALL_MODEL" \
    --dataset-name random \
    --random-input-len "$BENCH_RANDOM_INPUT_LEN" \
    --random-output-len "$BENCH_RANDOM_OUTPUT_LEN" \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --max-concurrency "$BENCH_MAX_CONCURRENCY"

  record_result "$LAB" "sglang-${dtype}-tp1" "$OUT_DIR/bench_${dtype}.jsonl" \
    --extra "dtype=$dtype" --extra "gpu_mem_mb=$mem_mb"

  stop_sglang_server
}

run_one_dtype float32
run_one_dtype bfloat16
run_one_dtype float16

section "Lab 02 summary"
python "$LABS_ROOT/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv" | grep -E "lab|$LAB" || true

echo
echo "Compare results/$LAB/sample_{float32,bfloat16,float16}.txt for output quality,"
echo "and the gpu_mem_mb / throughput columns in results/summary.csv for the"
echo "efficiency trade-off. On B200, bf16/fp16 tensor cores run at multiples of"
echo "FP32 throughput, and halving bytes-per-parameter roughly halves memory and"
echo "KV-cache pressure -- which indirectly increases the batch size SGLang can fit,"
echo "compounding the raw compute speedup."
