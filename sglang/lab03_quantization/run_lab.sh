#!/usr/bin/env bash
# Lab 03 - Quantization: FP8 (online) vs AWQ-INT4 vs GPTQ-INT4 (pre-quantized).
#
# Continues where lab02 left off: bf16 is our new "full precision" baseline
# (this lab is self-contained, so we re-measure it here rather than assuming
# you ran lab02 first). We then go one level lower in precision three ways:
#   - FP8: SGLang quantizes the bf16 checkpoint to FP8 on the fly at load
#     time (`--quantization fp8`). B200's tensor cores have native FP8
#     support, so this is usually close to a free lunch for throughput.
#   - AWQ-INT4 / GPTQ-INT4: pre-quantized checkpoints (quantized offline by
#     the community using calibration data) loaded directly; SGLang
#     auto-detects the quantization method from the checkpoint's config, so
#     no --quantization flag is needed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 1
trap_stop_server

LAB=lab03_quantization
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

PROMPT_FOR_EYEBALL="What is quantization in the context of neural network inference? Answer in two sentences."

run_one_config() {
  local config_name="$1" model_path="$2"; shift 2
  local extra_args=("$@")

  section "Quantization experiment: $config_name  (model=$model_path, args=${extra_args[*]:-none})"

  start_sglang_server "$OUT_DIR/server_${config_name}.log" \
    --model-path "$model_path" \
    --tp 1 \
    "${extra_args[@]}"

  local mem_mb
  mem_mb=$(gpu_mem_used_mb)
  log "GPU memory in use: ${mem_mb} MiB"

  curl -s "http://$SGLANG_HOST:$SGLANG_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
          "model": "'"$model_path"'",
          "messages": [{"role": "user", "content": "'"$PROMPT_FOR_EYEBALL"'"}],
          "max_tokens": 96,
          "temperature": 0
        }' | python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" \
    | tee "$OUT_DIR/sample_${config_name}.txt"

  run_bench "$OUT_DIR/bench_${config_name}.jsonl" \
    --model "$model_path" \
    --dataset-name random \
    --random-input-len "$BENCH_RANDOM_INPUT_LEN" \
    --random-output-len "$BENCH_RANDOM_OUTPUT_LEN" \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --max-concurrency "$BENCH_MAX_CONCURRENCY"

  record_result "$LAB" "$config_name" "$OUT_DIR/bench_${config_name}.jsonl" \
    --extra "gpu_mem_mb=$mem_mb" --extra "model=$model_path"

  stop_sglang_server
}

# 1. bf16 baseline (re-measured here for a self-contained lab)
run_one_config "bf16-baseline" "$SMALL_MODEL" --dtype bfloat16

# 2. FP8 online quantization of the same bf16 checkpoint
run_one_config "fp8-online" "$SMALL_MODEL" --dtype bfloat16 --quantization fp8

# 3. AWQ INT4, pre-quantized checkpoint (quantization auto-detected)
run_one_config "awq-int4" "$AWQ_MODEL"

# 4. GPTQ INT4, pre-quantized checkpoint (quantization auto-detected)
run_one_config "gptq-int4" "$GPTQ_MODEL"

section "Lab 03 summary"
python "$LABS_ROOT/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv" | grep -E "lab|$LAB" || true

echo
echo "Compare gpu_mem_mb and output_throughput_tok_s across the four rows for"
echo "'$LAB' in results/summary.csv, and eyeball results/$LAB/sample_*.txt for"
echo "quality drift. FP8 should be nearly lossless vs bf16 on B200; INT4"
echo "methods trade a bit more quality for the smallest memory footprint --"
echo "worth it when you're memory-bound (e.g. need to fit a bigger model or"
echo "much larger batch size on the same GPU)."
