#!/usr/bin/env bash
# Lab 05 - RadixAttention: automatic prefix caching.
#
# RadixAttention is SGLang's signature feature: it stores every request's KV
# cache in a radix tree keyed by token prefix, so any new request that
# shares a prefix with a previous one (e.g. the same system prompt, the same
# few-shot examples, a shared document) reuses that KV cache instead of
# recomputing it. This lab uses bench_serving's built-in
# `generated-shared-prefix` dataset -- long, repeated "system prompt" groups
# each followed by many different short questions -- which is exactly the
# workload shape (RAG, chat with a system prompt, few-shot prompting) where
# prefix caching wins big.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 1
trap_stop_server

LAB=lab05_radix_attention
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

# 8 distinct "system prompts" (e.g. 8 different customer-support bots, or 8
# different RAG documents), each reused across many short follow-up
# questions -- a very common real-world traffic shape.
GSP_NUM_GROUPS="${GSP_NUM_GROUPS:-8}"
GSP_PROMPTS_PER_GROUP="${GSP_PROMPTS_PER_GROUP:-32}"
GSP_SYSTEM_PROMPT_LEN="${GSP_SYSTEM_PROMPT_LEN:-2048}"
GSP_QUESTION_LEN="${GSP_QUESTION_LEN:-128}"
GSP_OUTPUT_LEN="${GSP_OUTPUT_LEN:-128}"
NUM_PROMPTS=$(( GSP_NUM_GROUPS * GSP_PROMPTS_PER_GROUP ))

run_config() {
  local config_name="$1"; shift
  local extra_args=("$@")

  section "RadixAttention experiment: $config_name"
  start_sglang_server "$OUT_DIR/server_${config_name}.log" \
    --model-path "$SMALL_MODEL" \
    --tp 1 \
    "${extra_args[@]}"

  run_bench "$OUT_DIR/bench_${config_name}.jsonl" \
    --model "$SMALL_MODEL" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups "$GSP_NUM_GROUPS" \
    --gsp-prompts-per-group "$GSP_PROMPTS_PER_GROUP" \
    --gsp-system-prompt-len "$GSP_SYSTEM_PROMPT_LEN" \
    --gsp-question-len "$GSP_QUESTION_LEN" \
    --gsp-output-len "$GSP_OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$BENCH_MAX_CONCURRENCY"

  record_result "$LAB" "$config_name" "$OUT_DIR/bench_${config_name}.jsonl" \
    --extra "gsp_groups=$GSP_NUM_GROUPS" --extra "gsp_system_len=$GSP_SYSTEM_PROMPT_LEN"

  stop_sglang_server
}

# 1. RadixAttention ENABLED (SGLang's default -- nothing special to pass)
run_config "radix-cache-enabled"

# 2. RadixAttention DISABLED, for comparison
run_config "radix-cache-disabled" --disable-radix-cache

section "Lab 05 summary"
python "$LABS_ROOT/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv" | grep -E "lab|$LAB" || true

echo
echo "Workload: $GSP_NUM_GROUPS distinct system prompts of ~$GSP_SYSTEM_PROMPT_LEN"
echo "tokens each, ~$GSP_PROMPTS_PER_GROUP different questions reusing each one"
echo "($NUM_PROMPTS requests total). With RadixAttention enabled, every request"
echo "after the first one in a group should skip prefill for the ~$GSP_SYSTEM_PROMPT_LEN"
echo "shared tokens entirely -- look for a large drop in median_ttft_ms and a"
echo "rise in output_throughput_tok_s for 'radix-cache-enabled' vs"
echo "'radix-cache-disabled' in results/summary.csv."
echo
echo "Also inspect results/$LAB/server_radix-cache-enabled.log for RadixAttention"
echo "cache-hit statistics logged during the run (search for 'cache hit')."
