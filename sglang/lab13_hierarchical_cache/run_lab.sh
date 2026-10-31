#!/usr/bin/env bash
# Lab 13 - HiCache: hierarchical (GPU -> CPU) KV cache.
#
# RadixAttention (lab05) only helps while the shared prefix is still in GPU
# memory. When the working set of distinct prefixes is larger than the GPU
# KV pool (many tenants, many long documents, long agentic sessions), the
# radix tree evicts and later requests pay full prefill again. HiCache adds
# a host-memory tier behind the radix tree: evicted KV pages are written to
# pinned CPU RAM and prefetched back on a hit, which is much cheaper than
# recomputing the prefill.
#
# To make the effect visible on a 178 GB B200 we deliberately shrink the GPU
# KV pool with --mem-fraction-static and use a working set of shared
# prefixes larger than what fits on the GPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 1
trap_stop_server

LAB=lab13_hierarchical_cache
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

# ~0.2 * 178 GB minus ~15 GB of BF16 weights leaves ~20 GB of KV, about
# 350K tokens for Qwen2.5-7B (57 KB/token). 64 groups x 8192-token prefixes
# is ~524K distinct prefix tokens, so the GPU tier alone must thrash.
MEM_FRACTION="${MEM_FRACTION:-0.2}"
GSP_NUM_GROUPS="${GSP_NUM_GROUPS:-64}"
GSP_PROMPTS_PER_GROUP="${GSP_PROMPTS_PER_GROUP:-8}"
GSP_SYSTEM_PROMPT_LEN="${GSP_SYSTEM_PROMPT_LEN:-8192}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-100}"
NUM_PROMPTS=$(( GSP_NUM_GROUPS * GSP_PROMPTS_PER_GROUP ))

run_config() {
  local name="$1"; shift
  section "HiCache experiment: $name"
  start_sglang_server "$OUT_DIR/server_${name}.log" \
    --model-path "$SMALL_MODEL" --tp 1 \
    --mem-fraction-static "$MEM_FRACTION" \
    "$@"
  run_bench "$OUT_DIR/bench_${name}.jsonl" \
    --model "$SMALL_MODEL" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups "$GSP_NUM_GROUPS" \
    --gsp-prompts-per-group "$GSP_PROMPTS_PER_GROUP" \
    --gsp-system-prompt-len "$GSP_SYSTEM_PROMPT_LEN" \
    --gsp-question-len 128 \
    --gsp-output-len 64 \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency 32
  record_result "$LAB" "$name" "$OUT_DIR/bench_${name}.jsonl" \
    --extra "mem_fraction=$MEM_FRACTION" --extra "prefix_tokens=$(( GSP_NUM_GROUPS * GSP_SYSTEM_PROMPT_LEN ))"
  stop_sglang_server
}

run_config "no-prefix-cache" --disable-radix-cache
run_config "radix-gpu-only"
run_config "radix-plus-hicache" \
  --enable-hierarchical-cache \
  --hicache-size "$HICACHE_SIZE_GB" \
  --hicache-write-policy write_through

section "Lab 13 summary"
python "$LABS_ROOT/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv" | grep -E "lab|$LAB" || true

echo
echo "Expect: radix-gpu-only beats no-prefix-cache only modestly (the working"
echo "set does not fit on the GPU, so prefixes get evicted before reuse), while"
echo "radix-plus-hicache recovers most of the prefix-cache win by loading"
echo "evicted pages back from CPU RAM. Compare median_ttft_ms across the three."
echo "llm-d's tiered-prefix-cache guide uses exactly this (--hicache-size 200)"
echo "for SGLang and vLLM's OffloadingConnector for vLLM."
