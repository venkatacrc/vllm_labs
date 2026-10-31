#!/usr/bin/env bash
# Lab 12 - Attention backends on Blackwell.
#
# SGLang lets you pick the attention kernel library with --attention-backend
# (and separately for each phase with --prefill-attention-backend /
# --decode-attention-backend). On B200 (SM100) the interesting choices for a
# standard GQA model like Qwen2.5-7B are:
#   fa4         FlashAttention 4 (CuTe-DSL, written for Blackwell)
#   flashinfer  FlashInfer (SGLang's long-time default on NVIDIA)
#   trtllm_mha  TensorRT-LLM's Blackwell MHA kernels exposed via FlashInfer
#   triton      Portable Triton kernels (reference point)
# Each backend is run against a prefill-heavy and a decode-heavy workload,
# because a kernel that is great at one phase is not necessarily great at
# the other. Backends that fail to start on this build are skipped.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 1
trap_stop_server

LAB=lab12_attention_backends
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

BACKENDS=(${ATTN_BACKENDS:-fa4 flashinfer trtllm_mha triton})
PREFILL_NUM_PROMPTS="${PREFILL_NUM_PROMPTS:-64}"
DECODE_NUM_PROMPTS="${DECODE_NUM_PROMPTS:-32}"

for backend in "${BACKENDS[@]}"; do
  section "Attention backend: $backend"
  if ! start_sglang_server "$OUT_DIR/server_${backend}.log" \
      --model-path "$SMALL_MODEL" --tp 1 --attention-backend "$backend"; then
    log_warn "backend=$backend failed to start; see $OUT_DIR/server_${backend}.log. Skipping."
    stop_sglang_server
    continue
  fi

  run_bench "$OUT_DIR/bench_${backend}_prefill.jsonl" \
    --model "$SMALL_MODEL" \
    --dataset-name random \
    --random-input-len 4000 \
    --random-output-len 32 \
    --num-prompts "$PREFILL_NUM_PROMPTS" \
    --max-concurrency 32
  record_result "$LAB" "${backend}-prefill" "$OUT_DIR/bench_${backend}_prefill.jsonl"

  run_bench "$OUT_DIR/bench_${backend}_decode.jsonl" \
    --model "$SMALL_MODEL" \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 512 \
    --num-prompts "$DECODE_NUM_PROMPTS" \
    --max-concurrency 4
  record_result "$LAB" "${backend}-decode" "$OUT_DIR/bench_${backend}_decode.jsonl"

  stop_sglang_server
done

section "Lab 12 summary"
python "$LABS_ROOT/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv" | grep -E "lab|$LAB" || true

echo
echo "Compare '*-prefill' rows on median_ttft_ms (prompt processing speed) and"
echo "'*-decode' rows on median_tpot_ms (per-token decode cost). fa4 and"
echo "trtllm_mha are the Blackwell-specific kernels; triton is the portable"
echo "baseline. Mixing is allowed: e.g. --prefill-attention-backend fa4"
echo "--decode-attention-backend trtllm_mha."
