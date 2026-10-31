#!/usr/bin/env bash
# Lab 07 - CUDA Graphs & torch.compile.
#
# The decode phase of LLM inference issues the same sequence of small CUDA
# kernels every single step (one step = one new token for every request in
# the batch). Each kernel launch has fixed CPU-side overhead (~microseconds,
# but there are dozens of kernels x hundreds of decode steps x many
# concurrent requests). CUDA graphs let SGLang record that whole sequence of
# kernel launches *once* per batch size and replay it as a single low-overhead
# operation on every subsequent step -- this matters most when decode steps
# are themselves cheap (small models, small batches) so launch overhead is a
# larger fraction of total step time.
#
# torch.compile is a separate, complementary optimization that fuses and
# specializes the *model's Python/eager code* itself into more efficient
# kernels. SGLang's docs currently flag `--enable-torch-compile` as an
# experimental / lightly-maintained feature that can error out on some
# model+version combinations, so this lab treats it as best-effort.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 1
trap_stop_server

LAB=lab07_cuda_graph_torch_compile
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

# Long output, moderate concurrency: emphasizes decode-step overhead, which
# is exactly what CUDA graphs target (as opposed to prefill, which is
# already compute-bound and dominated by matmul time, not launch overhead).
CG_OUTPUT_LEN="${CG_OUTPUT_LEN:-512}"
CG_CONCURRENCY="${CG_CONCURRENCY:-16}"
CG_NUM_PROMPTS="${CG_NUM_PROMPTS:-96}"

run_bench_cg() {
  local out_file="$1"
  run_bench "$out_file" \
    --model "$SMALL_MODEL" \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len "$CG_OUTPUT_LEN" \
    --num-prompts "$CG_NUM_PROMPTS" \
    --max-concurrency "$CG_CONCURRENCY"
}

section "[1/3] CUDA graphs ENABLED (default)"
start_sglang_server "$OUT_DIR/server_cudagraph_on.log" --model-path "$SMALL_MODEL" --tp 1
run_bench_cg "$OUT_DIR/bench_cudagraph_on.jsonl"
record_result "$LAB" "cuda-graph-on" "$OUT_DIR/bench_cudagraph_on.jsonl"
stop_sglang_server

section "[2/3] CUDA graphs DISABLED"
start_sglang_server "$OUT_DIR/server_cudagraph_off.log" --model-path "$SMALL_MODEL" --tp 1 \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
run_bench_cg "$OUT_DIR/bench_cudagraph_off.jsonl"
record_result "$LAB" "cuda-graph-off" "$OUT_DIR/bench_cudagraph_off.jsonl"
stop_sglang_server

section "[3/3] torch.compile ENABLED (experimental, best-effort -- may skip on failure)"
if start_sglang_server "$OUT_DIR/server_torch_compile.log" --model-path "$SMALL_MODEL" --tp 1 \
     --enable-torch-compile --torch-compile-max-bs "$CG_CONCURRENCY"; then
  run_bench_cg "$OUT_DIR/bench_torch_compile.jsonl"
  record_result "$LAB" "torch-compile-on" "$OUT_DIR/bench_torch_compile.jsonl"
  stop_sglang_server
else
  log_warn "torch.compile server failed to start (this is a known-experimental SGLang feature)."
  log_warn "See $OUT_DIR/server_torch_compile.log for details. Skipping this configuration."
fi

section "Lab 07 summary"
python "$LABS_ROOT/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv" | grep -E "lab|$LAB" || true

echo
echo "Compare median_tpot_ms (time per output token, i.e. decode step cost)"
echo "between cuda-graph-on and cuda-graph-off -- this is where the win should"
echo "concentrate, since prefill (dominated by big matmuls) barely benefits."
echo "If the torch-compile row is present, compare it to cuda-graph-on: gains"
echo "are typically largest for small models / small batch sizes where kernel"
echo "fusion reduces memory-bandwidth-bound overhead."
