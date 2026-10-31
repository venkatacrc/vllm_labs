#!/usr/bin/env bash
# Lab 06 - Wide expert parallelism (single-node version).
#
# llm-d's wide-EP guide serves DeepSeek-R1 on 32 H200 (prefill DP16 +
# decode DP16, DeepEP all-to-all, NIXL). The same building blocks fit on one
# 8xB200 node with an MoE model that fits: Qwen3-30B-A3B (128 experts, 8
# active per token).
#   tp8          MoE layers tensor-parallel across 8 GPUs (every GPU holds a
#                slice of every expert; all-reduce after every layer)
#   dp8-ep       attention data-parallel (8 independent batches), experts
#                sharded across GPUs (16 whole experts per GPU), tokens moved
#                with all-to-all dispatch/combine
#   dp8-ep-eplb  + expert-parallel load balancer (replicates hot experts)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

LAB=lab06_wide_ep
require_docker_gpus
trap_stop_all

export LLMD_MODEL="${EP_MODEL:-Qwen/Qwen3-30B-A3B}"
export LLMD_MAX_MODEL_LEN=8192
ALL_GPUS="0,1,2,3,4,5,6,7"
PORT=8000
CONCURRENCY="${CONCURRENCY:-512}"
REQUESTS="${REQUESTS:-3072}"

run_variant() {
  local config="$1"; shift
  section "$config"
  if ! { start_vllm "$config" "$ALL_GPUS" "$PORT" "$@" && \
         wait_http "http://127.0.0.1:$PORT/v1/models" "$SERVER_STARTUP_TIMEOUT" "$config"; }; then
    log_warn "$config failed to start:"; docker logs --tail 40 "${LLMD_CONTAINER_PREFIX}-$config" >&2 || true
    stop_all; return 0
  fi
  warmup "http://127.0.0.1:$PORT"
  loadgen "$LAB" "$config" --base-url "http://127.0.0.1:$PORT" \
    --workload random --prefix-tokens 1000 --suffix-tokens 0 --output-tokens 300 \
    --tenant "name=all,concurrency=$CONCURRENCY,requests=$REQUESTS" \
    --replica-metrics "http://127.0.0.1:$PORT" || true
  stop_all
}

run_variant tp8
TP_OVERRIDE=1 run_variant dp8-ep --data-parallel-size 8 --enable-expert-parallel
TP_OVERRIDE=1 run_variant dp8-ep-eplb --data-parallel-size 8 --enable-expert-parallel --enable-eplb

show_summary "$LAB"
cat <<EOF

- At high concurrency dp8-ep should beat tp8 on out_tok_per_s: attention runs
  on 8 independent batches (no per-layer all-reduce of activations, and each
  GPU holds whole experts so expert GEMMs are bigger and more efficient).
- tpot_p50 at low concurrency can favour tp8 (lower per-token latency): rerun with
  CONCURRENCY=8 to see the latency/throughput trade.
- Multi-node wide-EP adds DeepEP kernels (--all2all-backend deepep_high_throughput
  for prefill, deepep_low_latency for decode), P/D, and a DisaggregatedSet
  spanning nodes - see llm-d guides/wide-ep.
EOF
