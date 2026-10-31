#!/usr/bin/env bash
# Lab 04 - Tiered prefix cache: offload KV from HBM to CPU RAM.
#
# Multi-turn sessions whose combined context is larger than the pool's HBM
# KV cache. Every turn re-sends the whole conversation. Without offload,
# by the time a session comes back its prefix has been evicted from HBM and
# is recomputed. With vLLM's OffloadingConnector, evicted blocks were copied
# to pinned CPU memory and are DMA'd back instead.
#
# Two replicas (TP=2) behind the llm-d EPP, as in the upstream
# tiered-prefix-cache guide; GPU memory is capped so the effect shows up at
# lab scale.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

LAB=lab04_tiered_kv_cache
require_docker_gpus
trap_stop_all

export LLMD_REPLICAS="${TIER_REPLICAS:-2}"
GPU_UTIL="${GPU_UTIL:-0.45}"
CPU_BYTES="${CPU_BYTES:-214748364800}"   # 200 GiB of host RAM per replica
SESSIONS="${SESSIONS:-160}"
DOC_TOKENS="${DOC_TOKENS:-8000}"
TURNS="${TURNS:-4}"
REQUESTS=$(( SESSIONS * TURNS ))

write_endpoints "$LLMD_REPLICAS"

run_sessions() {
  local config="$1"
  start_epp "$LLMD_ROOT/configs/epp/optimized-baseline.yaml"
  start_envoy "$LLMD_ROOT/configs/envoy/envoy-epp.yaml"
  warmup "http://127.0.0.1:$ENVOY_PORT"
  loadgen "$LAB" "$config" --base-url "http://127.0.0.1:$ENVOY_PORT" \
    --workload multi-turn --sessions "$SESSIONS" --prefix-tokens "$DOC_TOKENS" \
    --suffix-tokens 150 --output-tokens 150 \
    --tenant "name=all,concurrency=32,requests=$REQUESTS" \
    --replica-metrics "$(replica_urls)"
  stop_router
  stop_all
}

section "Run 1: HBM-only prefix cache (gpu-memory-utilization $GPU_UTIL)"
start_replicas --gpu-memory-utilization "$GPU_UTIL"
run_sessions "hbm-only"

section "Run 2: HBM + CPU tier via OffloadingConnector ($(( CPU_BYTES / 1073741824 )) GiB/replica)"
start_replicas --gpu-memory-utilization "$GPU_UTIL" \
  --kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$CPU_BYTES}}"
run_sessions "hbm-plus-cpu"

show_summary "$LAB"
cat <<EOF

Working set: $SESSIONS sessions x ~$DOC_TOKENS-token documents, $TURNS turns each,
visited round-robin so each session returns after all others have run.
- hbm-only:     prefix_hit_rate collapses once the working set exceeds HBM.
- hbm-plus-cpu: KV evicted from HBM is reloaded from CPU RAM; ttft_p50 drops.
Per-replica 'ext_hits/ext_queries' in the JSON results count CPU-tier hits
(vllm:external_prefix_cache_*). The upstream guide additionally adds a
second prefix-cache scorer to the router so requests also prefer the replica
whose *CPU* tier holds their prefix.
EOF
