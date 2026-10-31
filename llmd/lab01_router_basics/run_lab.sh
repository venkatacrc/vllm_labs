#!/usr/bin/env bash
# Lab 01 - Why an inference-aware router: round-robin vs least-request vs
# the llm-d EPP, on a multi-tenant shared-prefix workload.
#
# Four Qwen3-32B replicas (TP=2 each, all 8 GPUs) serve the same traffic
# three times. Only the load balancer changes:
#   envoy-round-robin   classic L7 LB, cache- and load-blind
#   envoy-least-request load-aware, still cache-blind
#   llmd-epp            Envoy + llm-d EPP (prefix-cache + queue + KV scorers)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

LAB=lab01_router_basics
require_docker_gpus
trap_stop_all

GROUPS_N="${GROUPS_N:-48}"
PREFIX_TOKENS="${PREFIX_TOKENS:-6000}"
REQUESTS="${REQUESTS:-960}"
CONCURRENCY="${CONCURRENCY:-96}"

section "Start $LLMD_REPLICAS vLLM replicas (TP=$LLMD_TP)"
start_replicas
REPLICAS=$(replica_urls)

run_workload() {
  local config="$1"
  # Reset every replica's prefix cache so each router starts cold.
  local i
  for (( i = 0; i < LLMD_REPLICAS; i++ )); do
    curl --noproxy '*' -s -X POST "http://127.0.0.1:$(( LLMD_BASE_PORT + i ))/reset_prefix_cache" >/dev/null || true
  done
  warmup "http://127.0.0.1:$ENVOY_PORT"
  loadgen "$LAB" "$config" --base-url "http://127.0.0.1:$ENVOY_PORT" \
    --workload shared-prefix --groups "$GROUPS_N" --prefix-tokens "$PREFIX_TOKENS" \
    --suffix-tokens 100 --output-tokens 128 \
    --tenant "name=all,concurrency=$CONCURRENCY,requests=$REQUESTS" \
    --replica-metrics "$REPLICAS"
}

for policy in ROUND_ROBIN LEAST_REQUEST; do
  section "Envoy $policy"
  gen_envoy_lb "$policy" "$LLMD_REPLICAS" "$LLMD_WORK/envoy-lb.yaml"
  start_envoy "$LLMD_WORK/envoy-lb.yaml"
  run_workload "envoy-$(tr 'A-Z_' 'a-z-' <<<"$policy")"
  stop_router
done

section "Envoy + llm-d EPP (optimized baseline, file discovery)"
write_endpoints "$LLMD_REPLICAS"
start_epp "$LLMD_ROOT/configs/epp/optimized-baseline.yaml"
start_envoy "$LLMD_ROOT/configs/envoy/envoy-epp.yaml"
run_workload "llmd-epp"
docker logs "${LLMD_CONTAINER_PREFIX}-epp" > "$RESULTS_ROOT/$LAB/epp.log" 2>&1 || true

show_summary "$LAB"
cat <<EOF

How to read this:
  prefix_hit_rate   fraction of prompt tokens served from KV cache across
                    replicas. Round-robin scatters each of the $GROUPS_N prefix
                    groups over all $LLMD_REPLICAS replicas, so every replica has to
                    prefill (and store) every prefix; the EPP keeps a group on
                    the replica that already holds it.
  ttft_p50/p99      the payoff: cached prefixes skip ~$PREFIX_TOKENS tokens of prefill.
  max_replica_share should stay near 1/$LLMD_REPLICAS - affinity must not create hot spots
                    (the queue/KV scorers push back when one replica gets busy).
EPP decisions are logged in $RESULTS_ROOT/$LAB/epp.log.
EOF
