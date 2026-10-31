#!/usr/bin/env bash
# Lab 02 - Inside the EPP: scorer ablation and a live pool change.
#
# Same replicas and workload as lab01, but the EPP plugin config changes:
#   load-only            queue + KV-utilization scorers (no prefix awareness)
#   prefix-only          prefix-cache scorer alone (no load awareness)
#   optimized-baseline   no-k8s upstream mix (prefix x3, queue x2, kv x2, lru x2)
#   affinity-token-load  llm-d v1.0 k8s baseline (affinity filter + token-load)
# The workload is deliberately skewed (a few "hot" prefix groups) so that
# prefix-only routing creates hot spots.
#
# Then a live drain: the endpoints file shrinks from 4 to 3 replicas
# mid-run (atomic rename), and the EPP stops routing to the removed one
# without a restart.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

LAB=lab02_scheduler_ablation
require_docker_gpus
trap_stop_all

# Few groups + high concurrency = hot prefixes.
GROUPS_N="${GROUPS_N:-6}"
PREFIX_TOKENS="${PREFIX_TOKENS:-6000}"
REQUESTS="${REQUESTS:-960}"
CONCURRENCY="${CONCURRENCY:-128}"

section "Start $LLMD_REPLICAS vLLM replicas"
start_replicas
REPLICAS=$(replica_urls)
write_endpoints "$LLMD_REPLICAS"
start_envoy "$LLMD_ROOT/configs/envoy/envoy-epp.yaml"

reset_caches() {
  local i
  for (( i = 0; i < LLMD_REPLICAS; i++ )); do
    curl --noproxy '*' -s -X POST "http://127.0.0.1:$(( LLMD_BASE_PORT + i ))/reset_prefix_cache" >/dev/null || true
  done
}

for profile in load-only prefix-only optimized-baseline affinity-token-load; do
  section "EPP profile: $profile"
  if ! start_epp "$LLMD_ROOT/configs/epp/$profile.yaml"; then
    log_warn "EPP rejected $profile.yaml (plugin names differ in this EPP build?):"
    docker logs --tail 30 "${LLMD_CONTAINER_PREFIX}-epp" >&2 || true
    continue
  fi
  reset_caches
  warmup "http://127.0.0.1:$ENVOY_PORT"
  loadgen "$LAB" "$profile" --base-url "http://127.0.0.1:$ENVOY_PORT" \
    --workload shared-prefix --groups "$GROUPS_N" --prefix-tokens "$PREFIX_TOKENS" \
    --suffix-tokens 100 --output-tokens 128 \
    --tenant "name=all,concurrency=$CONCURRENCY,requests=$REQUESTS" \
    --replica-metrics "$REPLICAS" || log_warn "run failed for $profile"
  docker logs "${LLMD_CONTAINER_PREFIX}-epp" > "$RESULTS_ROOT/$LAB/epp-$profile.log" 2>&1 || true
done

section "Live pool change: drain vllm-$(( LLMD_REPLICAS - 1 )) while traffic flows"
start_epp "$LLMD_ROOT/configs/epp/optimized-baseline.yaml"
reset_caches
loadgen "$LAB" "live-drain" --base-url "http://127.0.0.1:$ENVOY_PORT" \
  --workload shared-prefix --groups 24 --prefix-tokens 4000 --output-tokens 128 \
  --tenant "name=all,concurrency=64,requests=1200" --replica-metrics "$REPLICAS" &
LG=$!
sleep 45
log "rewriting endpoints.yaml with $(( LLMD_REPLICAS - 1 )) replicas (atomic rename)"
write_endpoints "$(( LLMD_REPLICAS - 1 ))"
wait "$LG" || true
docker logs "${LLMD_CONTAINER_PREFIX}-epp" 2>&1 | grep -i "endpoints file changed" | tail -3 || true
write_endpoints "$LLMD_REPLICAS"

show_summary "$LAB"
cat <<EOF

How to read this:
  load-only           even spread (max_replica_share ~ 1/$LLMD_REPLICAS) but low prefix_hit_rate.
  prefix-only         highest hit rate, but hot groups overload a replica:
                      watch max_replica_share and ttft_p99.
  optimized-baseline  near prefix-only hit rate with load-only balance.
  affinity-token-load affinity until a replica saturates, then spills by queued
                      prompt tokens (peakPrefillThroughput gate).
  live-drain          the per-replica request counts show the drained replica
                      stopped receiving traffic after the file changed, with no failures.
EOF
