#!/usr/bin/env bash
# Lab 03 - Flow control: protect a quiet tenant from a noisy neighbour.
#
# Two tenants hit the same pool at once:
#   quiet  a latency-sensitive app: concurrency 4
#   noisy  a batch job: concurrency 384, far beyond what the pool can absorb
# Run 1: plain EPP. All requests go straight to vLLM; the noisy tenant fills
#        every replica's local queue and the quiet tenant waits behind it.
# Run 2: EPP with the flowControl feature gate. Requests are held in per-
#        tenant queues inside the EPP while the pool is saturated and
#        released round-robin across tenants, so the quiet tenant's requests
#        jump ahead of the noisy backlog.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

LAB=lab03_flow_control
require_docker_gpus
trap_stop_all

section "Start $LLMD_REPLICAS vLLM replicas"
start_replicas
REPLICAS=$(replica_urls)
write_endpoints "$LLMD_REPLICAS"
start_envoy "$LLMD_ROOT/configs/envoy/envoy-epp.yaml"

run_tenants() {
  local config="$1"
  warmup "http://127.0.0.1:$ENVOY_PORT"
  loadgen "$LAB" "$config" --base-url "http://127.0.0.1:$ENVOY_PORT" \
    --workload random --prefix-tokens 1500 --suffix-tokens 0 --output-tokens 256 \
    --flow-control-headers \
    --tenant "name=quiet,concurrency=4,requests=120" \
    --tenant "name=noisy,concurrency=384,requests=2400" \
    --replica-metrics "$REPLICAS"
}

section "Run 1: EPP without flow control"
start_epp "$LLMD_ROOT/configs/epp/optimized-baseline.yaml"
run_tenants "no-flow-control"

section "Run 2: EPP with flow control (per-tenant fair queuing)"
start_epp "$LLMD_ROOT/configs/epp/flow-control.yaml"
run_tenants "flow-control"
curl --noproxy '*' -s "http://127.0.0.1:${EPP_METRICS_PORT}/metrics" | grep -E "flow_control" | head -20 \
  > "$RESULTS_ROOT/$LAB/epp-flow-control-metrics.txt" || true

show_summary "$LAB"
cat <<EOF

Compare the 'quiet' rows: with flow control its ttft_p50/p99 should drop
dramatically while total throughput (ALL rows) stays about the same - the
backlog moved from vLLM's local queues (where it is FIFO across everyone)
into the EPP (where it is fair per tenant). EPP flow-control metrics are in
$RESULTS_ROOT/$LAB/epp-flow-control-metrics.txt.

Priorities (premium / standard / best-effort bands) are configured with
InferenceObjective resources and the x-llm-d-inference-objective header;
that part needs Kubernetes - see lab08 (flow-control guide).
EOF
