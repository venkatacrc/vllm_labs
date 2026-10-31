#!/usr/bin/env bash
# Lab 09 - Observability and SLO-aware autoscaling (KEDA + EPP metrics).
#
# 1. Prometheus + Grafana (upstream recipe) and the llm-d dashboards.
# 2. optimized-baseline with monitoring enabled (ServiceMonitors for EPP + vLLM).
# 3. A KEDA ScaledObject that scales the vLLM Deployment on the EPP's
#    in-flight request metric (upstream guides/workload-autoscaling/keda-epp).
# 4. Load ramps up -> watch replicas go 1 -> N; load stops -> scale back.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

LAB=lab09_k8s_autoscaling_observability
mkdir -p "$RESULTS_ROOT/$LAB"
export REPO_ROOT="$LLMD_UPSTREAM_DIR"
source "$REPO_ROOT/guides/env.sh"
export NAMESPACE=llm-d-optimized-baseline
export MONITORING_NAMESPACE=llm-d-monitoring
export MODEL=Qwen/Qwen3-32B
export TARGET_DEPLOYMENT=optimized-baseline-nvidia-gpu-vllm-decode
export SCALEDOBJECT_NAME=optimized-baseline-keda-epp
export HPA_NAME=keda-hpa-optimized-baseline
export EPP_SERVICE=optimized-baseline-epp
export INFERENCE_POOL=optimized-baseline
OVERLAY_ROOT="${REPO_ROOT}/guides/workload-autoscaling/keda-epp/optimized-baseline"
LOCAL_PORT=18081

section "1/4 Prometheus + Grafana + llm-d dashboards"
(cd "$REPO_ROOT" && ./guides/recipes/observability/install-prometheus-grafana.sh)
(cd "$REPO_ROOT" && ./guides/recipes/observability/load-llm-d-dashboards.sh) || log_warn "dashboard load failed"

section "2/4 optimized-baseline with monitoring"
MONITORING=true MODEL="$MODEL" bash "$SCRIPT_DIR/../lab08_k8s_well_lit_paths/deploy_guide.sh" optimized-baseline

section "3/4 KEDA ScaledObject on EPP in-flight requests"
kubectl scale deployment "$TARGET_DEPLOYMENT" -n "$NAMESPACE" --replicas=1
kubectl rollout status "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --timeout=15m
kubectl kustomize "${OVERLAY_ROOT}/overlays/k8s/inflight" \
  | envsubst '$NAMESPACE $MODEL $TARGET_DEPLOYMENT $EPP_SERVICE $INFERENCE_POOL' | kubectl apply -f -
kubectl wait --for=condition=Ready "scaledobject/${SCALEDOBJECT_NAME}" -n "$NAMESPACE" --timeout=120s || true
kubectl get scaledobject,hpa -n "$NAMESPACE"

section "4/4 Load ramp and scale timeline"
kubectl port-forward -n "$NAMESPACE" "service/${EPP_SERVICE}" "${LOCAL_PORT}:80" >/dev/null 2>&1 &
PF=$!
trap 'kill $PF ${WATCH:-} 2>/dev/null || true' EXIT
wait_http "http://127.0.0.1:${LOCAL_PORT}/v1/models" 120 "router"

TIMELINE="$RESULTS_ROOT/$LAB/replica_timeline.csv"
echo "elapsed_s,desired,ready" > "$TIMELINE"
( t0=$(date +%s); while true; do
    d=$(kubectl get deploy "$TARGET_DEPLOYMENT" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    r=$(kubectl get deploy "$TARGET_DEPLOYMENT" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
    echo "$(( $(date +%s) - t0 )),${d:-0},${r:-0}" >> "$TIMELINE"; sleep 10; done ) &
WATCH=$!

LLMD_MODEL="$MODEL" loadgen "$LAB" "ramp-high" --base-url "http://127.0.0.1:${LOCAL_PORT}" \
  --workload shared-prefix --groups 32 --prefix-tokens 3000 --output-tokens 256 \
  --tenant "name=all,concurrency=192,requests=6000" || true
log "load finished; watching scale-down for ${COOLDOWN_WATCH_S:-600}s"
sleep "${COOLDOWN_WATCH_S:-600}"
kill $WATCH 2>/dev/null || true

column -s, -t < "$TIMELINE" | tail -40
kubectl get hpa "$HPA_NAME" -n "$NAMESPACE" -o jsonpath='{.status.currentMetrics}' ; echo
cat <<EOF

The timeline shows desired vs ready replicas. Note the gap: a new vLLM pod
needs minutes (image pull, weight load, torch.compile, CUDA graphs) before
it is Ready. That startup latency, not the metric, usually dominates
autoscaling behaviour for LLMs - hence fast model actuation (sleep/wake),
weight caching on node-local disk, and scaling on *leading* signals
(in-flight/queued requests at the EPP) rather than GPU utilization.

Grafana: kubectl port-forward -n llm-d-monitoring svc/llmd-grafana 3000:80
(admin/admin) -> llm-d dashboards. Clean up with:
  bash ../lab08_k8s_well_lit_paths/deploy_guide.sh optimized-baseline --teardown
EOF
