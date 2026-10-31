#!/usr/bin/env bash
# Lab 08 - The llm-d well-lit paths on Kubernetes.
#
# Deploys each upstream guide in turn (unchanged manifests from the llm-d
# repo), verifies it the way the guide does, drives load through the router
# Service with loadgen.py, then tears it down to free the GPUs.
#
#   ./run_lab.sh                                   # default set below
#   ./run_lab.sh optimized-baseline pd-disaggregation
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

LAB=lab08_k8s_well_lit_paths
GUIDES=("$@")
(( ${#GUIDES[@]} )) || GUIDES=(optimized-baseline precise-prefix-cache-routing tiered-prefix-cache pd-disaggregation flow-control)
LOCAL_PORT=18081
mkdir -p "$RESULTS_ROOT/$LAB"

model_for() { [[ "$1" == "pd-disaggregation" ]] && echo "openai/gpt-oss-120b" || echo "Qwen/Qwen3-32B"; }

pod_counters() {
  local ns="$1" guide="$2" pod
  for pod in $(kubectl get pods -n "$ns" -l llm-d.ai/inference-serving=true -o jsonpath='{.items[*].metadata.name}'); do
    echo "== $pod ($(kubectl get pod -n "$ns" "$pod" -o jsonpath='{.metadata.labels.llm-d\.ai/role}'))"
    kubectl get --raw "/api/v1/namespaces/${ns}/pods/${pod}:8000/proxy/metrics" 2>/dev/null \
      | grep -E '^(vllm:prefix_cache_(hits|queries)_total|vllm:request_success_total|vllm:external_prefix_cache_hits_total)' || true
  done
}

for guide in "${GUIDES[@]}"; do
  ns="llm-d-${guide}"
  model=$(model_for "$guide")
  section "$guide"
  MODEL="$model" bash "$SCRIPT_DIR/deploy_guide.sh" "$guide"

  kubectl port-forward -n "$ns" "service/${guide}-epp" "${LOCAL_PORT}:80" >/dev/null 2>&1 &
  PF=$!
  trap 'kill $PF 2>/dev/null || true' EXIT
  wait_http "http://127.0.0.1:${LOCAL_PORT}/v1/models" 120 "router service"

  extra=()
  [[ "$guide" == "flow-control" ]] && extra=(--flow-control-headers \
      --tenant "name=premium,concurrency=4,requests=80,objective=premium-traffic" \
      --tenant "name=batch,concurrency=256,requests=1200,objective=best-effort-traffic")
  [[ "$guide" != "flow-control" ]] && extra=(--tenant "name=all,concurrency=64,requests=480")
  workload=(--workload shared-prefix --groups 24 --prefix-tokens 4000)
  [[ "$guide" == "tiered-prefix-cache" ]] && workload=(--workload multi-turn --sessions 120 --prefix-tokens 8000)
  [[ "$guide" == "pd-disaggregation" ]] && workload=(--workload random --prefix-tokens 8000 --suffix-tokens 0)

  LLMD_MODEL="$model" loadgen "$LAB" "$guide" --base-url "http://127.0.0.1:${LOCAL_PORT}" \
    "${workload[@]}" --output-tokens 128 "${extra[@]}" || log_warn "load run failed for $guide"
  pod_counters "$ns" "$guide" | tee "$RESULTS_ROOT/$LAB/${guide}-pod-metrics.txt"
  kubectl logs -n "$ns" "deployment/${guide}-epp" -c epp --tail 200 > "$RESULTS_ROOT/$LAB/${guide}-epp.log" 2>&1 || true

  kill $PF 2>/dev/null || true
  if [[ "${KEEP:-0}" != "1" ]]; then bash "$SCRIPT_DIR/deploy_guide.sh" "$guide" --teardown; fi
done

show_summary "$LAB"
echo "Per-pod prefix-cache/request counters: $RESULTS_ROOT/$LAB/*-pod-metrics.txt"
echo "Set KEEP=1 to leave a guide running for exploration (kubectl get all -n llm-d-<guide>)."
