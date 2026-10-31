#!/usr/bin/env bash
# Deploy (or tear down) one upstream llm-d guide on the current cluster,
# following the guide's README commands unchanged.
#
#   ./deploy_guide.sh <guide> [--teardown]
# guides: optimized-baseline precise-prefix-cache-routing tiered-prefix-cache
#         pd-disaggregation predicted-latency-routing flow-control
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

export GUIDE_NAME="${1:?usage: deploy_guide.sh <guide> [--teardown]}"
ACTION="${2:-deploy}"
export REPO_ROOT="$LLMD_UPSTREAM_DIR"
export NAMESPACE="llm-d-${GUIDE_NAME}"
export ACCELERATOR_TYPE=gpu
export MODEL_SERVER="${MODEL_SERVER:-vllm}"
export INFRA_PROVIDER=base
export MONITORING="${MONITORING:-false}"
export MONITORING_VALUES=""
[[ "$MONITORING" == "true" ]] && MONITORING_VALUES="-f ${REPO_ROOT}/guides/recipes/router/features/monitoring.values.yaml"
export MODEL="${MODEL:-Qwen/Qwen3-32B}"
export ROUTER_BASE_VALUES="${REPO_ROOT}/guides/recipes/router/base.values.yaml"
export ROUTER_VALUES="${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml"
MODELSERVER_DIR="${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}/${INFRA_PROVIDER}/"
EXTRA_VALUES=""

case "$GUIDE_NAME" in
  optimized-baseline|precise-prefix-cache-routing) ;;
  tiered-prefix-cache)
    export CONNECTOR="${CONNECTOR:-native}" VARIANT="${VARIANT:-cpu}"
    ROUTER_VALUES="${REPO_ROOT}/guides/${GUIDE_NAME}/router/single-host/${GUIDE_NAME}-cpu.values.yaml"
    MODELSERVER_DIR="${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/${MODEL_SERVER}/${CONNECTOR}/${VARIANT}/${INFRA_PROVIDER}/" ;;
  pd-disaggregation)
    export MODEL="${MODEL_PD:-openai/gpt-oss-120b}" ;;
  predicted-latency-routing)
    ROUTER_VALUES="${REPO_ROOT}/guides/${GUIDE_NAME}/router/predicted-latency.values.yaml" ;;
  flow-control)
    export MODEL_NAME="$MODEL" ;;
  *) log_err "unknown guide $GUIDE_NAME"; exit 1 ;;
esac
[[ "$MODEL_SERVER" == "sglang" && "$GUIDE_NAME" == "precise-prefix-cache-routing" ]] && \
  EXTRA_VALUES="-f ${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.sglang.values.yaml"

source "${REPO_ROOT}/guides/env.sh"

if [[ "$ACTION" == "--teardown" ]]; then
  section "Teardown $GUIDE_NAME"
  helm uninstall "$GUIDE_NAME" -n "$NAMESPACE" 2>/dev/null || true
  kubectl delete namespace "$NAMESPACE" --wait=true --ignore-not-found
  exit 0
fi

section "Deploy guide: $GUIDE_NAME (model $MODEL, engine $MODEL_SERVER) into $NAMESPACE"
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic llm-d-hf-token --from-literal="HF_TOKEN=${HF_TOKEN:-}" \
  --namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

if [[ "$GUIDE_NAME" == "pd-disaggregation" ]]; then
  kubectl apply -n "$NAMESPACE" -f "${REPO_ROOT}/guides/recipes/modelserver/components/model-cache/model-cache-pvc.yaml"
fi

log "router: helm ${ROUTER_STANDALONE_CHART} --version ${ROUTER_CHART_VERSION}"
# shellcheck disable=SC2086
helm upgrade --install "$GUIDE_NAME" "$ROUTER_STANDALONE_CHART" \
  -f "$ROUTER_BASE_VALUES" $MONITORING_VALUES -f "$ROUTER_VALUES" $EXTRA_VALUES \
  -n "$NAMESPACE" --version "$ROUTER_CHART_VERSION"

log "model servers"
if [[ "$GUIDE_NAME" == "flow-control" ]]; then
  kubectl kustomize "${REPO_ROOT}/guides/optimized-baseline/modelserver/gpu/vllm/${INFRA_PROVIDER}/" \
    | sed "s/optimized-baseline/${GUIDE_NAME}/g" | kubectl apply -n "$NAMESPACE" -f -
  kubectl apply -f "${REPO_ROOT}/guides/${GUIDE_NAME}/objectives.yaml" -n "$NAMESPACE"
else
  kubectl apply -n "$NAMESPACE" -k "$MODELSERVER_DIR"
fi
if [[ "$MONITORING" == "true" ]]; then
  kubectl apply -n "$NAMESPACE" -k "${REPO_ROOT}/guides/recipes/modelserver/components/monitoring"
fi

log "waiting for model-server pods (first start downloads weights and compiles; can take 10-20 min)"
kubectl wait pods -n "$NAMESPACE" -l llm-d.ai/inference-serving=true --for=condition=Ready --timeout=40m
kubectl get pods -n "$NAMESPACE" -o wide
log_ok "$GUIDE_NAME is up"
