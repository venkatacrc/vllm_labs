#!/usr/bin/env bash
# Lab 07 - A single-node Kubernetes cluster with GPUs for the llm-d guides.
#
# minikube (docker driver) with all 8 B200s passed through, then the
# cluster-wide prerequisites the guides share:
#   - Gateway API Inference Extension CRDs (InferencePool, ...)
#   - llm-d router CRDs (InferenceObjective, used by flow control)
#   - LeaderWorkerSet controller with DisaggregatedSet (P/D, wide-EP)
#   - KEDA (autoscaling, lab09)
#
#   ./run_lab.sh            create/verify the cluster
#   ./run_lab.sh --delete   tear it down
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

PROFILE="${MINIKUBE_PROFILE:-llmd}"
if [[ "${1:-}" == "--delete" ]]; then
  minikube delete -p "$PROFILE"; exit 0
fi

for t in minikube kubectl helm kustomize yq; do
  command -v "$t" >/dev/null || { log_err "$t missing: run ../lab00_setup/run_lab.sh --with-k8s-tools (minikube: https://minikube.sigs.k8s.io/docs/start/)"; exit 1; }
done
[[ -d "$LLMD_UPSTREAM_DIR" ]] || { log_err "llm-d repo not cloned; run lab00 first"; exit 1; }
export REPO_ROOT="$LLMD_UPSTREAM_DIR"
source "$REPO_ROOT/guides/env.sh"

section "1/5 minikube with GPUs (profile $PROFILE)"
if ! minikube status -p "$PROFILE" >/dev/null 2>&1; then
  # --gpus all needs the NVIDIA Container Toolkit configured for docker.
  # Give the node most of the host: model servers request 96-128 GiB each.
  minikube start -p "$PROFILE" --driver docker --container-runtime docker --gpus all \
    --cpus "${MINIKUBE_CPUS:-96}" --memory "${MINIKUBE_MEMORY:-1200g}" \
    --disk-size "${MINIKUBE_DISK:-2000g}" \
    --mount --mount-string "$HF_HOME:/var/cache/huggingface" \
    --kubernetes-version "${K8S_VERSION:-stable}"
fi
kubectl config use-context "$PROFILE"
log "waiting for the NVIDIA device plugin to advertise GPUs"
for _ in $(seq 1 60); do
  n=$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || true)
  [[ -n "$n" && "$n" != "0" ]] && break; sleep 5
done
log_ok "node allocatable nvidia.com/gpu=${n:-0}"

section "2/5 Gateway API Inference Extension CRDs ($GAIE_URL)"
kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/${GAIE_URL}/v1-manifests.yaml"

section "3/5 llm-d router CRDs (InferenceObjective)"
kubectl apply -f "https://github.com/llm-d/llm-d-router/${ROUTER_RELEASE_URL}/manifests.yaml"

section "4/5 LeaderWorkerSet $LWS_VERSION with DisaggregatedSet"
tmp=$(mktemp -d)
(cd "$tmp" && helm pull oci://registry.k8s.io/lws/charts/lws --version="${LWS_VERSION}" --untar)
kubectl apply --server-side --force-conflicts -f "$tmp/lws/crds"
helm upgrade --install lws oci://registry.k8s.io/lws/charts/lws \
  --version="${LWS_VERSION}" --namespace lws-system --create-namespace \
  --set enableDisaggregatedSet=true --wait --timeout 300s
rm -rf "$tmp"

section "5/5 KEDA"
helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1 || true
helm repo update kedacore >/dev/null
helm upgrade --install keda kedacore/keda --namespace keda --create-namespace --wait

kubectl get nodes -o wide
kubectl get crd | grep -E "inference|leaderworkerset|disaggregatedset|keda" || true
log_ok "Cluster ready. Next: ../lab08_k8s_well_lit_paths/run_lab.sh optimized-baseline"
