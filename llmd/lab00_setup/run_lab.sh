#!/usr/bin/env bash
# Lab 00 - Setup for the llm-d track.
#
# Checks Docker + GPU access, pulls the pinned images (vLLM, Envoy, llm-d
# EPP), pre-downloads the model, clones the llm-d repo for the Kubernetes
# labs and (optionally) installs the k8s client tools the guides expect.
#
#   ./run_lab.sh                      # docker labs prerequisites
#   ./run_lab.sh --with-k8s-tools     # also kubectl/helm/kustomize/yq/helmfile
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

WITH_K8S=0
[[ "${1:-}" == "--with-k8s-tools" ]] && WITH_K8S=1

section "1/5 Docker and GPU access"
require_docker_gpus
docker run --rm --gpus all --entrypoint nvidia-smi "$VLLM_IMAGE" -L \
  || { log_err "GPU containers failed. Install the NVIDIA Container Toolkit and run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"; exit 1; }
log_ok "containers can see the GPUs"

section "2/5 Pull images"
for img in "$VLLM_IMAGE" "$ENVOY_IMAGE" "${ROUTER_EPP_IMAGE}:${ROUTER_EPP_VERSION}"; do
  log "docker pull $img"; docker pull -q "$img"
done
docker pull -q "$SGLANG_IMAGE" || log_warn "SGLang image pull failed (only needed for the SGLang variant of the k8s guides)"

section "3/5 Python client deps (loadgen, P/D proxy)"
"$PYTHON" -c "import aiohttp" 2>/dev/null || "$PYTHON" -m pip install -q aiohttp
log_ok "aiohttp available in $("$PYTHON" -c 'import sys; print(sys.executable)')"

section "4/5 Download models into $HF_HOME"
MODELS=("$LLMD_MODEL" "openai/gpt-oss-120b" "Qwen/Qwen3-30B-A3B")
for m in "${MODELS[@]}"; do
  log "download $m"
  docker run --rm -e HF_TOKEN="${HF_TOKEN:-}" -e HF_HOME=/hf_home -v "$HF_HOME:/hf_home" \
    --entrypoint hf "$VLLM_IMAGE" download "$m" --exclude "original/*" >/dev/null \
    || log_warn "download of $m failed (set HF_TOKEN if gated); the lab using it will download on first start"
done

section "5/5 llm-d repo for the Kubernetes labs ($LLMD_UPSTREAM_REF)"
if [[ -d "$LLMD_UPSTREAM_DIR/.git" ]]; then
  git -C "$LLMD_UPSTREAM_DIR" fetch -q origin && git -C "$LLMD_UPSTREAM_DIR" checkout -q "$LLMD_UPSTREAM_REF" \
    && git -C "$LLMD_UPSTREAM_DIR" pull -q --ff-only || true
else
  git clone -q https://github.com/llm-d/llm-d.git "$LLMD_UPSTREAM_DIR"
  git -C "$LLMD_UPSTREAM_DIR" checkout -q "$LLMD_UPSTREAM_REF"
fi
log_ok "llm-d at $(git -C "$LLMD_UPSTREAM_DIR" log -1 --format='%h %cd' --date=short)"

if (( WITH_K8S )); then
  section "k8s client tools (upstream helpers/client-setup/install-deps.sh)"
  bash "$LLMD_UPSTREAM_DIR/helpers/client-setup/install-deps.sh"
fi
for t in kubectl helm kustomize yq helmfile minikube; do
  if command -v "$t" >/dev/null; then log_ok "$t: $(command -v "$t")"; else log_warn "$t not installed (needed only for labs 07-09)"; fi
done

log_ok "Setup complete. Next: cd ../lab01_router_basics && ./run_lab.sh"
