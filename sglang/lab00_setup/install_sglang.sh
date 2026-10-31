#!/usr/bin/env bash
# Creates a dedicated conda environment for SGLang and installs it.
#
# We deliberately use a *separate* conda environment from whatever env you
# already have vLLM/DeepSpeed/etc. in. SGLang pins specific torch / flashinfer
# / sgl-kernel wheel versions per release, and mixing those into an env that
# already has vllm's own torch/flash-attn pins is a common source of subtle
# CUDA symbol mismatches. Isolating them costs a few extra GB of disk (you
# have 27TB free) and saves hours of dependency debugging.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

section "Setting up conda env '$SGLANG_CONDA_ENV' for SGLang"

if ! command -v conda >/dev/null 2>&1; then
  log_err "conda not found on PATH. Install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -qE "^\s*${SGLANG_CONDA_ENV}\s"; then
  log "conda env '$SGLANG_CONDA_ENV' already exists, skipping creation."
else
  log "Creating conda env '$SGLANG_CONDA_ENV' (python 3.11)..."
  conda create -y -n "$SGLANG_CONDA_ENV" python=3.11
fi

conda activate "$SGLANG_CONDA_ENV"
log "Active python: $(which python) ($(python --version))"

log "Detecting CUDA toolkit version on the host..."
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VER="$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}')"
  log "nvidia-smi reports CUDA Version: ${CUDA_VER:-unknown}"
fi

python -m pip install --upgrade pip
python -m pip install -U uv

# As of SGLang's current release line, `pip install sglang` defaults to a
# CUDA 13 build, which matches this node's driver (CUDA 13.0). If you are on
# an older CUDA 12.x host, re-run with SGLANG_CUDA=cu129 to pull the CUDA 12
# compatible wheels instead (see docs.sglang.io/get_started/install for the
# up-to-date matrix -- these pins do shift between releases).
SGLANG_CUDA="${SGLANG_CUDA:-cu130}"
log "Installing sglang (target CUDA build: $SGLANG_CUDA)..."
uv pip install --prerelease=allow "sglang[all]"

if [[ "$SGLANG_CUDA" == "cu129" ]]; then
  log "Re-pinning torch + sgl-kernel for CUDA 12.9..."
  uv pip install --force-reinstall torch==2.11.0 torchaudio==2.11.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu129
  uv pip install --force-reinstall sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/
  uv pip install --force-reinstall sgl-deep-gemm --index-url https://docs.sglang.ai/whl/cu129/ --no-deps
fi

log "Installing flashinfer (attention kernels) explicitly to make sure the SM100 (Blackwell) kernels are present..."
python -m pip install --upgrade flashinfer-python --force-reinstall --no-deps || \
  log_warn "flashinfer reinstall failed; sglang will fall back to a bundled version or you can pass --attention-backend triton."
rm -rf ~/.cache/flashinfer || true

log "Installing extra packages used by the labs (hf_transfer for fast downloads, matplotlib for plots)..."
python -m pip install -U hf_transfer matplotlib pandas

section "Verifying installation"
python - <<'PY'
import torch, sglang
print("torch      :", torch.__version__, "cuda build:", torch.version.cuda)
print("sglang     :", sglang.__version__)
print("cuda avail :", torch.cuda.is_available())
print("gpu count  :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i}: {torch.cuda.get_device_name(i)} sm_{torch.cuda.get_device_capability(i)[0]}{torch.cuda.get_device_capability(i)[1]}")
PY

log_ok "SGLang installed in conda env '$SGLANG_CONDA_ENV'."
log "If sgl-kernel raises an 'expected SM100 variant' error at server start, see the README.md"
log "'Blackwell troubleshooting' section for the one-line fix (reinstall the matching cu130 sgl-kernel wheel)."
