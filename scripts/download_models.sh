#!/usr/bin/env bash
# Pre-fetch every model used by the labs into HF_HOME so lab runs aren't
# blocked on downloads mid-benchmark (some labs time server startup and
# will otherwise appear to hang while a 32B checkpoint downloads).
#
# Usage:
#   export HF_TOKEN=hf_...          # required for the gated meta-llama repos
#   bash scripts/download_models.sh
set -euo pipefail

: "${HF_HOME:=/raid/llm_rl_cache/hf_home}"
export HF_HOME
mkdir -p "$HF_HOME"
echo "Using HF_HOME=$HF_HOME"

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo ""
  echo "WARNING: HF_TOKEN is not set. The gated meta-llama/* models below will fail to download."
  echo "  1) Accept the license at:"
  echo "       https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
  echo "       https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"
  echo "  2) Create a token at https://huggingface.co/settings/tokens"
  echo "  3) export HF_TOKEN=hf_xxx"
  echo ""
fi

if command -v hf >/dev/null 2>&1; then
  DL_CMD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  DL_CMD=(huggingface-cli download)
else
  echo "Neither 'hf' nor 'huggingface-cli' found on PATH." >&2
  echo "Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
  "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
)

FAILED=()
for m in "${MODELS[@]}"; do
  echo ""
  echo "=== Downloading $m ==="
  if ! "${DL_CMD[@]}" "$m" --exclude "*.pth" "original/*"; then
    echo "!!! Failed to download $m (continuing with the rest)"
    FAILED+=("$m")
  fi
done

echo ""
echo "Done. Models cached under $HF_HOME"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "The following model(s) failed to download (labs that need them will fail too):"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
