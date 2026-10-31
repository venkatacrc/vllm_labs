#!/usr/bin/env bash
# Pre-downloads every model used across the curriculum into $HF_HOME so that
# later labs don't spend their first few minutes (or hit flaky network
# issues) downloading weights while you're trying to look at a benchmark.
#
# Total download size is roughly:
#   Qwen2.5-7B-Instruct            ~15 GB
#   Qwen2.5-7B-Instruct-AWQ        ~5.5 GB
#   Qwen2.5-7B-Instruct-GPTQ-Int4  ~5.5 GB
#   Qwen2.5-32B-Instruct           ~65 GB
#   Qwen3-30B-A3B                  ~60 GB
# ~150GB total -- trivial for the 27TB free on /raid.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env

MODELS=("$SMALL_MODEL" "$AWQ_MODEL" "$GPTQ_MODEL" "$MID_MODEL" "$MOE_MODEL")

section "Downloading ${#MODELS[@]} models into \$HF_HOME=$HF_HOME"

for m in "${MODELS[@]}"; do
  log "Downloading $m ..."
  python -c "
from huggingface_hub import snapshot_download
import sys
path = snapshot_download(repo_id='$m', ignore_patterns=['*.pth', 'original/*', '*.gguf'])
print('  -> cached at', path)
" || log_warn "Failed to download $m (will be re-attempted on first use by SGLang itself)."
done

log_ok "Model downloads complete. Cache size:"
du -sh "$HF_HOME" 2>/dev/null || true
