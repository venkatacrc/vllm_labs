#!/usr/bin/env bash
# Sanity check: boot the small model briefly with default settings and send
# one request through the OpenAI-compatible API, then shut it down. If this
# passes, every other lab in the curriculum should work.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 1
trap_stop_server

section "Smoke test: launching $SMALL_MODEL on 1 GPU"

LOG_DIR="$RESULTS_ROOT/lab00_setup"
mkdir -p "$LOG_DIR"

start_sglang_server "$LOG_DIR/server.log" \
  --model-path "$SMALL_MODEL" \
  --tp 1

log "Sending a test chat completion..."
RESPONSE=$(curl -s "http://$SGLANG_HOST:$SGLANG_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "'"$SMALL_MODEL"'",
        "messages": [{"role": "user", "content": "In one sentence, what is tensor parallelism?"}],
        "max_tokens": 64,
        "temperature": 0
      }')

echo "$RESPONSE" | python -m json.tool || { log_err "Server did not return valid JSON: $RESPONSE"; exit 1; }

if echo "$RESPONSE" | python -c "import json,sys; d=json.load(sys.stdin); assert d['choices'][0]['message']['content'].strip()" 2>/dev/null; then
  log_ok "Smoke test passed! SGLang is correctly installed and serving requests."
else
  log_err "Smoke test failed - response did not contain expected content."
  exit 1
fi
