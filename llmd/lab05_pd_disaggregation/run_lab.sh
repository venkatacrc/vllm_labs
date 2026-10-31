#!/usr/bin/env bash
# Lab 05 - Prefill/decode disaggregation with NIXL KV transfer.
#
# Upstream llm-d topology for NVIDIA: openai/gpt-oss-120b, 1 prefill
# server (TP=1) + 1 decode server (TP=4), KV moved by vLLM's NixlConnector
# (UCX; on one node this rides NVLink via cuda_ipc). The proxy
# (common/pd_proxy.py) plays the role of the llm-d router + routing sidecar.
#
# Runs:
#   aggregated-tp4   one normal vLLM server, TP=4 (GPUs 0-3)
#   pd-1p1d          prefill TP=1 (GPU 4) + decode TP=4 (GPUs 0-3)
#   pd-2p1d          two prefill servers (GPUs 4,5) + the same decoder
# on a prefill-heavy, long-prompt workload where prefill normally
# interrupts decode.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

LAB=lab05_pd_disaggregation
require_docker_gpus
trap_stop_all

export LLMD_MODEL="${PD_MODEL:-openai/gpt-oss-120b}"
export LLMD_MAX_MODEL_LEN="${PD_MAX_MODEL_LEN:-16384}"
IN_TOKENS="${IN_TOKENS:-8000}"
OUT_TOKENS="${OUT_TOKENS:-256}"
CONCURRENCY="${CONCURRENCY:-32}"
REQUESTS="${REQUESTS:-480}"
PROXY_PORT=8081
COMMON_ARGS=(--block-size=128 --no-disable-hybrid-kv-cache-manager)
NIXL_ROLE() { echo "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$1\",\"kv_buffer_device\":\"cuda\",\"kv_connector_extra_config\":{\"backends\":[\"UCX\"]}}"; }

run_load() {
  local config="$1" url="$2" metrics="$3"
  warmup "$url"
  loadgen "$LAB" "$config" --base-url "$url" \
    --workload random --prefix-tokens "$IN_TOKENS" --suffix-tokens 0 --output-tokens "$OUT_TOKENS" \
    --tenant "name=all,concurrency=$CONCURRENCY,requests=$REQUESTS" \
    --replica-metrics "$metrics"
}

nixl_env() { echo "-e VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 -e VLLM_NIXL_SIDE_CHANNEL_PORT=$1 -e UCX_TLS=all -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120"; }

section "Run 1: aggregated, TP=4"
start_vllm agg "0,1,2,3" 8200 "${COMMON_ARGS[@]}"
wait_http "http://127.0.0.1:8200/v1/models" "$SERVER_STARTUP_TIMEOUT" "aggregated"
run_load "aggregated-tp4" "http://127.0.0.1:8200" "http://127.0.0.1:8200"
stop_all

start_pd() {
  local n_prefill="$1" i
  VLLM_EXTRA_ENV="$(nixl_env 5700)" start_vllm decode "0,1,2,3" 8200 "${COMMON_ARGS[@]}" \
    --kv-transfer-config "$(NIXL_ROLE kv_consumer)"
  for (( i = 0; i < n_prefill; i++ )); do
    VLLM_EXTRA_ENV="$(nixl_env $(( 5600 - 50 * i )))" start_vllm "prefill-$i" "$(( 4 + i ))" "$(( 8100 + i ))" \
      "${COMMON_ARGS[@]}" --kv-transfer-config "$(NIXL_ROLE kv_producer)"
  done
  wait_http "http://127.0.0.1:8200/v1/models" "$SERVER_STARTUP_TIMEOUT" "decode"
  local pargs=()
  for (( i = 0; i < n_prefill; i++ )); do
    wait_http "http://127.0.0.1:$(( 8100 + i ))/v1/models" "$SERVER_STARTUP_TIMEOUT" "prefill-$i"
    pargs+=(--prefill "http://127.0.0.1:$(( 8100 + i ))")
  done
  "$PYTHON" "$LLMD_ROOT/common/pd_proxy.py" --port "$PROXY_PORT" "${pargs[@]}" \
    --decode http://127.0.0.1:8200 > "$RESULTS_ROOT/$LAB/proxy.log" 2>&1 &
  wait_http "http://127.0.0.1:$PROXY_PORT/health" 60 "P/D proxy"
}

mkdir -p "$RESULTS_ROOT/$LAB"
section "Run 2: P/D 1 prefill (TP=1) + 1 decode (TP=4)"
start_pd 1
run_load "pd-1p1d" "http://127.0.0.1:$PROXY_PORT" "http://127.0.0.1:8200,http://127.0.0.1:8100"
docker logs "${LLMD_CONTAINER_PREFIX}-decode" 2>&1 | grep -iE "nixl|kv transfer" | tail -5 || true
stop_all

section "Run 3: P/D 2 prefill + 1 decode"
start_pd 2
run_load "pd-2p1d" "http://127.0.0.1:$PROXY_PORT" "http://127.0.0.1:8200,http://127.0.0.1:8100,http://127.0.0.1:8101"
stop_all

show_summary "$LAB"
cat <<EOF

What to compare (prompts ~$IN_TOKENS tokens, outputs $OUT_TOKENS tokens):
- tpot_p50_ms: decode-only servers never pause decoding to run someone
  else's 8K prefill, so time-per-output-token is lower and steadier with P/D.
- ttft_p50_ms: P/D adds a KV transfer hop; with one TP=1 prefiller it can be
  *worse* than aggregated once prefill saturates - pd-2p1d shows how scaling
  the prefill pool independently fixes that.
- GPUs used: 4 vs 5 vs 6. Normalise out_tok_per_s by GPU count before
  declaring a winner. P/D pays off for long prompts, large models and
  strict TPOT SLOs; for short prompts aggregated serving usually wins.
EOF
