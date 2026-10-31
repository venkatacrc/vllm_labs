#!/usr/bin/env bash
# Helpers shared by the llm-d labs. Source after env.sh.
#
# Everything runs as Docker containers on the host network so that Envoy,
# the EPP and every vLLM replica can reach each other on 127.0.0.1, exactly
# like the upstream no-Kubernetes deployment guide.

C_BLUE='\033[0;34m'; C_GREEN='\033[0;32m'; C_YELLOW='\033[0;33m'; C_RED='\033[0;31m'; C_OFF='\033[0m'
log()      { echo -e "${C_BLUE}[$(date '+%H:%M:%S')]${C_OFF} $*"; }
log_ok()   { echo -e "${C_GREEN}[$(date '+%H:%M:%S')] OK:${C_OFF} $*"; }
log_warn() { echo -e "${C_YELLOW}[$(date '+%H:%M:%S')] WARN:${C_OFF} $*"; }
log_err()  { echo -e "${C_RED}[$(date '+%H:%M:%S')] ERROR:${C_OFF} $*" >&2; }
section()  { echo -e "\n${C_BLUE}==== $* ====${C_OFF}"; }

LLMD_CONTAINER_PREFIX="llmd-lab"

require_docker_gpus() {
  command -v docker >/dev/null || { log_err "docker not found (see lab00_setup)"; exit 1; }
  docker info >/dev/null 2>&1 || { log_err "docker daemon not reachable (are you in the docker group?)"; exit 1; }
}

# wait_http <url> [timeout_s] [name]
wait_http() {
  local url="$1" timeout="${2:-$SERVER_STARTUP_TIMEOUT}" name="${3:-$1}" waited=0
  while (( waited < timeout )); do
    if curl --noproxy '*' -sf -o /dev/null "$url"; then
      log_ok "$name is up after ${waited}s"
      return 0
    fi
    sleep 5; waited=$((waited + 5))
    (( waited % 60 == 0 )) && log "... waiting for $name (${waited}s)"
  done
  log_err "$name not ready after ${timeout}s"
  return 1
}

# gpu_set <replica_index> <gpus_per_replica> -> "0,1"
gpu_set() {
  local i="$1" n="$2" first=$(( $1 * $2 )) out=() g
  for (( g = first; g < first + n; g++ )); do out+=("$g"); done
  local IFS=,; echo "${out[*]}"
}

# start_vllm <name> <gpus csv> <port> [extra vllm serve args...]
start_vllm() {
  local name="$1" gpus="$2" port="$3"; shift 3
  local cname="${LLMD_CONTAINER_PREFIX}-${name}"
  docker rm -f "$cname" >/dev/null 2>&1 || true
  log "vLLM $name on GPUs [$gpus] port $port: $LLMD_MODEL $*"
  docker run -d --name "$cname" --network host --ipc host \
    --gpus "\"device=${gpus}\"" --shm-size 20g \
    -e HF_TOKEN="${HF_TOKEN:-}" -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME=/hf_home -v "$HF_HOME:/hf_home" \
    -e VLLM_SERVER_DEV_MODE=1 \
    ${VLLM_EXTRA_ENV:-} \
    --entrypoint vllm "$VLLM_IMAGE" \
    serve "$LLMD_MODEL" \
    --port "$port" \
    --tensor-parallel-size "${TP_OVERRIDE:-$(awk -F, '{print NF}' <<<"$gpus")}" \
    --max-model-len "$LLMD_MAX_MODEL_LEN" \
    --disable-access-log-for-endpoints=/health,/metrics,/v1/models \
    "$@" >/dev/null
}

# start_replicas [extra vllm args...] -> LLMD_REPLICAS replicas of LLMD_TP GPUs
start_replicas() {
  local i
  for (( i = 0; i < LLMD_REPLICAS; i++ )); do
    start_vllm "vllm-$i" "$(gpu_set "$i" "$LLMD_TP")" "$(( LLMD_BASE_PORT + i ))" "$@"
  done
  for (( i = 0; i < LLMD_REPLICAS; i++ )); do
    wait_http "http://127.0.0.1:$(( LLMD_BASE_PORT + i ))/v1/models" "$SERVER_STARTUP_TIMEOUT" "vllm-$i" || {
      docker logs --tail 80 "${LLMD_CONTAINER_PREFIX}-vllm-$i" >&2; return 1; }
  done
}

replica_urls() {
  local i out=()
  for (( i = 0; i < ${1:-$LLMD_REPLICAS}; i++ )); do out+=("http://127.0.0.1:$(( LLMD_BASE_PORT + i ))"); done
  local IFS=,; echo "${out[*]}"
}

# write_endpoints <n_replicas> [label_key=value ...]: writes the file-discovery
# inventory atomically (temp file + rename, as the EPP watcher expects).
write_endpoints() {
  local n="$1"; shift
  mkdir -p "$LLMD_WORK/epp"
  local tmp="$LLMD_WORK/epp/endpoints.yaml.tmp" i
  {
    echo "endpoints:"
    for (( i = 0; i < n; i++ )); do
      echo "  - name: vllm-$i"
      echo "    address: 127.0.0.1"
      echo "    port: \"$(( LLMD_BASE_PORT + i ))\""
      echo "    labels:"
      echo "      model: $LLMD_MODEL"
    done
  } > "$tmp"
  mv "$tmp" "$LLMD_WORK/epp/endpoints.yaml"
}

# start_epp <config.yaml>: stages the plugin config and runs the EPP with
# file discovery (no Kubernetes).
start_epp() {
  local cfg="$1"
  mkdir -p "$LLMD_WORK/epp"
  cp "$cfg" "$LLMD_WORK/epp/config.yaml"
  docker rm -f "${LLMD_CONTAINER_PREFIX}-epp" >/dev/null 2>&1 || true
  log "EPP with $(basename "$cfg")"
  docker run -d --name "${LLMD_CONTAINER_PREFIX}-epp" --network host \
    -v "$LLMD_WORK/epp:/etc/epp:ro" \
    "${ROUTER_EPP_IMAGE}:${ROUTER_EPP_VERSION}" \
    --config-file=/etc/epp/config.yaml \
    --pool-name=file-discovery \
    --pool-namespace=default \
    --grpc-port="$EPP_GRPC_PORT" \
    --grpc-health-port="$EPP_HEALTH_PORT" \
    --metrics-port="$EPP_METRICS_PORT" \
    --secure-serving=false \
    --health-checking=true \
    --v=2 >/dev/null
  wait_http "http://127.0.0.1:${EPP_METRICS_PORT}/metrics" 120 "EPP"
}

# start_envoy <envoy.yaml>
start_envoy() {
  local cfg="$1"
  mkdir -p "$LLMD_WORK/envoy"
  cp "$cfg" "$LLMD_WORK/envoy/envoy.yaml"
  docker rm -f "${LLMD_CONTAINER_PREFIX}-envoy" >/dev/null 2>&1 || true
  log "Envoy with $(basename "$cfg")"
  docker run -d --name "${LLMD_CONTAINER_PREFIX}-envoy" --network host \
    -v "$LLMD_WORK/envoy/envoy.yaml:/etc/envoy/envoy.yaml:ro" \
    "$ENVOY_IMAGE" \
    --service-node envoy-proxy --log-level warn --concurrency 8 \
    --drain-strategy immediate --drain-time-s 60 \
    -c /etc/envoy/envoy.yaml >/dev/null
  wait_http "http://127.0.0.1:${ENVOY_ADMIN_PORT}/ready" 60 "Envoy"
}

# gen_envoy_lb <policy> <n_replicas> <out.yaml>: a plain L7 load balancer
# (no EPP) - ROUND_ROBIN or LEAST_REQUEST - as the cache-blind baseline.
gen_envoy_lb() {
  local policy="$1" n="$2" out="$3" i
  {
    cat <<EOF
admin:
  address: { socket_address: { address: 127.0.0.1, port_value: ${ENVOY_ADMIN_PORT} } }
static_resources:
  listeners:
    - name: vllm
      address: { socket_address: { address: 0.0.0.0, port_value: ${ENVOY_PORT} } }
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: lb
                route_config:
                  virtual_hosts:
                    - name: all
                      domains: ["*"]
                      routes:
                        - match: { prefix: "/" }
                          route: { cluster: replicas, timeout: 86400s, idle_timeout: 86400s }
                http_filters:
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
  clusters:
    - name: replicas
      type: STATIC
      connect_timeout: 5s
      lb_policy: ${policy}
      circuit_breakers:
        thresholds: [{ max_connections: 40000, max_pending_requests: 40000, max_requests: 40000 }]
      load_assignment:
        cluster_name: replicas
        endpoints:
          - lb_endpoints:
EOF
    for (( i = 0; i < n; i++ )); do
      echo "              - endpoint: { address: { socket_address: { address: 127.0.0.1, port_value: $(( LLMD_BASE_PORT + i )) } } }"
    done
  } > "$out"
}

stop_router() {
  docker rm -f "${LLMD_CONTAINER_PREFIX}-envoy" "${LLMD_CONTAINER_PREFIX}-epp" >/dev/null 2>&1 || true
}

stop_all() {
  local ids
  ids=$(docker ps -aq --filter "name=${LLMD_CONTAINER_PREFIX}-" 2>/dev/null)
  [[ -n "$ids" ]] && docker rm -f $ids >/dev/null 2>&1
  pkill -f "llmd/common/pd_proxy.py" 2>/dev/null || true
  return 0
}

trap_stop_all() { trap 'stop_all' EXIT INT TERM; }

# warmup <base_url>: one tiny request so first-request costs don't pollute results
warmup() {
  curl --noproxy '*' -s -o /dev/null "$1/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$LLMD_MODEL\", \"prompt\": \"hello\", \"max_tokens\": 4}" || true
}

# loadgen <lab> <config> [loadgen args...]: drives traffic through Envoy
loadgen() {
  local lab="$1" config="$2"; shift 2
  "$PYTHON" "$LLMD_ROOT/common/loadgen.py" \
    --model "$LLMD_MODEL" --lab "$lab" --config "$config" \
    --out-dir "$RESULTS_ROOT/$lab" --summary-csv "$SUMMARY_CSV" "$@"
}

show_summary() {
  local lab="$1"
  [[ -f "$SUMMARY_CSV" ]] || return 0
  section "Results: $lab"
  { head -1 "$SUMMARY_CSV"; grep "^$lab," "$SUMMARY_CSV"; } | column -s, -t
}
