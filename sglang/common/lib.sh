#!/usr/bin/env bash
# Shared bash helpers for launching/benchmarking/tearing down SGLang servers.
# Source this AFTER common/env.sh from every lab's run_lab.sh:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"
#   source "$LABS_ROOT/common/lib.sh"

set -uo pipefail

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
COLOR_BLUE='\033[1;34m'
COLOR_GREEN='\033[1;32m'
COLOR_RED='\033[1;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_RESET='\033[0m'

log()      { echo -e "${COLOR_BLUE}[$(date '+%H:%M:%S')]${COLOR_RESET} $*"; }
log_ok()   { echo -e "${COLOR_GREEN}[$(date '+%H:%M:%S')] OK:${COLOR_RESET} $*"; }
log_warn() { echo -e "${COLOR_YELLOW}[$(date '+%H:%M:%S')] WARN:${COLOR_RESET} $*"; }
log_err()  { echo -e "${COLOR_RED}[$(date '+%H:%M:%S')] ERROR:${COLOR_RESET} $*" >&2; }
section()  { echo -e "\n${COLOR_BLUE}=================================================================${COLOR_RESET}"; echo -e "${COLOR_BLUE}  $*${COLOR_RESET}"; echo -e "${COLOR_BLUE}=================================================================${COLOR_RESET}"; }

# ---------------------------------------------------------------------------
# Conda helpers
# ---------------------------------------------------------------------------
# Runs a command inside the sglang-labs conda env without requiring the
# caller's shell to already have conda activated.
run_in_sglang_env() {
  local conda_base
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -z "$conda_base" ]]; then
    log_err "conda not found on PATH. Install miniconda/anaconda first (see lab00)."
    return 1
  fi
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$SGLANG_CONDA_ENV"
  "$@"
}

require_sglang_env() {
  local conda_base
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -z "$conda_base" ]]; then
    log_err "conda not found. Run lab00_setup first."
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  if ! conda env list | grep -qE "^\s*${SGLANG_CONDA_ENV}\s"; then
    log_err "conda env '$SGLANG_CONDA_ENV' not found. Run lab00_setup/run_lab.sh first."
    exit 1
  fi
  conda activate "$SGLANG_CONDA_ENV"
  if ! python -c "import sglang" >/dev/null 2>&1; then
    log_err "sglang is not importable inside '$SGLANG_CONDA_ENV'. Run lab00_setup/run_lab.sh first."
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
gpu_mem_used_mb() {
  # Sum of memory.used (MiB) across all visible GPUs.
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}'
}

require_gpus() {
  local needed="$1"
  local have
  have="$(nvidia-smi -L | wc -l | tr -d ' ')"
  if (( have < needed )); then
    log_err "This lab needs $needed GPU(s) but only $have are visible. Set CUDA_VISIBLE_DEVICES or reduce TP/DP size."
    exit 1
  fi
}

# Wait until the GPUs used by the last server are (mostly) freed so back to
# back labs don't OOM each other.
wait_for_gpu_memory_free() {
  local max_wait="${1:-60}"
  local waited=0
  while (( waited < max_wait )); do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)"
    if (( used < 2000 )); then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  log_warn "GPUs still show ${used:-?}MiB used after ${max_wait}s; continuing anyway."
}

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
SGLANG_SERVER_PID=""
SGLANG_SERVER_LOG=""

# start_sglang_server <log_file> <extra launch_server args...>
# Launches `python -m sglang.launch_server` in the background inside the
# sglang-labs conda env, streams logs to <log_file>, and blocks until the
# /health endpoint responds (or SERVER_STARTUP_TIMEOUT is hit).
start_sglang_server() {
  local log_file="$1"; shift
  local conda_base
  conda_base="$(conda info --base 2>/dev/null)"
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$SGLANG_CONDA_ENV"

  log "Launching: python -m sglang.launch_server --host $SGLANG_HOST --port $SGLANG_PORT $*"
  mkdir -p "$(dirname "$log_file")"
  nohup python -m sglang.launch_server \
    --host "$SGLANG_HOST" --port "$SGLANG_PORT" \
    "$@" \
    > "$log_file" 2>&1 &
  SGLANG_SERVER_PID=$!
  SGLANG_SERVER_LOG="$log_file"
  log "Server PID=$SGLANG_SERVER_PID, logging to $log_file"

  local waited=0
  while (( waited < SERVER_STARTUP_TIMEOUT )); do
    if ! kill -0 "$SGLANG_SERVER_PID" 2>/dev/null; then
      log_err "Server process died during startup. Tail of $log_file:"
      tail -n 60 "$log_file" >&2 || true
      return 1
    fi
    if curl -s -o /dev/null -w "%{http_code}" "http://$SGLANG_HOST:$SGLANG_PORT/health" 2>/dev/null | grep -q "200"; then
      log_ok "Server is healthy after ${waited}s"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
    if (( waited % 30 == 0 )); then
      log "... still waiting for server to come up (${waited}s elapsed, timeout ${SERVER_STARTUP_TIMEOUT}s)"
    fi
  done

  log_err "Server did not become healthy within ${SERVER_STARTUP_TIMEOUT}s. Tail of $log_file:"
  tail -n 60 "$log_file" >&2 || true
  return 1
}

# stop_sglang_server: kill the server started by start_sglang_server and wait
# for GPU memory to be released.
stop_sglang_server() {
  if [[ -n "$SGLANG_SERVER_PID" ]] && kill -0 "$SGLANG_SERVER_PID" 2>/dev/null; then
    log "Stopping server PID=$SGLANG_SERVER_PID"
    kill "$SGLANG_SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SGLANG_SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$SGLANG_SERVER_PID" 2>/dev/null || true
  fi
  # Belt-and-suspenders: SGLang spawns scheduler/detokenizer subprocesses that
  # do not always die with the parent.
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  pkill -9 -f "sglang::scheduler" 2>/dev/null || true
  SGLANG_SERVER_PID=""
  wait_for_gpu_memory_free 60
}

# Ensure the server is always stopped, even if the lab script errors out.
trap_stop_server() {
  trap 'stop_sglang_server' EXIT INT TERM
}

# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
# run_bench <output_jsonl_path> <extra bench_serving args...>
run_bench() {
  local out_file="$1"; shift
  mkdir -p "$(dirname "$out_file")"
  log "Running bench_serving -> $out_file"
  python -m sglang.bench_serving \
    --backend sglang \
    --host "$SGLANG_HOST" --port "$SGLANG_PORT" \
    --output-file "$out_file" \
    "$@" 2>&1 | tee "${out_file%.jsonl}.log"
}

record_result() {
  # record_result <lab> <config_label> <jsonl_file> [extra_json_fields...]
  local lab="$1" config="$2" jsonl="$3"; shift 3
  python "$LABS_ROOT/common/bench_lib.py" record \
    --lab "$lab" --config "$config" --jsonl "$jsonl" \
    --summary-csv "$RESULTS_ROOT/summary.csv" "$@"
}

# record_result_raw <lab> <config_label> [--metric k=v ...] [--extra k=v ...]
record_result_raw() {
  local lab="$1" config="$2"; shift 2
  python "$LABS_ROOT/common/bench_lib.py" record-raw \
    --lab "$lab" --config "$config" \
    --summary-csv "$RESULTS_ROOT/summary.csv" "$@"
}
