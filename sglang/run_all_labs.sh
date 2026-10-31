#!/usr/bin/env bash
# Top-level runner for the whole SGLang optimization curriculum.
#
# Usage:
#   ./run_all_labs.sh                       # run every lab, in order
#   ./run_all_labs.sh --list                # list labs and exit
#   ./run_all_labs.sh --only lab01,lab03     # run just these labs (prefix match)
#   ./run_all_labs.sh --skip lab04,lab10     # run everything except these
#   ./run_all_labs.sh --from lab05           # resume starting at this lab
#   ./run_all_labs.sh --dry-run              # print what would run, don't run it
#
# Each lab is fully self-contained (its own run_lab.sh + README.md) and can
# also be run directly, e.g. `cd lab03_quantization && ./run_lab.sh`.
# This script just sequences them, with logging and a final summary.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common/env.sh"
source "$SCRIPT_DIR/common/lib.sh"

LABS=(
  lab00_setup
  lab01_fp32_baseline
  lab02_precision
  lab03_quantization
  lab04_tensor_parallelism
  lab05_radix_attention
  lab06_scheduling_batching
  lab07_cuda_graph_torch_compile
  lab08_speculative_decoding
  lab09_structured_output
  lab10_moe_expert_parallel
  lab11_capstone
  lab12_attention_backends
  lab13_hierarchical_cache
)

ONLY=""
SKIP=""
FROM=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      printf '%s\n' "${LABS[@]}"
      exit 0
      ;;
    --only) ONLY="$2"; shift 2 ;;
    --skip) SKIP="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      log_err "Unknown argument: $1 (use --help)"
      exit 1
      ;;
  esac
done

matches_csv() {
  # matches_csv <name> <csv> -> 0 if any csv entry is a prefix of name
  local name="$1" csv="$2" item
  IFS=',' read -ra items <<< "$csv"
  for item in "${items[@]}"; do
    [[ "$name" == "$item"* ]] && return 0
  done
  return 1
}

TO_RUN=()
STARTED=0
for lab in "${LABS[@]}"; do
  if [[ -n "$FROM" ]]; then
    if [[ "$STARTED" -eq 0 ]]; then
      matches_csv "$lab" "$FROM" && STARTED=1
      [[ "$STARTED" -eq 0 ]] && continue
    fi
  fi
  if [[ -n "$ONLY" ]] && ! matches_csv "$lab" "$ONLY"; then continue; fi
  if [[ -n "$SKIP" ]] && matches_csv "$lab" "$SKIP"; then continue; fi
  TO_RUN+=("$lab")
done

section "SGLang optimization curriculum: 8xB200, FP32 baseline -> optimized serving"
log "Labs to run (${#TO_RUN[@]}):"
printf '  - %s\n' "${TO_RUN[@]}"
log "Results will be written under: $RESULTS_ROOT"
log "Model config: SMALL_MODEL=$SMALL_MODEL MID_MODEL=$MID_MODEL MOE_MODEL=$MOE_MODEL"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Dry run requested, exiting without executing anything."
  exit 0
fi

mkdir -p "$RESULTS_ROOT/logs"
OVERALL_START=$(date +%s)
FAILED=()

for lab in "${TO_RUN[@]}"; do
  lab_dir="$SCRIPT_DIR/$lab"
  log_file="$RESULTS_ROOT/logs/${lab}.log"
  section "Running $lab  (full log: $log_file)"
  t0=$(date +%s)
  if bash "$lab_dir/run_lab.sh" 2>&1 | tee "$log_file"; then
    t1=$(date +%s)
    log_ok "$lab finished in $((t1 - t0))s"
  else
    t1=$(date +%s)
    log_err "$lab FAILED after $((t1 - t0))s -- see $log_file"
    FAILED+=("$lab")
    log_warn "Continuing to the next lab. Re-run just this one later with:"
    log_warn "  cd $lab_dir && ./run_lab.sh"
  fi
  # Belt and suspenders: make sure nothing is left running/holding GPU memory
  # between labs even if a lab's own cleanup trap didn't fire cleanly.
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  wait_for_gpu_memory_free 30
done

OVERALL_END=$(date +%s)
section "All done in $(( (OVERALL_END - OVERALL_START) / 60 ))m $(( (OVERALL_END - OVERALL_START) % 60 ))s"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  log_err "${#FAILED[@]} lab(s) failed: ${FAILED[*]}"
else
  log_ok "Every lab completed successfully."
fi

if [[ -f "$RESULTS_ROOT/summary.csv" ]]; then
  echo
  python "$SCRIPT_DIR/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv"
fi

exit $(( ${#FAILED[@]} > 0 ? 1 : 0 ))
