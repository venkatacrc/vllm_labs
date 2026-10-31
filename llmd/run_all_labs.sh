#!/usr/bin/env bash
# Runner for the llm-d track.
#
#   ./run_all_labs.sh                 # docker labs 00-06
#   ./run_all_labs.sh --k8s           # also the Kubernetes labs 07-09
#   ./run_all_labs.sh --only lab01,lab05
#   ./run_all_labs.sh --list
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common/env.sh"
source "$SCRIPT_DIR/common/lib.sh"

DOCKER_LABS=(lab00_setup lab01_router_basics lab02_scheduler_ablation lab03_flow_control
             lab04_tiered_kv_cache lab05_pd_disaggregation lab06_wide_ep)
K8S_LABS=(lab07_k8s_cluster lab08_k8s_well_lit_paths lab09_k8s_autoscaling_observability)

LABS=("${DOCKER_LABS[@]}")
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --k8s) LABS+=("${K8S_LABS[@]}"); shift ;;
    --only) ONLY="$2"; LABS=("${DOCKER_LABS[@]}" "${K8S_LABS[@]}"); shift 2 ;;
    --list) printf '%s\n' "${DOCKER_LABS[@]}" "${K8S_LABS[@]}"; exit 0 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) log_err "unknown arg $1"; exit 1 ;;
  esac
done

FAILED=()
for lab in "${LABS[@]}"; do
  if [[ -n "$ONLY" ]]; then
    keep=0; IFS=',' read -ra want <<< "$ONLY"
    for w in "${want[@]}"; do [[ "$lab" == "$w"* ]] && keep=1; done
    (( keep )) || continue
  fi
  section "llm-d track: $lab"
  mkdir -p "$RESULTS_ROOT/logs"
  if bash "$SCRIPT_DIR/$lab/run_lab.sh" 2>&1 | tee "$RESULTS_ROOT/logs/$lab.log"; then
    log_ok "$lab done"
  else
    log_err "$lab failed (log: $RESULTS_ROOT/logs/$lab.log)"; FAILED+=("$lab")
  fi
  stop_all
done

[[ -f "$SUMMARY_CSV" ]] && column -s, -t < "$SUMMARY_CSV"
(( ${#FAILED[@]} )) && { log_err "failed: ${FAILED[*]}"; exit 1; }
exit 0
