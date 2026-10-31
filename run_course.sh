#!/usr/bin/env bash
# Run the whole course (or one track) on the 8xB200 node.
#
#   ./run_course.sh                 # vllm, then sglang, then llm-d (docker labs)
#   ./run_course.sh vllm [args]     # python run_all_labs.py [args]
#   ./run_course.sh sglang [args]   # sglang/run_all_labs.sh [args]
#   ./run_course.sh llmd [args]     # llmd/run_all_labs.sh [args]  (--k8s for labs 07-09)
#   ./run_course.sh --list
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_track() {
  local track="$1"; shift
  case "$track" in
    vllm)   (cd "$ROOT" && python run_all_labs.py "$@") ;;
    sglang) (cd "$ROOT/sglang" && ./run_all_labs.sh "$@") ;;
    llmd)   (cd "$ROOT/llmd" && ./run_all_labs.sh "$@") ;;
    *) echo "unknown track: $track (vllm | sglang | llmd)" >&2; return 2 ;;
  esac
}

if [[ "${1:-}" == "--list" ]]; then
  echo "== vllm";   (cd "$ROOT" && python run_all_labs.py --list)
  echo "== sglang"; (cd "$ROOT/sglang" && ./run_all_labs.sh --list)
  echo "== llmd";   (cd "$ROOT/llmd" && ./run_all_labs.sh --list)
  exit 0
fi

if [[ $# -gt 0 ]]; then
  run_track "$@"; exit $?
fi

status=0
for t in vllm sglang llmd; do
  echo "################ track: $t ################"
  run_track "$t" || status=1
done
exit $status
