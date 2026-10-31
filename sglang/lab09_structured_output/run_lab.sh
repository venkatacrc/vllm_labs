#!/usr/bin/env bash
# Lab 09 - Structured output / grammar-guided decoding.
#
# Many production use cases need output that's guaranteed to parse as JSON
# matching a schema (tool calling, data extraction, form filling). Asking
# nicely in the prompt gets you *usually*-valid JSON; SGLang's grammar
# backends (xgrammar by default) can *guarantee* it by constraining the
# token sampling distribution at every step to only allow tokens consistent
# with the schema. This lab measures both the reliability win and the
# latency cost of that constraint.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
require_gpus 1
trap_stop_server

LAB=lab09_structured_output
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

section "Launching SGLang with xgrammar backend available"
start_sglang_server "$OUT_DIR/server.log" \
  --model-path "$SMALL_MODEL" \
  --tp 1 \
  --grammar-backend xgrammar

section "[1/2] Unconstrained: ask nicely for JSON in the prompt"
python "$SCRIPT_DIR/json_schema_client.py" \
  --base-url "http://$SGLANG_HOST:$SGLANG_PORT" --model "$SMALL_MODEL" \
  --constrained false --out "$OUT_DIR/unconstrained.json"

section "[2/2] Constrained: enforce JSON Schema via response_format"
python "$SCRIPT_DIR/json_schema_client.py" \
  --base-url "http://$SGLANG_HOST:$SGLANG_PORT" --model "$SMALL_MODEL" \
  --constrained true --out "$OUT_DIR/constrained.json"

# A throughput-under-load view: does grammar constraint meaningfully slow
# down a batch of concurrent structured-output requests?
section "[extra] Throughput under load, unconstrained vs constrained (bench_serving)"
run_bench "$OUT_DIR/bench_plain.jsonl" \
  --model "$SMALL_MODEL" --dataset-name random \
  --random-input-len "$BENCH_RANDOM_INPUT_LEN" --random-output-len 128 \
  --num-prompts 100 --max-concurrency 32
record_result "$LAB" "plain-decoding-load" "$OUT_DIR/bench_plain.jsonl"

if python -m sglang.bench_serving --help 2>&1 | grep -q "json-schema"; then
  run_bench "$OUT_DIR/bench_json.jsonl" \
    --model "$SMALL_MODEL" --dataset-name random \
    --random-input-len "$BENCH_RANDOM_INPUT_LEN" --random-output-len 128 \
    --num-prompts 100 --max-concurrency 32 \
    --json-schema "$(python -c "import json;print(json.dumps({'type':'object','properties':{'name':{'type':'string'},'age':{'type':'integer'},'profession':{'type':'string'}},'required':['name','age','profession']}))")"
  record_result "$LAB" "constrained-decoding-load" "$OUT_DIR/bench_json.jsonl"
else
  log_warn "This sglang version's bench_serving has no --json-schema flag; skipping the load-test comparison (the per-request comparison above still covers the core lesson)."
fi

stop_sglang_server

section "Lab 09 summary"
echo "Reliability (per-request comparison):"
python -c "
import json
u = json.load(open('$OUT_DIR/unconstrained.json'))
c = json.load(open('$OUT_DIR/constrained.json'))
print(f\"  unconstrained : {u['num_valid_json']}/{u['num_requests']} valid JSON, mean latency {u['mean_latency_s']:.2f}s\")
print(f\"  constrained   : {c['num_valid_json']}/{c['num_requests']} valid JSON, mean latency {c['mean_latency_s']:.2f}s\")
"
python "$LABS_ROOT/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv" | grep -E "lab|$LAB" || true

echo
echo "The core trade-off: constrained decoding should hit (or nearly hit) 100%"
echo "valid JSON, vs whatever error rate the unconstrained prompt-only approach"
echo "shows, at the cost of some extra per-token latency for grammar-mask"
echo "computation. For any pipeline that would otherwise need retry logic or"
echo "manual JSON repair, that latency cost is almost always worth paying."
