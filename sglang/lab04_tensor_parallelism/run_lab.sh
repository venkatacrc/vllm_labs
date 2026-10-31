#!/usr/bin/env bash
# Lab 04 - Tensor Parallelism (TP) scaling.
#
# Uses a bigger model ($MID_MODEL, Qwen2.5-32B-Instruct, bf16 ~65GB) that
# fits on a single B200 already, specifically so that any speedup you see
# from TP=1 -> TP=8 is a genuine measurement of parallel compute scaling and
# inter-GPU communication overhead, not an artifact of "the model didn't fit
# before." Each linear layer's weight matrix is sharded column/row-wise
# across the TP ranks, with an all-reduce after each attention/MLP block to
# resync activations -- that all-reduce is the tax you pay for parallelism,
# and it's why scaling is sub-linear.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/env.sh"
source "$SCRIPT_DIR/../common/lib.sh"

require_sglang_env
trap_stop_server

LAB=lab04_tensor_parallelism
OUT_DIR="$RESULTS_ROOT/$LAB"
mkdir -p "$OUT_DIR"

MAX_TP="$NUM_GPUS"
log "This node reports $NUM_GPUS visible GPUs; will test TP sizes up to $MAX_TP."

run_tp() {
  local tp="$1"
  if (( tp > MAX_TP )); then
    log_warn "Skipping TP=$tp: only $MAX_TP GPU(s) available."
    return
  fi
  section "Tensor Parallelism: TP=$tp"

  start_sglang_server "$OUT_DIR/server_tp${tp}.log" \
    --model-path "$MID_MODEL" \
    --tp "$tp"

  # IMPORTANT: concurrency and num_prompts are held FIXED across every TP
  # size. This is what makes the comparison a clean measurement of "how much
  # faster does the same request load run as we add GPUs to the TP group",
  # isolating parallel-compute scaling and NCCL all-reduce overhead. (In
  # real production you'd typically *also* raise concurrency as you add
  # GPUs, since more GPUs means more KV-cache headroom for extra concurrent
  # requests -- that's a separate, additional win on top of what we measure
  # here, deliberately left out to keep this experiment controlled.)
  run_bench "$OUT_DIR/bench_tp${tp}.jsonl" \
    --model "$MID_MODEL" \
    --dataset-name random \
    --random-input-len "$BENCH_RANDOM_INPUT_LEN" \
    --random-output-len "$BENCH_RANDOM_OUTPUT_LEN" \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --max-concurrency "$BENCH_MAX_CONCURRENCY"

  record_result "$LAB" "tp${tp}" "$OUT_DIR/bench_tp${tp}.jsonl" \
    --extra "tp=$tp" --extra "concurrency=$BENCH_MAX_CONCURRENCY"

  stop_sglang_server
}

run_tp 1
run_tp 2
run_tp 4
run_tp 8

section "Lab 04 summary"
python "$LABS_ROOT/common/bench_lib.py" show --summary-csv "$RESULTS_ROOT/summary.csv" | grep -E "lab|$LAB" || true

python - "$RESULTS_ROOT/summary.csv" "$LAB" <<'PY'
import csv, sys
path, lab = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open(path)) if r["lab"] == lab]
print(f"\n{'TP':>4} | {'tok/s (total)':>14} | {'tok/s per GPU':>14} | {'scaling efficiency':>18}")
print("-" * 60)
base = None
for r in sorted(rows, key=lambda r: int(r["extra"].split('"tp": "')[1].split('"')[0]) if '"tp"' in r["extra"] else 0):
    try:
        tp = int(r["extra"].split('"tp": "')[1].split('"')[0])
        tput = float(r["output_throughput_tok_s"])
    except Exception:
        continue
    per_gpu = tput / tp
    if base is None:
        base = per_gpu
    eff = 100.0 * per_gpu / base
    print(f"{tp:>4} | {tput:>14.1f} | {per_gpu:>14.1f} | {eff:>17.1f}%")
PY

echo
echo "A perfectly-scaling system would show 100% scaling efficiency at every TP"
echo "size. Expect it to drop as TP grows -- that gap is inter-GPU NCCL"
echo "all-reduce communication overhead, which is why you should only use as"
echo "much TP as you need to (a) fit the model or (b) hit a latency target,"
echo "and prefer NVLink-connected GPUs (which this 8xB200 node has) for TP."
