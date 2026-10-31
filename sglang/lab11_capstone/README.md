# Lab 11 - Capstone: Stack the Wins

**Goal:** combine the broadly-applicable optimizations from the earlier labs
into one server configuration, and put a single number on the total
speedup from "naive FP32, one request at a time" to "SGLang, tuned."

## Prerequisite

Run `lab01_fp32_baseline` at least once before this lab -- the final
comparison table reads its results from `results/summary.csv`. If you
haven't run every lab, that's fine; this lab still measures the optimized
recipe's absolute numbers, it just can't print every intermediate
comparison row.

## The recipe

```
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tp 1 \
  --dtype bfloat16 \
  --quantization fp8
```

Notice what's *not* an explicit flag here: RadixAttention prefix caching,
continuous batching with the overlap scheduler, chunked prefill, and CUDA
graph capture are all **on by default** in SGLang. The only things this
recipe adds explicitly on top of an out-of-the-box launch are the precision
(bf16 base + fp8 quantization) choices from labs 02-03. That's deliberate:
this capstone's point is that most of SGLang's value is already there the
moment you use it as your serving engine (lab01's lesson), and the highest
-leverage manual choice on top of that is picking the right numeric
precision for your hardware (labs 02-03).

Deliberately left out of the default recipe, because their benefit is
workload-dependent rather than universal:

- **Speculative decoding** (lab08) -- big win for repetitive/structured
  output, smaller for open-ended generation.
- **Tensor/expert/data parallelism** (labs 04, 10) -- only relevant once a
  model doesn't fit on (or isn't fast enough on) one GPU.
- **Non-default scheduler tuning** (lab06) -- `lpm` scheduling only helps
  prefix-heavy traffic; chunked prefill's default is already reasonable for
  most workloads.

## What this lab does

1. Launches the recipe above.
2. Benchmarks it at concurrency=1 (directly comparable to lab01's naive HF
   loop and naive SGLang-FP32-concurrency=1 numbers) and at
   concurrency=`$BENCH_MAX_CONCURRENCY` (comparable to lab01's SGLang-FP32
   batching number).
3. Prints a final comparison table with speedup multipliers relative to the
   naive HF baseline.
4. Calls `common/plot_results.py` to render bar charts across every lab
   you've run into `results/plots/`.

## Run it

```bash
cd vllm_labs/sglang/lab11_capstone
./run_lab.sh
```

## Expected takeaways

- The concurrency=1 comparison isolates the pure "better engine + better
  precision" win with no batching involved -- expect this alone to be a
  multi-x speedup over the naive HF loop.
- The concurrency=`$BENCH_MAX_CONCURRENCY` comparison additionally captures
  continuous batching's throughput multiplier on top of that -- expect this
  to be the largest speedup number in the whole curriculum.
- Look at `results/plots/throughput.png` for a single chart spanning every
  lab you've run, from the naive FP32 baseline all the way to this recipe.
- The real lesson: optimization is not one silver bullet, it's a stack of
  mostly-independent, mostly-compatible levers (engine choice, precision,
  parallelism, scheduling, caching, speculative decoding), and the right
  combination depends on your model size, hardware, and traffic pattern --
  which is exactly why each lab isolated one lever at a time instead of
  just handing you a single "optimal" config.

You've now worked through the full FP32-to-optimized-serving journey on
SGLang. Re-run any lab with different `$SMALL_MODEL` / `$MID_MODEL` /
`$MOE_MODEL` values, different `BENCH_*` load parameters, or on a different
GPU count to build intuition for how these trade-offs shift.
