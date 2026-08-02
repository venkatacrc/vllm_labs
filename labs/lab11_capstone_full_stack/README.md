# Lab 11: Capstone - the full optimization stack

## Concept

This lab doesn't introduce a new technique - it pulls together everything
from Labs 00-10 into the "fp32 -> fully optimized" journey, on both models.

One thing worth internalizing by this point: **most of the optimizations
in this suite are already the default in modern vLLM.** Continuous
batching, paged KV cache, prefix caching, chunked prefill, and CUDA graphs
are all on unless you explicitly disable them. The knobs you actually have
to reach for in practice are the ones this capstone varies: **precision /
quantization**, and (for models too large for one GPU) **tensor
parallelism**. Everything else, vLLM is already doing for you.

## What this lab does

**8B journey** (reuses saved results from earlier labs where available,
reruns anything missing):

1. `fp32` (Lab 00)
2. `bfloat16` (Lab 01)
3. `fp8_weights_and_kv` (Lab 08) - the "fully optimized single-GPU" config

**32B journey** (Qwen2.5-32B-Instruct, TP=4):

1. `tp4_bf16` (Lab 10)
2. `tp4_fp8` - adds FP8 weight quantization on top of TP=4

Both journeys get a cumulative speedup column relative to their first
entry, and everything is appended to `results/SUMMARY.md` as the final
section - the one-page summary of the whole suite.

## Run it

```bash
python labs/lab11_capstone_full_stack/lab.py
python labs/lab11_capstone_full_stack/lab.py --quick
```

Running the full suite first (`python run_all_labs.py`) means this lab
reuses everyone else's saved numbers instead of re-running servers; run
standalone and it will backfill whatever's missing itself.

## What to look for

- The **total** speedup from `fp32` to `fp8_weights_and_kv` on the 8B
  model - this is the number that answers the original question ("how much
  can I actually speed up serving a base fp32 model with vLLM on B200?").
- Whether `tp4_fp8` beats `tp4_bf16` by roughly the same margin FP8 gave
  the 8B model - quantization and parallelism are largely orthogonal
  levers and should compose.
- Check `results/SUMMARY.md` after this lab runs: it now contains every
  lab's table plus this final rollup, in one file.
