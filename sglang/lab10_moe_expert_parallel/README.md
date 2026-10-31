# Lab 10 - Mixture-of-Experts: Expert Parallelism & DP Attention

**Goal:** understand the two parallelism strategies unique to Mixture-of-Experts
(MoE) models, and why plain tensor parallelism is often not the best fit for
them.

## Background

`$MOE_MODEL` (Qwen3-30B-A3B) is a MoE model: it has ~30B total parameters
organized into many small expert MLPs, but a per-token router activates
only a handful of experts per forward pass (~3B "active" parameters worth
of compute per token). This changes the parallelism trade-offs vs a dense
model like the ones in labs 01-04:

- **Plain TP** shards *every* expert's weight matrix across all GPUs (same
  as it would for a dense model's single MLP). Simple, but wasteful: each
  GPU stores and computes against a slice of experts that a given token may
  not have even been routed to.
- **Expert Parallelism (EP)** instead places whole experts on specific
  GPUs, and uses an all-to-all communication step to route each token's
  hidden state to whichever GPU(s) hold its selected expert(s), then
  another all-to-all to bring results back. In SGLang, `--ep` "subdivides"
  the TP group (`--tp 8 --ep 8` uses the 8-GPU TP group to host 8 distinct
  expert shards). This avoids redundant expert compute, at the cost of
  all-to-all communication instead of all-reduce.
- **DP Attention** (`--enable-dp-attention --dp-size N`) targets a
  different inefficiency: attention layers are relatively small compared to
  a MoE model's total FLOPs, so tensor-parallelizing them (which requires an
  all-reduce and duplicates KV cache management overhead across ranks) is
  wasteful. DP attention instead runs each GPU's attention layers on its own
  independent slice of requests (true data parallelism, no cross-GPU
  communication for attention), while the larger FFN/expert layers still use
  TP or EP.

## A real, current gotcha (worth knowing about)

Combining DP-attention with EP>1 at the same time has an **open, known
correctness bug** on bf16 MoE checkpoints that aren't using the DeepEP
kernel path (see
[sgl-project/sglang#23729](https://github.com/sgl-project/sglang/issues/23729)).
Since `$MOE_MODEL` is bf16, this lab deliberately measures EP and DP
attention **separately** rather than stacking them, and this README calls
that decision out explicitly. The combined "DPA + EP" recipe is the
documented, recommended production setup for models like DeepSeek-V3 when
running an FP8 checkpoint with `--moe-a2a-backend deepep` -- but "the docs
say to combine these flags" and "this specific model+precision+version
combination actually works correctly" are different claims, and the second
one is always worth verifying yourself before shipping. This is a good
general lesson for working with a fast-moving inference engine: always
sanity-check output quality (not just throughput numbers) after stacking
multiple advanced parallelism flags.

## What this lab does

Requires all 8 GPUs. Runs `$MOE_MODEL` three ways against identical
benchmark load:

1. `tp8-baseline` -- `--tp 8`.
2. `tp8-ep8` -- `--tp 8 --ep 8`.
3. `tp8-dp8-attention` -- `--tp 8 --dp-size 8 --enable-dp-attention`.

## Run it

```bash
cd vllm_labs/sglang/lab10_moe_expert_parallel
./run_lab.sh
```

## Expected takeaways

- `tp8-ep8` should generally outperform `tp8-baseline` for this model, since
  it eliminates redundant per-GPU expert computation -- the gap should be
  larger at higher concurrency (more tokens per all-to-all round = better
  amortized communication cost) and smaller (or even reversed) at very low
  concurrency.
- `tp8-dp8-attention` should show its benefit mainly in memory efficiency
  (no duplicated KV cache management across the attention TP group) and can
  particularly help at high concurrency / long context, where attention's
  KV cache footprint becomes a bigger fraction of total memory pressure.
- Try re-running with `BENCH_MAX_CONCURRENCY=16 ./run_lab.sh` and then
  `BENCH_MAX_CONCURRENCY=128 ./run_lab.sh` to see how the ranking between
  these three strategies shifts with load -- there is no universally "best"
  MoE parallelism strategy independent of your expected traffic pattern.

Next: `lab11_capstone` combines the single best setting from each earlier
lab into one "production recipe" server and measures the cumulative
speedup versus lab01's naive FP32 baseline.
