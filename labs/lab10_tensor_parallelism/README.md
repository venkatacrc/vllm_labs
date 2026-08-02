# Lab 10: Tensor parallelism

## Concept

Every other lab uses the 8B primary model, which comfortably fits on one
B200. This lab switches to `Qwen/Qwen2.5-32B-Instruct` (ungated, ~64GB in
BF16) specifically to make tensor parallelism (TP) worth doing.

**Tensor parallelism** shards individual weight matrices (attention
projections, MLP layers) across multiple GPUs, so each GPU holds and
computes on only `1/TP` of every layer. Every layer requires an all-reduce
(or similar collective) across the TP group to combine partial results, so
TP trades GPU memory/compute headroom for inter-GPU communication -
which is exactly why it matters that B200s in this node are connected over
fast NVLink rather than PCIe alone.

## What this lab does

1. Serves Qwen2.5-32B-Instruct in BF16 at `--tensor-parallel-size` 1, 2, 4,
   and 8 (using the first N GPUs from your pool for each), running the same
   benchmark workload at each size and recording throughput/latency.
2. **Bonus (full run only, skipped under `--quick`)**: a memory-pressure
   check - serving the *same* 32B model at **FP32** with `TP=1`. At FP32,
   weights alone are `32B x 4 bytes = ~128GB`, leaving only ~50GB of a
   B200's 179GB for the framework, activations, and KV cache. This either
   succeeds with a very small usable KV cache (few concurrent requests /
   short context) or fails outright with an out-of-memory error - both
   outcomes are informative, and this variant is wrapped so a failure here
   doesn't fail the whole lab.

## Run it

```bash
python labs/lab10_tensor_parallelism/lab.py
python labs/lab10_tensor_parallelism/lab.py --quick        # TP=1,2 only, skips the fp32 bonus
python labs/lab10_tensor_parallelism/lab.py --gpus 0,1,2,3,4,5,6,7   # explicit GPU pool
```

## What to look for

- Throughput scaling from TP=1 -> 2 -> 4 -> 8: expect solid gains from 1->2
  and 1->4, with diminishing returns by TP=8 as all-reduce communication
  overhead starts to compete with the compute time saved per GPU.
- `mean_tpot_ms` at each TP size: more GPUs means each one does less
  compute per step, but pays a fixed communication cost every layer -
  latency doesn't necessarily improve as much as raw throughput does.
- Whether the FP32/TP=1 bonus variant succeeded or OOM'd, and if it
  succeeded, how small the achievable KV cache / concurrency was compared
  to the BF16 variants - this is the "why you'd reach for TP or
  quantization" moment for large models.
