# Lab 04 - Tensor Parallelism (TP) Scaling

**Goal:** measure how throughput scales as you shard a model's weights
across more GPUs with `--tp`, and understand why that scaling is sub-linear.

## Background

Tensor parallelism splits individual weight matrices (attention QKVO
projections, MLP up/down projections) across GPUs, so each GPU only holds
`1/TP` of the parameters and does `1/TP` of the matmul FLOPs for those
layers. After each attention block and each MLP block, the partial results
must be combined with an **all-reduce** across the TP group before the next
layer can run. That all-reduce is pure communication overhead with no
useful compute -- it's the tax you pay for splitting the model.

On this node, all 8 B200s should be NVLink-connected (verify with
`nvidia-smi topo -m`), which is exactly the hardware TP wants: low-latency,
high-bandwidth interconnect for those frequent, small-ish all-reduces.
Multi-*node* TP over Ethernet/InfiniBand is far more sensitive to this
overhead -- generally avoid TP across nodes if you can, and prefer it
within a node.

We use `$MID_MODEL` (Qwen2.5-32B-Instruct, bf16, ~65GB) specifically because
it fits on **one** B200 already. That means TP=1 is a valid, meaningful
baseline, and any speedup at TP=2/4/8 reflects real parallel scaling -- not
"it didn't even fit before."

## What this lab does

Launches the same model at `--tp 1`, `--tp 2`, `--tp 4`, `--tp 8` (skipping
sizes larger than the GPUs you have available), and for each, runs
`bench_serving` with an **identical, fixed** request load (same
`--num-prompts`, same `--max-concurrency`). Holding load fixed is what makes
the comparison a clean measurement of parallel-compute scaling rather than
"more GPUs -> more room for concurrent requests" (a real and separate
benefit you'd also chase in production, just not what we're isolating here).

## Run it

```bash
cd vllm_labs/sglang/lab04_tensor_parallelism
./run_lab.sh
```

This needs multiple GPUs to see the interesting part; it will auto-skip TP
sizes beyond what `nvidia-smi` reports. On your 8xB200 node it will run all
four configurations.

## Expected takeaways

- Total throughput should increase from TP=1 to TP=8, but **per-GPU**
  throughput (`total_throughput / TP`) should *decrease* as TP grows -- the
  script prints a "scaling efficiency" table making this explicit.
- The drop-off is the NCCL all-reduce tax. It grows with TP size (more
  ranks to synchronize) and shrinks relative to compute as the model gets
  bigger (more FLOPs to amortize the fixed communication cost against) --
  this is why TP is usually reserved for models that don't fit on fewer
  GPUs, or where you specifically need the lower per-request latency that
  splitting compute across more GPUs buys you.
- Compare TTFT (`median_ttft_ms`) across TP sizes too: TP mainly helps
  *latency* per request (the prefill/decode for one sequence is now spread
  over more GPUs), which is a different axis from the throughput-via-batching
  win you saw in lab01. Production sizing is a joint decision over both.

Next: `lab05_radix_attention` looks at SGLang's signature feature --
RadixAttention prefix caching -- which needs no extra GPUs at all.
