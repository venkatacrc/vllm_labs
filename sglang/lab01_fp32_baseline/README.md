# Lab 01 - The Naive FP32 Baseline

**Goal:** establish honest "before" numbers for the rest of the curriculum.
Every subsequent lab reports its speedup relative to what you measure here.

## Background

The task says "take a base FP32 precision model." In practice almost no
model is published in true FP32 anymore (checkpoints on the Hub are usually
BF16), so we treat FP32 as our deliberately-naive starting point: we force
the model to load and run its weights and activations in `float32`
(`--dtype float32` in SGLang, `torch_dtype=torch.float32` in transformers).
This is exactly what you'd get if someone handed you a model and you served
it with zero thought about efficiency.

We use `Qwen/Qwen2.5-7B-Instruct` (`$SMALL_MODEL`) for this and the next several
labs because at FP32 it is ~30GB of weights, comfortably fits on one B200 (183GB)
with room for KV cache, and downloads quickly.

## What this lab measures

| Step | What it is | What it isolates |
|---|---|---|
| 1. `hf_naive_baseline.py` | Plain `transformers` `model.generate()` in a Python for-loop, one request at a time, FP32 | The true floor: no server, no batching, no paged KV cache |
| 2. SGLang FP32, concurrency=1 | Same FP32 weights, but behind `sglang.launch_server`, one request in flight at a time | How much do SGLang's fused kernels / scheduler / RadixAttention alone help, even with no batching opportunity? |
| 3. SGLang FP32, concurrency=64 | Same server, 64 concurrent requests | How much does continuous batching alone add, with precision still unchanged? |

Notice precision never changes within this lab -- that axis is lab02. This
lab isolates the "just use a real inference engine" axis, which in our
experience is often the single biggest lever people forget to pull.

## Run it

```bash
cd vllm_labs/sglang/lab01_fp32_baseline
./run_lab.sh
```

Takes ~5-10 minutes (the HF sequential loop is the slow part, by design --
that's the point). Results land in `results/lab01_fp32_baseline/` and are
appended to `results/summary.csv`.

## Things to look at afterwards

- `results/lab01_fp32_baseline/hf_naive.json` -- per-request latency and
  overall `output_throughput` (tokens/sec) for the naive loop.
- `results/lab01_fp32_baseline/bench_conc1.log` and `bench_conc64.log` --
  console output from `sglang.bench_serving`, including TTFT (time to first
  token) and TPOT (time per output token) percentiles.
- Compare `output_throughput_tok_s` across the three rows in
  `results/summary.csv` for `lab=lab01_fp32_baseline`.

## Expected takeaways

- SGLang at concurrency=1 should already beat the naive HF loop by a
  meaningful margin, purely from better attention kernels (FlashInfer) and
  avoiding Python-loop overhead -- with **zero** precision or batching
  changes.
- SGLang at concurrency=64 should show a dramatically higher
  `output_throughput_tok_s` than concurrency=1, at the cost of higher
  per-request latency (TTFT/TPOT) as requests share GPU time. This
  throughput/latency trade-off from continuous batching is the central
  design point of every modern LLM serving engine, including SGLang.
- Peak GPU memory in the naive HF run vs. what `nvidia-smi` shows for the
  SGLang server (check the server log) -- SGLang pre-allocates most of GPU
  memory in `--mem-fraction-static` for KV cache up front; the naive loop
  under-uses memory badly.

Move on to `lab02_precision` once you've looked at these numbers.
