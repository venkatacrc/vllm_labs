# Lab 02 - Precision: FP32 vs FP16 vs BF16

**Goal:** isolate the effect of numeric precision alone on memory footprint,
throughput, and output quality, holding model, hardware, and load pattern
fixed.

## Background

B200 (Blackwell) tensor cores deliver dramatically higher FLOPs for
`bf16`/`fp16` than for `fp32` (and higher still for `fp8`/`fp4`, covered in
lab03). Halving bytes-per-parameter (4 bytes -> 2 bytes) also halves:

- The GPU memory needed to hold the weights.
- The size of the KV cache per token, which means SGLang's memory
  allocator (`--mem-fraction-static`) can fit a larger `max_running_requests`
  and a deeper `RadixAttention` prefix-cache tree in the same GPU memory.

That second effect compounds with the first: more concurrent requests
fitting in memory means continuous batching has more work to overlap,
so throughput gains from precision reduction are often *larger* than the
raw FLOPs ratio would suggest.

`bf16` and `fp16` have the same bit width (16 bits) but split it differently
(`bf16`: 8 exponent / 7 mantissa bits, matching fp32's dynamic range with
less precision; `fp16`: 5 exponent / 10 mantissa bits, more precision but a
much smaller dynamic range that can overflow on some activations). Modern
LLMs are almost universally trained and released in `bf16` for this reason.

## What this lab does

Launches the same `$SMALL_MODEL`, TP=1, three times with `--dtype float32`,
`--dtype bfloat16`, `--dtype float16`, and for each:

1. Records GPU memory used right after model load.
2. Grabs one qualitative sample completion (temperature 0, so deterministic)
   so you can eyeball whether reduced precision visibly changed the answer.
3. Runs `sglang.bench_serving` with identical load (`$BENCH_NUM_PROMPTS`
   prompts, `$BENCH_MAX_CONCURRENCY` concurrency) and records throughput/latency.

## Run it

```bash
cd vllm_labs/sglang/lab02_precision
./run_lab.sh
```

## Expected takeaways

- `bfloat16` and `float16` should use roughly half the GPU memory of
  `float32` for weights, and show meaningfully higher
  `output_throughput_tok_s` in `results/summary.csv`.
- The three qualitative samples in `results/lab02_precision/sample_*.txt`
  should be very close in content (Qwen2.5 was trained in bf16, so bf16 is
  the "native" precision; fp32 is a lossless upcast of those same weights;
  fp16 can occasionally diverge more due to its narrower dynamic range,
  though for a 7B chat model at greedy decoding this is usually
  imperceptible).
- This is the cheapest possible win in the entire curriculum: one flag
  change, no retraining, no accuracy-tuning required to go from FP32 to
  BF16. Always start here before reaching for quantization.

Move on to `lab03_quantization` for the next rung down: FP8 and INT4.
