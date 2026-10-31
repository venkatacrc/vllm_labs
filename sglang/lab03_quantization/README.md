# Lab 03 - Quantization: FP8 and INT4

**Goal:** understand the two flavors of quantization SGLang supports --
*online* (quantize a full-precision checkpoint at load time) and *offline /
pre-quantized* (load someone else's already-quantized checkpoint) -- and see
their memory/throughput/quality trade-offs relative to bf16.

## Background

- **FP8 (`--quantization fp8`)**: SGLang casts the bf16 weights to FP8
  (`E4M3`) on the fly when the server starts. B200's tensor cores have
  *native* FP8 matmul support (2x the FLOPs of bf16 on the same silicon), so
  this is typically the best throughput-per-accuracy-lost trade available on
  Blackwell. No calibration dataset or offline step required.
- **AWQ (Activation-aware Weight Quantization) / GPTQ**: both quantize
  weights down to INT4, but need an offline calibration pass against sample
  data to decide how to round weights with minimal output error. We use
  community pre-quantized checkpoints (`Qwen/Qwen2.5-7B-Instruct-AWQ` and
  `-GPTQ-Int4`) so this lab doesn't need to run its own calibration job --
  SGLang auto-detects the quantization method from the checkpoint's
  `config.json`, so you do **not** pass `--quantization` for these.
- Weight-only INT4 quantization keeps activations/KV-cache at higher
  precision and dequantizes weights on the fly during the matmul, so it
  saves memory more than it saves compute -- useful when you're
  memory-bound rather than compute-bound.

## What this lab does

Runs four server configurations against the same benchmark load:

1. `bf16-baseline` -- re-measured here so this lab stands on its own.
2. `fp8-online` -- `$SMALL_MODEL` loaded with `--quantization fp8`.
3. `awq-int4` -- `$AWQ_MODEL` (pre-quantized, auto-detected).
4. `gptq-int4` -- `$GPTQ_MODEL` (pre-quantized, auto-detected).

For each: GPU memory footprint, one deterministic qualitative sample, and a
full `bench_serving` throughput/latency run.

## Run it

```bash
cd vllm_labs/sglang/lab03_quantization
./run_lab.sh
```

## Expected takeaways

- Memory: `bf16` > `fp8` > `awq-int4` ~= `gptq-int4`, roughly in a 4 : 2 : 1
  : 1 ratio for weights (KV cache size is unaffected by weight
  quantization, so total memory savings will be somewhat less dramatic than
  a pure 4x).
- Throughput: on B200, `fp8-online` should show a clear throughput win over
  `bf16-baseline` with a barely perceptible quality difference in
  `sample_fp8-online.txt` vs `sample_bf16-baseline.txt`. INT4 methods often
  show *even higher* throughput at low/medium concurrency (memory-bandwidth
  bound decode phase moves less data per token), but the gap narrows or can
  even reverse at very high concurrency where compute-bound prefill
  dominates and dequantization overhead starts to matter.
- Read all four `sample_*.txt` files side by side. This is the most
  important habit to build for production quantization work: never ship a
  quantized model without eyeballing (and ideally running a proper eval
  harness against) its outputs.

Next: `lab04_tensor_parallelism` moves from "shrink the model" to "spread
the model across more GPUs."
