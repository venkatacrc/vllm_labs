# Lab 01: Precision reduction (FP32 -> FP16 / BF16)

## Concept

The single biggest, lowest-risk win in serving efficiency is simply not
using FP32. Halving the bytes per parameter:

- Halves weight memory (32GB -> 16GB for the 8B model), leaving far more
  headroom for KV cache (i.e. more concurrent requests).
- Roughly doubles achievable throughput, since most modern GPU tensor cores
  (including B200's) are dramatically faster at FP16/BF16 matmuls than FP32.
- Unlocks vLLM's fast attention backends entirely: FlashInfer and
  FlashAttention only support FP16/BF16 (see Lab 00's note and Lab 03).

**BF16 vs FP16**: same memory/throughput profile, different tradeoff.
BF16 keeps FP32's 8-bit exponent (same dynamic range, less mantissa
precision) so it rarely needs loss scaling and is what most modern LLMs
are trained/released in. FP16 has more mantissa bits (more precision) but a
much smaller dynamic range, which can matter for numerical stability. On
Blackwell both run at the same tensor-core throughput, so BF16 is the
practical default; FP16 is included here mainly so you can see that for
yourself.

## What this lab does

Serves `meta-llama/Llama-3.1-8B-Instruct` with `--dtype float16` and
`--dtype bfloat16` (same GPU, same synthetic workload as Lab 00), then
prints a combined table alongside Lab 00's saved FP32 numbers with speedup
ratios.

## Run it

```bash
python labs/lab01_precision_bf16_fp16/lab.py
python labs/lab01_precision_bf16_fp16/lab.py --quick
```

Run Lab 00 first (or at least once) so this lab has a baseline row to
compare against; if it hasn't been run yet, this lab still works, it just
skips the comparison row.

## What to look for

- Memory drop after server start (`nvidia-smi` snapshot) - should be
  roughly half of Lab 00's.
- `output_throughput` (tokens/sec) - typically ~1.5-2x FP32 on B200.
- Which attention backend vLLM selects now that FP16/BF16 is available
  (check the server log) - this is worth comparing to Lab 00's log.
