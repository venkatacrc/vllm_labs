# Lab 08: FP8 quantization (Blackwell-native)

## Concept

B200 (Blackwell, SM100) has native FP8 tensor cores - unlike AWQ/GPTQ INT4
(Lab 07), FP8 quantization here needs no pre-quantized checkpoint or
calibration step: vLLM can quantize a BF16 checkpoint to FP8 **dynamically
at load time** with `--quantization fp8` (per-tensor/per-channel dynamic
scales computed on the fly). This gets you roughly half the weight memory
of BF16 with much less accuracy risk than INT4, backed by hardware that
runs FP8 matmuls at full speed.

FP8 also applies to the **KV cache** independently of weight precision
(`--kv-cache-dtype fp8_e4m3`): halving KV cache bytes-per-token means
either more concurrent sequences at the same memory budget, or a larger
usable context length.

This connects directly to Lab 03: on Blackwell, vLLM's default attention
backend (FlashInfer) ships a decode-optimized `trtllm-gen` kernel path with
native FP8 (and NVFP4) support - so FP8 here isn't just "smaller", it's
running through hardware-accelerated kernels rather than a generic
quantize/dequantize fallback.

## What this lab does

Serves three configurations of the primary model:

1. BF16 (reused from Lab 01 if available).
2. `--quantization fp8` (FP8 weights, BF16 KV cache).
3. `--quantization fp8 --kv-cache-dtype fp8_e4m3` (FP8 weights **and** KV
   cache).

## Run it

```bash
python labs/lab08_fp8_quantization/lab.py
python labs/lab08_fp8_quantization/lab.py --quick
```

## What to look for

- GPU memory after server start: FP8 weights should be roughly half of
  BF16's; adding FP8 KV cache reduces it further, most visibly at higher
  concurrency where KV cache (not weights) dominates memory.
- `output_throughput`: FP8 matmuls on B200 should outperform BF16, unlike
  a non-native-FP8 GPU where the dequantization overhead could eat the
  memory savings.
- Check the server log for which attention backend was selected - if
  FlashInfer is available, FP8 attention should be running through the
  `trtllm-gen` path discussed in Lab 03.
