---
title: Number formats and quantization
track: Foundations
order: 5
---

# Number formats and quantization

## Floating-point formats

| Format | Bits (sign/exp/mantissa) | Range | Notes |
|---|---|---|---|
| FP32 | 1/8/23 | ±3.4e38 | training master weights; no tensor-core fast path |
| TF32 | 1/8/10 | FP32 range | tensor-core FP32 "lite" |
| BF16 | 1/8/7 | FP32 range | default for inference; no overflow surprises |
| FP16 | 1/5/10 | ±65504 | more precision, can overflow activations |
| FP8 E4M3 | 1/4/3 | ±448 | weights/activations/KV |
| FP8 E5M2 | 1/5/2 | ±57344 | gradients; KV option |
| FP6 / FP4 E2M1 | 1/3/2, 1/2/1 | tiny | needs fine-grained scales |

Halving bits halves weight bytes, which (decode being memory-bound) nearly
halves decode step time, and lets more KV cache fit.

## Scaling: why FP8/FP4 work

Low-bit formats have little range, so values are stored as `q × scale`.
The question is how many values share one scale:

- **Per-tensor**: one scale for the whole matrix. Simple; outliers hurt.
- **Per-channel / per-token**: one scale per output channel (weights) or per
  token (activations).
- **Per-block**: one scale per 128×128 weight block and 1×128 activation
  block (DeepSeek-V3 style, vLLM `fp8_per_block`).
- **Microscaling (MX)**: an 8-bit power-of-two scale (E8M0) per 32
  elements: MXFP8, MXFP6, MXFP4 (OCP standard). NVIDIA's **NVFP4** uses an
  FP8 scale per 16 elements plus a per-tensor scale. Blackwell tensor cores
  apply these block scales in hardware.

## Weight-only vs weight-and-activation

- **Weight-only (W4A16, e.g. AWQ, GPTQ)**: weights stored in INT4,
  dequantized to BF16 in the kernel (Marlin). Cuts memory traffic; math
  stays BF16. Great for memory-bound decode at small batch.
- **W8A8 FP8**: weights and activations in FP8; matmuls run on FP8 tensor
  cores (2x BF16 FLOPs). Helps both prefill (compute) and decode (bytes).
- **W4A4 (NVFP4/MXFP4)**: 4x fewer bytes than BF16 and FP4 tensor cores;
  needs good calibration or quantization-aware training to keep accuracy.

## Offline vs online

- **Online / dynamic** (vLLM `--quantization fp8_per_tensor|fp8_per_block|
  mxfp8|mxfp4`): quantize BF16 weights at load time; activation scales
  computed on the fly. No calibration data.
- **Offline / calibrated** (AWQ, GPTQ, llm-compressor, ModelOpt): run
  calibration data through the model, pick scales (or rotate/smooth
  outliers), save a quantized checkpoint. Better accuracy at 4 bits.

## KV-cache quantization

`--kv-cache-dtype fp8_e4m3` stores K/V in FP8 with per-tensor or per-head
scales: half the KV bytes → ~2x the concurrent tokens. Attention backends
with native FP8 KV (FlashInfer/TRT-LLM on Blackwell) avoid a dequantize
pass. Sensitive layers can be skipped (`--kv-cache-dtype-skip-layers`).

## Always measure quality

Quantization is a throughput/quality trade: check perplexity or task
accuracy (e.g. GSM8K, MMLU via lm-eval) for each scheme, not just tokens/s.

## Flashcards

Q: Why is BF16 preferred over FP16 for inference?
A: BF16 has FP32's exponent range (8 bits), so activations don't overflow; FP16 has more mantissa precision but a max of 65504.

Q: What is microscaling (MX)?
A: A block format where each group of 32 elements shares an 8-bit power-of-two (E8M0) scale, e.g. MXFP8, MXFP4; Blackwell tensor cores apply the scales in hardware.

Q: Difference between W4A16 and W8A8?
A: W4A16 stores weights in 4 bits but computes in 16-bit (saves bytes); W8A8 quantizes weights and activations to 8 bits and uses FP8 tensor cores (saves bytes and FLOPs).

Q: Online vs offline quantization?
A: Online quantizes BF16 weights at load time with dynamic activation scales (no calibration); offline uses calibration data to choose scales and saves a quantized checkpoint (AWQ, GPTQ, llm-compressor).

Q: What does FP8 KV cache buy you?
A: Half the KV bytes per token, so roughly twice as many tokens/sequences fit in the same memory, plus less bandwidth per decode step.

Q: Why do outliers matter for per-tensor scaling?
A: One large value forces a large scale, so most small values lose precision; finer-grained (per-channel/per-block) scales contain the damage.

Q: Why does weight quantization speed up decode almost proportionally?
A: Decode is memory-bound on reading weights, so fewer bytes per weight means proportionally less HBM time per step.
