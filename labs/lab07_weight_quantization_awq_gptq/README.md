# Lab 07: Weight-only quantization (AWQ / GPTQ)

## Concept

BF16 halved FP32's memory footprint; weight-only quantization goes further
by storing weights in 4 bits instead of 16, while keeping activations in a
higher-precision compute dtype. Two popular post-training quantization
methods:

- **AWQ** (Activation-aware Weight Quantization): protects the small
  fraction of "salient" weight channels most sensitive to quantization
  error (identified from activation statistics) by keeping them at higher
  effective precision, quantizing the rest to INT4.
- **GPTQ**: a one-shot weight quantization method that greedily minimizes
  layer-wise reconstruction error using second-order (Hessian) information,
  quantizing to INT4 (or INT8) with typically small accuracy loss.

Both need a calibration/quantization pass done ahead of time against a
calibration dataset - this lab uses ready-made community checkpoints rather
than re-deriving them, since that's how most teams actually consume
quantized models. The interesting comparison for a serving lab is memory
and throughput, not the quantization algorithm's internals.

## What this lab does

Serves three checkpoints of the same base model back to back:

1. BF16 (reuses Lab 01's saved result if available, otherwise reruns it).
2. `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`
3. `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4`

vLLM auto-detects the quantization method from each checkpoint's config, so
no explicit `--quantization` flag is needed.

## Run it

```bash
python labs/lab07_weight_quantization_awq_gptq/lab.py
python labs/lab07_weight_quantization_awq_gptq/lab.py --quick
```

## What to look for

- GPU memory snapshot right after each server starts: weights should drop
  from ~16GB (bf16) to roughly 5-6GB (INT4) for the 8B model.
- `output_throughput`: INT4 quantization reduces memory-bandwidth pressure
  (fewer bytes moved per weight read), which often *increases* throughput
  even though there's dequantization overhead - especially at
  memory-bandwidth-bound batch sizes.
- This lab does not measure output quality/accuracy - that requires a task
  benchmark (e.g. lm-eval-harness), which is out of scope here. Quantization
  always trades some accuracy for memory/speed; AWQ and GPTQ are both
  designed to minimize that loss, not eliminate it.
