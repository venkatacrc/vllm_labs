# Lab 00: FP32 baseline

## Concept

Every other lab in this suite measures itself against this one. We serve
the primary model (`meta-llama/Llama-3.1-8B-Instruct`) with vLLM at full
FP32 precision - no quantization, no speculative decoding, nothing clever -
and record:

- **Memory**: weights alone are `8B params x 4 bytes = ~32GB`.
- **Throughput**: output tokens/sec and requests/sec under a fixed synthetic
  workload.
- **Latency**: mean/P99 time-to-first-token (TTFT) and time-per-output-token
  (TPOT).

Every later lab (precision, quantization, caching, CUDA graphs, attention
backends, ...) reruns the *same* workload shape and reports its speedup
relative to these numbers.

## Why FP32 at all?

Nobody serves LLMs in FP32 in production - it's 2x the memory and compute
cost of BF16 for (in practice) no measurable quality benefit for inference.
We start here on purpose: it's the "naive PyTorch default" starting point
most people would reach for if they just called `.float()` on a checkpoint,
and it makes the payoff of each subsequent optimization visible in the
numbers instead of asserted in prose.

## A preview of things to come

Watch the server log (`results/logs/*fp32*.log`) at startup. Because
FP32 activations aren't supported by vLLM's fastest attention kernels
(FlashInfer and FlashAttention both require FP16/BF16), vLLM will silently
fall back to a slower attention backend (Triton or FlexAttention) just to
run this lab at all. That's your first real hint about the relationship
between precision and kernel choice - which is exactly what Lab 03 digs
into.

## What this lab does

1. Launches `vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype float32`
   on one GPU.
2. Runs `vllm bench serve` with a synthetic (`random` dataset) workload:
   ~512 input tokens / 128 output tokens per request, fired at unlimited
   request rate (`--request-rate inf`) so the server's own scheduler is the
   bottleneck.
3. Saves the result to `results/lab00_fp32_baseline/results.json` for every
   later lab to load and compare against, and appends a section to
   `results/SUMMARY.md`.

## Run it

```bash
python labs/lab00_fp32_baseline/lab.py            # full run
python labs/lab00_fp32_baseline/lab.py --quick     # fast smoke test
python labs/lab00_fp32_baseline/lab.py --gpus 1    # pin to a specific GPU
```

## What to look for

- Peak GPU memory usage (`nvidia-smi` snapshot printed before/after server
  start) - this is your reference point for "how much does precision /
  quantization actually save".
- Note the achieved throughput is already using vLLM's continuous batching
  and paged KV cache - the "fp32" in this lab's name refers *only* to model
  precision, not to which serving techniques are enabled. Lab 02 is where we
  isolate the batching technique itself.
