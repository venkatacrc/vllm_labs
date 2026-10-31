---
title: GPU architecture for inference
track: Foundations
order: 1
---

# GPU architecture for inference

## The parts that matter

A GPU is a large array of **streaming multiprocessors (SMs)** sharing a
pool of high-bandwidth memory.

```
            ┌────────────── GPU (B200: two dies, one logical GPU) ──────────────┐
            │  SM  SM  SM  SM  ... (~148 SMs)                                   │
            │  each SM: 4 warp schedulers, CUDA cores, 5th-gen tensor cores,     │
            │           registers, shared memory / L1 (~228 KB), TMEM (256 KB)   │
            │                         L2 cache (~126 MB total)                   │
            │                HBM3e: ~180 GB @ ~8 TB/s                            │
            └────────────────────── NVLink 5: 1.8 TB/s ──────────────────────────┘
```

- **Threads run in warps of 32**; a warp executes one instruction for all
  32 threads at once (SIMT). Blocks of warps are scheduled onto SMs.
- **Tensor cores** do small matrix multiply-accumulates per instruction.
  They are where almost all LLM FLOPs happen. Blackwell's 5th-generation
  tensor cores add FP4/FP6 and microscaling (MX) formats, keep accumulators
  in a dedicated **tensor memory (TMEM)**, and can pair two SMs on one MMA.
- **Memory hierarchy**: registers → shared memory/L1 (per SM, ~tens of
  TB/s) → L2 (shared) → **HBM** (~8 TB/s on B200) → host DRAM over PCIe
  Gen5/C2C (~64-450 GB/s) → NVMe/network.

Approximate B200 peak numbers (dense, no sparsity):

| Format | Tensor-core throughput |
|---|---|
| BF16 / FP16 | ~2.2 PFLOP/s |
| FP8 | ~4.5 PFLOP/s |
| FP4 (NVFP4/MXFP4) | ~9 PFLOP/s |
| FP32 (non-tensor) | ~80 TFLOP/s |

The FP32 row is why lab 00's FP32 baseline is so slow: FP32 matmuls can't
use the fast tensor-core paths at all (TF32 can, at reduced precision).

## The roofline model

Every kernel needs some **FLOPs** and moves some **bytes**. Its
**arithmetic intensity** is FLOPs / bytes. The GPU's **ridge point** is
peak FLOP/s / peak bytes/s:

```
ridge = 2.2e15 FLOP/s ÷ 8e12 B/s ≈ 280 FLOP/byte   (BF16)
```

- Intensity **below** the ridge → **memory-bound**: time = bytes /
  bandwidth. Faster math doesn't help; moving fewer bytes does
  (quantization, caching, fusion).
- Intensity **above** the ridge → **compute-bound**: time = FLOPs / peak.
  Lower-precision tensor cores and better kernels help.

## Why decode is memory-bound

For a linear layer with a d × d BF16 weight matrix and
a batch of B tokens: FLOPs = 2·B·d², bytes (weights dominate) ≈ 2·d².
Intensity ≈ B FLOP/byte.

- **Decode** processes one new token per sequence per step, so B = number
  of running sequences (often 1-256) - far below 280. Each step must stream
  *all the weights* from HBM. Lower bound for an 8B BF16 model:
  16 GB / 8 TB/s ≈ **2 ms per step**, whatever the batch size up to the
  ridge. That is why batching many sequences is nearly free throughput.
- **Prefill** processes the whole prompt at once: B = thousands of
  tokens → compute-bound.

Attention during decode is even more memory-bound: each sequence reads its
entire KV cache for one query token, and that cannot be shared across the
batch.

## Occupancy, launch overhead, and small batches

A kernel launch costs a few microseconds of CPU time. A decode step for a
32-layer model launches hundreds of kernels; at small batch the GPU work
per kernel can be shorter than the launch cost, so the GPU idles waiting
for the CPU. **CUDA graphs** (vLLM lab 06, SGLang lab 07) record the launch
sequence once and replay it in one call.

## Flashcards

Q: What is the arithmetic-intensity ridge point of a B200 in BF16, roughly?
A: About 280 FLOP/byte (≈2.2 PFLOP/s ÷ ≈8 TB/s). Kernels below it are memory-bound, above it compute-bound.

Q: Why is LLM decode memory-bound?
A: Each step generates one token per sequence, so a linear layer does about B FLOPs per weight byte (B = batch size), far below the ridge point; the step time is dominated by streaming all weights (and each sequence's KV cache) from HBM.

Q: Lower bound on decode step time for an 8B BF16 model on one B200?
A: About 2 ms: 16 GB of weights ÷ 8 TB/s of HBM bandwidth.

Q: Why does batching more sequences increase decode throughput almost for free?
A: The weights are read once per step regardless of batch size, so until the batch approaches the ridge point, extra sequences add FLOPs without adding much memory traffic.

Q: What are tensor cores and which formats do Blackwell's support?
A: Matrix-multiply-accumulate units inside each SM that do almost all LLM FLOPs; Blackwell's 5th-gen tensor cores support FP64/TF32/BF16/FP16/FP8/FP6/FP4 including microscaling (MX) block formats.

Q: Why is FP32 inference so slow on modern GPUs?
A: FP32 math runs on regular CUDA cores (~80 TFLOP/s on B200) instead of tensor cores (~2.2 PFLOP/s BF16), and it moves twice the bytes of BF16.

Q: What problem do CUDA graphs solve?
A: CPU launch overhead: a decode step launches hundreds of small kernels; a graph replays the recorded sequence with one launch so the GPU is not starved at small batch sizes.

Q: Name the GPU memory hierarchy from fastest to slowest.
A: Registers → shared memory/L1 (and TMEM on Blackwell) → L2 → HBM → host DRAM (PCIe/C2C) → NVMe/network.
