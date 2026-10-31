---
title: CUDA execution, kernels and compilation
track: Foundations
order: 4
---

# CUDA execution, kernels and compilation

## Host and device

Python/PyTorch runs on the **host** (CPU) and enqueues work onto **CUDA
streams**; the **device** executes kernels from each stream in order,
asynchronously. The CPU only waits when it needs a result
(`.item()`, `.cpu()`, synchronize). An inference engine is mostly a
CPU-side scheduler that keeps the GPU's queue full.

```
CPU: schedule batch ─ prepare inputs ─ launch k1 k2 k3 ... k800 ─ sample ─ detokenize ─ schedule next
GPU:                                   ███ ███ ███ ...  ███
                                       (gaps = GPU idle waiting for launches)
```

Engines hide CPU work with:

- **CUDA graphs**: capture a whole forward pass for a fixed batch shape and
  replay it with one launch. vLLM captures graphs for a set of batch sizes
  (`cudagraph_capture_sizes`) and pads batches up to the next size.
  `PIECEWISE` graphs leave attention outside the graph; `FULL` graphs include
  it (needs graph-safe attention metadata).
- **Overlap / async scheduling**: prepare step N+1 on the CPU while step N
  runs on the GPU (SGLang's overlap scheduler, lab 06; vLLM async
  scheduling).
- **Multi-process design**: API server, tokenizer/detokenizer and engine
  core in separate processes so Python HTTP handling doesn't stall the GPU.

## Kernels that matter

- **GEMMs** (cuBLAS, CUTLASS, DeepGEMM, FlashInfer/TRT-LLM kernels):
  linear layers and expert MLPs. Quantized variants (FP8, MXFP4, INT4 AWQ/
  Marlin) dequantize in registers.
- **Attention kernels**: FlashAttention tiles Q/K/V through shared memory so
  the N×N score matrix never hits HBM; FA3 targets Hopper (TMA, warp
  specialization), FA4 targets Blackwell (CuTe-DSL, TMEM, 2-CTA MMA).
  FlashInfer and TRT-LLM provide paged-KV decode kernels. MLA models use
  FlashMLA / TRT-LLM MLA.
- **Fused small ops**: RMSNorm + quantize, SiLU-and-mul, rotary embedding,
  residual add, all-reduce + norm. Each fusion removes an HBM round trip.
- **MoE**: routing (top-k), permute/unpermute, grouped GEMM, all-to-all
  (DeepEP, pplx, NCCL).

## torch.compile and Triton

`torch.compile` traces the model into an FX graph, and Inductor generates
fused **Triton** kernels for the pointwise and reduction parts. vLLM wraps
this as optimization levels `-O0`…`-O3` plus custom fusion passes and caches
the compiled artifacts (`~/.cache/vllm/torch_compile_cache`), so later
startups skip compilation. **Triton** is a Python DSL for writing GPU
kernels at the block level; it is portable across NVIDIA/AMD and is the
fallback attention backend in both engines.

## Startup cost

Model load + `torch.compile` + CUDA graph capture can take tens of seconds
to minutes. Serving systems care because it determines how fast autoscaling
can add capacity (llm-d lab 09): `--enforce-eager` starts fastest but runs
slowest; compile caches, pre-captured graphs and sleep/wake (keeping weights
in host memory) make cold starts warm.

## Flashcards

Q: What is a CUDA stream?
A: An ordered queue of GPU work; kernels in one stream run in order and asynchronously to the CPU, and different streams can overlap.

Q: Why do engines pad decode batches to fixed sizes?
A: CUDA graphs are captured for specific shapes; padding a batch of 13 to the captured size 16 lets the step replay a graph instead of launching kernels individually.

Q: Difference between PIECEWISE and FULL CUDA graph modes in vLLM?
A: PIECEWISE captures everything except attention (which runs eagerly between graph pieces); FULL also captures attention, which needs a graph-compatible attention backend.

Q: What does FlashAttention avoid?
A: Materializing the N×N attention score matrix in HBM: it tiles Q, K, V through on-chip memory and computes softmax online.

Q: What do vLLM's -O0 to -O3 flags control?
A: Optimization level: -O0 none, -O1 simple fusions and PIECEWISE graphs, -O2 (default) more fusions and FULL_AND_PIECEWISE graphs, -O3 aggressive (currently same as -O2).

Q: What is the overlap scheduler?
A: Preparing the next batch on the CPU while the GPU executes the current one, hiding scheduling overhead (SGLang's default; vLLM has async scheduling).

Q: Why does startup time matter for serving?
A: It bounds how quickly autoscaling can add capacity and how fast a crashed replica recovers; compile caches, graph capture and weight loading dominate it.

Q: What is Triton?
A: A Python-embedded language and compiler for writing block-level GPU kernels, used by torch.compile's Inductor and as a portable attention backend.
