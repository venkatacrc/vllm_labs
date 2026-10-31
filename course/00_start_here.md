---
title: Start here
track: Orientation
order: 0
---

# Start here: LLM serving, from one GPU to a global service

This course takes you from a naive FP32 model running in a Python loop to a
globally deployed, autoscaled, multi-tenant LLM service. It has five parts:

1. **Foundations** - the systems knowledge every later lab leans on: how a
   GPU executes work, why decode is memory-bound, what the KV cache costs,
   how GPUs talk to each other, how CUDA launches work, number formats,
   HTTP streaming, containers, Kubernetes, and how to measure latency
   properly.
2. **Engine labs** - make *one* model server fast, twice, with two engines:
   - **vLLM track** (`labs/`, 12 labs): FP32 baseline, precision, continuous
     batching and paged KV cache, attention backends (FlashAttention 4,
     FlashInfer, Triton), prefix caching, chunked prefill, CUDA graphs and
     compilation levels, INT4 AWQ/GPTQ, FP8/MXFP8/MXFP4, speculative decoding
     (draft model, EAGLE-3, n-gram), tensor parallelism, capstone.
   - **SGLang track** (`sglang/`, 14 labs): the same journey in SGLang plus
     RadixAttention, the overlap scheduler, structured output, MoE expert
     parallelism, attention backends and HiCache.
3. **Fleet labs** - the **llm-d track** (`llmd/`, 10 labs): inference-aware
   routing (EPP), scorer design, flow control, tiered KV cache, prefill/
   decode disaggregation, wide expert parallelism, then the upstream
   Kubernetes guides and autoscaling.
4. **Cloud & global deployment** - regions, DNS and anycast, edge and
   gateways, capacity and cost, reliability and security.
5. **How it fits together** - one request's life from DNS to GPU and back,
   a reference architecture, and a symptom-to-fix optimization playbook.

## Hardware assumed

One node with 8x NVIDIA B200 (Blackwell, SM100, ~180 GB HBM3e each,
NVLink 5 through NVSwitch), CUDA 13, Python 3.11, vLLM 0.26+ in a conda env
for the vLLM track, a separate `sglang-labs` env for SGLang, and Docker
(+ minikube for the Kubernetes labs) for llm-d.

## Suggested order

| Week | Read | Do |
|---|---|---|
| 1 | Foundations 1-4 | vLLM labs 00-06 |
| 2 | Foundations 5-8 | vLLM labs 07-11, SGLang labs 00-07 |
| 3 | How it fits together 1 (request lifecycle) | SGLang labs 08-13, llm-d labs 00-03 |
| 4 | Cloud 1-5, How it fits together 2-3 | llm-d labs 04-09 |

## Using the app

Every module and lab README is a **lesson**. Modules end with
**flashcards**: use *Learn* to flip through them and *Test* to quiz
yourself; cards you mark *Review* come back first. On a phone, swipe left/
right to move between cards. Your progress is stored on the device.

## Flashcards

Q: What are the three layers this course optimizes, from inside out?
A: A single model server (engine: vLLM or SGLang), a fleet of replicas behind an inference-aware router (llm-d), and the global service around it (regions, edge, gateways, autoscaling).

Q: Which track covers prefill/decode disaggregation and inference-aware routing?
A: The llm-d track (`llmd/`).

Q: Why is there both a vLLM and an SGLang track?
A: They are the two leading open-source engines; doing the same optimizations in both shows which ideas are universal (batching, paging, caching, quantization, speculation) and which are engine-specific (RadixAttention, overlap scheduler, HiCache vs OffloadingConnector).
