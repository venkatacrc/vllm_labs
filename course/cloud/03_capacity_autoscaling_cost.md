---
title: GPU capacity, autoscaling and cost
track: Cloud & Global
order: 3
---

# GPU capacity, autoscaling and cost

## Capacity planning from first principles

1. **Traffic model**: peak requests/s, input and output token distributions,
   prefix reuse, diurnal pattern per region.
2. **Per-replica capacity at SLO**: from benchmarks (the labs), the maximum
   requests/s one replica sustains while meeting p99 TTFT and TPOT
   targets - *goodput*, not peak throughput.
3. **Replicas** = peak load / per-replica goodput, plus headroom for
   failures (N+1 per AZ/region) and deploys.
4. **Memory check**: weights + KV for the concurrency implied by Little's
   law (concurrency = rate × latency) must fit, per replica.

## Cost per million tokens

```
cost per 1M output tokens = (GPU $/hour × GPUs) ÷ (output tokens/s × 3600) × 1,000,000
```

Illustrative: 8 GPUs at $5/GPU-hour = $40/h. At 20,000 output tokens/s
sustained that is 72M tokens/h → **≈ $0.56 per 1M output tokens**. At 30%
average utilization the effective cost is ~3x higher. Every optimization
in the engine labs moves this number: FP8 (~1.5-2x tokens/s), prefix
caching (fewer prefill FLOPs), speculative decoding (lower latency at low
load), and routing (higher cache hit rates per GPU).

Input tokens are cheaper than output tokens (prefill is parallel and
compute-efficient), which is why providers price them separately and why
cached input tokens are discounted further.

## Buying GPUs

| Option | Trade-off |
|---|---|
| On-demand | flexible, most expensive, may be unavailable for large GPUs |
| Reservations / committed use | cheaper per hour, you pay whether used or not |
| Spot / preemptible | cheapest, can be reclaimed with short notice: batch and offline inference |
| Multi-cloud / neoclouds | more capacity, more operational surface |

Common pattern: reserved capacity for the baseline, on-demand or another
region for peaks, spot for batch jobs (llm-d's batch gateway / async
processing fits here).

## Autoscaling LLM services

- **Scale on leading signals**: router in-flight requests, queue size, pool
  saturation, KV-cache pressure - not GPU utilization, which reads ~100%
  under any batching load (llm-d lab 09).
- **Startup latency dominates**: image pull + weight load + compile + CUDA
  graph capture can take minutes. Mitigations: pre-pulled images, weights on
  node-local NVMe or a fast distributed cache, compile caches, model
  streaming loaders, **sleep/wake** (fast model actuation: keep a vLLM
  process with weights parked in host memory, wake in seconds).
- **Scale-down carefully**: drain in-flight requests (`--shutdown-timeout`,
  preStop hooks); scaling a pool with warm prefix caches down loses those
  caches.
- **Cluster autoscaler / node provisioning** adds GPU nodes when pods are
  pending; for GPUs this can take many minutes, so keep warm spare nodes for
  bursty traffic.
- **Scale to zero** suits rarely used models (KEDA), at the cost of a cold
  start on the first request.

## Multi-model and multi-tenant efficiency

- **LoRA adapters**: serve many fine-tunes on one base model; the router can
  prefer replicas with the adapter loaded.
- **Model variants** across accelerator types (H100, B200, MI355X) with the
  router picking the cheapest pool that meets the SLO.
- **Priority tiers**: interactive traffic first, batch fills the gaps.

## Flashcards

Q: Formula for cost per 1M output tokens?
A: (GPU $/hour × number of GPUs) ÷ (output tokens/s × 3600) × 1,000,000.

Q: 8 GPUs at $5/hour sustaining 20k output tokens/s: cost per 1M tokens?
A: $40/h ÷ 72M tokens/h ≈ $0.56 per 1M output tokens (at full utilization).

Q: Why is GPU utilization a bad autoscaling signal for LLM serving?
A: Continuous batching keeps the GPU busy (~100%) at almost any load, so it neither rises before latency degrades nor distinguishes light from heavy load.

Q: What dominates autoscaling reaction time for LLM pods?
A: Startup latency: image pull, loading tens of GB of weights, torch.compile and CUDA graph capture.

Q: What is fast model actuation (sleep/wake)?
A: Keeping a model-server process with weights offloaded to host memory so it can wake and serve in seconds instead of cold-starting.

Q: Where do spot GPUs fit in an LLM platform?
A: Interruptible batch/offline inference and evaluation jobs; not latency-critical interactive serving.

Q: How do you size replicas for a service?
A: Peak load ÷ per-replica goodput at the SLO (from benchmarks), plus headroom for failures and deployments, checked against KV memory for the implied concurrency.

Q: Why are input tokens priced lower than output tokens?
A: Prefill processes prompt tokens in parallel and is compute-efficient, while each output token needs its own memory-bound decode step.
