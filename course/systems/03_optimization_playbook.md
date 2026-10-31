---
title: The optimization playbook
track: How it fits together
order: 3
---

# The optimization playbook

A checklist that maps symptoms to the lab that fixes them.

## Single replica (vLLM / SGLang)

| Symptom | Likely cause | Try | Lab |
|---|---|---|---|
| Very low tokens/s, GPU underused | FP32 or naive HF loop, static batching | BF16, a real engine with continuous batching | vLLM 00-02, SGLang 01-02 |
| Few concurrent sequences fit | KV cache too big | FP8 KV, GQA model, smaller max-model-len, more GPUs | vLLM 08, Foundations 2 |
| High TTFT with shared system prompts | prefix recomputation | prefix caching / RadixAttention | vLLM 04, SGLang 05 |
| TPOT spikes when long prompts arrive | prefill interrupting decode | chunked prefill, tune max-num-batched-tokens; later P/D | vLLM 05, SGLang 06, llm-d 05 |
| Slow decode at low concurrency | launch overhead, eager mode | CUDA graphs, -O2, overlap scheduling | vLLM 06, SGLang 07 |
| Slow decode at low concurrency, graphs on | memory-bound weights | FP8/MXFP8/FP4 or INT4 weights; speculative decoding | vLLM 07-09, SGLang 03, 08 |
| Long-prompt prefill slow | attention kernel | FA4 / FlashInfer / TRT-LLM backends | vLLM 03, SGLang 12 |
| Model doesn't fit / latency too high | single-GPU limits | tensor parallelism (inside NVLink) | vLLM 10, SGLang 04 |
| MoE model inefficient | TP slicing experts | DP attention + expert parallelism, EPLB | SGLang 10, llm-d 06 |
| JSON outputs invalid | unconstrained sampling | structured outputs (grammar backends) | SGLang 09 |

## Fleet (llm-d)

| Symptom | Try | Lab |
|---|---|---|
| Prefix hit rate low despite shared prompts | prefix-cache-aware routing | llm-d 01 |
| One replica hot, others idle | load-aware scorers with affinity, calibrate peakPrefillThroughput | llm-d 02 |
| One tenant's burst hurts everyone | flow control with fairness + priorities | llm-d 03 |
| Multi-turn sessions miss cache after a while | CPU/NVMe KV tier + tier-aware routing | llm-d 04, SGLang 13 |
| Approximate cache routing mispredicts at scale | precise prefix-cache routing (KV events) | llm-d 08 |
| Long prompts + strict TPOT | P/D disaggregation, scale P and D separately | llm-d 05, 08 |
| Big MoE across nodes | wide-EP with DeepEP + P/D | llm-d 06 |
| Latency SLOs differ per request | predicted-latency routing with SLO headers | llm-d 08 |
| Capacity lags demand | KEDA on EPP signals, warm pools, sleep/wake | llm-d 09 |

## Order of operations

1. Measure a baseline with the target workload shape (lengths, prefix
   reuse), not just random tokens.
2. Get the engine basics right: BF16 → FP8, prefix caching, chunked prefill,
   CUDA graphs, right TP.
3. Scale out with DP replicas behind an inference-aware router.
4. Add the specialised paths only when their symptom shows up: P/D, KV
   tiers, wide-EP, flow control, predicted latency.
5. Re-measure after each change, including quality.

## Flashcards

Q: First fix for TTFT problems with shared system prompts?
A: Turn on prefix caching (vLLM) / RadixAttention (SGLang); at fleet level, add prefix-cache-aware routing.

Q: Fix for TPOT spikes when long prompts arrive?
A: Chunked prefill with a tuned token budget; if that's not enough, prefill/decode disaggregation.

Q: Fix when one tenant's burst raises everyone's latency?
A: Router flow control: per-tenant fair queues with priority bands, plus gateway rate limits.

Q: Fix when multi-turn sessions lose their cache over time?
A: Add a CPU (and optionally NVMe/shared FS) KV tier and route with tier awareness.

Q: What should you measure before optimizing?
A: A baseline on the real workload shape (input/output lengths, prefix reuse, concurrency), with latency percentiles and quality.
