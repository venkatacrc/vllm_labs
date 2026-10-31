---
title: The life of one request
track: How it fits together
order: 1
---

# The life of one request

A user in Berlin sends a chat message to `api.example.com`. Here is every
hop, what it costs, and which lab or module covers it.

```
 ① DNS          ② Edge PoP          ③ Regional gateway      ④ llm-d router          ⑤ Model server         ⑥ GPU
 GeoDNS ──► anycast TLS/WAF ──► auth, quotas, tenant ──► Envoy + EPP pick ──► vLLM/SGLang schedule ──► prefill, decode
   ~10 ms*      ~5-20 ms           ~1-5 ms                 ~1-3 ms                queue: 0 .. seconds     ms .. seconds
                                                                                    ◄──── tokens stream back over SSE ────
```
\* cached DNS answers cost ~0.

1. **DNS** (Cloud 1). GeoDNS answers with the anycast IP of the global
   load balancer; the EU region is chosen for latency and data residency.
2. **Edge** (Cloud 1). The nearest PoP terminates TCP+TLS (saving ~2 round
   trips to the origin), applies WAF/DDoS rules, and forwards over the
   provider backbone to `eu-west`.
3. **API gateway** (Cloud 2, 5). Validates the API key/JWT, looks up the
   tenant (tier: premium), checks token-rate limits, strips any client
   `x-llm-d-*` headers and injects `x-llm-d-inference-fairness-id: acme`
   and `x-llm-d-inference-objective: premium-traffic`, starts a trace, maps
   model `chat-large` → InferencePool `qwen3-32b-prod`.
4. **llm-d router** (llm-d labs 01-03). Envoy receives the request and
   streams it to the EPP over `ext_proc`. The EPP:
   - classifies the flow (tenant + priority) and, if the pool is saturated,
     queues it fairly (flow control);
   - hashes the prompt's blocks: the 3,000-token system prompt of this app
     is cached on replicas 3 and 7 (prefix-cache affinity);
   - filters to those, scores by queued prompt tokens (token load), picks
     replica 7, and returns `x-gateway-destination-endpoint: 10.4.2.17:8000`.
   In P/D mode it picks a prefill pod and a decode pod instead (llm-d lab 05).
5. **Model server** (vLLM/SGLang labs). The API server applies the chat
   template and tokenizes. The scheduler admits the request into the next
   continuous batch if KV blocks are available (else it waits or preempts).
   Prefix caching finds 3,000 tokens already cached; only the new 150 tokens
   are prefilled (chunked so running decodes aren't stalled).
6. **GPU** (Foundations 1-5). Prefill runs compute-bound GEMMs and
   FlashAttention-4 kernels; then each decode step streams all weights
   (FP8: half the bytes) plus this sequence's KV from HBM, inside a captured
   CUDA graph, verifying 3 EAGLE-3 draft tokens per step when acceptance
   is high.
7. **Streaming back**. Each detokenized chunk is sent as an SSE event through
   Envoy, the gateway and the edge. The gateway counts output tokens for
   billing; the trace records TTFT and TPOT per hop.
8. **After the response**. The KV blocks stay in the prefix cache (and are
   offloaded to CPU RAM when evicted) so the user's next turn hits the cache
   - as long as the router sends it back to replica 7 or its CPU tier.

## Where the latency goes (typical 32B model, warm cache)

| Component | Budget |
|---|---|
| Network to region (TLS at edge) | 20-60 ms |
| Gateway + router | 2-8 ms |
| Queueing | 0 ms at healthy load; seconds when overloaded |
| Prefill 150 new tokens (3K cached) | ~20-40 ms |
| Prefill 3K tokens uncached | ~150-300 ms |
| Decode, per token | 10-30 ms |

The single biggest lever on TTFT is usually **not recomputing the prefix**
(cache + cache-aware routing); on TPOT it is **bytes per step** (precision,
batch, speculation); on tail latency it is **queueing** (capacity, flow
control, autoscaling).

## Flashcards

Q: What does the API gateway inject for llm-d flow control?
A: The authoritative x-llm-d-inference-fairness-id (tenant) and x-llm-d-inference-objective (priority class) headers, after stripping client-supplied ones.

Q: How does the EPP tell Envoy which replica to use?
A: It returns the x-gateway-destination-endpoint header with the chosen pod's address; Envoy's ORIGINAL_DST cluster forwards there.

Q: Biggest lever on TTFT for multi-turn chat?
A: Avoiding prefix recomputation: prefix caching on the server plus routing the next turn to the replica (or CPU tier) that holds the conversation's KV.

Q: Biggest lever on TPOT?
A: Bytes moved per decode step: lower-precision weights/KV, good batching, CUDA graphs, and speculative decoding.

Q: What drives tail latency under load?
A: Queueing: too little capacity, unfair FIFO queues, and slow autoscaling.

Q: Why does the edge PoP reduce latency even though GPUs are in one region?
A: TCP and TLS handshakes complete with a nearby PoP (short RTT), and the request then travels over a warm, optimized backbone connection.
