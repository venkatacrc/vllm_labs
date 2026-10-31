---
title: Load balancers, API gateways and admission control
track: Cloud & Global
order: 2
---

# Load balancers, API gateways and admission control

## L4 vs L7

- **L4 (transport) load balancers** forward TCP/UDP connections by IP/port.
  Fast and simple, but one long-lived HTTP/2 connection carrying thousands
  of requests lands on one backend.
- **L7 (application) load balancers / proxies** (Envoy, NGINX, cloud ALBs)
  parse HTTP, so they can route per request, by path or header, retry,
  apply timeouts, and terminate TLS. Classic policies: round-robin,
  least-request, consistent hashing (session affinity).
- **Inference-aware L7** (llm-d EPP via Envoy `ext_proc`, Gateway API
  Inference Extension) adds model-server state: queue depth, KV-cache
  usage, prefix-cache locality, predicted latency, P/D roles, LoRA
  adapters loaded. llm-d lab 01 shows why this beats round-robin and
  least-request.

## The API gateway layer

In front of the inference router, a production API usually has an **API
gateway** that does what the GPUs shouldn't:

| Concern | How |
|---|---|
| Authentication | API keys, OAuth/JWT validation, mTLS for service-to-service |
| Tenant identity | derive tenant/tier from the token, **strip** client-supplied `x-llm-d-*` headers, inject authoritative `x-llm-d-inference-fairness-id` / `x-llm-d-inference-objective` |
| Rate limiting | requests/min **and tokens/min** per key (token-bucket); prompt tokens can be counted up front, output tokens charged as they stream |
| Quotas & billing | per-tenant token accounting, usage events to a billing pipeline |
| Model routing | map public model names to pools/versions, A/B and canary splits |
| Safety | input/output moderation, PII filtering, prompt-injection defenses |
| Observability | request ids, traces (W3C traceparent), access logs |

## Admission control and backpressure

Accepting more work than you can serve only converts overload into
timeouts. Layers of defense:

1. **Gateway rate limits** per tenant (protects everyone from one tenant).
2. **Router flow control** (llm-d lab 03): per-tenant fair queues with
   priorities; hold requests in the router while the pool is saturated;
   reject (429) when a band's queue or TTL limits are hit.
3. **Engine limits**: `--max-num-seqs`, `--max-num-batched-tokens`,
   preemption when KV runs out.
4. **Client behavior**: retries with exponential backoff and jitter, and
   respecting `Retry-After`; otherwise retry storms amplify overload.

Shedding low-priority (batch, best-effort) traffic first keeps interactive
latency within SLO during spikes.

## Streaming through proxies

Every proxy on the path must stream SSE without buffering, allow long
request durations (minutes for long generations), and propagate client
disconnects so the engine can abort the request and free its KV cache.

## Flashcards

Q: Why is L4 load balancing a poor fit for LLM APIs?
A: It balances connections, not requests; long-lived HTTP/2 connections carrying many requests pin to one backend, and it can't see request cost or server state.

Q: What does an inference-aware router consider that a classic L7 LB doesn't?
A: Model-server queue depth, KV-cache utilization, which replica holds the prompt's prefix, predicted TTFT/TPOT, P/D role and loaded adapters.

Q: Why should the API gateway strip client-supplied x-llm-d-* headers?
A: Otherwise clients could claim another tenant's identity or premium priority; the gateway derives tenant/tier from the authenticated token and injects trusted headers.

Q: Why rate-limit tokens per minute, not just requests?
A: Request cost varies by orders of magnitude with prompt and output length; token limits track actual GPU work.

Q: What is the point of router-level flow control vs engine queues?
A: The router can queue fairly per tenant with priorities and dispatch to the best replica when capacity frees up, instead of FIFO queues stuck inside individual replicas.

Q: Why must proxies propagate client disconnects?
A: So the engine aborts the generation and frees its KV cache instead of computing tokens nobody will read.

Q: What client behavior prevents retry storms?
A: Exponential backoff with jitter and honoring Retry-After on 429/503 responses.
