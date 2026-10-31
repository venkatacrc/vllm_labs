# llm-d Lab 01: Why LLM traffic needs an inference-aware router

## Concept

A classic load balancer assumes requests are roughly interchangeable and
replicas are stateless. LLM serving breaks both assumptions:

- **Requests differ by orders of magnitude.** A 50-token chat turn and a
  30,000-token RAG prompt cost wildly different amounts of prefill compute,
  and output length is unknown up front. "Number of open connections" is a
  poor proxy for load.
- **Replicas are stateful.** Each vLLM replica holds a prefix cache (KV
  blocks for prompts it has seen). Sending a request to the replica that
  already holds its 6,000-token system prompt skips that prefill entirely.
  Sending it anywhere else recomputes it *and* evicts something useful.

The llm-d **EPP** (Endpoint Picker) is plugged into Envoy as an `ext_proc`
filter. For every request Envoy streams the headers/body to the EPP, which
runs a **filter → score → pick** pipeline over the replicas and answers
with the chosen endpoint in the `x-gateway-destination-endpoint` header;
Envoy's `ORIGINAL_DST` cluster then forwards the request there.

The no-Kubernetes optimized-baseline profile used here combines:

| Scorer | Weight | Signal |
|---|---|---|
| `prefix-cache-scorer` | 3 | How much of this prompt's prefix the EPP predicts is cached on each replica (it hashes prompt blocks and remembers where it sent them). |
| `queue-scorer` | 2 | Waiting requests per replica, scraped from vLLM `/metrics`. |
| `kv-cache-utilization-scorer` | 2 | KV-cache memory pressure per replica. |
| `no-hit-lru-scorer` | 2 | For prompts with no cache hit anywhere, spread them to the least-recently-used replica so new prefixes don't pile up. |

## Setup

```
client (loadgen.py) ──> Envoy :8081 ──ext_proc──> EPP :9002
                           │                       │ scrapes /metrics
                           └──> vllm-0 :8000  (GPUs 0,1)
                                vllm-1 :8001  (GPUs 2,3)
                                vllm-2 :8002  (GPUs 4,5)
                                vllm-3 :8003  (GPUs 6,7)
```

The EPP gets its replica list from `endpoints.yaml` (file discovery). The
configs are copied verbatim from upstream
`guides/no-kubernetes-deployment/router/`.

## Workload

48 "tenants", each with a ~6,000-token system prompt, sending 960 short
questions in random order at concurrency 96 (`loadgen.py --workload
shared-prefix`). The total prefix working set (~290K tokens) fits in the
pool's combined KV cache only if each prefix lives on *one* replica.

## Run it

```bash
cd llmd/lab01_router_basics && ./run_lab.sh
# smaller model / more replicas:
LLMD_MODEL=meta-llama/Llama-3.1-8B-Instruct LLMD_TP=1 LLMD_REPLICAS=8 ./run_lab.sh
```

## What to look for

- `prefix_hit_rate`: round-robin should be far below the EPP.
- `ttft_p50_ms` / `ttft_p99_ms`: the EPP should cut TTFT sharply.
- `max_replica_share`: should stay close to 0.25 with the EPP; prefix
  affinity without load awareness would create hot replicas (see lab 02).
- Least-request vs round-robin: load awareness alone fixes queueing
  imbalance but not cache misses.
