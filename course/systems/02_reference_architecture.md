---
title: Reference architecture for a global LLM service
track: How it fits together
order: 2
---

# Reference architecture for a global LLM service

```
                           ┌──────────────────────── Global ────────────────────────┐
  clients ─► GeoDNS ─► anycast edge (TLS, WAF, DDoS) ─► global LB (latency/geo/capacity)
                           └───────────────┬──────────────────────┬─────────────────┘
                                           ▼                      ▼
            ┌────────────── Region us-east ──────────┐   ┌──── Region eu-west ────┐
            │ API gateway: authn, quotas, tenant hdrs│   │ (same stack; EU data   │
            │ model routing, moderation, metering    │   │  stays in EU)          │
            │                   │                    │   └────────────────────────┘
            │   ┌───────── Kubernetes cluster(s) ─────────────────────────┐      │
            │   │ llm-d router (Envoy + EPP, 2+ replicas, HA)              │      │
            │   │   flow control → filters → scorers → picker              │      │
            │   │ InferencePool A: Qwen3-32B FP8, aggregated, TP=2 × N     │      │
            │   │ InferencePool B: DeepSeek MoE, P/D + wide-EP (LWS)       │      │
            │   │ KV tiers: HBM → CPU RAM → NVMe/shared FS                  │      │
            │   │ KEDA autoscaling on EPP metrics; warm/sleeping replicas  │      │
            │   └──────────────────────────────────────────────────────────┘      │
            │ Model registry mirror · image registry · secrets manager           │
            │ Prometheus/Grafana · OTel traces · logs · SLO alerts               │
            └─────────────────────────────────────────────────────────────────────┘
  Control plane: GitOps (Helm/Kustomize) · CI with perf + eval gates · canary controller
```

## Layer responsibilities

| Layer | Owns | Course |
|---|---|---|
| Global traffic | DNS, anycast, edge TLS, region selection, failover | Cloud 1 |
| Gateway | identity, quotas, tenant/priority headers, model mapping, safety, billing | Cloud 2, 5 |
| Inference router (llm-d) | replica choice (cache/load/latency aware), P/D scheduling, fair queuing, pool discovery | llm-d 01-05, 08 |
| Model server (vLLM/SGLang) | batching, KV paging and caching, kernels, quantization, speculation, parallelism | vLLM & SGLang tracks |
| Accelerators | HBM, tensor cores, NVLink, RDMA NICs | Foundations 1-3 |
| Platform | Kubernetes, LWS, GPU operators, autoscaling, observability, rollouts | Foundations 7, llm-d 07-09, Cloud 3-4 |

## Design decisions and defaults

- **Engine**: vLLM for breadth of models, hardware and features; SGLang for
  RadixAttention-heavy workloads, structured generation and some
  MoE/DeepSeek optimizations. llm-d supports both behind the same router.
- **Precision**: BF16 to validate, then FP8 (W8A8 + FP8 KV) as the
  production default on Hopper/Blackwell; FP4 (NVFP4/MXFP4) where evals
  allow.
- **Topology**: smallest TP that fits with KV headroom; DP replicas behind
  the llm-d optimized baseline. Add P/D when prompts are long and TPOT SLOs
  strict; wide-EP for large MoE models across nodes.
- **Caching**: prefix caching always on; cache-aware routing; CPU KV tier for
  multi-turn/agentic traffic; precise prefix routing when the pool is large.
- **Admission**: gateway rate limits + router flow control with priority
  bands; batch traffic as best-effort or through a batch API.
- **Scaling**: KEDA on EPP queue/saturation signals; warm capacity sized to
  startup time × growth rate.
- **Release safety**: perf and eval gates in CI; canaries per region; model,
  template, engine and flags versioned together.

## Flashcards

Q: Which layer decides which GPU replica serves a request?
A: The inference router: llm-d's EPP (behind Envoy) using cache, load and latency signals over the InferencePool.

Q: Which layer enforces tenant identity and quotas?
A: The API gateway, before traffic reaches the inference router.

Q: Default precision recommendation on Blackwell for production?
A: FP8 weights and activations plus FP8 KV cache, after validating quality against BF16; FP4 where evals allow.

Q: When should you add P/D disaggregation?
A: Long prompts, large models and strict TPOT SLOs where prefill interference hurts decode; for short prompts aggregated serving usually wins.

Q: Why run the router with multiple replicas?
A: It's on the request path for every request; HA (2+ replicas) avoids a single point of failure. Precise prefix routing has every router replica subscribe to every pod's KV events.

Q: What goes in the platform control plane?
A: GitOps-managed manifests (Helm/Kustomize), CI with performance and quality gates, and a canary/rollout controller.
