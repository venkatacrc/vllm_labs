---
title: Reliability, rollouts and observability
track: Cloud & Global
order: 4
---

# Reliability, rollouts and observability

## SLOs and error budgets

- **SLI** (indicator): a measured ratio, e.g. "fraction of requests with
  TTFT < 1 s and no 5xx".
- **SLO** (objective): the target, e.g. 99.5% over 28 days.
- **Error budget**: 1 − SLO = 0.5% of requests may miss. While budget
  remains, ship features; when it's burning fast, freeze risky changes and
  fix reliability. Alert on **burn rate** (budget consumed per hour), not
  on single blips.

For LLM APIs, define separate SLOs for availability (non-5xx), TTFT and
TPOT, per tier (interactive vs batch).

## Failure modes specific to LLM serving

| Failure | Symptom | Mitigation |
|---|---|---|
| GPU Xid / ECC / NVLink errors | pod crash, NCCL hang | health checks, node auto-repair, LWS group restart |
| KV cache exhaustion | preemptions, TTFT spikes | admission control, KV offload, more replicas |
| Hot prefix / hot tenant | one replica saturated | load-aware scoring (not pure affinity), flow control |
| Slow cold starts | autoscale too late | warm pools, sleep/wake, scale on queue signals |
| Model/tokenizer mismatch | garbage output, failed tool parsing | pinned images and artifacts, canaries with eval checks |
| Retry storms | load doubles under failure | backoff + jitter, retry budgets, 429 with Retry-After |
| Long requests during deploy | dropped streams | graceful drain (`--shutdown-timeout`, preStop), connection draining |

## Safe rollouts

- **Rolling update**: replace pods a few at a time; readiness probes gate
  traffic.
- **Blue/green**: stand up the new version beside the old, switch traffic,
  keep the old one for instant rollback.
- **Canary**: send 1-5% of traffic (weighted HTTPRoute, gateway split) to
  the new model/engine version and compare SLIs and **quality metrics**
  (eval scores, refusal rates, tool-call parse errors) before ramping.
- **Shadow traffic**: mirror requests to the new version without returning
  its responses, to measure performance safely.
- Treat **model weights, chat templates, engine version and flags** as one
  versioned artifact - a change in any of them is a deploy.

## Observability

- **Metrics** (Prometheus): engine metrics (`vllm:num_requests_running`,
  `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`,
  `vllm:prefix_cache_hits_total`, TTFT/ITL histograms), router metrics (EPP
  in-flight, flow-control queues, scheduling decisions), GPU metrics (DCGM
  exporter: SM activity, memory, power, Xid errors), gateway metrics.
- **Logs**: structured, sampled; never log prompts by default (privacy).
- **Traces** (OpenTelemetry → Jaeger/Tempo): one trace per request across
  gateway → Envoy → EPP → model server shows where time went (queueing vs
  prefill vs decode vs network).
- **Dashboards** per layer and an SLO dashboard; llm-d ships Grafana
  dashboards for the router and model servers.

## Disaster recovery

Define **RTO** (how fast you must recover) and **RPO** (how much data you
may lose). Inference is mostly stateless, so DR is about capacity and
artifacts: model weights replicated to every region's registry, infra as
code to rebuild a cluster, tested regional failover (game days), and
enough quota in the fallback regions.

## Flashcards

Q: What is an error budget?
A: The allowed unreliability (1 − SLO); e.g. a 99.5% SLO allows 0.5% of requests to miss. It governs the trade-off between shipping changes and reliability work.

Q: Why alert on burn rate instead of individual errors?
A: Burn rate measures how fast the error budget is being consumed, catching sustained problems quickly while ignoring harmless blips.

Q: What should a canary for a new model version check besides latency?
A: Quality: eval scores, refusal rates, tool-call/JSON parse errors and user feedback signals.

Q: Which vLLM metrics show overload before users complain?
A: vllm:num_requests_waiting rising, vllm:kv_cache_usage_perc near 1, preemptions, and TTFT histogram tails.

Q: What do distributed traces add over metrics?
A: Per-request breakdown across hops (gateway, router, queue, prefill, decode), showing where an individual slow request spent its time.

Q: Define RTO and RPO.
A: RTO: maximum acceptable time to restore service. RPO: maximum acceptable data loss measured in time.

Q: Why treat weights, chat template, engine version and flags as one artifact?
A: Any of them can change outputs or performance; versioning them together makes rollouts and rollbacks reproducible.
