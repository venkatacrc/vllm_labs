# llm-d Lab 03: Flow control and multi-tenancy

## Concept

When offered load exceeds capacity, requests queue *somewhere*. Without
flow control they queue inside each vLLM replica's scheduler, which is
FIFO across all users: a batch job that dumps 2,000 requests makes every
interactive user wait behind it. Once a request is committed to a replica's
queue it is also stuck there even if another replica frees up first.

llm-d **Flow Control** moves the queue into the router:

- Each request is classified into a **flow** = fairness id
  (`x-llm-d-inference-fairness-id` header, e.g. tenant) + priority (from the
  `x-llm-d-inference-objective` header, resolved via `InferenceObjective`
  resources on Kubernetes).
- Flows live in **priority bands**. Each dispatch cycle serves the highest
  band first, rotates between tenants inside a band (**fairness policy**,
  e.g. round-robin) and orders requests within a flow (**ordering policy**,
  e.g. FCFS or SLO-aware).
- A **saturation detector** (`concurrency-detector`: max in-flight requests
  per replica) gates dispatch. While the pool is saturated, requests wait
  in the EPP; when capacity frees up they go to the *best* replica at that
  moment ("no-regret scheduling").
- Bands have byte/request limits and a TTL, so excess load is rejected
  early (HTTP 429/503) instead of timing out after minutes.

Production pattern: the API gateway strips client-supplied `x-llm-d-*`
headers and injects authoritative tenant/objective headers from the
authenticated API key or JWT.

## What this lab does

Runs a quiet tenant (concurrency 4) and a noisy tenant (concurrency 384)
concurrently, first through a plain EPP, then through an EPP with the
`flowControl` feature gate (`configs/epp/flow-control.yaml`, adapted from
upstream `guides/flow-control`).

## Run it

```bash
cd llmd/lab03_flow_control && ./run_lab.sh
```

## What to look for

- `quiet` tenant `ttft_p99_ms`: large drop with flow control.
- `ALL` throughput: about the same - flow control reorders work, it does
  not create capacity.
- Tune `maxConcurrency` in the config: too high and nothing queues in the
  EPP; too low and replicas idle.
