# llm-d Lab 02: Inside the scheduler - filters, scorers, pickers

## Concept

For each request the EPP runs one or more **scheduling profiles**. A profile
is a pipeline:

1. **Filters** drop ineligible endpoints (wrong role in P/D, unhealthy,
   saturated, not cache-warm enough).
2. **Scorers** give each remaining endpoint a score in [0, 1]; scores are
   combined with the configured **weights**.
3. A **picker** chooses (`max-score-picker`, `weighted-random-picker`, ...).

Data feeding the scorers comes from **producers** and the **data layer**:
scraped model-server metrics (`metrics-data-source` +
`core-metrics-extractor`: queue length, KV-cache usage, running requests),
in-flight load tracked by the EPP itself (`inflight-load-producer`), and the
prefix index (`approx-prefix-cache-producer`, or exact KV events in the
precise prefix-cache path).

Two production profiles ship upstream:

- **No-Kubernetes optimized baseline**: `prefix-cache-scorer` (3) +
  `queue-scorer` (2) + `kv-cache-utilization-scorer` (2) +
  `no-hit-lru-scorer` (2).
- **v1.0 optimized baseline** (Kubernetes guide):
  `prefix-cache-affinity-filter` + `token-load-scorer`. The filter keeps
  each prefix group on its cache-warm endpoints *until* their estimated
  prefill backlog exceeds what they can absorb (`peakPrefillThroughput`,
  calibrated per model/accelerator); then the token-load scorer picks the
  endpoint with the fewest queued prompt tokens.

## What this lab does

1. Runs a **skewed** shared-prefix workload (6 hot groups, concurrency 128)
   under four EPP configs: `load-only`, `prefix-only`, `optimized-baseline`,
   `affinity-token-load` (configs in `llmd/configs/epp/`).
2. **Live pool change**: while traffic flows, rewrites `endpoints.yaml`
   (atomic rename) with one replica fewer. The EPP's file watcher reloads
   the pool and stops routing to the drained replica - the same mechanism a
   Kubernetes `InferencePool` gives you via pod readiness.

## Run it

```bash
cd llmd/lab02_scheduler_ablation && ./run_lab.sh
```

## What to look for

- The tension between **cache affinity** and **load balance**: prefix-only
  wins on `prefix_hit_rate` but loses on `ttft_p99_ms` and
  `max_replica_share`; load-only is the reverse; the combined profiles
  should get most of both.
- Calibration: the affinity filter's default `peakPrefillThroughput`
  (15928 tokens/s) was measured for Qwen3-32B TP=2 on H100. A B200 has
  ~2x the prefill throughput, so the filter spills too early. Re-measure with
  upstream `guides/recipes/router/calibration` and set it in
  `configs/epp/affinity-token-load.yaml`.
- `epp-*.log`: per-request scheduling decisions at `--v=2`.
