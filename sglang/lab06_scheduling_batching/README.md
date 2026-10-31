# Lab 06 - Continuous Batching & Scheduler Knobs

**Goal:** understand the three scheduler settings that most affect
throughput/latency trade-offs once RadixAttention and precision are already
handled: overlap scheduling, chunked prefill, and queue ordering policy.

## Background

SGLang's scheduler runs a loop: at every iteration, decide which requests
(new prefills and in-flight decodes) to run together as one GPU batch. Three
knobs shape this:

- **Overlap scheduling** (on by default): the CPU thread that decides the
  *next* batch runs concurrently with the GPU executing the *current* batch,
  instead of waiting for the GPU to finish first. This hides scheduling and
  Python-side overhead behind GPU compute -- pure upside on GPUs with fast
  CPUs, at the cost of one extra batch of "lookahead" latency for aborts/rate
  limiting to take effect.
- **Chunked prefill** (`--chunked-prefill-size N`): normally, prefilling a
  very long prompt is one big matmul-heavy step. If a 4096-token prompt
  needs prefilling right when 30 other requests are mid-decode, the naive
  approach makes those 30 requests wait for the entire prefill to finish.
  Chunked prefill splits long prefills into `N`-token pieces and interleaves
  them with other requests' decode steps, trading a bit of total throughput
  for dramatically better tail latency.
- **Schedule policy** (`--schedule-policy`): controls which queued request
  is picked next. `fcfs` (first-come-first-served) is simple and fair.
  `lpm` (longest-prefix-match) instead prioritizes requests that share a
  RadixAttention prefix with what's currently cached/running, improving
  cache locality (fewer evictions, more reuse) under prefix-heavy traffic
  at some cost to strict request-arrival fairness.

## What this lab does

Three back-to-back A/B comparisons, each holding everything else fixed:

- **[A]** `--disable-overlap-schedule` off vs on, random workload.
- **[B]** `--chunked-prefill-size -1` (disabled) vs `2048` (enabled), with
  4096-token input prompts, so a real prefill/decode contention scenario
  actually occurs.
- **[C]** `--schedule-policy fcfs` vs `lpm`, on the same
  `generated-shared-prefix` workload from lab05.

## Run it

```bash
cd vllm_labs/sglang/lab06_scheduling_batching
./run_lab.sh
```

## Expected takeaways

- [A] Overlap scheduling should give a small, consistent throughput/TPOT
  improvement with no real downside -- there's rarely a reason to disable it
  in production; it exists mainly as a debugging knob.
- [B] Look at `p99_ttft_ms` in `results/lab06_scheduling_batching/bench_chunked_off.log`
  vs `bench_chunked_on.log`: disabling chunked prefill should show occasional
  very high tail latencies for requests unlucky enough to be queued behind a
  long prompt's prefill. Total throughput may be similar or even slightly
  higher without chunking (less scheduling overhead) -- this is a genuine
  throughput-vs-tail-latency trade-off, not a strict win either way.
- [C] `lpm` should match or beat `fcfs` on throughput/TTFT for the
  prefix-heavy workload, since it actively schedules cache-friendly requests
  together instead of processing strictly in arrival order.
- None of these knobs require touching the model or its weights at all --
  they're pure serving-layer tuning, and usually the *first* place to look
  before reaching for more GPUs or a smaller model.

Next: `lab07_cuda_graph_torch_compile` looks at kernel-launch-level
optimizations: CUDA graphs and `torch.compile`.
