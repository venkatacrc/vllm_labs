---
title: Measuring LLM serving performance
track: Foundations
order: 8
---

# Measuring LLM serving performance

## Latency metrics

```
request sent ──► first token ──► token 2 ──► ... ──► last token
|<--- TTFT --->|<- ITL ->|                         |
|<------------------------- E2E latency ---------->|
TPOT = (E2E − TTFT) / (output_tokens − 1)
```

- **TTFT** (time to first token): queueing + prefill (+ network). What a chat
  user feels as "responsiveness".
- **ITL** (inter-token latency): gap between consecutive streamed tokens;
  **TPOT** (time per output token) is its per-request average. Reading speed
  is ~5-10 tokens/s, so TPOT under ~50 ms feels instant for chat; agents
  and code generation want much lower.
- **E2E latency**: total; dominates for non-streaming and agentic loops.

Report **percentiles** (p50, p90, p99), not just means: the p99 is what one
in a hundred users sees, and in agentic workflows with many calls per task,
tail latency compounds.

## Throughput metrics

- **Output tokens/s** (decode throughput) and **total tokens/s** (input +
  output).
- **Requests/s** at a given latency SLO.
- **Goodput**: requests/s that *meet* the SLO (e.g. TTFT < 1 s and
  TPOT < 50 ms). Throughput at unbounded latency is not useful.
- **Per-GPU** or **per-dollar** normalization when comparing configs with
  different GPU counts (TP=8 vs 8 x TP=1).

## Load generation

- **Closed loop** (fixed concurrency, `--max-concurrency`): each client sends
  the next request when the previous finishes. Good for sweeping.
- **Open loop** (fixed arrival rate, `--request-rate`, Poisson arrivals):
  closer to real traffic; queues grow without bound past capacity.
- Workload shape matters more than anything: input/output lengths, prefix
  sharing, multi-turn reuse, burstiness. Benchmarks with random tokens get
  0% prefix reuse; real chat traffic gets a lot.

## Queueing intuition

**Little's law**: L = λ × W (items in system = arrival rate × time in
system). At 20 req/s and 5 s average latency, 100 requests are in flight -
that's the concurrency your servers must hold (and the KV cache must fit).

As utilization approaches 100%, waiting time grows sharply (an M/M/1 queue
waits ~ρ/(1−ρ) service times; at 90% utilization that's 9x). That is why
latency-sensitive services run with headroom, and why the router should
queue centrally (flow control) instead of overfilling replica queues.

## Benchmarking hygiene

- Warm up (compile caches, CUDA graphs, first-request costs) before measuring.
- Fix seeds, input/output lengths, and `ignore_eos` for comparable output
  lengths.
- Change one variable at a time; record engine version, flags, GPU clocks.
- Watch for client bottlenecks (a single Python client can't drive 10k tok/s
  without async I/O).
- Check quality as well as speed when changing numerics.

## Flashcards

Q: Define TTFT, TPOT and ITL.
A: TTFT: time from sending the request to the first output token. ITL: time between consecutive streamed tokens. TPOT: average time per output token after the first, (E2E − TTFT)/(n − 1).

Q: What is goodput?
A: The rate of requests that meet the latency SLO; throughput that violates the SLO doesn't count.

Q: State Little's law and apply it to 20 req/s at 5 s latency.
A: L = λ × W, so 20 × 5 = 100 requests in flight on average.

Q: Closed-loop vs open-loop load generation?
A: Closed loop keeps a fixed number of requests outstanding (concurrency); open loop sends at a fixed arrival rate regardless of completions, which exposes queue growth past capacity.

Q: Why report p99 rather than mean latency?
A: Means hide the tail; p99 is what 1% of users see, and tails compound across multi-call agentic workflows.

Q: Why do random-token benchmarks understate prefix caching?
A: Random prompts share no prefixes, so cache hit rate is ~0; real chat/RAG/agent traffic reuses system prompts, documents and history heavily.

Q: Why run inference services below 100% utilization?
A: Queueing delay grows like ρ/(1−ρ): at 90% utilization waits are ~9 service times, so headroom keeps tail latency bounded.
