# Lab 05: Chunked prefill

## Concept

Prefill (processing a prompt's tokens for the first time) and decode
(generating one new token per sequence) have very different compute
profiles: prefill is compute-bound and scales with prompt length, decode is
comparatively cheap but latency-sensitive. Without chunked prefill, a
single very long prompt's prefill can occupy an entire scheduler step,
during which no other sequence's decode step runs - so a handful of
short, latency-sensitive requests can get stuck behind one long document
being ingested.

**Chunked prefill** splits a long prompt's prefill into smaller chunks and
interleaves them with other sequences' decode steps across multiple
scheduler iterations, instead of processing the whole prompt in one go.
Short requests keep making progress every step instead of queueing behind
a long one.

## What this lab does

Simulates realistic mixed traffic by firing **two concurrent workloads at
the same running server** (via `common/bench.run_concurrent`):

- A trickle of long documents: ~6000-token prompts, low request rate.
- A stream of short, latency-sensitive requests: ~100-token prompts, higher
  request rate.

Runs this mixed workload twice: once with `--no-enable-chunked-prefill`,
once with chunked prefill enabled (vLLM's current default). The metric
that matters is the **short-request stream's TTFT** while the long
requests are in flight.

## Run it

```bash
python labs/lab05_chunked_prefill/lab.py
python labs/lab05_chunked_prefill/lab.py --quick
```

## What to look for

- `chunked_prefill_off_short` vs `chunked_prefill_on_short`: look at
  `p99_ttft_ms` specifically - the tail latency of short requests is where
  the "blocked behind a long prefill" effect shows up most clearly.
- The `_long` rows are there for context (throughput on the long-document
  stream itself), not the main point of the lab.
