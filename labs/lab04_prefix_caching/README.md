# Lab 04: Automatic prefix caching (APC)

## Concept

Many real workloads send requests that share a long common prefix: a
system prompt, a few-shot example block, a RAG context chunk reused across
several questions, or successive turns of the same chat. Without prefix
caching, vLLM re-runs the full prefill computation for that shared prefix
on every single request. With **automatic prefix caching**, the paged KV
blocks for a prefix are kept around (as long as memory allows) and reused
by any later request whose tokens start with the same prefix - turning an
O(prefix length) prefill into an O(1) cache lookup for every request after
the first.

The benefit shows up almost entirely as reduced **time-to-first-token
(TTFT)**: less new computation before the response begins, not more output
tokens/sec.

## What this lab does

Uses `vllm bench serve`'s built-in `random` dataset with
`--random-prefix-len 1500` - every synthetic request shares an identical
1500-token random prefix, plus its own random 200-token suffix, and expects
a 100-token completion. Runs this workload once with
`--no-enable-prefix-caching` and once with prefix caching enabled (vLLM's
current default), same server otherwise.

Within a single run, only the *first* request pays the full 1500-token
prefill; every request after it should hit the cache when caching is
enabled. With ~80 requests sharing the prefix, that dominates the mean/P99
TTFT - which is exactly what you should see move between the two variants.

## Run it

```bash
python labs/lab04_prefix_caching/lab.py
python labs/lab04_prefix_caching/lab.py --quick
```

## What to look for

- `mean_ttft_ms` and `p99_ttft_ms`: should drop substantially with prefix
  caching on, since ~79/80 requests skip re-prefilling the shared 1500
  tokens.
- `output_throughput`: caching frees up GPU compute that was being spent on
  redundant prefill, so overall throughput should also improve, not just
  latency.
