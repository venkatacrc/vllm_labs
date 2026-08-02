# Lab 02: Continuous batching + paged KV cache

## Concept

This is the idea that made vLLM famous, so it's worth being precise about
what it actually is (a scheduling + memory-management technique) versus
what it's called in the code today (see the callout below).

**Static batching (the naive approach)**: collect a batch of N requests,
run `model.generate()` until every sequence in the batch is done, return
all N results, repeat. The problem: if one sequence in the batch needs 500
tokens and the rest need 20, the whole batch - and the GPU - is stuck
waiting on that one straggler. Short requests that finished early still
occupy a batch slot and its KV cache reservation until the whole batch
retires.

**Continuous batching**: at every scheduler step, evict sequences that just
finished and admit new sequences waiting in the queue, so the batch
composition changes every step instead of being fixed for its whole
lifetime. Throughput is limited by GPU compute, not by the slowest member
of an arbitrarily-drawn batch.

**Paged KV cache**: the memory technique that makes continuous batching
practical. Instead of reserving one contiguous `max_seq_len`-sized KV
buffer per sequence (mostly wasted for short sequences), the cache is
divided into fixed-size blocks allocated on demand, like OS virtual memory
pages. This is what lets sequences of wildly different lengths share GPU
memory efficiently and lets the scheduler admit/evict sequences cheaply.

## A note on the name "PagedAttention"

You may have heard that PagedAttention is "deprecated" - this is true at
the implementation level and misleading at the concept level. vLLM v0.25.0
(July 2026) deleted the original, standalone 2023 PagedAttention CUDA
kernel. It didn't disappear because the idea failed; it disappeared because
the *technique* - block-based paged KV memory - is now built directly into
every modern attention backend (FlashInfer, FlashAttention, Triton,
FlexAttention) under vLLM's Model Runner V2, which has been the default
execution path for a while. There's no longer a separate "paged attention
toggle"; every backend paged the KV cache. Lab 03 is dedicated to comparing
those backends against each other.

## What this lab does

1. Runs a deliberately naive HF Transformers static-batch baseline
   (`common/hf_baseline.py`, launched as its own subprocess) against a set
   of prompts with intentionally bimodal output lengths (half one-sentence
   answers, half multi-paragraph essays) - this bimodality is what makes
   the "batch straggler" effect visible.
2. Serves the same model with vLLM and fires the *same number* of requests
   at once via `vllm bench serve --max-concurrency`, in two configurations:
   - `--max-num-seqs 8`: capped at the same concurrency as the HF batch
     size, isolating the effect of continuous batching itself (same
     concurrency budget, no static-batch bubble).
   - default `--max-num-seqs`: letting vLLM use its normal scheduling
     headroom.

## Run it

```bash
python labs/lab02_continuous_batching_paged_kv_cache/lab.py
python labs/lab02_continuous_batching_paged_kv_cache/lab.py --quick
```

## What to look for

- `hf_naive_static_batch` vs `vllm_max_num_seqs_8`: same effective
  concurrency limit, but vLLM should still substantially outperform HF's
  `generate()` loop because it isn't blocked by the slowest sequence in
  each fixed batch.
- `vllm_max_num_seqs_8` vs `vllm_default`: shows the throughput you leave on
  the table by artificially constraining concurrency.
