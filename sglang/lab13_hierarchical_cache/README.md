# Lab 13: HiCache - hierarchical KV cache (GPU -> CPU)

## Concept

RadixAttention (Lab 05) reuses KV cache for shared prefixes, but only while
those prefixes are still resident in GPU memory. Real traffic often has a
working set of prefixes much larger than HBM: thousands of tenants with
their own system prompts, RAG over many documents, long multi-turn agent
sessions that come back minutes later. Once the radix tree evicts a prefix,
the next request pays the full prefill again.

**HiCache** puts a host-memory tier (and optionally a storage tier such as
3FS, Mooncake or a file system via `--hicache-storage-backend`) behind the
radix tree:

- On eviction, KV pages are copied to pinned CPU RAM instead of dropped
  (`--hicache-write-policy write_through` copies eagerly, `write_back`
  copies only on eviction).
- On a hit in the host tier, pages are prefetched back to the GPU over
  PCIe/C2C - far cheaper than recomputing thousands of tokens of prefill.
- `--hicache-size` sets the host tier in GB per TP rank (or
  `--hicache-ratio` as a multiple of the GPU pool).

This is SGLang's equivalent of vLLM's `OffloadingConnector` / LMCache, and
it is exactly what llm-d's *tiered prefix cache* guide enables for SGLang
pods.

## What this lab does

Shrinks the GPU KV pool (`--mem-fraction-static 0.2`) and sends a
`generated-shared-prefix` workload whose distinct prefixes (~524K tokens)
exceed it. Three runs:

1. `no-prefix-cache` - `--disable-radix-cache` (floor).
2. `radix-gpu-only` - default RadixAttention, GPU tier only.
3. `radix-plus-hicache` - `--enable-hierarchical-cache --hicache-size 100`.

## Run it

```bash
cd sglang/lab13_hierarchical_cache && ./run_lab.sh
```

## What to look for

- `median_ttft_ms`: HiCache should sit close to what lab05 shows for a
  fully-resident cache, while GPU-only radix falls back towards the
  no-cache floor.
- Server log of the HiCache run: host-tier hit/load statistics.
- The KV math in the script header - redo it for your model
  (layers x KV heads x head_dim x 2 x bytes) to size GPU vs host tiers.
