# llm-d Lab 04: Tiered prefix cache (HBM -> CPU RAM -> storage)

## Concept

A B200 has 178 GB of HBM, but a node has ~2 TB of DRAM and tens of TB of
NVMe. KV cache for Qwen3-32B costs ~256 KB per token (64 layers x 8 KV
heads x 128 dim x 2 (K,V) x 2 bytes), so one 8K-token conversation is
~2 GB. A few hundred concurrent agentic sessions overflow HBM quickly.

Recomputing an evicted 8K prefix costs a full prefill (hundreds of ms on a
busy replica). Copying 2 GB back from pinned host RAM over PCIe Gen5 / C2C
takes tens of ms and doesn't use the GPU's compute. That is the whole idea
of **KV offloading**:

| Tier | Typical size per node | Who manages it |
|---|---|---|
| HBM (GPU) | 100s of GB | vLLM prefix cache / SGLang RadixAttention |
| CPU DRAM | 1-2 TB | vLLM `OffloadingConnector` (CPU tier), SGLang HiCache, LMCache |
| Local NVMe / shared FS | 10s of TB | `TieringOffloadingSpec` secondary tiers, LMCache, Mooncake store |

vLLM's native connector:

```bash
vllm serve Qwen/Qwen3-32B --kv-transfer-config \
  '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"cpu_bytes_to_use":322122547200}}'
```

`cpu_bytes_to_use` is the total across workers. Blocks are copied to the
CPU tier asynchronously (DMA via `cudaMemcpyAsync`) as they complete, and
promoted back on a hit. The upstream llm-d guide gives each replica 300 GiB
and adds a router scorer that tracks the CPU tier.

## What this lab does

Two Qwen3-32B replicas (TP=2) behind the EPP, with GPU memory capped at
45% so the HBM tier is small. 160 sessions x 8K-token documents x 4 turns,
visited round-robin. Run once HBM-only, once with a 200 GiB CPU tier per
replica.

## Run it

```bash
cd llmd/lab04_tiered_kv_cache && ./run_lab.sh
GPU_UTIL=0.6 SESSIONS=300 ./run_lab.sh    # change the pressure
```

## What to look for

- `prefix_hit_rate` and `ttft_p50_ms`: should improve sharply with the
  CPU tier.
- `ext_hits` per replica in `results/lab04_tiered_kv_cache/*.json`.
- Host memory: `free -g` during run 2 shows the pinned CPU tier.
- Compare with SGLang lab 13 (HiCache), the same idea in SGLang.
