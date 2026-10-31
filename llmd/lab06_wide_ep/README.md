# llm-d Lab 06: Wide expert parallelism

## Concept

Mixture-of-Experts models (DeepSeek-V3/R1, Qwen3-MoE, gpt-oss, Kimi) replace
each dense FFN with many experts, of which a router activates a few per
token. DeepSeek-R1 has 671B parameters but only ~37B are active per token,
so the model needs a lot of memory but relatively little compute per token.

How to spread it over GPUs:

- **Tensor parallelism (TP)** slices every expert across all GPUs. Each
  expert's GEMM becomes tiny and every layer ends with an all-reduce.
- **Expert parallelism (EP)** gives each GPU whole experts. Tokens are sent
  to the GPU that owns their expert (**all-to-all dispatch**) and the
  results come back (**combine**). Attention runs **data-parallel (DP)**:
  each GPU processes its own batch of requests with a full copy of the
  attention weights.
- **Wide EP** = EP across many GPUs and nodes (EP=32..96), so each GPU holds
  few experts and has HBM left over for a large KV cache and batch.
  All-to-all over NVLink + RDMA becomes the critical path. **DeepEP**
  kernels provide a high-throughput mode (prefill) and a low-latency,
  CUDA-graph-compatible mode (decode), which is why wide-EP is paired with
  P/D disaggregation.
- **EPLB** (expert-parallel load balancing) replicates hot experts and
  rebalances placement periodically, because real traffic routes unevenly.

vLLM flags:

```bash
vllm serve <moe-model> --data-parallel-size 8 --enable-expert-parallel \
  [--all2all-backend deepep_low_latency] [--enable-eplb]
# EP size = TP x DP; without --enable-expert-parallel MoE layers use TP over TP x DP GPUs
```

llm-d's `wide-ep` guide: DeepSeek-R1-0528 on 32 H200, prefill DP16 +
decode DP16 (2 nodes each), DeepEP + NIXL, deployed as a `DisaggregatedSet`
of LeaderWorkerSets.

## What this lab does

Single node, `Qwen/Qwen3-30B-A3B`: TP=8 vs DP=8+EP vs DP=8+EP+EPLB, at high
concurrency.

## Run it

```bash
cd llmd/lab06_wide_ep && ./run_lab.sh
CONCURRENCY=8 REQUESTS=64 ./run_lab.sh     # latency regime
```

## What to look for

- Throughput at high concurrency: DP+EP > TP.
- Latency at low concurrency: TP can win.
- EPLB: small gain on synthetic random tokens (expert routing is close to
  uniform); larger gains on real skewed traffic.
