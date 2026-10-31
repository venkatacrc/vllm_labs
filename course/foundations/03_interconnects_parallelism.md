---
title: Interconnects, collectives and parallelism
track: Foundations
order: 3
---

# Interconnects, collectives and parallelism

## How GPUs talk

| Link | Typical bandwidth | Where |
|---|---|---|
| NVLink 5 (B200) | 1.8 TB/s per GPU (bidirectional) | GPU↔GPU inside a node, via NVSwitch (all-to-all, non-blocking) |
| PCIe Gen5 x16 | ~64 GB/s per direction | GPU↔CPU, GPU↔NIC, NVMe |
| Grace C2C (GB200) | ~900 GB/s | GPU↔CPU on Grace-Blackwell |
| InfiniBand NDR / XDR | 400 / 800 Gb/s per NIC (50 / 100 GB/s) | node↔node, RDMA |
| RoCE v2 / AWS EFA | 100-400 Gb/s per NIC | node↔node, RDMA over Ethernet |
| Datacenter TCP | 25-100 Gb/s | everything else |

Inside an 8-GPU HGX node every GPU reaches every other at full NVLink
bandwidth through NVSwitch. Between nodes the bandwidth drops by an order
of magnitude, so **parallelism that needs a collective every layer stays
inside the node**.

**RDMA** (remote direct memory access) lets a NIC read/write remote memory
without involving the remote CPU; **GPUDirect RDMA** lets it read GPU memory
directly. That is how NIXL ships KV cache between prefill and decode pods
(llm-d lab 05), and how NCCL runs collectives across nodes.

## Collectives (NCCL)

| Collective | What it does | Used by |
|---|---|---|
| all-reduce | every rank ends with the sum of everyone's tensor | tensor parallelism (after attention and MLP) |
| all-gather | every rank ends with everyone's shards concatenated | TP/sequence parallel, FSDP |
| reduce-scatter | sum, then each rank keeps one shard | TP variants, FSDP |
| all-to-all | each rank sends a different chunk to each other rank | expert parallelism (dispatch/combine) |
| send/recv | point-to-point | pipeline parallelism, KV transfer |

A ring all-reduce moves about 2·(N−1)/N × the tensor size per GPU; on
NVLink that's microseconds for decode-sized activations, but it happens
twice per layer, so it adds up.

## The parallelism menu

- **Tensor parallelism (TP)**: split each weight matrix across GPUs
  (column-parallel then row-parallel) so each GPU holds 1/N of every layer.
  Two all-reduces per layer. Cuts per-GPU weight memory and per-token
  latency; scales well inside NVLink, poorly across nodes. (vLLM lab 10,
  SGLang lab 04.)
- **Pipeline parallelism (PP)**: split layers into stages on different
  GPUs/nodes; activations flow stage to stage. Little communication, but
  bubbles unless many micro-batches are in flight. Used to span nodes.
- **Data parallelism (DP)**: independent replicas, each with full weights,
  serving different requests. No communication; this is what a fleet behind
  a router is. **DP attention** (SGLang lab 10, vLLM `--data-parallel-size`)
  runs attention data-parallel inside one MoE deployment.
- **Expert parallelism (EP)**: place whole MoE experts on different GPUs;
  route tokens to their experts with all-to-all. **Wide EP** spreads experts
  over dozens of GPUs (llm-d lab 06) using DeepEP kernels.
- **Context / sequence parallelism (CP)**: split one long sequence's tokens
  across GPUs for prefill of very long contexts.

Rule of thumb for serving: use the **smallest TP that fits the model with
enough KV headroom**, then add **DP replicas** behind a router. TP beyond
that buys latency, not throughput per GPU.

## Flashcards

Q: Why should tensor parallelism stay inside one node?
A: TP needs two all-reduces per layer; NVLink/NVSwitch gives ~1.8 TB/s per GPU inside a node, but inter-node links are ~10x slower, so cross-node TP is communication-bound.

Q: Which collective does expert parallelism depend on?
A: All-to-all: tokens are dispatched to the GPUs that own their experts and the results combined back.

Q: What is RDMA and why does P/D disaggregation need it?
A: Remote direct memory access: the NIC reads or writes remote (GPU) memory without the remote CPU. Moving GBs of KV cache per request between prefill and decode pods needs that bandwidth and latency.

Q: Difference between TP and DP for serving?
A: TP splits each layer across GPUs to serve one batch faster (lower latency, communication every layer); DP runs independent replicas that serve different requests (more throughput, no communication).

Q: What does NVSwitch provide in an HGX B200 node?
A: Full-bandwidth all-to-all NVLink connectivity between all 8 GPUs (1.8 TB/s per GPU).

Q: Rule of thumb for choosing TP size?
A: The smallest TP that fits the weights with enough KV-cache headroom; then scale out with data-parallel replicas behind a router.

Q: What is pipeline parallelism's main inefficiency?
A: Pipeline bubbles: stages sit idle unless enough micro-batches are in flight.
