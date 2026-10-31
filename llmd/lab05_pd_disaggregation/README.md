# llm-d Lab 05: Prefill/decode (P/D) disaggregation

## Concept

Prefill and decode want different hardware behaviour:

| | Prefill | Decode |
|---|---|---|
| Work per step | thousands of tokens of one prompt | one token for each of many sequences |
| Bottleneck | tensor-core FLOPs (compute-bound) | HBM bandwidth reading weights + KV (memory-bound) |
| Best parallelism | low TP, many replicas | higher TP / large batch |
| SLO it drives | TTFT | TPOT / ITL |

In an aggregated server both phases share a GPU. Chunked prefill (vLLM lab
05) limits the damage, but every long prompt that arrives still steals
step time from in-flight decodes, so TPOT jitters with prompt traffic.

**Disaggregation** runs them on separate pools:

1. The router picks a **prefill** server (llm-d: `prefix-cache-affinity-filter`
   + `token-load-scorer`) and sends the request with `max_tokens=1` and
   `kv_transfer_params.do_remote_decode=true`.
2. The prefill server computes the KV cache, holds the blocks for a lease
   (`kv_lease_duration`), and returns `kv_transfer_params` (engine id,
   block ids, host, port).
3. The router picks a **decode** server (llm-d: `active-request-scorer`)
   and sends the request with those params. The decoder pulls the blocks via
   **NIXL** (`NixlConnector`, UCX transport: NVLink `cuda_ipc` on a node,
   RDMA over InfiniBand/RoCE/EFA across nodes) and streams the output.

In llm-d the router schedules every request twice (once per pool) and a
routing sidecar next to each decode pod runs the P-then-D protocol. Both
pools are one `InferencePool`, deployed as a LeaderWorkerSet
`DisaggregatedSet` so prefill and decode roll out as one version. The pools
can use different TP sizes (NIXL supports heterogeneous TP) and scale
independently.

## What this lab does

Upstream reference topology on one node: `openai/gpt-oss-120b`, 1 prefill
(TP=1) + 1 decode (TP=4). `common/pd_proxy.py` implements the router side
of the protocol in ~100 lines so you can read it. Three runs with ~8K-token
prompts: aggregated TP=4, P/D 1P1D, P/D 2P1D.

## Run it

```bash
cd llmd/lab05_pd_disaggregation && ./run_lab.sh
IN_TOKENS=2000 ./run_lab.sh     # see where P/D stops paying off
```

## What to look for

- `tpot_p50_ms` lower/steadier with P/D, at the price of a transfer hop in TTFT.
- Prefill-pool saturation in 1P1D and the fix in 2P1D: the
  "scale P and D independently" argument.
- Decoder log lines about NIXL handshakes and transfers.
- Kubernetes version of this lab: `lab08` deploys the upstream
  `pd-disaggregation` guide unchanged.
