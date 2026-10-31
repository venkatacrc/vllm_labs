# llm-d Lab 08: The well-lit paths on Kubernetes

## Concept

llm-d packages its optimizations as **well-lit paths**: guides with tested
manifests, a validated reference model/topology, and nightly end-to-end CI.
Each guide is the same three steps (`deploy_guide.sh` does them for you):

```bash
# 1. router: EPP + Envoy sidecar, configured by layered Helm values
helm install <guide> oci://ghcr.io/llm-d/charts/llm-d-router-standalone \
  -f guides/recipes/router/base.values.yaml -f guides/<guide>/router/<guide>.values.yaml \
  -n llm-d-<guide> --version v0
# 2. model servers: Kustomize overlay per accelerator / engine / provider
kubectl apply -n llm-d-<guide> -k guides/<guide>/modelserver/gpu/vllm/base/
# 3. verify through the router Service (port 80 -> Envoy 8081)
```

| Guide | Reference topology (NVIDIA) | What it adds |
|---|---|---|
| `optimized-baseline` | Qwen3-32B, 2 replicas x TP=2 | `prefix-cache-affinity-filter` + `token-load-scorer`; the default to start from. |
| `precise-prefix-cache-routing` | same | Model servers publish **KV-cache events over ZMQ**; the router's KV indexer knows exactly which blocks each pod holds. The router renders (tokenizes) each prompt to look it up. |
| `tiered-prefix-cache` | same, 300 GiB CPU tier per pod | vLLM `OffloadingConnector` (or SGLang HiCache / LMCache) plus a router scorer for the CPU tier. |
| `pd-disaggregation` | gpt-oss-120b, 1P (TP=1) + 1D (TP=4) | Prefill/decode pools in one `DisaggregatedSet`, NIXL KV transfer, router schedules each request twice. |
| `predicted-latency-routing` | Qwen3-32B, 2 x TP=2 | XGBoost sidecars trained online predict TTFT/TPOT per pod; optional SLO headers (`x-llm-d-slo-ttft-ms`). |
| `flow-control` | Qwen3-32B | Per-tenant fair queuing with priority bands (`InferenceObjective`). |
| `wide-ep` | DeepSeek-R1, 32 GPUs | DP+EP across nodes with DeepEP + P/D (multi-node; see lab 06 for the single-node version). |

## What this lab does

Deploys, load-tests and tears down (to free the GPUs) the guides above that
fit on one 8xB200 node: `optimized-baseline`,
`precise-prefix-cache-routing`, `tiered-prefix-cache`,
`pd-disaggregation`, `flow-control`. Each guide gets a workload that shows
off its feature. Per-pod vLLM counters are read through the API server
proxy, the same way the guides' verification steps do.

## Run it

```bash
cd llmd/lab08_k8s_well_lit_paths
./run_lab.sh                                     # all five, sequentially
KEEP=1 ./run_lab.sh precise-prefix-cache-routing # leave it running to explore
MODEL_SERVER=sglang ./deploy_guide.sh optimized-baseline   # SGLang pods instead
```

## What to look for

- `*-pod-metrics.txt`: requests and prefix hits concentrated per prefix
  group (optimized-baseline, precise), CPU-tier hits (tiered), prefill vs
  decode roles (P/D).
- `flow-control`: `premium` tenant latency vs `batch`.
- Compare against the Docker labs 01-05: same mechanisms, now with
  discovery, health probes, rollouts and Helm-managed config.
- Calibration (`guides/recipes/router/calibration`) - the reference
  `peakPrefillThroughput` is for H100; measure it for B200.
