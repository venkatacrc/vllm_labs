# llm-d Labs: from one fast replica to a fast fleet

The vLLM and SGLang tracks make a single model server fast. This track is
about what happens when you run **many** of them behind one endpoint:
which replica should serve each request, how to split prefill from decode,
how to keep KV cache across tiers and replicas, how to keep tenants from
hurting each other, and how to scale the fleet with demand.

Everything targets the same single 8xB200 node, with upstream
[llm-d](https://github.com/llm-d/llm-d) v1.0 components (router/EPP, Envoy,
vLLM `v0.31.0`, SGLang `v0.5.22`).

| # | Lab | Runs on | What it teaches |
|---|-----|---------|-----------------|
| 00 | [`lab00_setup`](lab00_setup/) | Docker | Images, models, llm-d repo, k8s tools |
| 01 | [`lab01_router_basics`](lab01_router_basics/) | Docker | Round-robin vs least-request vs llm-d EPP on shared-prefix traffic |
| 02 | [`lab02_scheduler_ablation`](lab02_scheduler_ablation/) | Docker | Filters/scorers/pickers, affinity vs load, live pool changes |
| 03 | [`lab03_flow_control`](lab03_flow_control/) | Docker | Fair queuing in the router: quiet vs noisy tenant |
| 04 | [`lab04_tiered_kv_cache`](lab04_tiered_kv_cache/) | Docker | HBM → CPU KV offload (vLLM OffloadingConnector) |
| 05 | [`lab05_pd_disaggregation`](lab05_pd_disaggregation/) | Docker | Prefill/decode split with NIXL KV transfer (gpt-oss-120b, 1P+1D) |
| 06 | [`lab06_wide_ep`](lab06_wide_ep/) | Docker | MoE: TP vs DP+EP vs EPLB (Qwen3-30B-A3B) |
| 07 | [`lab07_k8s_cluster`](lab07_k8s_cluster/) | minikube | Single-node GPU Kubernetes + GAIE/LWS/KEDA |
| 08 | [`lab08_k8s_well_lit_paths`](lab08_k8s_well_lit_paths/) | minikube | Upstream guides: optimized baseline, precise prefix routing, tiered cache, P/D, flow control |
| 09 | [`lab09_k8s_autoscaling_observability`](lab09_k8s_autoscaling_observability/) | minikube | Prometheus/Grafana, KEDA on EPP metrics, startup latency |

## Architecture in one picture

```
client ─► Envoy (proxy) ──ext_proc gRPC──► EPP (Endpoint Picker)
             │                               │  filters → scorers → picker
             │  x-gateway-destination-       │  data: /metrics scrape, in-flight
             │  endpoint header              │  load, prefix index, KV events
             ▼                               ▼
     vLLM / SGLang replicas  ◄──── InferencePool (k8s) or endpoints.yaml (no k8s)
     (aggregated, or prefill + decode pools connected by NIXL)
```

## Prerequisites

- Docker + NVIDIA Container Toolkit (labs 00-06).
- Python with `aiohttp` for `common/loadgen.py` and `common/pd_proxy.py`
  (lab00 installs it into the current env).
- `HF_TOKEN` for gated models (recommended).
- For labs 07-09: minikube, kubectl, helm, kustomize, yq, helmfile
  (`lab00_setup/run_lab.sh --with-k8s-tools`).

## Run

```bash
cd llmd
./run_all_labs.sh            # 00-06
./run_all_labs.sh --k8s      # plus 07-09
```

Results: `llmd/results/summary.csv` (one row per run, per tenant) and a JSON
file per run under `llmd/results/<lab>/`.

## Shared tooling

- `common/loadgen.py`: async OpenAI-API load generator (shared-prefix,
  multi-turn, random workloads; multiple tenants with flow-control headers;
  per-replica prefix-cache accounting from `/metrics`).
- `common/pd_proxy.py`: the P/D request protocol in ~100 lines.
- `configs/epp/*.yaml`: EPP plugin configs (upstream no-k8s baseline,
  v1.0 affinity + token-load, ablations, flow control).
- `configs/envoy/envoy-epp.yaml`: upstream no-k8s Envoy config.
