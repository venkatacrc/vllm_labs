# llm-d Lab 09: Observability and SLO-aware autoscaling

## Concept

**Why CPU/GPU utilization is the wrong autoscaling signal for LLMs.**
During continuous batching the GPU reports close to 100% utilization almost
regardless of load, so utilization says nothing about whether users are
waiting. By the time latency visibly degrades, requests are already
queueing. Good signals are **leading** and inference-aware:

| Signal | Source | Meaning |
|---|---|---|
| in-flight requests | EPP (`llm_d_epp_inflight_requests`) | demand currently in the system |
| flow-control queue size | EPP (`llm_d_epp_flow_control_queue_size`) | demand the pool cannot absorb right now |
| pool saturation | EPP (`llm_d_epp_flow_control_pool_saturation`) | fraction of capacity used |
| waiting requests / KV-cache usage | vLLM (`vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`) | per-replica pressure |

llm-d's current autoscaling paths all use **KEDA** (the Workload Variant
Autoscaler is deprecated): KEDA's Prometheus scaler queries one of the EPP
metrics above, publishes it to the external metrics API, and the HPA that
KEDA manages scales the model-server Deployment. KEDA can also scale to
zero.

**The startup problem.** A new replica must pull a multi-GB image, load
tens of GB of weights, compile, and capture CUDA graphs before serving, which
can take minutes. Mitigations: node-local weight caches, pre-pulled images,
the torch.compile cache, *fast model actuation* (keep a sleeping vLLM
process with weights in CPU memory: `/sleep` and `/wake_up`), and scaling
early on queue growth rather than late on latency.

**Observability stack.** Prometheus scrapes the EPP and every vLLM pod
(ServiceMonitors/PodMonitors), Grafana shows the llm-d dashboards, and
OpenTelemetry + Jaeger give per-request traces across Envoy → EPP → vLLM.

## What this lab does

1. Installs Prometheus + Grafana and the llm-d dashboards (upstream scripts).
2. Deploys `optimized-baseline` with monitoring.
3. Applies the upstream KEDA ScaledObject (`keda-epp`, in-flight signal).
4. Starts at 1 replica, drives a heavy load ramp, records desired vs ready
   replicas every 10 s to `replica_timeline.csv`, then watches scale-down.

## Run it

```bash
cd llmd/lab09_k8s_autoscaling_observability && ./run_lab.sh
```

## What to look for

- Lag between `desired` rising and `ready` rising (startup cost).
- KEDA's `currentMetrics` vs the target value.
- In Grafana: queueing at the EPP vs inside vLLM as replicas come up.
