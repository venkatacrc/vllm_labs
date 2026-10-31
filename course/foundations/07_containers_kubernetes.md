---
title: Containers and Kubernetes for GPU serving
track: Foundations
order: 7
---

# Containers and Kubernetes for GPU serving

## Containers

A container is a process with its own filesystem image, namespaces (PID,
network, mount) and cgroup limits (CPU, memory). For GPUs:

- The **NVIDIA Container Toolkit** injects the driver libraries and device
  nodes: `docker run --gpus '"device=0,1"' ...` exposes GPUs 0 and 1 (seen
  inside as 0 and 1).
- Images pin the whole user-space stack (CUDA runtime, PyTorch, vLLM, NCCL,
  NIXL/UCX), which is why the llm-d guides pin `vllm/vllm-openai:v0.31.0`.
- `--network host` shares the host network stack (used in the no-k8s
  labs so everything talks over 127.0.0.1); `--ipc host` or `--shm-size`
  for shared memory.
- Mount model weights from a cache volume instead of baking 60 GB into the
  image.

## Kubernetes in one page

| Object | Purpose |
|---|---|
| **Pod** | one or more containers scheduled together on a node, sharing network/IPC (e.g. vLLM + routing sidecar) |
| **Deployment** | N identical pods, rolling updates, self-healing |
| **Service** | stable virtual IP/DNS name load-balancing to pods selected by labels |
| **ConfigMap / Secret** | config files and credentials (HF token) |
| **PersistentVolumeClaim** | storage (model cache) |
| **Node resources** | `nvidia.com/gpu: 2` requests whole GPUs via the NVIDIA device plugin; **DRA** (`resource.k8s.io/v1`) is the newer claim-based model |
| **Probes** | startup/readiness/liveness; readiness gates traffic until the model is loaded |
| **HPA** | Horizontal Pod Autoscaler; scales replicas on metrics. **KEDA** feeds it external metrics (Prometheus) and can scale to zero |
| **CRD + controller (operator)** | extend the API with new kinds, e.g. `InferencePool`, `LeaderWorkerSet`, `DisaggregatedSet`, `InferenceObjective` |

Kubernetes is **declarative**: you apply desired state (YAML), controllers
reconcile actual state toward it continuously.

## Kubernetes objects used by llm-d

- **InferencePool** (Gateway API Inference Extension): the set of
  model-server pods (by label selector) plus a reference to the EPP that
  picks among them.
- **LeaderWorkerSet (LWS)**: a replica made of a group of pods (leader +
  workers) that start, fail and roll out together - needed when one model
  replica spans several nodes (wide-EP, multi-node TP/PP).
- **DisaggregatedSet**: prefill and decode LWS roles managed as one versioned
  unit (P/D, wide-EP).
- **Gateway API** (`Gateway`, `HTTPRoute`): standard L7 ingress; llm-d can
  run in *standalone* mode (Envoy sidecar in the EPP pod) or *gateway* mode
  (Istio, agentgateway, GKE Gateway in front).

## Packaging: Helm and Kustomize

- **Helm** charts are templated packages with values files; llm-d's router
  ships as `llm-d-router-standalone` / `llm-d-router-gateway` charts, and
  guides layer values (`base.values.yaml` + `<guide>.values.yaml`).
- **Kustomize** composes plain YAML with overlays and patches; llm-d model
  servers are overlays per accelerator / engine / provider
  (`modelserver/gpu/vllm/base/`).

## Flashcards

Q: What does the NVIDIA Container Toolkit do?
A: Injects the host GPU driver libraries and device files into containers so `--gpus` works, without baking the driver into images.

Q: What does a Kubernetes readiness probe protect against in LLM serving?
A: Routing traffic to a pod whose model is still loading/compiling; the pod receives requests only after the probe (e.g. GET /v1/models) succeeds.

Q: What is an InferencePool?
A: A Gateway API Inference Extension CRD naming the set of model-server pods (label selector) and the endpoint picker (EPP) that chooses among them.

Q: Why does wide-EP need LeaderWorkerSet?
A: One model replica spans several pods/nodes that must be scheduled, restarted and upgraded as a unit; LWS manages that group.

Q: Helm vs Kustomize?
A: Helm renders templated charts from values files; Kustomize patches and overlays plain YAML. llm-d uses Helm for the router and Kustomize for model servers.

Q: What does KEDA add to the HPA?
A: Scalers that read external metrics (e.g. Prometheus queries on EPP metrics), exposing them to the HPA, plus scale-to-zero.

Q: How do pods request GPUs?
A: Via the extended resource `nvidia.com/gpu: N` advertised by the NVIDIA device plugin (or DRA resource claims).
