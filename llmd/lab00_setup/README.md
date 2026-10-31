# llm-d Lab 00: Setup

## What llm-d is

[llm-d](https://github.com/llm-d/llm-d) (v1.0, CNCF sandbox) is a
Kubernetes-native distributed inference stack built *around* vLLM and
SGLang rather than replacing them. vLLM/SGLang make one replica fast; llm-d
makes a **fleet of replicas** fast and cheap:

- **llm-d Router** = a proxy (Envoy, or agentgateway) + the **EPP**
  (Endpoint Picker), an Envoy `ext_proc` gRPC service that decides which
  model-server replica gets each request using live state: queue depth, KV
  cache utilization, which replica already caches this prompt's prefix,
  predicted latency.
- **InferencePool** (Gateway API Inference Extension CRD) = the set of
  model-server pods the EPP may pick from.
- **Well-lit paths**: tested recipes for intelligent scheduling,
  prefill/decode disaggregation, precise prefix-cache routing, tiered KV
  cache, wide expert parallelism, flow control and autoscaling.

## Two ways to run the labs on one 8xB200 box

1. **No Kubernetes** (labs 01-06): vLLM replicas, the EPP and Envoy run as
   Docker containers on the host network. The EPP reads its replica list
   from a YAML file (`file-discovery` plugin) instead of watching an
   `InferencePool`. This is an upstream-supported deployment and is the
   fastest way to *see* routing decisions.
2. **Kubernetes** (labs 07-09): a single-node minikube cluster with GPU
   passthrough, then the upstream guides deployed unchanged with
   `helm` + `kustomize`.

## What this lab does

- Verifies Docker can start GPU containers (NVIDIA Container Toolkit).
- Pulls the images the guides pin: `vllm/vllm-openai:v0.31.0`,
  `envoyproxy/envoy:distroless-v1.33.2`,
  `ghcr.io/llm-d/llm-d-router-endpoint-picker`, `lmsysorg/sglang:v0.5.22`.
- Downloads `Qwen/Qwen3-32B` (routing labs), `openai/gpt-oss-120b` (P/D,
  the upstream guide's model) and `Qwen/Qwen3-30B-A3B` (wide-EP) into
  `$HF_HOME`.
- Clones `llm-d/llm-d` into `llmd/.upstream/llm-d` for the Kubernetes labs.

## Run it

```bash
cd llmd/lab00_setup
export HF_TOKEN=hf_...            # recommended
./run_lab.sh                      # add --with-k8s-tools for labs 07-09
```

## Configuration

All knobs live in `llmd/common/env.sh` and can be overridden from the
environment: `LLMD_MODEL`, `LLMD_TP`, `LLMD_REPLICAS`, image tags,
`ROUTER_EPP_VERSION`, `LLMD_UPSTREAM_REF`, `HF_HOME`.
