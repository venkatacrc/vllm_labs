# llm-d Lab 07: A Kubernetes cluster for llm-d on one 8xB200 node

## Concept: what Kubernetes adds

Labs 01-06 wired everything by hand: container names, ports, GPU indices,
an endpoints file. That works on one box, but a production fleet needs:

- **Scheduling onto GPUs**: the NVIDIA device plugin advertises
  `nvidia.com/gpu` as a node resource; pods request `nvidia.com/gpu: 2` and
  the scheduler places them. (DRA, `resource.k8s.io/v1`, is the newer
  model for richer device claims.)
- **Self-healing and rollouts**: Deployments/LeaderWorkerSets restart crashed
  replicas, probes gate traffic until the model is loaded, rollouts drain
  gracefully (`--shutdown-timeout`, `preStop`).
- **Service discovery**: the EPP watches an **InferencePool** (pods selected
  by label) instead of a YAML file.
- **Multi-pod replicas**: a **LeaderWorkerSet** (LWS) treats a group of pods
  (e.g. one model replica spanning 2 nodes) as one unit; a
  **DisaggregatedSet** bundles prefill and decode LWS roles so they roll out
  together.
- **Autoscaling**: HPA/KEDA scale replicas on metrics (lab 09).

The llm-d guides are **Kustomize-first** for model servers
(`guides/<guide>/modelserver/<accelerator>/<engine>/<provider>/`) and use
**Helm** charts for the router (`llm-d-router-standalone` = EPP + Envoy
sidecar; `llm-d-router-gateway` = EPP behind a Kubernetes Gateway).
Requirements: Kubernetes 1.29+ (1.33+ recommended), helm 3.12+, kubectl,
kustomize 5+, yq, helmfile.

## What this lab does

1. `minikube start --driver docker --container-runtime docker --gpus all`
   (single node, all 8 GPUs, `$HF_HOME` mounted at
   `/var/cache/huggingface`).
2. Installs the shared CRDs/controllers: Gateway API Inference Extension,
   llm-d router CRDs (`InferenceObjective`), LeaderWorkerSet with
   DisaggregatedSet, KEDA.

Alternatives on bare metal: k3s or kubeadm + the NVIDIA GPU Operator. The
guides themselves are cluster-agnostic (`INFRA_PROVIDER=base`).

## Run it

```bash
cd llmd/lab00_setup && ./run_lab.sh --with-k8s-tools   # once
cd ../lab07_k8s_cluster && ./run_lab.sh
./run_lab.sh --delete                                   # when done
```

Prerequisite on the host: NVIDIA Container Toolkit configured for Docker
(`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).
