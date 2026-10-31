#!/usr/bin/env bash
# Shared configuration for the llm-d track. Source, don't execute.
# Override anything from the environment, e.g.
#   LLMD_MODEL=meta-llama/Llama-3.1-8B-Instruct LLMD_TP=1 LLMD_REPLICAS=8 ./run_lab.sh

export LLMD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RESULTS_ROOT="${RESULTS_ROOT:-$LLMD_ROOT/results}"
export SUMMARY_CSV="$RESULTS_ROOT/summary.csv"

# Where the EPP/Envoy configs for the running lab are staged (mounted into
# the containers). Kept out of /etc so no sudo is needed.
export LLMD_WORK="${LLMD_WORK:-/tmp/llmd-labs}"

export HF_HOME="${HF_HOME:-/raid/llm_rl_cache/hf_home}"

# Images pinned to what the llm-d guides use (guides/recipes/.../images and
# guides/no-kubernetes-deployment).
export VLLM_IMAGE="${VLLM_IMAGE:-docker.io/vllm/vllm-openai:v0.31.0}"
export SGLANG_IMAGE="${SGLANG_IMAGE:-docker.io/lmsysorg/sglang:v0.5.22}"
export ENVOY_IMAGE="${ENVOY_IMAGE:-docker.io/envoyproxy/envoy:distroless-v1.33.2}"
export ROUTER_EPP_IMAGE="${ROUTER_EPP_IMAGE:-ghcr.io/llm-d/llm-d-router-endpoint-picker}"
export ROUTER_EPP_VERSION="${ROUTER_EPP_VERSION:-main}"

# The llm-d repo (manifests, Helm values, Kustomize overlays) for the k8s labs.
export LLMD_UPSTREAM_DIR="${LLMD_UPSTREAM_DIR:-$LLMD_ROOT/.upstream/llm-d}"
export LLMD_UPSTREAM_REF="${LLMD_UPSTREAM_REF:-main}"

# Default topology: the optimized-baseline shape (Qwen3-32B, TP=2) replicated
# across all 8 B200s -> 4 replicas, so the router has real choices to make.
export LLMD_MODEL="${LLMD_MODEL:-Qwen/Qwen3-32B}"
export LLMD_TP="${LLMD_TP:-2}"
export LLMD_REPLICAS="${LLMD_REPLICAS:-4}"
export LLMD_MAX_MODEL_LEN="${LLMD_MAX_MODEL_LEN:-16384}"
export LLMD_BASE_PORT="${LLMD_BASE_PORT:-8000}"

# Ports used by the router stack (match the upstream no-k8s configs).
export ENVOY_PORT=8081
export ENVOY_ADMIN_PORT=19000
export EPP_GRPC_PORT=9002
export EPP_HEALTH_PORT=9003
export EPP_METRICS_PORT=9090

export SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-1500}"
export PYTHON="${PYTHON:-python}"

mkdir -p "$RESULTS_ROOT" "$LLMD_WORK" 2>/dev/null || true
