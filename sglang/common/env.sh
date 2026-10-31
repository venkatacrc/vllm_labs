#!/usr/bin/env bash
# Central configuration shared by every lab.
# Edit this file once to change models, ports, cache locations etc. for the
# whole curriculum, or override any variable via the environment before
# calling run_all_labs.sh / an individual lab's run_lab.sh, e.g.:
#   SMALL_MODEL=Qwen/Qwen2.5-3B-Instruct ./run_lab.sh
#
# This file is meant to be `source`d, not executed.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Root of the SGLang track (vllm_labs/sglang) (works no matter where you source this from)
export LABS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RESULTS_ROOT="${RESULTS_ROOT:-$LABS_ROOT/results}"

# Put the HF cache on the big /raid volume, not the root disk, as recommended
# by the environment check the user already ran.
export HF_HOME="${HF_HOME:-/raid/llm_rl_cache/hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
mkdir -p "$HF_HOME" "$RESULTS_ROOT"

# Conda/venv environment name that lab00 creates and every other lab expects.
# We intentionally use a *separate* environment from the existing PyTorch/vLLM
# env described in the task, because SGLang pins its own torch/flashinfer/
# sgl-kernel versions and mixing the two can break both installs.
export SGLANG_CONDA_ENV="${SGLANG_CONDA_ENV:-sglang-labs}"

# ---------------------------------------------------------------------------
# Models used across the curriculum (all ungated, no HF token required)
# ---------------------------------------------------------------------------
# Small dense model: used for labs 00-03, 05-09. FP32 ~30GB, BF16 ~15GB,
# FP8 ~8GB, INT4 ~5GB -> comfortably fits on a single B200 (183GB) at every
# precision, which keeps those labs single-GPU and fast to iterate on.
export SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# Mid-size dense model: used for the tensor-parallelism scaling lab (04).
# BF16 ~65GB. Fits on 1 GPU already, so any speedup you see going from
# TP=1 -> TP=8 is a genuine measurement of parallel scaling, not a
# memory-pressure artifact.
export MID_MODEL="${MID_MODEL:-Qwen/Qwen2.5-32B-Instruct}"

# MoE model: used for the expert-parallelism / DP-attention lab (10).
# SGLang's --enable-dp-attention currently supports DeepSeek-V2/V3 and
# Qwen 2/3 MoE architectures, so we standardize on Qwen3's small MoE.
export MOE_MODEL="${MOE_MODEL:-Qwen/Qwen3-30B-A3B}"

# Pre-quantized checkpoints (offline quantization) used in lab03.
export AWQ_MODEL="${AWQ_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
export GPTQ_MODEL="${GPTQ_MODEL:-Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4}"

# ---------------------------------------------------------------------------
# Server / networking
# ---------------------------------------------------------------------------
export SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
export SGLANG_PORT="${SGLANG_PORT:-30000}"
export SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-1200}" # seconds; big models take a while to load

# ---------------------------------------------------------------------------
# Benchmark defaults (override per-lab as needed)
# ---------------------------------------------------------------------------
export BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-200}"
export BENCH_RANDOM_INPUT_LEN="${BENCH_RANDOM_INPUT_LEN:-512}"
export BENCH_RANDOM_OUTPUT_LEN="${BENCH_RANDOM_OUTPUT_LEN:-256}"
export BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-64}"

# Number of physical GPUs on this node (auto-detected, override if needed).
if command -v nvidia-smi >/dev/null 2>&1; then
  export NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
else
  export NUM_GPUS="${NUM_GPUS:-8}"
fi
