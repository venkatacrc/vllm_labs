"""Shared configuration for the vLLM optimization labs.

Every lab imports this module instead of hardcoding model IDs, paths, or
ports, so the whole suite can be retargeted (different model, different
cache dir, different GPU) by changing environment variables or this one
file.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Primary model: fp32 weights are ~32GB, comfortably fits on a single B200
# for every lab except the tensor-parallelism one.
MODEL_PRIMARY = os.environ.get("LAB_MODEL_PRIMARY", "meta-llama/Llama-3.1-8B-Instruct")

# Large model used only by the tensor-parallelism lab (and referenced by the
# capstone). Ungated, so no HF license-approval wait. At fp32 (~128GB) it
# barely fits a single B200; TP is what makes it comfortable.
MODEL_LARGE = os.environ.get("LAB_MODEL_LARGE", "Qwen/Qwen2.5-32B-Instruct")

# Small draft model for speculative decoding.
MODEL_DRAFT = os.environ.get("LAB_MODEL_DRAFT", "meta-llama/Llama-3.2-1B-Instruct")

# Pre-quantized checkpoints for the weight-quantization lab (no local
# calibration/quantization step needed).
MODEL_AWQ = os.environ.get("LAB_MODEL_AWQ", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
MODEL_GPTQ = os.environ.get("LAB_MODEL_GPTQ", "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4")

ALL_MODELS = [MODEL_PRIMARY, MODEL_LARGE, MODEL_DRAFT, MODEL_AWQ, MODEL_GPTQ]

# ---------------------------------------------------------------------------
# Paths / environment
# ---------------------------------------------------------------------------
# Make sure every subprocess we launch (vllm serve, vllm bench serve, HF
# downloads) agrees on where the model cache lives.
HF_HOME = os.environ.setdefault("HF_HOME", "/raid/llm_rl_cache/hf_home")

RESULTS_DIR = Path(os.environ.get("LAB_RESULTS_DIR", str(REPO_ROOT / "results")))
LOGS_DIR = RESULTS_DIR / "logs"
SUMMARY_PATH = RESULTS_DIR / "SUMMARY.md"

for _d in (RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Server defaults
# ---------------------------------------------------------------------------
DEFAULT_PORT = int(os.environ.get("LAB_BASE_PORT", "8000"))
DEFAULT_GPU = os.environ.get("LAB_DEFAULT_GPU", "0")
# 8B fp32 can take a few minutes to load + a while for CUDA-graph capture;
# 32B TP jobs take even longer. Be generous.
SERVER_STARTUP_TIMEOUT_S = int(os.environ.get("LAB_SERVER_TIMEOUT", "900"))
DEFAULT_GPU_MEMORY_UTILIZATION = float(os.environ.get("LAB_GPU_MEM_UTIL", "0.85"))
DEFAULT_MAX_MODEL_LEN = int(os.environ.get("LAB_MAX_MODEL_LEN", "4096"))

# ---------------------------------------------------------------------------
# Benchmark defaults (individual labs override as needed; all are shrunk
# under --quick so the whole suite can be smoke-tested in minutes).
# ---------------------------------------------------------------------------
DEFAULT_NUM_PROMPTS = 200
QUICK_NUM_PROMPTS = 24
DEFAULT_SEED = 0


def num_prompts(quick: bool, full: int = DEFAULT_NUM_PROMPTS, quick_n: int = QUICK_NUM_PROMPTS) -> int:
    """Pick a request count: shrink it way down under --quick."""
    return quick_n if quick else full


def scale_quick(value: int, quick: bool, factor: float = 0.25, minimum: int = 2) -> int:
    """Generic helper to shrink any per-lab count (prompts, output tokens, ...) under --quick."""
    return max(minimum, int(value * factor)) if quick else value
