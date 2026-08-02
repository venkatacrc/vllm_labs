#!/usr/bin/env python
"""Sanity-check the environment before running any lab.

Mirrors the style of a typical environment-check script: each check prints
PASS/WARN/FAIL and a short detail string. WARN means "the suite will still
run, but a specific lab or backend will be skipped/slower". FAIL means "fix
this before running the labs".

Usage: python scripts/check_env.py
"""
from __future__ import annotations

import importlib
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS = []


def check(name: str, fn):
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001
        status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
    RESULTS.append((status, name, detail))
    print(f"  [{status:4s}] {name:<55s} {detail}")


def check_python():
    return "PASS", sys.version.split()[0]


def check_torch_cuda():
    import torch

    if not torch.cuda.is_available():
        return "FAIL", "torch.cuda.is_available() is False"
    return "PASS", f"torch {torch.__version__}, {torch.cuda.device_count()} device(s) visible"


def check_gpu_inventory():
    import torch

    n = torch.cuda.device_count()
    if n == 0:
        return "FAIL", "no CUDA devices visible"
    descs = []
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        descs.append(f"gpu{i}={props.name} ({props.total_memory / 1e9:.0f}GB, sm_{props.major}{props.minor})")
    status = "PASS" if n >= 1 else "FAIL"
    return status, "; ".join(descs)


def check_bf16():
    import torch

    ok = torch.cuda.is_bf16_supported()
    return ("PASS" if ok else "WARN"), f"bf16_supported={ok}"


def check_nccl():
    import torch.distributed as dist

    ok = dist.is_nccl_available()
    return ("PASS" if ok else "WARN"), f"is_nccl_available()={ok} (needed for lab10 tensor parallelism)"


def check_vllm():
    import vllm

    return "PASS", f"vllm {vllm.__version__}"


def check_vllm_bench_cli():
    proc = subprocess.run(["vllm", "bench", "serve", "--help"], capture_output=True, text=True)
    if proc.returncode != 0:
        return "FAIL", "`vllm bench serve --help` failed; is vLLM installed correctly?"
    return "PASS", "`vllm bench serve` CLI is available"


def check_transformers():
    import transformers

    return "PASS", f"transformers {transformers.__version__}"


def check_flash_attn():
    try:
        import flash_attn

        return "PASS", f"flash_attn {flash_attn.__version__} (fallback attention backend, lab03/lab08)"
    except ImportError:
        return "WARN", "not installed; FlashAttention backend variants will be skipped"


def check_flashinfer():
    try:
        import flashinfer

        version = getattr(flashinfer, "__version__", "unknown")
        return "PASS", f"flashinfer {version} (vLLM's default Blackwell attention backend, lab03/lab08)"
    except ImportError:
        return "WARN", (
            "not installed; lab03/lab08 will skip the FlashInfer variant and vLLM will "
            "fall back to FlashAttention on Blackwell. Install with: pip install flashinfer-python"
        )


def check_hf_home():
    from common import config

    hf_home = config.HF_HOME
    on_raid = hf_home.startswith("/raid")
    status = "PASS" if on_raid else "WARN"
    return status, f"HF_HOME={hf_home}"


def check_disk_space():
    from common import config

    path = Path(config.HF_HOME)
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1e9
    status = "PASS" if free_gb > 300 else "WARN"
    return status, f"{free_gb:.0f}GB free at {path}"


def check_hf_token():
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return "PASS", "HF_TOKEN is set (needed for the gated meta-llama repos)"
    return "WARN", "HF_TOKEN not set; gated meta-llama/* downloads will fail. export HF_TOKEN=hf_..."


def check_port_free():
    from common import config

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        in_use = s.connect_ex(("127.0.0.1", config.DEFAULT_PORT)) == 0
    if in_use:
        return "WARN", f"port {config.DEFAULT_PORT} already in use; something else may be running"
    return "PASS", f"port {config.DEFAULT_PORT} free"


def check_nvidia_smi():
    if shutil.which("nvidia-smi") is None:
        return "FAIL", "nvidia-smi not found on PATH"
    proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    n_gpus = len(proc.stdout.strip().splitlines())
    return "PASS", f"{n_gpus} GPU(s) reported by nvidia-smi"


def main() -> int:
    print("vLLM Optimization Labs - environment check")
    print("=" * 70)
    check("python version", check_python)
    check("torch import + CUDA available", check_torch_cuda)
    check("GPU inventory", check_gpu_inventory)
    check("bf16 support", check_bf16)
    check("NCCL backend (lab10 tensor parallelism)", check_nccl)
    check("nvidia-smi", check_nvidia_smi)
    check("import vllm", check_vllm)
    check("`vllm bench serve` CLI", check_vllm_bench_cli)
    check("import transformers", check_transformers)
    check("flash-attn (optional)", check_flash_attn)
    check("flashinfer (optional, Blackwell default backend)", check_flashinfer)
    check("HF_HOME location", check_hf_home)
    check("disk space on HF cache volume", check_disk_space)
    check("HF_TOKEN set (gated models)", check_hf_token)
    check("benchmark port free", check_port_free)

    n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    n_warn = sum(1 for s, _, _ in RESULTS if s == "WARN")
    print("=" * 70)
    print(f"{len(RESULTS)} checks: {len(RESULTS) - n_fail - n_warn} PASS, {n_warn} WARN, {n_fail} FAIL")
    if n_fail:
        print("Fix the FAIL item(s) above before running the labs.")
        return 1
    if n_warn:
        print("You can proceed; WARN items just mean some lab variants will be skipped/slower.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
