# Lab 00 - Environment Setup

**Goal:** get a clean, isolated SGLang install working on your 8xB200 node, and
pre-download every model the rest of the curriculum needs.

## Why a separate conda env?

Your node already has a conda env with `torch 2.11.0+cu130`, `vllm 0.26.0`,
`deepspeed`, etc. SGLang ships its own tight pins on `torch`, `flashinfer`,
and `sgl-kernel` (its custom CUDA kernel package, which must match your CUDA
toolkit **and** GPU architecture -- B200 is `sm_100`, Blackwell). Installing
SGLang into an env that already has vLLM's own flash-attn / kernel stack is
one of the most common sources of "works in one process, segfaults in
another" bugs. We create a new env, `sglang-labs`, and leave your existing
env untouched.

## What this lab does

1. `install_sglang.sh` - creates the `sglang-labs` conda env (Python 3.11)
   and installs `sglang[all]` targeting the CUDA 13.0 build that matches your
   driver (`nvidia-smi` reported `CUDA Version: 13.0`). It also (re)installs
   `flashinfer-python`, which provides the fused attention kernels SGLang
   uses by default on sm75+ GPUs including B200.
2. `download_models.sh` - pre-fetches every checkpoint used later:
   - `Qwen/Qwen2.5-7B-Instruct` - the small model used in labs 00-03, 05-09
   - `Qwen/Qwen2.5-7B-Instruct-AWQ` and `-GPTQ-Int4` - pre-quantized variants for lab03
   - `Qwen/Qwen2.5-32B-Instruct` - the mid-size model used for TP scaling in lab04
   - `Qwen/Qwen3-30B-A3B` - a Mixture-of-Experts model used in lab10
   All models are ungated (no HF token needed) and land in `$HF_HOME`
   (defaults to `/raid/llm_rl_cache/hf_home`, matching your existing cache
   location, ~150GB total).
3. `verify_install.sh` - boots the small model on 1 GPU, sends one chat
   completion request, and shuts the server down again.

## Run it

```bash
cd vllm_labs/sglang/lab00_setup
./run_lab.sh
```

This takes ~10-30 minutes depending on network speed (mostly model
downloads). Re-running is safe; each step skips work that's already done.

## Blackwell (B200 / sm_100) troubleshooting

SGLang's custom kernel package, `sgl-kernel`, ships architecture-specific
wheels. If `verify_install.sh` fails at server startup with an error like:

```
RuntimeError: ... expected SM100 variant ... sgl_kernel ...
```

it means pip resolved a `sgl-kernel` build that doesn't include Blackwell
(sm_100) kernels. Fix it by force-reinstalling the CUDA-13-tagged wheel
directly from the SGLang wheel index (adjust the version to whatever
`python -c "import sglang; print(sglang.__version__)"` reports):

```bash
conda activate sglang-labs
pip uninstall -y sgl-kernel
pip install "sgl-kernel==<version>+cu130" \
  --extra-index-url https://docs.sglang.ai/whl/cu130/
```

If pip/uv installation continues to be painful, the SGLang team's officially
supported path for CUDA 13 hosts is the prebuilt Docker image, which bundles
matching torch/flashinfer/sgl-kernel versions:

```bash
docker run --gpus all --shm-size 32g --ipc=host \
  -p 30000:30000 \
  -v /raid/llm_rl_cache/hf_home:/root/.cache/huggingface \
  lmsysorg/sglang:latest-cu130-runtime \
  python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000
```

If you go the Docker route, every other lab's `run_lab.sh` still works -
just replace the `start_sglang_server` call in `common/lib.sh` with the
equivalent `docker run ...` invocation, or run `docker exec` into a
long-lived container that has the `sglang-labs` conda env's equivalent
packages installed.

## What you should understand after this lab

- Why SGLang benefits from an isolated environment on multi-framework
  research nodes.
- The two SGLang benchmarking entry points you'll use throughout the
  curriculum: `python -m sglang.launch_server` (the OpenAI-compatible
  server) and `python -m sglang.bench_serving` (the async load generator).
- Where model weights and results live (`$HF_HOME`, `results/`).
