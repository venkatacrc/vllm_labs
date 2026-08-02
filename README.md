# vLLM Optimization Labs

Twelve self-contained labs that take a base **FP32** model and progressively
apply the major serving optimizations available in vLLM, on an 8xB200 node -
ending with a capstone that shows the full fp32-to-fully-optimized speedup.

Every lab launches a real `vllm serve` process, drives it with `vllm bench
serve`, tears it down, and saves its results - no notebooks, no mocked
timings. Each lab is runnable on its own, and `run_all_labs.py` runs the
whole curriculum end to end and builds a consolidated report.

## Prerequisites

- 8x NVIDIA B200 (this was built/tuned against `nvidia-smi` reporting
  `sm_100`, 178GB/GPU - fewer GPUs work for every lab except Lab 10).
- A working vLLM + CUDA environment. Known-good versions (see
  [requirements.txt](requirements.txt)): `vllm==0.26.0`, `torch==2.11.0`,
  `transformers==5.14.1`.
- `HF_HOME` pointed at fast, large local storage (labs assume
  `/raid/llm_rl_cache/hf_home`, override with the `HF_HOME` env var).
- An `HF_TOKEN` with access to the gated `meta-llama/*` repos (accept the
  license on the model pages first).
- Optional but recommended on Blackwell: `flashinfer-python` (vLLM's
  default attention backend on B200 - Lab 03 and Lab 08 skip the FlashInfer
  variant gracefully if it's missing, but you'll want it for realistic
  numbers).

## Quickstart

```bash
# 1. Check your environment is ready
python scripts/check_env.py

# 2. Pre-fetch every model used by the labs (avoids downloads mid-benchmark)
export HF_TOKEN=hf_...
bash scripts/download_models.sh

# 3. Run everything
python run_all_labs.py

# ...or just smoke-test the whole suite quickly first
python run_all_labs.py --quick

# ...or run one lab standalone
python labs/lab04_prefix_caching/lab.py
```

Results land in `results/`: each lab writes `results/<lab_name>/results.json`,
and every lab appends a section to `results/SUMMARY.md`, which becomes the
one-page report of the whole fp32 -> optimized journey once you've run
everything (Lab 11 adds the final rollup).

## The labs

| # | Lab | What it teaches | GPUs |
|---|---|---|---|
| 00 | [FP32 baseline](labs/lab00_fp32_baseline/) | Reference point every other lab compares against | 1 |
| 01 | [Precision (fp32 -> fp16/bf16)](labs/lab01_precision_bf16_fp16/) | The single biggest, lowest-risk win | 1 |
| 02 | [Continuous batching + paged KV cache](labs/lab02_continuous_batching_paged_kv_cache/) | Naive HF static batching vs vLLM's scheduler | 1 |
| 03 | [High-efficiency attention backends](labs/lab03_attention_backends/) | FlashInfer vs FlashAttention vs Triton vs FlexAttention | 1 |
| 04 | [Automatic prefix caching](labs/lab04_prefix_caching/) | Reusing KV cache across requests with shared prefixes | 1 |
| 05 | [Chunked prefill](labs/lab05_chunked_prefill/) | Keeping short requests responsive alongside long ones | 1 |
| 06 | [CUDA graphs / compilation](labs/lab06_cuda_graphs_compilation/) | Kernel-launch overhead at small batch sizes | 1 |
| 07 | [Weight quantization (AWQ / GPTQ)](labs/lab07_weight_quantization_awq_gptq/) | INT4 checkpoints, memory vs throughput | 1 |
| 08 | [FP8 quantization (Blackwell-native)](labs/lab08_fp8_quantization/) | Dynamic FP8 weights + KV cache, no calibration needed | 1 |
| 09 | [Speculative decoding](labs/lab09_speculative_decoding/) | Draft-model and n-gram proposers, latency vs throughput regime | 1-2 |
| 10 | [Tensor parallelism](labs/lab10_tensor_parallelism/) | Scaling a 32B model across TP=1/2/4/8 | up to 8 |
| 11 | [Capstone: full stack](labs/lab11_capstone_full_stack/) | The complete fp32 -> optimized journey, both models | up to 4 |

See [`AGENTS.md`-style detail in each lab's own README](labs/) for the
concept explanation, exact experiment, and what to look for in the results.

### Why 12 labs instead of a single "PagedAttention" lab?

If you've read that PagedAttention is deprecated - you read correctly.
vLLM v0.25.0 (July 2026) deleted the original 2023 standalone PagedAttention
kernel. The *technique* (paged, block-based KV cache management) didn't go
away: it's now built into every modern attention backend (FlashInfer,
FlashAttention, Triton, FlexAttention) under Model Runner V2, which has been
the default execution path for a while. Lab 02 teaches the underlying
concept; Lab 03 is a dedicated shootout between the backends that now carry
it, including the FlashInfer `trtllm-gen` path B200 uses by default.

## How the suite is wired

- `common/config.py` - every model ID, path, and default lives here.
  Override any of it via environment variables without touching code
  (e.g. `LAB_MODEL_PRIMARY=some/other-model`).
- `common/server.py` - starts/stops a `vllm serve` subprocess, polls
  `/health`, and guarantees teardown (SIGTERM then SIGKILL, process-group
  wide) so GPU memory is actually free before the next lab starts.
- `common/bench.py` - wraps `vllm bench serve`, parses its JSON result file,
  and can fire multiple benchmark workloads concurrently at the same server
  (used by Lab 05's mixed-traffic simulation).
- `common/report.py` - console tables, speedup-vs-baseline columns, and the
  `results/SUMMARY.md` rollup.
- `common/hf_baseline.py` - the deliberately naive HF `transformers.generate()`
  static-batching baseline used only by Lab 02, run in an isolated
  subprocess.
- Every `labs/<name>/lab.py` exposes `run(quick: bool, gpus: str | None)` and
  a standalone CLI (`--quick`, `--gpus`), so `run_all_labs.py` and direct
  invocation share the exact same code path.

## `run_all_labs.py` options

```bash
python run_all_labs.py --list              # show the lab order and exit
python run_all_labs.py --labs 0,1,3        # run a subset
python run_all_labs.py --quick             # shrink every workload for a fast smoke test
python run_all_labs.py --gpus 1            # override the default single GPU used by most labs
python run_all_labs.py --skip-download     # suppress the download reminder
```

A single failing lab doesn't stop the suite - `run_all_labs.py` logs the
failure, moves on, and reports every failure at the end.

## Notes on runtime

A full (non `--quick`) run of the entire suite involves ~30+ server
start/stop cycles (several of them loading a 32B model across multiple
GPUs), so budget a few hours for `python run_all_labs.py` end to end,
depending on model download state and disk speed. `--quick` shrinks every
workload for a fast correctness smoke test (tens of minutes), not for
numbers you should trust.

## Customizing

Nothing here is hardcoded beyond `common/config.py`. Swap in different
models, ports, or GPU defaults via the environment variables documented at
the top of that file, e.g.:

```bash
export LAB_MODEL_PRIMARY=Qwen/Qwen2.5-7B-Instruct
export LAB_DEFAULT_GPU=2
python run_all_labs.py --labs 0,1
```
