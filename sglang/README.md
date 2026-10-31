# SGLang Optimization Labs

A hands-on curriculum for taking a base FP32-precision LLM to an efficiently
served, production-tuned deployment on [SGLang](https://github.com/sgl-project/sglang),
built for a single 8xNVIDIA B200 node.

Each lab is **self-contained**: it has its own `README.md` (concept
explanation + what to expect), its own `run_lab.sh` (launches SGLang
server(s), runs benchmarks, records results, cleans up after itself), and
can be run independently in any order after `lab00_setup`. A top-level
`run_all_labs.sh` runs the whole curriculum end to end.

## The curriculum

| # | Lab | What it teaches |
|---|-----|------------------|
| 00 | [`lab00_setup`](lab00_setup/) | Isolated SGLang install, model downloads, smoke test |
| 01 | [`lab01_fp32_baseline`](lab01_fp32_baseline/) | Naive HF loop vs SGLang server, the "before" numbers |
| 02 | [`lab02_precision`](lab02_precision/) | FP32 vs BF16 vs FP16 |
| 03 | [`lab03_quantization`](lab03_quantization/) | FP8 (online) vs AWQ/GPTQ INT4 (pre-quantized) |
| 04 | [`lab04_tensor_parallelism`](lab04_tensor_parallelism/) | TP scaling (1/2/4/8 GPUs), communication overhead |
| 05 | [`lab05_radix_attention`](lab05_radix_attention/) | RadixAttention automatic prefix caching |
| 06 | [`lab06_scheduling_batching`](lab06_scheduling_batching/) | Overlap scheduler, chunked prefill, queue policy |
| 07 | [`lab07_cuda_graph_torch_compile`](lab07_cuda_graph_torch_compile/) | CUDA graphs & torch.compile |
| 08 | [`lab08_speculative_decoding`](lab08_speculative_decoding/) | N-gram speculative decoding (no draft model needed) |
| 09 | [`lab09_structured_output`](lab09_structured_output/) | Grammar-guided JSON Schema decoding |
| 10 | [`lab10_moe_expert_parallel`](lab10_moe_expert_parallel/) | MoE: Expert Parallelism & DP Attention |
| 11 | [`lab11_capstone`](lab11_capstone/) | Stack the wins, measure total speedup, final charts |
| 12 | [`lab12_attention_backends`](lab12_attention_backends/) | FA4 vs FlashInfer vs TRT-LLM MHA vs Triton on Blackwell |
| 13 | [`lab13_hierarchical_cache`](lab13_hierarchical_cache/) | HiCache: GPU -> CPU tiered KV cache behind RadixAttention |

Every lab writes its raw results to `results/<lab_name>/` and appends a row
per configuration to `results/summary.csv`, so `lab11_capstone` (and you) can
compare across the whole curriculum at the end.

## Prerequisites

- The 8xB200 node you described (`nvidia-smi` showing 8x B200, 183GB each,
  CUDA 13.0 driver).
- `conda` on `PATH` (lab00 creates a dedicated `sglang-labs` environment; it
  intentionally does not touch whatever env your existing vLLM/DeepSpeed/etc.
  stack lives in).
- Internet access to Hugging Face Hub and PyPI (or a pre-warmed
  `$HF_HOME` cache and pip mirror, if this node is air-gapped, in which case
  see `lab00_setup/README.md` for the Docker-based alternative).
- ~150GB free disk for model weights (you have 27TB free on `/raid` --
  trivial) and a few more GB for the `sglang-labs` conda environment.

## Quick start

```bash
cd vllm_labs/sglang
./run_all_labs.sh              # runs lab00 through lab13, in order (a few hours total)
```

Or work through it incrementally, reading each README as you go:

```bash
cd vllm_labs/sglang/lab00_setup && ./run_lab.sh   # do this first, always
cd ../lab01_fp32_baseline && ./run_lab.sh
cd ../lab02_precision && ./run_lab.sh
# ...etc, or:
cd .. && ./run_all_labs.sh --from lab02
```

Useful `run_all_labs.sh` flags:

```bash
./run_all_labs.sh --list                  # show the lab order
./run_all_labs.sh --only lab03,lab08      # run just these labs
./run_all_labs.sh --skip lab04,lab10      # skip the multi/8-GPU-heavy labs
./run_all_labs.sh --from lab05            # resume from a specific lab
./run_all_labs.sh --dry-run               # print the plan without running anything
```

## Configuration

Every knob lives in [`common/env.sh`](common/env.sh) with sensible
defaults, and can be overridden via environment variables without editing
any file, e.g.:

```bash
SMALL_MODEL=Qwen/Qwen2.5-3B-Instruct BENCH_MAX_CONCURRENCY=128 ./run_all_labs.sh --only lab02
```

Key variables: `SMALL_MODEL`, `MID_MODEL`, `MOE_MODEL` (the three
checkpoints used across the curriculum -- see `lab00_setup/README.md` for
why these were chosen), `HF_HOME`, `SGLANG_CONDA_ENV`,
`BENCH_NUM_PROMPTS` / `BENCH_MAX_CONCURRENCY` / `BENCH_RANDOM_INPUT_LEN` /
`BENCH_RANDOM_OUTPUT_LEN` (default benchmark load shape).

## How each lab is structured

```
labNN_topic/
  README.md      <- read this first: background, what the lab measures, expected results
  run_lab.sh      <- launches server(s), runs sglang.bench_serving, records + tears down
  *.py            <- any custom client/benchmark scripts specific to that lab
```

Shared infrastructure lives in `common/`:

- `env.sh` -- central configuration (models, ports, paths, benchmark defaults).
- `lib.sh` -- bash helpers: start/stop an SGLang server with health-check
  polling, GPU memory helpers, benchmark runner, result recording.
- `bench_lib.py` -- parses `sglang.bench_serving` JSON output into
  `results/summary.csv`.
- `plot_results.py` -- renders `results/summary.csv` into comparison bar
  charts (used by `lab11_capstone`).

## Why this structure

- **Self-contained labs**: every lab can be understood and re-run on its
  own -- read one `README.md`, run one `run_lab.sh`. Nothing requires
  reverse-engineering a shared framework to understand what a single lab is
  doing.
- **One variable at a time**: each lab changes exactly one axis (precision,
  quantization, parallelism, caching, scheduling, speculative decoding,
  structured output, MoE parallelism) while holding everything else fixed,
  so the throughput/latency numbers you see are attributable to that one
  change.
- **Honest baselines**: lab01 measures a genuinely naive serving path (plain
  `transformers.generate()` in a loop) so later speedup numbers mean
  something, instead of comparing two already-optimized configurations
  against each other.
- **Real SGLang docs, real flags**: every server flag used across these labs
  was verified against SGLang's current `server_arguments` and feature docs
  (linked from each lab's README) rather than guessed from memory --
  SGLang's CLI surface changes fast across releases, so if a flag has been
  renamed by the time you run this, `python -m sglang.launch_server --help`
  is always the ground truth.

## Troubleshooting

- **Server fails to start on Blackwell (`sgl_kernel` / SM100 errors)**: see
  the "Blackwell troubleshooting" section in `lab00_setup/README.md`.
- **OOM during a lab**: lower `BENCH_MAX_CONCURRENCY`, or add
  `--mem-fraction-static 0.7` (or lower) to that lab's `start_sglang_server`
  call in its `run_lab.sh`.
- **A lab hangs waiting for the server to become healthy**: check
  `results/<lab>/server_*.log` for the actual error; big models can
  legitimately take several minutes to load + capture CUDA graphs, so
  `SERVER_STARTUP_TIMEOUT` in `common/env.sh` is generous (1200s) by
  default, but you can raise it further if needed.
- **Leftover GPU memory between runs**: `common/lib.sh`'s `stop_sglang_server`
  and `run_all_labs.sh`'s between-lab cleanup both `pkill` any stray
  `sglang.launch_server` processes and wait for `nvidia-smi` to report
  freed memory, but if you Ctrl-C mid-lab you may need to do this by hand:
  `pkill -9 -f sglang.launch_server`.
