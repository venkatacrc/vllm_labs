# Lab 06: CUDA graphs / compilation

## Concept

Every forward pass through a model involves launching hundreds of small
CUDA kernels. At small batch sizes (the common case for low-latency,
low-concurrency decode), the CPU-side overhead of *launching* those kernels
can rival or exceed the GPU time actually spent computing them.

**CUDA graphs** capture a fixed sequence of kernel launches once and replay
the whole graph with a single call, eliminating per-kernel launch overhead.
vLLM also runs `torch.compile` over the model to fuse ops (e.g. RMSNorm +
quant, SiLU-mul + quant, allreduce + RMSNorm) before graph capture.

### Optimization levels (current vLLM)

| Flag | What you get |
|---|---|
| `--enforce-eager` | No compilation and no CUDA graphs. Fastest startup, slowest decode. |
| `-O0` | No optimizations. |
| `-O1` | Fast compilation and simple fusions, `PIECEWISE` CUDA graphs. |
| `-O2` (default) | More compile ranges and fusions, `FULL_AND_PIECEWISE` CUDA graphs. |
| `-O3` | Aggressive; currently equal to `-O2`. |

The CUDA graph mode is the single knob in
`--compilation-config '{"cudagraph_mode": "..."}'`, one of `NONE`,
`PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE`:

- `PIECEWISE` graphs everything *except* attention (attention runs eagerly
  between graph pieces) - works with every attention backend.
- `FULL` graphs the whole forward pass including attention - needs the
  attention backend to support graph capture.
- `FULL_AND_PIECEWISE` uses full graphs for uniform decode batches and
  piecewise graphs for mixed prefill+decode batches (the default).
- `FULL_DECODE_ONLY` uses full graphs for decode, eager for mixed batches -
  less memory spent on captured graphs, useful with P/D disaggregation.

(The old `{"level": 3}` compilation-config key has been replaced by
`-O<n>` / `"mode"`.)

## What this lab does

Runs five variants (`--enforce-eager`, `-O0`, `-O1`, `-O2`, and `-O2` with
`FULL_DECODE_ONLY`) at two concurrency levels:

- **Low concurrency** (`max-concurrency=1`): isolates per-token decode
  latency, where CUDA graphs matter most.
- **High concurrency** (`max-concurrency=64`): the workload becomes more
  compute-bound, so the relative benefit of CUDA graphs should shrink.

## Run it

```bash
python labs/lab06_cuda_graphs_compilation/lab.py
python labs/lab06_cuda_graphs_compilation/lab.py --quick
```

## What to look for

- `eager_low` vs `O2_default_low`: `mean_tpot_ms` should be noticeably
  higher (worse) under `--enforce-eager`.
- `O1_piecewise` vs `O2_default`: the extra gain from capturing attention
  inside full graphs.
- The `_high` runs: the gaps shrink as each step does more GPU work.
- Startup time in `results/logs/lab06_*.log`: compilation and graph capture
  add tens of seconds (cached under `~/.cache/vllm/torch_compile_cache` on
  later runs) - the tradeoff for faster steady-state decode.
