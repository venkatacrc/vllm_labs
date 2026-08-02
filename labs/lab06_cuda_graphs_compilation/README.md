# Lab 06: CUDA graphs / compilation

## Concept

Every forward pass through a model involves launching hundreds of small
CUDA kernels. At small batch sizes (the common case for low-latency,
low-concurrency decode), the CPU-side overhead of *launching* those kernels
can rival or exceed the GPU time actually spent computing them.

**CUDA graphs** let you capture a fixed sequence of kernel launches once
and replay the whole graph with a single call, eliminating per-kernel
launch overhead. vLLM captures CUDA graphs for its decode step shapes by
default. **`--enforce-eager`** disables this, falling back to normal
PyTorch eager-mode execution (one Python-dispatched kernel launch at a
time) - useful for debugging, much slower for small-batch decode.

vLLM also uses `torch.compile` (piecewise compilation around the CUDA
graph regions) to fuse/optimize the surrounding non-graphed ops. This lab's
third variant attempts a more aggressive `--compilation-config` level; if
your vLLM build's exact compilation-config schema differs, that variant is
skipped gracefully rather than failing the lab.

## What this lab does

Compares `--enforce-eager` vs the default (CUDA graphs + compilation) at
two concurrency levels:

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

- `eager_low` vs `cudagraph_default_low`: `mean_tpot_ms` should be
  noticeably higher (worse) under `--enforce-eager`.
- `eager_high` vs `cudagraph_default_high`: the gap should be smaller than
  in the low-concurrency case - CUDA graphs help most when GPU work per
  step is small relative to launch overhead.
- Startup time in the server log: CUDA graph capture adds to server startup
  time (it has to capture graphs for a range of batch sizes) - that's the
  tradeoff for faster steady-state decode.
