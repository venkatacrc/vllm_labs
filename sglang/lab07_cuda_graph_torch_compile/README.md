# Lab 07 - CUDA Graphs & torch.compile

**Goal:** understand kernel-launch-overhead optimizations, which matter most
for the decode phase of serving (many cheap, repetitive steps).

## Background

- **CUDA graphs**: capture the full sequence of GPU kernel launches for a
  given batch shape once, then replay the whole graph with a single
  lightweight API call on every subsequent step, instead of re-dispatching
  each kernel individually from Python/C++ every time. This eliminates
  per-kernel CPU launch overhead, which matters a lot during decode (many
  small kernels, once per token, per request in the batch) and much less
  during prefill (a handful of large matmuls that are compute-bound
  regardless of launch overhead). SGLang captures graphs for a set of common
  batch sizes at server startup (you'll see a "Capturing cuda graph" log
  line listing batch sizes) and falls back to eager execution for
  uncaptured shapes.
- **torch.compile**: PyTorch's JIT compiler traces the model's forward pass
  and generates fused, specialized kernels (via TorchInductor) for it. This
  is complementary to CUDA graphs (which just replay whatever kernels
  already exist) since it can create *better* kernels in the first place.
  SGLang's docs currently mark `--enable-torch-compile` as an experimental,
  lightly-maintained feature that can fail for some model/version
  combinations -- this lab treats it as best-effort and continues even if
  it fails to start.

## What this lab does

Three server configurations against `$SMALL_MODEL`, benchmarked with a
decode-heavy workload (long output length, moderate concurrency, so
per-step overhead is a meaningful fraction of total time):

1. `cuda-graph-on` -- SGLang's default.
2. `cuda-graph-off` -- `--cuda-graph-backend-decode disabled
   --cuda-graph-backend-prefill disabled`.
3. `torch-compile-on` -- `--enable-torch-compile` (skipped automatically if
   it fails to start on your SGLang version).

## Run it

```bash
cd vllm_labs/sglang/lab07_cuda_graph_torch_compile
./run_lab.sh
```

Startup for the CUDA-graph-enabled configurations takes noticeably longer
than a normal server start -- SGLang is capturing graphs for a range of
batch sizes before it reports healthy. Watch the server logs
(`results/lab07_cuda_graph_torch_compile/server_*.log`) to see this happen.

## Expected takeaways

- `median_tpot_ms` (time per output token = decode step cost) should be
  visibly higher with CUDA graphs disabled -- this is the metric CUDA graphs
  target directly.
- `median_ttft_ms` (time to first token = prefill cost) should barely move
  between the two CUDA-graph configurations, confirming that prefill is
  matmul-bound rather than launch-overhead-bound.
- If the `torch-compile-on` row is present in `results/summary.csv`,
  compare `median_tpot_ms` against `cuda-graph-on` -- gains beyond what CUDA
  graphs already provide are usually modest for a 7B model at these batch
  sizes, since CUDA graphs already remove most launch overhead; torch.compile
  mainly helps when there's additional kernel-fusion opportunity (e.g.
  fusing normalization + activation ops) that raw graph replay can't provide.

Next: `lab08_speculative_decoding` attacks the decode phase from a
completely different angle -- generating multiple tokens per forward pass.
