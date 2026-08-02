# Lab 03: High-efficiency attention backends

## Concept

As covered in Lab 02: the original standalone PagedAttention kernel is
gone (removed in vLLM v0.25.0, July 2026). What's left to tune is *which
attention backend* vLLM uses, since paged KV-cache management now lives
inside each backend's own kernels. On NVIDIA Blackwell (B200, SM100), the
backends available today are:

| Backend | Notes |
|---|---|
| `FLASHINFER` | vLLM's **default on Blackwell**. Ships a decode-optimized `trtllm-gen` path with native FP8/NVFP4 support (relevant again in Lab 08). Requires the `flashinfer-python` package. |
| `FLASH_ATTN` | FlashAttention, FA4 by default on SM100. vLLM's automatic fallback if FlashInfer isn't installed. |
| `TRITON_ATTN` | Pure-Triton attention kernel. Slower than the two above but has no extra native-kernel dependency, and supports dtypes/shapes the others don't (e.g. FP32, used unavoidably in Lab 00). |
| `FLEX_ATTENTION` | PyTorch's `torch.nn.attention.flex_attention`, compiled via `torch.compile`. Flexible (custom masks/score-mods) but generally the slowest of the four for standard decoder-only attention. |

You select a backend with the `VLLM_ATTENTION_BACKEND` environment
variable, e.g. `VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve ...`. If
`flashinfer-python` isn't installed in your environment, the `FLASHINFER`
variant below is skipped automatically (vLLM would otherwise fail to start)
- run `python scripts/check_env.py` to see if you have it.

## Why prefill vs decode matters

Attention backends don't all have the same strengths. A backend can be
excellent at prefill (processing a long prompt's tokens in parallel, compute
-bound) and mediocre at decode (one new token at a time per sequence,
latency/launch-overhead-bound), or vice versa. This lab runs two separate
workload profiles against every backend:

- **Prefill-heavy**: long input (~4000 tokens), short output (~32 tokens),
  many concurrent requests - stresses the prefill path.
- **Decode-heavy**: short input (~128 tokens), long output (~512 tokens), low
  concurrency (4) - stresses the per-token decode path (this is where the
  `trtllm-gen`/FlashInfer decode kernel is specifically optimized on SM100).

## What this lab does

For each of the four backends: launch `vllm serve` with
`VLLM_ATTENTION_BACKEND=<backend>`, run the prefill-heavy and decode-heavy
benchmarks, tear down, move on. Any backend that fails to start (missing
dependency, unsupported config) is reported as skipped rather than failing
the whole lab.

## Run it

```bash
python labs/lab03_attention_backends/lab.py
python labs/lab03_attention_backends/lab.py --quick
```

## What to look for

- Whether `FLASHINFER` is actually available in your environment (check
  `results/logs/lab03_FLASHINFER_*.log` if it's skipped).
- Decode-heavy `mean_tpot_ms` (time per output token) is usually where the
  backends differentiate most sharply - this is the per-step, per-token
  cost dominated by kernel launch overhead and memory-access patterns.
- Prefill-heavy `output_throughput` shows which backend is best at chewing
  through long prompts.
