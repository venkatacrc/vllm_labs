# Lab 03: High-efficiency attention backends

## Concept

As covered in Lab 02: the original standalone PagedAttention kernel is
gone (removed in vLLM v0.25.0, July 2026). What's left to tune is *which
attention backend* vLLM uses, since paged KV-cache management now lives
inside each backend's own kernels. On NVIDIA Blackwell (B200, SM100), the
backends you can pick for a standard GQA decoder like Llama-3.1-8B are:

| Backend | Notes |
|---|---|
| `FLASH_ATTN` | FlashAttention. Version is selectable with `--attention-config.flash_attn_version=2/3/4`; **FA4 is the default on SM100+**, FA3 on Hopper, FA2 elsewhere. Automatic selection tries FlashAttention first. |
| `FLASHINFER` | FlashInfer. On SM100 it uses the `trtllm-gen` decode kernels (with attention-sink support and native FP8 KV); disable with `--attention-config.use_trtllm_attention=0`. Part of vLLM's Blackwell fallback order (TRT-LLM ragged → FlashInfer → ...). |
| `TRITON_ATTN` | Pure-Triton attention kernel. Slower than the two above but has no extra native-kernel dependency and supports dtypes/shapes the others don't (e.g. FP32, used unavoidably in Lab 00). |
| `FLEX_ATTENTION` | PyTorch's `flex_attention`, compiled via `torch.compile`. Flexible (custom masks/score-mods) but generally the slowest for plain causal attention. |

Multimodal models get *composite* backends such as `TRITON_FLASHINFER`
(Triton for bidirectional image-prefix attention, FlashInfer for causal text),
picked automatically - not relevant for this text-only lab, but worth knowing.

### Selecting a backend (current vLLM)

The old `VLLM_ATTENTION_BACKEND` environment variable has been replaced by a
CLI flag / structured config (the two forms are mutually exclusive):

```bash
vllm serve <model> --attention-backend FLASHINFER
vllm serve <model> --attention-config '{"backend": "FLASH_ATTN", "flash_attn_version": 2}'
vllm serve <model> -ac.backend TRITON_ATTN
```

When you pick a backend explicitly, vLLM validates it against the model
(dtype, head size, compute capability) and errors out with the reason if it
is incompatible; this lab reports that as "skipped". With no flag, vLLM walks
a priority list and takes the first compatible backend.

## Why prefill vs decode matters

Attention backends don't all have the same strengths. A backend can be
excellent at prefill (processing a long prompt's tokens in parallel,
compute-bound) and mediocre at decode (one new token at a time per sequence,
memory-bandwidth/launch-overhead-bound), or vice versa. This lab runs two
workload profiles against every backend:

- **Prefill-heavy**: long input (~4000 tokens), short output (~32 tokens),
  many concurrent requests - stresses the prefill path.
- **Decode-heavy**: short input (~128 tokens), long output (~512 tokens), low
  concurrency (4) - stresses the per-token decode path.

## What this lab does

For each variant (FlashInfer, FlashAttention 4, FlashAttention 2, Triton,
FlexAttention): launch `vllm serve` with the backend flag, run the
prefill-heavy and decode-heavy benchmarks, tear down, move on. Any variant
that fails to start is reported as skipped rather than failing the lab.

## Run it

```bash
python labs/lab03_attention_backends/lab.py
python labs/lab03_attention_backends/lab.py --quick
```

## What to look for

- FA4 vs FA2 on the prefill-heavy run: FA4 is written for Blackwell's tensor
  memory and async TMA pipeline, so the gap should be largest on long prompts.
- Decode-heavy `mean_tpot_ms` (time per output token) is usually where the
  backends differentiate most sharply.
- Check `results/logs/lab03_<backend>_*.log` for the line where vLLM reports
  the backend it actually selected.
