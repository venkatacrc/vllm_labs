# Lab 12: Attention backends on Blackwell

## Concept

The attention kernel is the one part of the transformer forward pass whose
cost grows with context length, and it is also the part that reads the
paged KV cache. SGLang exposes the kernel library as a server argument:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --attention-backend fa4
# or per phase:
python -m sglang.launch_server ... --prefill-attention-backend fa4 --decode-attention-backend trtllm_mha
```

Current choices include `triton`, `torch_native`, `flex_attention`, `fa3`,
`fa4`, `flashinfer`, `trtllm_mha`, and MLA-specific backends (`flashmla`,
`trtllm_mla`, `cutedsl_mla`, ...) for DeepSeek-style models. On B200 the
four worth comparing for a GQA model are:

| Backend | What it is |
|---|---|
| `fa4` | FlashAttention 4, written in CuTe-DSL for Blackwell's tensor memory (TMEM) and 2-CTA MMA. |
| `flashinfer` | FlashInfer's paged attention (JIT-compiled, plan/run split for CUDA graphs). |
| `trtllm_mha` | TensorRT-LLM's hand-tuned Blackwell MHA kernels, exposed via FlashInfer. |
| `triton` | Portable Triton kernels - the reference point. |

## What this lab does

For each backend: start the server, run a **prefill-heavy** workload (4000
in / 32 out, concurrency 32) and a **decode-heavy** workload (128 in / 512
out, concurrency 4), stop the server. A backend that is not supported by
your SGLang build is skipped.

## Run it

```bash
cd sglang/lab12_attention_backends && ./run_lab.sh
ATTN_BACKENDS="fa4 trtllm_mha" ./run_lab.sh    # subset
```

## What to look for

- Prefill rows: `median_ttft_ms`. FA4 should lead on long prompts.
- Decode rows: `median_tpot_ms`. `trtllm_mha` is tuned for small-batch decode.
- This is the same experiment as vLLM lab 03 - compare the two engines'
  numbers for similar backends.
