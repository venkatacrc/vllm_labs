---
title: Transformer inference and the KV cache
track: Foundations
order: 2
---

# Transformer inference and the KV cache

## One forward pass

A decoder-only transformer turns token ids into embeddings, runs them
through L identical layers, and projects the last hidden state to
vocabulary logits:

```
tokens → embed → [ RMSNorm → Attention(Q,K,V) → + → RMSNorm → MLP / MoE → + ] × L → norm → LM head → logits → sample
```

- **Attention** mixes information across positions: each token's query
  Q is compared with the keys K of all earlier tokens, and the
  softmax-weighted values V are summed.
- **MLP** (or **MoE**: a router picks k of E expert MLPs per token) is
  where most parameters and FLOPs live.

## Prefill and decode

Generation has two phases with opposite performance profiles:

| | Prefill | Decode |
|---|---|---|
| Input per step | the whole prompt (N tokens) | 1 new token per sequence |
| Bound by | compute (big matmuls) | memory bandwidth (weights + KV) |
| Sets | **TTFT** (time to first token) | **TPOT / ITL** (time per output token) |
| Scales with | prompt length (attention: ~N²) | batch size, context length |

## The KV cache

Attention at step t needs K and V for every earlier token. Rather
than recompute them, engines store them: the **KV cache**. Its size per
token is

```
bytes/token = 2 × layers × kv_heads × head_dim × bytes_per_element
```

(2 for K and V). Examples in BF16:

| Model | L | KV heads | head dim | KV per token | 8K-token sequence |
|---|---|---|---|---|---|
| Llama-3.1-8B | 32 | 8 | 128 | 128 KiB | 1 GiB |
| Qwen2.5-7B | 28 | 4 | 128 | 56 KiB | 448 MiB |
| Qwen3-32B | 64 | 8 | 128 | 256 KiB | 2 GiB |

So after weights, **KV cache is what limits how many sequences you can run
at once**, and therefore throughput. Every major serving optimization
touches it:

- **Paged KV cache** (vLLM lab 02): store KV in fixed-size blocks (pages)
  allocated on demand, like virtual memory, instead of one contiguous slab
  per sequence sized for the max length. Near-zero fragmentation means far
  more concurrent sequences.
- **Prefix caching / RadixAttention** (vLLM lab 04, SGLang lab 05): blocks are
  content-addressed by the tokens they hold, so requests that share a
  prefix (system prompt, document, conversation history) reuse them and
  skip that part of prefill.
- **KV quantization** (FP8 KV, vLLM lab 08): half the bytes per token.
- **Offloading / tiering** (SGLang lab 13, llm-d lab 04): spill KV to CPU
  RAM or disk instead of dropping it.
- **Prefix-aware routing** (llm-d labs 01-02): send each request to the
  replica that already holds its prefix.
- **P/D disaggregation** (llm-d lab 05): compute KV on prefill GPUs, ship it
  to decode GPUs.

## Attention variants that shrink KV

- **MHA**: one KV head per query head. **GQA** (Llama 3, Qwen): groups of
  query heads share a KV head (8 KV heads for 32-64 query heads) → 4-8x
  smaller KV. **MQA**: a single KV head.
- **MLA** (DeepSeek-V2/V3): compresses K/V into a low-rank latent vector per
  token, decompressed inside the attention kernel - very small KV, needs
  special kernels (FlashMLA, TRT-LLM MLA).
- **Sliding-window / hybrid** layers keep only recent tokens in some layers.

## Continuous batching

Static batching waits for a batch of requests and runs them to completion,
so short requests wait for the longest one. **Continuous (iteration-level)
batching** re-forms the batch every step: finished sequences leave, waiting
ones join (their prefill is scheduled into the step). All modern engines do
this; it is the single biggest throughput win over a naive loop (vLLM lab 02).

## Sampling

Logits → temperature/top-p/top-k → token. Greedy (temperature 0) is
deterministic given identical numerics. **Structured output** (SGLang lab 09)
masks logits so only tokens valid under a JSON schema/grammar can be
sampled. **Speculative decoding** (vLLM lab 09, SGLang lab 08) drafts
several tokens cheaply and verifies them in one target-model pass.

## Flashcards

Q: Formula for KV-cache bytes per token?
A: 2 × layers × KV heads × head dim × bytes per element (2 for K and V).

Q: KV cache per token for Llama-3.1-8B in BF16?
A: 2 × 32 × 8 × 128 × 2 B = 128 KiB, so an 8K-token sequence needs about 1 GiB.

Q: Which phase sets TTFT and which sets TPOT?
A: Prefill (processing the prompt) sets time to first token; decode (one token per step) sets time per output token.

Q: What problem does a paged KV cache solve?
A: Fragmentation and over-reservation: KV is stored in fixed-size blocks allocated on demand instead of one contiguous max-length buffer per sequence, so many more sequences fit.

Q: How does GQA reduce KV cache size?
A: Several query heads share one key/value head (e.g. 32 query heads, 8 KV heads), shrinking KV by the group factor.

Q: What is MLA?
A: Multi-head Latent Attention (DeepSeek): K and V are compressed into a small per-token latent vector, cutting KV size dramatically; it needs dedicated kernels.

Q: What is continuous batching?
A: Re-forming the batch at every decode step so finished sequences leave and new ones join immediately, instead of waiting for a whole static batch to finish.

Q: Why does prefix caching speed up TTFT?
A: Requests sharing a prompt prefix reuse already-computed KV blocks, so prefill only runs on the new suffix.

Q: In MoE models, what does the router do?
A: For each token it selects the top-k of E expert MLPs (e.g. 8 of 128), so only a fraction of parameters is active per token.
