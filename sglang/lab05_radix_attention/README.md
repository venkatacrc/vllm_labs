# Lab 05 - RadixAttention: Automatic Prefix Caching

**Goal:** see, quantitatively, why RadixAttention is SGLang's headline
feature: automatic reuse of KV cache across requests that share a token
prefix, with zero application-level changes.

## Background

Every LLM request starts with a "prefill" phase (process the whole input
prompt to build its KV cache) followed by a "decode" phase (generate output
tokens one at a time, extending that KV cache). Prefill cost scales with
prompt length -- for a workload with long, repeated system prompts (chat
assistants, RAG pipelines, few-shot prompting, agent scaffolding), most GPUs
end up repeatedly re-computing KV cache for tokens that are *byte-for-byte
identical* to what a previous request already computed.

SGLang's **RadixAttention** stores every request's KV cache pages in a radix
tree (a trie) keyed by token id sequence. When a new request arrives,
SGLang walks the tree to find the longest matching prefix already cached,
reuses that KV cache directly, and only computes the (usually much shorter)
suffix that's actually new. This is on by default -- no code changes, no
special client-side caching logic, it just works based on token-level
prefix matching.

## What this lab does

Uses `sglang.bench_serving --dataset-name generated-shared-prefix`, a
built-in synthetic workload generator that creates `$GSP_NUM_GROUPS`
distinct "system prompts" (~$GSP_SYSTEM_PROMPT_LEN tokens each) and issues
`$GSP_PROMPTS_PER_GROUP` different short questions against each one -- e.g.
8 groups x 32 questions = 256 requests, but only 8 unique ~2048-token
prefixes.

Runs this workload twice against `$SMALL_MODEL`:

1. `radix-cache-enabled` -- SGLang's default.
2. `radix-cache-disabled` -- launched with `--disable-radix-cache`, which
   forces every request to prefill its full prompt from scratch even if an
   identical prefix was just processed.

## Run it

```bash
cd vllm_labs/sglang/lab05_radix_attention
./run_lab.sh
```

## Expected takeaways

- `median_ttft_ms` (time to first token) should be dramatically lower with
  RadixAttention enabled: after the first request in each group "warms" the
  cache for that system prompt, every subsequent request in the same group
  only needs to prefill the short, unique question -- not the full
  ~2048-token shared prefix.
- `output_throughput_tok_s` should also be higher with caching enabled,
  since GPU time that would have been spent redundantly re-prefilling
  identical tokens is instead available for decoding other requests.
- This benefit is "free" -- it required zero changes to how you structure
  requests, as long as your client sends the *same* token sequence for the
  shared portion (same system prompt text, same tokenizer, no
  non-deterministic templating). It's why designing your prompts to put
  static/shared content first and per-request/dynamic content last is a
  best practice for any RadixAttention-based server.
- Try increasing `GSP_SYSTEM_PROMPT_LEN` (e.g. `GSP_SYSTEM_PROMPT_LEN=8192
  ./run_lab.sh`) to see the gap widen further -- the longer the shared
  prefix relative to the unique suffix, the bigger RadixAttention's win.

Next: `lab06_scheduling_batching` tunes the knobs that control *how*
SGLang's continuous batching scheduler packs requests together.
