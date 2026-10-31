# Lab 09: Speculative decoding

## Concept

Normal decoding generates exactly one token per forward pass per sequence.
**Speculative decoding** has a cheap "proposer" guess several tokens ahead,
then verifies all of them in a single forward pass of the real (target)
model - if the target model agrees with a guessed token, you get it for
free; the first disagreement is the fallback point. Because verification is
one batched forward pass regardless of how many tokens were proposed, you
can get multiple tokens per target-model step when the proposer guesses
well, at zero cost in output distribution (the outputs are mathematically
identical to standard decoding, provided the acceptance rule is used
correctly).

Three proposer strategies compared here, all configured through
`--speculative-config '{"method": ..., ...}'`:

- **Draft-model speculation** (`"method": "draft_model"`): a small, fast
  model (`Llama-3.2-1B-Instruct`) proposes tokens for the target model
  (`Llama-3.1-8B-Instruct`) to verify. Works well when the draft model's
  outputs correlate well with the target's.
- **EAGLE-3** (`"method": "eagle3"`): instead of a separate LM, a tiny
  one-layer head (`RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3`) reads
  the target model's own hidden states (from low, middle and high layers)
  and predicts the next few tokens. Much cheaper than a draft LM and usually
  the best acceptance rate - this is what production deployments use today.
- **N-gram / prompt-lookup speculation**: no draft model at all - propose
  tokens by looking for repeated n-grams earlier in the prompt/generation
  so far. Free to run, and surprisingly effective for tasks with a lot of
  copying/repetition (code editing, summarization, structured extraction).

## Why concurrency matters here

Speculative decoding helps most when the target model is **not** already
compute-saturated - i.e. low concurrency, latency-sensitive serving. At high
concurrency, the GPU is already busy doing useful work for many sequences,
so the extra verification forward passes (and a rejected guess is pure
waste) can make speculative decoding neutral or even a net throughput
*loss*. This lab measures both regimes deliberately.

## What this lab does

Four variants (no speculation baseline, draft-model, EAGLE-3, n-gram), each
benchmarked at:

- **Low concurrency** (`max-concurrency=2`): the latency-sensitive regime
  where speculative decoding should win.
- **High concurrency** (`max-concurrency=64`): the throughput regime where
  the benefit should shrink or reverse.

## Run it

```bash
python labs/lab09_speculative_decoding/lab.py
python labs/lab09_speculative_decoding/lab.py --quick
```

## What to look for

- `no_spec_low` vs `draft_model_spec_low` / `ngram_spec_low`:
  `mean_tpot_ms` should drop (fewer wall-clock ms per output token) when
  speculation is accepted often.
- The same comparison at `_high`: the gap should be much smaller, or even
  favor `no_spec`.
- vLLM logs speculative acceptance metrics (mean acceptance length,
  per-position acceptance rate) in the server log and on `/metrics`
  (`vllm:spec_decode_*`) - compare EAGLE-3 vs the draft model there.
- Beyond this lab, upstream also has MTP heads (for models trained with
  multi-token prediction such as DeepSeek-V3), suffix decoding, and
  *dynamic* speculation (`num_speculative_tokens_per_batch_size`) that
  shrinks `k` as concurrency grows.
- `ngram_spec` works with zero extra model-loading cost - worth comparing
  its win rate against the draft-model approach on this specific synthetic
  (random-token) workload, which has little natural repetition, so
  n-gram speculation is a harder case here than on a real repetitive task.
