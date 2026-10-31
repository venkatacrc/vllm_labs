# Lab 08 - Speculative Decoding (N-gram)

**Goal:** understand how speculative decoding turns "1 forward pass = 1
token" into "1 forward pass = several tokens (when the guesses are right)",
using a method that needs zero extra models to demonstrate.

## Background

Decoding is memory-bandwidth bound: generating one token means reading the
*entire* model's weights (and KV cache) from HBM for a batch that's often
small relative to the GPU's compute capacity. Speculative decoding exploits
this by having a cheap "drafter" propose several candidate future tokens,
then running **one** forward pass of the real target model to verify all of
them simultaneously (a form of prefill-shaped, parallel computation). Since
that verification pass costs about the same as generating a single token
normally, every additional accepted draft token is close to free.

Common drafters:

- **EAGLE / EAGLE3**: a small, separately-trained neural network predicts
  draft tokens using the target model's own hidden states. Highest quality
  speedups, but requires a compatible pretrained EAGLE checkpoint for your
  exact target model (not always available, and out of scope for a
  zero-extra-downloads lab).
- **STANDALONE**: use a smaller full LLM as the drafter (e.g. a 1B model
  drafting for a 70B target). Needs a second model server-side.
- **NGRAM** (what we use here): no model at all. SGLang keeps a cache of
  recently-seen token n-grams (from the prompt and from what's already been
  generated) and proposes continuations by pattern-matching against it.
  Zero setup cost, but only helps when output has local redundancy: copying
  or lightly editing input text, structured/repetitive formats, code with
  repeated tokens, etc. For open-ended creative generation the hit rate (and
  thus the speedup) is much smaller.

## What this lab does

Two server configurations against `$SMALL_MODEL`:

1. `no-spec` -- plain baseline.
2. `ngram-spec` -- `--speculative-algorithm NGRAM --speculative-num-draft-tokens 16 --speculative-ngram-max-bfs-breadth 10`.

For each, two benchmarks:

- **general**: `bench_serving` with generic random-content prompts (natural
  language has *some* redundancy, so expect a modest win).
- **repetitive**: `repetitive_task_client.py` sends prompts explicitly
  designed to produce highly redundant output (numbered lists, verbatim
  repetition, tables) -- the best case for n-gram speculation.

## Run it

```bash
cd vllm_labs/sglang/lab08_speculative_decoding
./run_lab.sh
```

Note: NGRAM speculative decoding is CUDA-only and automatically disables
the overlap scheduler and mixed chunked prefill internally (SGLang handles
this for you; you don't need extra flags) -- so part of what you measure
here also reflects that trade-off, not purely the speculative-decoding win.

## Expected takeaways

- `ngram-spec-general` vs `no-spec-general`: expect a real but modest
  `output_throughput_tok_s` improvement.
- `ngram-spec-repetitive` vs `no-spec-repetitive`: expect a much larger
  improvement -- this is the workload shape where speculative decoding
  earns its keep in production (e.g. code-editing assistants, "reformat
  this data" tools, autocomplete).
- Check the server log (`server_ngram.log`) for any lines mentioning
  acceptance rate / accept length if your SGLang version logs them --
  higher accepted-tokens-per-step directly explains the throughput gain.
- If you have access to a compatible EAGLE checkpoint for your target model
  in a real deployment, prefer it over NGRAM: EAGLE's learned drafter
  generalizes far better across content types than n-gram matching.

Next: `lab09_structured_output` looks at constrained decoding (JSON
schemas, regex) and its throughput cost.
