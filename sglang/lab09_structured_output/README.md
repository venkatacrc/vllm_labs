# Lab 09 - Structured Output / Grammar-Guided Decoding

**Goal:** measure the reliability gain and latency cost of enforcing a JSON
Schema on model output via SGLang's grammar backend, versus just asking
nicely in the prompt.

## Background

LLMs are surprisingly good at producing JSON when asked, but "surprisingly
good" isn't "guaranteed" -- extra prose before/after the JSON, trailing
commas, unescaped quotes, or truncated output at `max_tokens` are all common
failure modes that show up more often than you'd like in production at
scale.

SGLang's grammar backends (default: **xgrammar**, also supports
**outlines** and **llguidance**) solve this differently: at every decoding
step, they compute a token mask that only allows tokens which keep the
output on a path consistent with your schema (compiled to a pushdown
automaton / FSM), then apply that mask to the logits before sampling. The
model literally cannot emit a token that would break the schema. This
costs some extra per-step CPU/GPU work to compute and apply the mask, which
we measure here.

## What this lab does

Launches `$SMALL_MODEL` once with `--grammar-backend xgrammar`, then:

1. **Unconstrained**: `json_schema_client.py --constrained false` asks the
   model in plain English to "respond with only a JSON object" and checks
   how often the raw output actually parses as valid JSON with the required
   fields.
2. **Constrained**: same task, same prompts, but with
   `response_format={"type": "json_schema", ...}` set on the request,
   letting SGLang enforce the schema.
3. An optional throughput-under-load comparison via `bench_serving`
   (skipped automatically if your SGLang version's `bench_serving` doesn't
   expose a `--json-schema` flag -- check the console output).

## Run it

```bash
cd vllm_labs/sglang/lab09_structured_output
./run_lab.sh
```

## Expected takeaways

- `constrained` should show `num_valid_json / num_requests` at or very
  close to 100%. `unconstrained` will likely show at least one failure
  (extra commentary text, a stray field, wrong type for `age`, etc.) even
  though the model is generally well-behaved -- this gap is exactly why
  production extraction/tool-calling pipelines should use constrained
  decoding instead of prompt-and-pray plus a JSON parser wrapped in
  retry logic.
- `mean_latency_s` should be modestly higher for the constrained case (extra
  grammar-mask computation per step), but usually far cheaper than the
  latency of a failed parse + retry round-trip would have been.
- If the load-test rows are present in `results/summary.csv`, compare
  `median_tpot_ms` between `plain-decoding-load` and
  `constrained-decoding-load` to see the per-token overhead in a
  concurrent, throughput-oriented setting rather than one-request-at-a-time.

Next: `lab10_moe_expert_parallel` moves from dense models to
Mixture-of-Experts, and the parallelism strategies specific to them.
