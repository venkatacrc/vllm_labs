"""A deliberately 'naive' HF Transformers static-batch generation loop.

Used only by lab02 as the point of comparison against vLLM's continuous
batching + paged KV cache: `model.generate()` on a fixed-size batch must
wait for every sequence in that batch to finish (hit EOS or max_new_tokens)
before the call returns and the next batch can start, wasting compute on
sequences that finished early ("bubbles"). vLLM instead evicts finished
sequences and admits new ones every scheduler step.

This is intentionally simple/unoptimized (that's the point) - it is not
meant to be a fair "best effort HF serving" baseline, just the naive
approach most people write first.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_TOPICS = [
    "the history of the Roman Empire", "quantum computing", "the water cycle",
    "renewable energy sources", "the French Revolution", "black holes",
    "machine learning", "the human immune system", "ancient Egyptian pyramids",
    "climate change", "the global stock market", "photosynthesis", "the Silk Road",
    "artificial intelligence ethics", "the Great Barrier Reef", "volcanoes",
    "the invention of the printing press", "space exploration", "the Cold War",
    "coral reef ecosystems", "the history of the internet", "electric vehicles",
    "the Renaissance", "genetics and DNA",
]


def default_prompts(n: int, seed: int = 0) -> list[str]:
    """Prompts with deliberately bimodal implied output length.

    Short factual questions finish in a handful of tokens; long "write a
    detailed essay" requests run for hundreds. Mixing them in the same
    static batch is exactly what creates the classic "batch straggler"
    problem that continuous batching solves.
    """
    import random

    prompts = []
    for i in range(n):
        topic = _TOPICS[i % len(_TOPICS)]
        if i % 2 == 0:
            prompts.append(f"In one short sentence, what is {topic}?")
        else:
            prompts.append(
                f"Write a detailed, multi-paragraph explanation of {topic}, "
                "covering its origins, key facts, and significance."
            )
    random.Random(seed).shuffle(prompts)
    return prompts


def run_naive_static_batches(
    model_id: str,
    prompts: list[str],
    *,
    max_new_tokens: int = 128,
    batch_size: int = 16,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    torch_dtype = getattr(torch, dtype)

    print(f"[hf_baseline] loading {model_id} ({dtype}) on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    total_output_tokens = 0
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    print(f"[hf_baseline] running {len(prompts)} prompts in {n_batches} static batch(es) of {batch_size}")

    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        total_output_tokens += int((new_tokens != tokenizer.pad_token_id).sum().item())
    elapsed = time.time() - t0

    del model
    gc.collect()
    torch.cuda.empty_cache()

    n = len(prompts)
    return {
        "label": "hf_naive_static_batch",
        "completed": n,
        "duration": elapsed,
        "request_throughput": n / elapsed if elapsed else None,
        "output_throughput": total_output_tokens / elapsed if elapsed else None,
        "total_token_throughput": total_output_tokens / elapsed if elapsed else None,
        "mean_ttft_ms": None,  # not meaningful: this loop has no streaming/TTFT concept
        "p99_ttft_ms": None,
        "mean_tpot_ms": (elapsed * 1000 * batch_size / total_output_tokens) if total_output_tokens else None,
    }


def _main() -> None:
    """CLI entrypoint so lab02 can run this in an isolated subprocess
    (own CUDA_VISIBLE_DEVICES, own process lifetime) rather than in the
    main driver's long-lived Python process.
    """
    import argparse
    import json

    p = argparse.ArgumentParser(description="Naive HF static-batch generation baseline.")
    p.add_argument("--model", required=True)
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    prompts = default_prompts(args.num_prompts, seed=args.seed)
    result = run_naive_static_batches(
        args.model,
        prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
    )
    Path(args.output).write_text(json.dumps(result))
    print(f"[hf_baseline] wrote {args.output}")


if __name__ == "__main__":
    _main()
