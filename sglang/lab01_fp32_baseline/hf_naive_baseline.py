#!/usr/bin/env python3
"""
The "before" picture: serve the base FP32 model the naive way, with plain
HuggingFace `transformers`, one request at a time, no batching, no KV-cache
reuse across requests, no scheduler.

This is intentionally the dumbest possible way to serve an LLM: a Python
for-loop calling `model.generate()`. It exists so that lab01's SGLang runs
have something honest to compare against -- every later lab's speedup number
is relative to *this*.

Usage:
    python hf_naive_baseline.py --model Qwen/Qwen2.5-7B-Instruct \
        --num-prompts 20 --max-new-tokens 256 --out results/lab01/hf_naive.json
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain the difference between tensor parallelism and pipeline parallelism.",
    "Write a short poem about the number zero.",
    "What are three ways to reduce the memory footprint of an LLM during inference?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Give me a recipe for a simple tomato pasta sauce.",
    "What is the time complexity of binary search, and why?",
    "Describe how a transformer's self-attention mechanism works.",
    "Translate 'Good morning, how are you?' into French, Spanish, and Japanese.",
    "List five debugging strategies for a program that crashes intermittently.",
    "What is the difference between latency and throughput in serving systems?",
]


def build_prompts(n):
    return [PROMPTS[i % len(PROMPTS)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model} in {args.dtype} onto a single GPU (this is the naive path: "
          f"no tensor parallel, no continuous batching, no paged KV cache)...")
    t_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cuda:0"
    )
    model.eval()
    t_load = time.time() - t_load_start
    peak_mem_load_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Model loaded in {t_load:.1f}s, using {peak_mem_load_gb:.1f} GB GPU memory.")

    prompts = build_prompts(args.num_prompts)
    per_request = []

    total_output_tokens = 0
    t_bench_start = time.time()
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda:0")
        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        dt = time.time() - t0
        n_new = output.shape[1] - input_ids.shape[1]
        total_output_tokens += n_new
        per_request.append({"request": i, "latency_s": dt, "output_tokens": n_new})
        print(f"  [{i+1}/{len(prompts)}] {dt:.2f}s, {n_new} tokens ({n_new/dt:.1f} tok/s, this request only)")

    total_time = time.time() - t_bench_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    result = {
        "backend": "transformers-naive-sequential",
        "model": args.model,
        "dtype": args.dtype,
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "load_time_s": t_load,
        "total_wall_time_s": total_time,
        "total_output_tokens": total_output_tokens,
        "output_throughput": total_output_tokens / total_time,
        "mean_latency_s": total_time / len(prompts),
        "peak_gpu_memory_gb": peak_mem_gb,
        "per_request": per_request,
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n===== Naive HF baseline summary =====")
    print(f"  model               : {args.model} ({args.dtype})")
    print(f"  requests            : {len(prompts)} (sequential, batch size 1)")
    print(f"  total wall time     : {total_time:.1f}s")
    print(f"  output throughput   : {result['output_throughput']:.1f} tok/s")
    print(f"  mean latency/request: {result['mean_latency_s']:.2f}s")
    print(f"  peak GPU memory     : {peak_mem_gb:.1f} GB")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
