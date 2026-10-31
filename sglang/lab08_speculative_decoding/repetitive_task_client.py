#!/usr/bin/env python3
"""
Speculative decoding shines brightest when the model's output has a lot of
local redundancy -- e.g. reformatting/copying input text, structured output,
or repeated patterns -- because that's exactly what an n-gram cache built
from recent tokens can predict correctly. This script sends a handful of
deliberately repetitive-output prompts to a running SGLang server and
reports wall-clock tokens/sec, so you can see the "best case" speedup number
in addition to bench_serving's "general workload" number.

Usage:
    python repetitive_task_client.py --base-url http://127.0.0.1:30000 --model Qwen/Qwen2.5-7B-Instruct --out results/.../repetitive.json
"""
import argparse
import json
import time

import requests

PROMPTS = [
    "Write the numbers from 1 to 150, one per line, in the exact format 'Item 001: value'.",
    "Repeat the following paragraph exactly, word for word, three times in a row, separated by '---':\n\n"
    "The quick brown fox jumps over the lazy dog near the riverbank every single morning before sunrise.",
    "Generate a markdown table with columns Name, ID, Status for 40 rows named row_1 through row_40, "
    "all with Status='OK' and ID equal to the row number.",
    "Correct the grammar in the following text but otherwise copy it verbatim, sentence by sentence: "
    "'Yesterday I goes to the market and buys three apple, two banana, and some bread for breakfast tomorrow.' "
    "Repeat this exercise with slight rewording five times.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    total_tokens = 0
    t_start = time.time()
    per_request = []
    for i, prompt in enumerate(PROMPTS):
        t0 = time.time()
        resp = requests.post(
            f"{args.base_url}/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        dt = time.time() - t0
        usage = data.get("usage", {})
        n_tokens = usage.get("completion_tokens", args.max_tokens)
        total_tokens += n_tokens
        per_request.append({"prompt_idx": i, "latency_s": dt, "output_tokens": n_tokens, "tok_s": n_tokens / dt})
        print(f"  [{i+1}/{len(PROMPTS)}] {dt:.2f}s, {n_tokens} tokens ({n_tokens/dt:.1f} tok/s)")

    total_time = time.time() - t_start
    result = {
        "model": args.model,
        "num_prompts": len(PROMPTS),
        "total_wall_time_s": total_time,
        "total_output_tokens": total_tokens,
        "output_throughput": total_tokens / total_time,
        "per_request": per_request,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nOverall: {result['output_throughput']:.1f} tok/s on repetitive-output tasks. Wrote {args.out}")


if __name__ == "__main__":
    main()
