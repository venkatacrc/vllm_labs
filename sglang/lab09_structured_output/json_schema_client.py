#!/usr/bin/env python3
"""
Compares unconstrained vs grammar-constrained (JSON Schema) generation on the
same extraction task: given a short bio, extract {name, age, profession}.

Unconstrained: we ask nicely in the prompt for JSON and hope the model
complies. Constrained: we pass `response_format={"type": "json_schema", ...}`
which SGLang enforces token-by-token via its grammar backend (xgrammar by
default), guaranteeing schema-valid output.

Usage:
    python json_schema_client.py --base-url http://127.0.0.1:30000 \
        --model Qwen/Qwen2.5-7B-Instruct --constrained {true,false} \
        --out results/.../unconstrained.json
"""
import argparse
import json
import time

import requests

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "profession": {"type": "string"},
    },
    "required": ["name", "age", "profession"],
}

BIOS = [
    "Dr. Amara Osei, 41, has spent the last decade leading clinical trials for a biotech startup in Boston.",
    "Kenji Watanabe just turned 29 and recently switched careers from investment banking to become a pastry chef.",
    "At 67, retired airline pilot Susan Whitfield now restores vintage motorcycles in her spare time.",
    "Marco, a 23-year-old street artist from Naples, was commissioned to paint a mural for the city hall.",
    "Priya Nair, 55, is a civil engineer who has designed bridges across three continents.",
    "The band's newest member, 19-year-old drummer Theo Larsson, joined right after finishing high school.",
    "Fatima Al-Sayed, aged 34, runs a nonprofit that provides microloans to women entrepreneurs.",
    "Grandpa Joe, 82, still tends his own vegetable garden every morning before breakfast.",
]


def build_prompt(bio: str) -> str:
    return (
        "Extract the person's name, age, and profession from the bio below. "
        "Respond with ONLY a JSON object with keys 'name', 'age', 'profession'.\n\n"
        f"Bio: {bio}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--constrained", choices=["true", "false"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    constrained = args.constrained == "true"

    results = []
    t_start = time.time()
    for bio in BIOS:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": build_prompt(bio)}],
            "max_tokens": 128,
            "temperature": 0,
        }
        if constrained:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": SCHEMA, "strict": True},
            }
        t0 = time.time()
        resp = requests.post(f"{args.base_url}/v1/chat/completions", json=payload, timeout=120)
        dt = time.time() - t0
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        valid = False
        parsed = None
        try:
            parsed = json.loads(content)
            valid = all(k in parsed for k in ("name", "age", "profession"))
        except json.JSONDecodeError:
            valid = False

        results.append({"bio": bio, "latency_s": dt, "raw_output": content, "valid_json": valid, "parsed": parsed})
        status = "OK " if valid else "FAIL"
        print(f"  [{status}] {dt:.2f}s  -> {content[:80]!r}")

    total_time = time.time() - t_start
    n_valid = sum(1 for r in results if r["valid_json"])
    summary = {
        "constrained": constrained,
        "model": args.model,
        "num_requests": len(BIOS),
        "num_valid_json": n_valid,
        "valid_json_rate": n_valid / len(BIOS),
        "total_wall_time_s": total_time,
        "mean_latency_s": total_time / len(BIOS),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'CONSTRAINED' if constrained else 'UNCONSTRAINED'}: "
          f"{n_valid}/{len(BIOS)} valid JSON outputs, mean latency {summary['mean_latency_s']:.2f}s")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
