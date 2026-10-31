#!/usr/bin/env python
"""Workload generator for the llm-d labs.

`vllm bench serve` is great for single-server benchmarks, but the routing
labs need things it does not express directly: many tenants with their own
concurrency and HTTP headers, conversations that come back after their
prefix may have been evicted, and per-replica accounting (which replica got
which request, and how much of its prefix cache was hit). This script does
exactly that against any OpenAI-compatible endpoint (Envoy, a vLLM replica,
the P/D proxy) and prints/saves the same latency metrics as vllm bench.

Workloads
  shared-prefix  G groups, each with a long shared "system prompt" and many
                 distinct short questions (multi-tenant chat / RAG).
  multi-turn     S sessions, each a long document followed by T turns; every
                 turn re-sends the whole history (agentic / chat traffic).
  random         independent random prompts (no reuse at all).

Tenants (--tenant, repeatable) run concurrently with their own concurrency,
request count and optional llm-d flow-control headers:
  --tenant name=quiet,concurrency=2,requests=40
  --tenant name=noisy,concurrency=64,requests=600,objective=best-effort-traffic

Example
  python loadgen.py --base-url http://127.0.0.1:8081 --model Qwen/Qwen3-32B \\
      --workload shared-prefix --groups 32 --prefix-tokens 4000 \\
      --tenant name=all,concurrency=64,requests=800 \\
      --replica-metrics http://127.0.0.1:8000,http://127.0.0.1:8001 \\
      --lab lab01_router_basics --config epp-optimized-baseline
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

WORDS = (
    "the of and to in is for on with as by at from that this be are was it an or "
    "model token cache prefix router decode prefill memory bandwidth kernel tensor "
    "latency throughput request server replica queue batch schedule gpu cluster node "
    "region service deploy scale metric policy tenant session document answer question"
).split()

# English text averages roughly 1.3 tokens per word for Llama/Qwen tokenizers.
TOKENS_PER_WORD = 1.3


def make_text(rng: random.Random, approx_tokens: int) -> str:
    n_words = max(1, int(approx_tokens / TOKENS_PER_WORD))
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


@dataclass
class Tenant:
    name: str
    concurrency: int
    requests: int
    objective: str | None = None
    results: list[dict] = field(default_factory=list)


def parse_tenant(spec: str) -> Tenant:
    kv = dict(part.split("=", 1) for part in spec.split(","))
    return Tenant(
        name=kv["name"],
        concurrency=int(kv.get("concurrency", 16)),
        requests=int(kv.get("requests", 100)),
        objective=kv.get("objective"),
    )


class Workload:
    """Produces prompts. Multi-turn sessions are stateful (history grows)."""

    def __init__(self, args: argparse.Namespace, seed: int):
        self.args = args
        self.rng = random.Random(seed)
        self.kind = args.workload
        if self.kind == "shared-prefix":
            self.prefixes = [
                f"System prompt {g}: " + make_text(self.rng, args.prefix_tokens) for g in range(args.groups)
            ]
        elif self.kind == "multi-turn":
            self.sessions = [
                [f"Document {s}: " + make_text(self.rng, args.prefix_tokens)] for s in range(args.sessions)
            ]
            self.turn_cursor = 0

    def next_prompt(self) -> tuple[str, int | None]:
        a = self.args
        if self.kind == "shared-prefix":
            g = self.rng.randrange(len(self.prefixes))
            return self.prefixes[g] + "\nQuestion: " + make_text(self.rng, a.suffix_tokens), None
        if self.kind == "multi-turn":
            # Visit sessions round-robin so that by the time a session comes
            # back, many other sessions' KV has been written in between.
            s = self.turn_cursor % len(self.sessions)
            self.turn_cursor += 1
            history = self.sessions[s]
            history.append("\nUser: " + make_text(self.rng, a.suffix_tokens))
            return "".join(history), s
        return make_text(self.rng, a.prefix_tokens + a.suffix_tokens), None

    def record_answer(self, session: int | None, text: str) -> None:
        if session is not None:
            self.sessions[session].append("\nAssistant: " + text)


async def one_request(http: aiohttp.ClientSession, args, tenant: Tenant, prompt: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if tenant.objective or args.flow_control_headers:
        headers["x-llm-d-inference-fairness-id"] = tenant.name
        if tenant.objective:
            headers["x-llm-d-inference-objective"] = tenant.objective
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.output_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    ttft = None
    token_times: list[float] = []
    text_parts: list[str] = []
    out_tokens = 0
    status = 0
    try:
        async with http.post(f"{args.base_url}/v1/completions", json=body, headers=headers) as resp:
            status = resp.status
            if status != 200:
                await resp.read()
                return {"ok": False, "status": status, "tenant": tenant.name}
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    out_tokens = chunk["usage"].get("completion_tokens", out_tokens)
                for choice in chunk.get("choices", []):
                    if choice.get("text"):
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        token_times.append(now)
                        text_parts.append(choice["text"])
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return {"ok": False, "status": status, "error": str(exc), "tenant": tenant.name}
    e2e = time.perf_counter() - t0
    n = out_tokens or len(token_times)
    tpot = (e2e - ttft) / (n - 1) if ttft is not None and n > 1 else None
    return {
        "ok": True,
        "tenant": tenant.name,
        "ttft_ms": ttft * 1000 if ttft is not None else None,
        "tpot_ms": tpot * 1000 if tpot is not None else None,
        "e2e_ms": e2e * 1000,
        "output_tokens": n,
        "text": "".join(text_parts),
    }


async def run_tenant(http, args, tenant: Tenant, workload: Workload) -> None:
    sem = asyncio.Semaphore(tenant.concurrency)
    lock = asyncio.Lock()

    async def worker() -> None:
        async with sem:
            async with lock:
                prompt, session = workload.next_prompt()
            res = await one_request(http, args, tenant, prompt)
            if res.get("ok"):
                async with lock:
                    workload.record_answer(session, res.pop("text"))
            tenant.results.append(res)

    await asyncio.gather(*(worker() for _ in range(tenant.requests)))


def scrape(url: str) -> dict[str, float]:
    """Sum the few vLLM/SGLang counters we care about from a /metrics page."""
    import urllib.request

    wanted = {
        "vllm:prefix_cache_hits_total": "hits",
        "vllm:prefix_cache_queries_total": "queries",
        "vllm:request_success_total": "requests",
        "vllm:external_prefix_cache_hits_total": "ext_hits",
        "vllm:external_prefix_cache_queries_total": "ext_queries",
        "sglang:num_requests_total": "requests",
    }
    out = {v: 0.0 for v in wanted.values()}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{url}/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001 - metrics are best-effort
        return out
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in wanted:
            try:
                out[wanted[name]] += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = min(len(values) - 1, max(0, int(round(p / 100 * (len(values) - 1)))))
    return values[k]


def summarize(results: list[dict], duration: float) -> dict:
    ok = [r for r in results if r.get("ok")]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None]
    e2e = [r["e2e_ms"] for r in ok]
    out_tok = sum(r["output_tokens"] for r in ok)
    return {
        "requests": len(results),
        "failed": len(results) - len(ok),
        "duration_s": round(duration, 2),
        "req_per_s": round(len(ok) / duration, 3) if duration else None,
        "out_tok_per_s": round(out_tok / duration, 1) if duration else None,
        "ttft_p50_ms": round(statistics.median(ttft), 1) if ttft else None,
        "ttft_p99_ms": round(pct(ttft, 99), 1) if ttft else None,
        "tpot_p50_ms": round(statistics.median(tpot), 2) if tpot else None,
        "e2e_p50_ms": round(statistics.median(e2e), 1) if e2e else None,
        "e2e_p99_ms": round(pct(e2e, 99), 1) if e2e else None,
    }


CSV_FIELDS = [
    "lab", "config", "tenant", "requests", "failed", "duration_s", "req_per_s", "out_tok_per_s",
    "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "e2e_p50_ms", "e2e_p99_ms",
    "prefix_hit_rate", "max_replica_share",
]


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


async def main_async(args) -> int:
    tenants = [parse_tenant(t) for t in args.tenant] or [Tenant("all", 32, 200)]
    replicas = [u for u in (args.replica_metrics or "").split(",") if u]
    before = {u: scrape(u) for u in replicas}

    workloads = {t.name: Workload(args, seed=args.seed + i) for i, t in enumerate(tenants)}
    if args.workload != "multi-turn" and args.share_prefixes:
        shared = workloads[tenants[0].name]
        workloads = {t.name: shared for t in tenants}

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=0)
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http:
        await asyncio.gather(*(run_tenant(http, args, t, workloads[t.name]) for t in tenants))
    duration = time.perf_counter() - t0

    after = {u: scrape(u) for u in replicas}
    per_replica = {}
    for u in replicas:
        d = {k: after[u][k] - before[u].get(k, 0.0) for k in after[u]}
        per_replica[u] = d
    total_req = sum(d["requests"] for d in per_replica.values())
    total_hits = sum(d["hits"] for d in per_replica.values())
    total_q = sum(d["queries"] for d in per_replica.values())
    hit_rate = round(total_hits / total_q, 3) if total_q else None
    max_share = round(max(d["requests"] for d in per_replica.values()) / total_req, 3) if total_req else None

    all_results = [r for t in tenants for r in t.results]
    overall = summarize(all_results, duration)
    report = {
        "lab": args.lab,
        "config": args.config,
        "workload": args.workload,
        "overall": overall,
        "tenants": {t.name: summarize(t.results, duration) for t in tenants},
        "prefix_hit_rate": hit_rate,
        "max_replica_share": max_share,
        "per_replica": per_replica,
    }

    print(f"\n=== {args.lab} / {args.config} ({args.workload}) ===")
    for name, s in [("ALL", overall)] + list(report["tenants"].items()):
        print(
            f"{name:>12}: {s['requests']} req ({s['failed']} failed), {s['req_per_s']} req/s, "
            f"{s['out_tok_per_s']} out tok/s, TTFT p50/p99 {s['ttft_p50_ms']}/{s['ttft_p99_ms']} ms, "
            f"TPOT p50 {s['tpot_p50_ms']} ms"
        )
    if replicas:
        print(f"  prefix-cache hit rate across replicas: {hit_rate}; busiest replica got {max_share} of requests")
        for u, d in per_replica.items():
            print(f"    {u}: {int(d['requests'])} requests, hits/queries {int(d['hits'])}/{int(d['queries'])}")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.config}.json").write_text(json.dumps(report, indent=2))
    if args.summary_csv:
        rows = [("ALL", overall)] + (list(report["tenants"].items()) if len(tenants) > 1 else [])
        for name, s in rows:
            append_csv(Path(args.summary_csv), {
                "lab": args.lab, "config": args.config, "tenant": name, **s,
                "prefix_hit_rate": hit_rate, "max_replica_share": max_share,
            })
    return 1 if overall["failed"] == overall["requests"] else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--workload", choices=["shared-prefix", "multi-turn", "random"], default="shared-prefix")
    p.add_argument("--groups", type=int, default=32, help="shared-prefix: number of distinct prefixes")
    p.add_argument("--sessions", type=int, default=64, help="multi-turn: number of conversations")
    p.add_argument("--prefix-tokens", type=int, default=4000)
    p.add_argument("--suffix-tokens", type=int, default=100)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--tenant", action="append", default=[], help="name=..,concurrency=..,requests=..[,objective=..]")
    p.add_argument("--share-prefixes", action="store_true", help="all tenants draw from the same prefix pool")
    p.add_argument("--flow-control-headers", action="store_true", help="always send the fairness-id header")
    p.add_argument("--replica-metrics", default="", help="comma list of replica base URLs to scrape /metrics")
    p.add_argument("--request-timeout", type=float, default=1800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lab", default="adhoc")
    p.add_argument("--config", default="run")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--summary-csv", default=None)
    args = p.parse_args()
    args.base_url = args.base_url.rstrip("/")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
