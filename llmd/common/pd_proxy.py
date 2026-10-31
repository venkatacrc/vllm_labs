#!/usr/bin/env python
"""Minimal prefill/decode proxy for vLLM's NixlConnector.

This is what the llm-d router + decode-side routing sidecar do for you in a
real deployment, stripped to the protocol so you can see it:

  1. Send the request to a *prefill* server with max_tokens=1 and
     kv_transfer_params={"do_remote_decode": true}. The prefill server
     computes the prompt's KV cache, keeps the blocks pinned for a lease,
     and returns kv_transfer_params describing where they live
     (engine id, block ids, host, port).
  2. Send the original request to a *decode* server with those
     kv_transfer_params attached. The decode server pulls the KV blocks
     over NIXL (NVLink/RDMA via UCX), skips prefill, and streams tokens.

Prefill and decode servers are picked round-robin (llm-d instead picks the
prefill pod by prefix affinity + queued prompt tokens and the decode pod by
active requests).

Usage:
  python pd_proxy.py --port 8081 --prefill http://127.0.0.1:8100 \\
      --decode http://127.0.0.1:8200 [--decode http://127.0.0.1:8201]
"""
from __future__ import annotations

import argparse
import itertools
import json

import aiohttp
from aiohttp import web


def build_app(prefill: list[str], decode: list[str]) -> web.Application:
    p_cycle = itertools.cycle(prefill)
    d_cycle = itertools.cycle(decode)

    async def on_startup(app: web.Application) -> None:
        app["http"] = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600), connector=aiohttp.TCPConnector(limit=0)
        )

    async def on_cleanup(app: web.Application) -> None:
        await app["http"].close()

    async def handle(request: web.Request) -> web.StreamResponse:
        http: aiohttp.ClientSession = request.app["http"]
        path = request.path
        req = await request.json()

        prefill_req = dict(req)
        prefill_req["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        prefill_req["stream"] = False
        prefill_req["max_tokens"] = 1
        if "max_completion_tokens" in prefill_req:
            prefill_req["max_completion_tokens"] = 1
        prefill_req.pop("stream_options", None)

        async with http.post(f"{next(p_cycle)}{path}", json=prefill_req) as r:
            if r.status != 200:
                return web.Response(status=r.status, body=await r.read())
            prefill_resp = await r.json()

        decode_req = dict(req)
        decode_req["kv_transfer_params"] = prefill_resp.get("kv_transfer_params")

        out = web.StreamResponse(status=200)
        out.content_type = "text/event-stream" if req.get("stream") else "application/json"
        await out.prepare(request)
        async with http.post(f"{next(d_cycle)}{path}", json=decode_req) as r:
            async for chunk in r.content.iter_any():
                await out.write(chunk)
        await out.write_eof()
        return out

    async def models(request: web.Request) -> web.Response:
        http: aiohttp.ClientSession = request.app["http"]
        async with http.get(f"{decode[0]}/v1/models") as r:
            return web.json_response(json.loads(await r.text()), status=r.status)

    async def health(_: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/completions", handle)
    app.router.add_post("/v1/chat/completions", handle)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--prefill", action="append", required=True)
    p.add_argument("--decode", action="append", required=True)
    args = p.parse_args()
    web.run_app(build_app(args.prefill, args.decode), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
