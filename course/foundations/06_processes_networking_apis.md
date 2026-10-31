---
title: Processes, networking and the serving API
track: Foundations
order: 6
---

# Processes, networking and the serving API

## Linux processes for servers

- A server is a **process** listening on a **TCP port** (vLLM 8000, SGLang
  30000, Envoy 8081). `ss -ltnp` shows who listens where; a port can only be
  bound by one process per address.
- Engines spawn **child processes** (engine core, TP workers, tokenizer).
  Killing only the parent can leave orphans holding GPU memory - the labs
  start servers in their own **process group** (`setsid`) and kill the group
  (`SIGTERM`, then `SIGKILL`).
- **Signals**: `SIGTERM` asks for graceful shutdown (finish in-flight
  requests: vLLM `--shutdown-timeout`), `SIGKILL` cannot be caught,
  `SIGINT` is Ctrl-C.
- **Shared memory** (`/dev/shm`): NCCL and multi-process engines use it; in
  containers it must be enlarged (`--shm-size` or `--ipc host`).
- **Health endpoints**: `/health` (alive), `/v1/models` (model loaded),
  `/metrics` (Prometheus text format).

## HTTP, streaming and gRPC

- **HTTP/1.1** keeps a TCP connection alive for multiple requests;
  **HTTP/2** multiplexes many streams over one connection (Envoy uses it to
  talk gRPC to the EPP).
- **Server-Sent Events (SSE)**: the response is `text/event-stream`; the
  server writes `data: {...}\n\n` chunks as tokens are produced and ends
  with `data: [DONE]`. This is how TTFT becomes visible to users.
- **gRPC** = protobuf messages over HTTP/2. Envoy's `ext_proc` filter
  streams request/response headers and bodies to an external processor
  (the llm-d EPP) over gRPC and lets it mutate headers - e.g. set
  `x-gateway-destination-endpoint` to the chosen replica.

## The OpenAI-compatible API

Both engines expose:

```http
POST /v1/completions        {"model", "prompt", "max_tokens", "temperature", "stream"}
POST /v1/chat/completions   {"model", "messages":[{"role","content"}], "tools", "response_format", "stream"}
GET  /v1/models
```

The server applies the model's **chat template** (turns `messages` into one
prompt string with special tokens), tokenizes, generates, detokenizes
incrementally, and parses tool calls/reasoning if configured
(`--tool-call-parser`, `--reasoning-parser`). Engine-specific extras ride in
the same JSON (`ignore_eos`, `kv_transfer_params`, `priority`).

## Timeouts and keep-alive

Every hop (client → CDN → gateway → Envoy → model server) has its own
connection and idle timeouts. Long generations need long timeouts at every
hop; mismatched keep-alive between a proxy and a server causes reset
connections (the llm-d P/D guide sets `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120` to
exceed the sidecar's 90 s idle timeout).

## Flashcards

Q: Why do the labs kill the whole process group of a server?
A: Engines spawn child processes (workers, engine core) that hold GPU memory; killing only the parent can leave orphans, so they start with setsid and signal the group.

Q: What is SSE and why does LLM serving use it?
A: Server-Sent Events: an HTTP response that streams `data:` chunks; it lets clients see tokens as they are generated (so TTFT matters to users).

Q: What does Envoy's ext_proc filter do in llm-d?
A: It streams each request's headers/body to the EPP over gRPC; the EPP replies with a header naming the destination replica, which Envoy's ORIGINAL_DST cluster forwards to.

Q: What does a chat template do?
A: Converts the messages list into the single prompt string (with role markers and special tokens) the model was trained on.

Q: Why must containers running NCCL get a large /dev/shm?
A: NCCL and multi-process engines use shared memory for intra-node communication; the Docker default (64 MB) is too small.

Q: SIGTERM vs SIGKILL?
A: SIGTERM requests graceful shutdown (drain in-flight requests); SIGKILL terminates immediately and cannot be handled.

Q: Why can mismatched keep-alive timeouts break proxies?
A: If the server closes an idle connection before the proxy's idle timeout, the proxy may reuse a closed connection and get a TCP reset.
