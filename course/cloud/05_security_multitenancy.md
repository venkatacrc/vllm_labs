---
title: Security, privacy and multi-tenancy
track: Cloud & Global
order: 5
---

# Security, privacy and multi-tenancy

## Threats to an LLM service

- **Abuse**: credential stuffing, scraping, denial of wallet (expensive long
  prompts), DDoS. → WAF/DDoS at the edge, auth, token-based rate limits.
- **Tenant isolation failures**: one tenant seeing another's data. Shared
  prefix caches are safe in content terms (a cache hit requires the exact
  same tokens) but can leak *timing* (a fast TTFT reveals that someone sent
  that prefix). Mitigations: per-tenant cache salting/namespaces for
  sensitive tenants, dedicated pools for regulated customers.
- **Prompt injection / tool abuse**: untrusted content steering an agent to
  call tools. → least-privilege tools, output validation, human approval for
  risky actions. (Structured outputs help parsing, not trust.)
- **Supply chain**: malicious model files (pickle), compromised images or
  Python packages. → safetensors only, signed and scanned images, pinned
  digests, private registries and model mirrors.
- **Secrets**: HF tokens, API keys in env vars or logs. → secret managers,
  Kubernetes Secrets with RBAC, never in images or command lines.

## Data handling

- Encrypt in transit (TLS at the edge, mTLS inside the mesh) and at rest
  (model registry, logs, KV offload tiers on disk).
- Minimize retention: don't log prompts/responses by default; if you must
  (debugging, abuse review), redact PII and set short retention.
- **Data residency**: keep prompts, logs and KV tiers in the user's region;
  route by geography at DNS/gateway level.
- Offloaded KV on shared filesystems is user data - apply the same access
  controls and encryption as any other data store.

## Network and cluster hygiene

- Model servers and the EPP should not be reachable from the internet; only
  the gateway is. Use NetworkPolicies so only the router can reach model
  pods, and only the EPP can scrape them.
- RBAC: the EPP only needs to watch pods/InferencePools in its namespace.
- Run containers as non-root where possible; restrict privileged mounts
  (RDMA devices, host paths) to the pods that need them.

## Flashcards

Q: Can a shared prefix cache leak data across tenants?
A: Not content directly (a hit needs identical tokens), but timing can reveal that a prefix was recently sent; sensitive tenants can get salted cache keys or dedicated pools.

Q: What is "denial of wallet"?
A: Abuse that drives up cost (e.g. very long prompts or outputs) rather than availability; mitigated with token-based quotas and limits.

Q: Why prefer safetensors over pickle checkpoints?
A: Pickle files can execute arbitrary code when loaded; safetensors is a pure tensor format.

Q: Which component should be internet-facing in an llm-d deployment?
A: Only the gateway/edge; the router (Envoy/EPP) and model servers stay internal, protected by network policies.

Q: Where do prompts end up that need data-residency controls?
A: Model-server memory, KV caches and offload tiers, logs, traces, and any request analytics pipeline.
