---
title: Regions, DNS and global traffic
track: Cloud & Global
order: 1
---

# Regions, DNS and global traffic

## Building blocks

- **Region**: a geographic area (e.g. `us-east`, `eu-west`) with several
  **availability zones (AZs)**: independent datacenters with separate power
  and networking, a few ms apart. Spread replicas across AZs to survive a
  datacenter failure; across regions to survive a regional outage and to be
  close to users.
- **Latency floor**: light in fiber covers ~200 km per ms, so a round trip
  New York ↔ Frankfurt (~6,200 km) is ≥ 60 ms, and real paths are ~80-90 ms.
  For a chatbot that's noise compared to prefill; for voice agents and tool
  loops with dozens of round trips it adds up. Put inference near users,
  or at least near the systems that call it.
- **GPU geography**: GPU capacity is unevenly distributed across regions and
  clouds, so "near the user" competes with "where the GPUs are". Many
  deployments serve from a few GPU-rich regions and route globally.

## Getting users to a region

1. **DNS**: the client resolves `api.example.com`. **GeoDNS / latency-based
   DNS** returns the IP of the nearest healthy region; health checks remove
   failed regions. Caveat: DNS answers are cached for the record TTL (keep
   it short, 30-60 s, for failover) and resolvers may be far from users.
2. **Anycast**: the same IP is announced via BGP from many locations; the
   internet routes each client to the topologically nearest one. Global
   load balancers and CDNs use anycast front ends that terminate TLS at the
   edge and forward over the provider's backbone.
3. **Edge / CDN**: points of presence close to users terminate TCP+TLS
   (saving round trips), apply WAF/DDoS protection, and keep long-lived
   connections to origin regions. LLM responses aren't cacheable like
   images, but edge termination still cuts the connection-setup latency and
   protects the origin. Streaming (SSE) must pass through without buffering.

```
user ─► DNS (GeoDNS) ─► anycast edge PoP (TLS, WAF, DDoS) ─► provider backbone
     ─► regional L7 load balancer / API gateway ─► inference cluster (llm-d router) ─► GPUs
```

## Routing policy across regions

- **Latency-based**: nearest region by measured RTT.
- **Weighted**: split traffic (e.g. 90/10 for a canary region).
- **Failover**: primary/secondary with health checks.
- **Capacity-aware**: send overflow to regions with spare GPUs (needs a
  global view of queue depth or saturation - the same signals the llm-d
  EPP uses inside a cluster, aggregated up).
- **Data residency**: EU users' prompts must stay in EU regions (GDPR,
  customer contracts), which overrides latency.

## Multi-region active-active

Each region runs a full stack (gateway, router, model servers) and serves
live traffic. Requirements: identical model versions everywhere (artifact
registry replication), configuration as code, per-region capacity for
failover (N+1: if one region of N fails, the rest absorb its load), and
state that is either regional (conversation KV caches are regional and
rebuildable) or replicated (user accounts, quotas).

## Flashcards

Q: What is an availability zone?
A: An independent datacenter (separate power, cooling, network) within a region, a few ms from the others, used to survive single-datacenter failures.

Q: Minimum round-trip time New York ↔ Frankfurt?
A: About 60 ms at the speed of light in fiber (~200 km/ms for ~6,200 km each way); real paths are ~80-90 ms.

Q: How does GeoDNS route users?
A: It returns different IPs depending on the resolver's location (or measured latency) and region health, sending users to the nearest healthy region.

Q: What is anycast?
A: Announcing the same IP from many locations via BGP so each client reaches the topologically nearest one; used by global load balancers and CDNs.

Q: Why use an edge PoP for LLM APIs if responses aren't cacheable?
A: Terminating TCP/TLS near the user saves round trips, the edge filters attacks (WAF, DDoS), and traffic rides the provider's backbone to the GPU region.

Q: Why keep DNS TTLs short for failover?
A: Clients and resolvers cache answers for the TTL; a long TTL keeps sending users to a failed region.

Q: What does N+1 capacity mean for multi-region?
A: Provision enough spare capacity that if any one region fails, the remaining regions can absorb its traffic.

Q: What can override latency-based routing?
A: Data residency/sovereignty rules (e.g. EU prompts must be processed in EU regions) and capacity constraints.
