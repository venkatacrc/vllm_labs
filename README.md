# LLM Serving Labs

A hands-on course that takes you from a naive FP32 model in a Python loop to
a globally deployed, autoscaled, multi-tenant LLM service, built for a single
8x NVIDIA B200 node.

**Read it as an app:** [venkatacrc.github.io/vllm_labs](https://venkatacrc.github.io/vllm_labs/)
(installable on Android/Pixel, works offline) or build the Mac app below.
Every concept module and lab README is a lesson, with flash cards to learn
and test yourself.

## What's inside

| Track | Where | What you do |
|---|---|---|
| Foundations | [`course/foundations/`](course/foundations/) | GPU architecture, transformer inference and the KV cache, interconnects and parallelism, CUDA execution, number formats, serving APIs, containers and Kubernetes, performance metrics |
| vLLM labs (12) | [`labs/`](labs/) | FP32 baseline → BF16 → continuous batching and paged KV → attention backends (FA4, FlashInfer, Triton, Flex) → prefix caching → chunked prefill → CUDA graphs and `-O` levels → AWQ/GPTQ → FP8/MXFP8/MXFP4 → speculative decoding (draft, EAGLE-3, n-gram) → tensor parallelism → capstone |
| SGLang labs (14) | [`sglang/`](sglang/) | The same journey in SGLang plus RadixAttention, the overlap scheduler, structured output, MoE expert parallelism, attention backends and HiCache |
| llm-d labs (10) | [`llmd/`](llmd/) | Inference-aware routing (Envoy + EPP), scorer ablation, flow control, tiered KV cache, prefill/decode disaggregation, wide expert parallelism, then the upstream Kubernetes guides and autoscaling |
| Cloud & global | [`course/cloud/`](course/cloud/) | Regions, DNS and anycast, edge and gateways, capacity and cost, reliability, security and multi-tenancy |
| How it fits together | [`course/systems/`](course/systems/) | One request's life from DNS to GPU, a reference architecture, and a symptom-to-fix optimization playbook |

Start with [`course/00_start_here.md`](course/00_start_here.md) for the
suggested order.

## Running the labs

```bash
./run_course.sh --list          # show tracks
./run_course.sh vllm            # python run_all_labs.py   (see labs/README.md)
./run_course.sh sglang          # sglang/run_all_labs.sh    (see sglang/README.md)
./run_course.sh llmd            # llmd/run_all_labs.sh      (see llmd/README.md)
./run_course.sh                 # all three tracks, in order
```

Each track has its own prerequisites (a vLLM conda env, an isolated
`sglang-labs` env, Docker and optionally minikube for llm-d); read the track
README first. Every lab is self-contained: one README, one runnable script,
results written under `results/`.

## The apps

The course content is plain Markdown. `tools/build_course.py` turns the
modules and every lab README into `web/course.json` (pre-rendered lessons
plus flash cards: modules carry `Q:`/`A:` pairs under `## Flashcards`; each
lab gets cards from its concept and "what to look for" sections).

- **Web app / PWA** (`web/`): open the GitHub Pages URL on your phone and
  choose *Add to Home screen*. Read mode for lessons, Cards mode with
  Learn/Test, Known/Review marks, swipe between cards, search, offline cache.
  Try it locally with `python3 tools/build_course.py && python3 -m http.server -d web`.
- **Mac app** (`app/`, SwiftUI, macOS 14+): `scripts/build_app.sh` builds
  `build/LLMServingLabs.app` with the course bundled.
- **Publishing**: `scripts/publish_web.sh` rebuilds the course and pushes
  `web/` to the `gh-pages` branch.

## Repository layout

```
course/          concept modules (foundations, cloud, systems) with flash cards
labs/            vLLM track (Python labs, run_all_labs.py at repo root)
common/          shared Python helpers for the vLLM track
sglang/          SGLang track (bash labs + common/)
llmd/            llm-d track (Docker no-k8s labs 01-06, Kubernetes labs 07-09)
tools/           build_course.py
web/             PWA (index.html, app.js, app.css, sw.js, course.json, icons)
app/             SwiftUI Mac app
scripts/         check_env.py, download_models.sh, build_app.sh, make_icon.swift, publish_web.sh
run_course.sh    runs one or all lab tracks
```
