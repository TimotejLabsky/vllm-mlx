# Fork documentation

Fork-owned docs live here, separate from upstream's `docs/` tree to keep the
rebase conflict surface small. The patch list itself — rationale, measurements,
rebase history — lives in [`PATCHES.md`](../../PATCHES.md) at the repo root.

| Doc | What it covers |
|---|---|
| [`continuous-batching-hybrid-caching.md`](continuous-batching-hybrid-caching.md) | Why BatchedEngine's prefix cache gets zero hits on hybrid (attention+SSM) models, what a fix would take (work items A–D), and why we deliberately stay on SimpleEngine + system-KV for now |
| [`DESIGN-system-kv-lru.md`](DESIGN-system-kv-lru.md) | Design for the multi-slot LRU system-KV snapshot cache (patch #13) |
| [`DESIGN-system-kv-ssd.md`](DESIGN-system-kv-ssd.md) | Design for SSD persistence of system-KV snapshots (patch #16); implementation diverges on two points noted in PATCHES.md |
| [`qwen38-looping-investigation.md`](qwen38-looping-investigation.md) | The 2026-08 Qwen3.8 agentic-looping hunt: the real cause (the thinking phase machine walked the prompt, patch #79), the **three disproven mechanisms** that preceded it, and the method/operational gotchas the search turned up |
| [`prefix-caching-landscape-2026-08.md`](prefix-caching-landscape-2026-08.md) | How vLLM/SGLang/LMCache/Marconi do prefix caching, which of their axes are structurally N/A here, and the one real gap in ours |
| [`improvement-roadmap-2026-08.md`](improvement-roadmap-2026-08.md) | 2026-08 ecosystem survey and ranked improvement levers |
| [`vision-caching.md`](vision-caching.md) | Which caches apply on vision routes, why, and the per-arch sweep verdicts |
