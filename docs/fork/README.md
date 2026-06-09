# Fork documentation

Fork-owned docs live here, separate from upstream's `docs/` tree to keep the
rebase conflict surface small. The patch list itself — rationale, measurements,
rebase history — lives in [`PATCHES.md`](../../PATCHES.md) at the repo root.

| Doc | What it covers |
|---|---|
| [`continuous-batching-hybrid-caching.md`](continuous-batching-hybrid-caching.md) | Why BatchedEngine's prefix cache gets zero hits on hybrid (attention+SSM) models, what a fix would take (work items A–D), and why we deliberately stay on SimpleEngine + system-KV for now |
| [`DESIGN-system-kv-lru.md`](DESIGN-system-kv-lru.md) | Design for the multi-slot LRU system-KV snapshot cache (patch #13) |
| [`DESIGN-system-kv-ssd.md`](DESIGN-system-kv-ssd.md) | Design for SSD persistence of system-KV snapshots (patch #16); implementation diverges on two points noted in PATCHES.md |
