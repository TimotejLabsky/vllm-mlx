# CLAUDE.md

This is the **TimotejLabsky fork** of `waybarrios/vllm-mlx`, maintained as a
patch stack for the homelab Mac Studio inference server. It is not a normal
feature branch — read this before changing anything.

## Key documents

- [`PATCHES.md`](PATCHES.md) — **single source of truth**: every patch with
  rationale, measurements, rebase history, upstreaming status. Update it in the
  same commit as any patch change.
- [`docs/fork/`](docs/fork/) — fork design docs and investigations:
  - [`continuous-batching-hybrid-caching.md`](docs/fork/continuous-batching-hybrid-caching.md)
    — why BatchedEngine's stock prefix cache gets zero hits on hybrid
    (attention+SSM) models, and the 2026-07-02 update: the fork built its own
    hybrid-safe batched cache (patches #29–#40, `batched_system_kv.py`).
    **Since 2026-07-09 the ENTIRE text fleet runs BatchedEngine**
    (`--continuous-batching --text-only` + batched system-KV + patch-#48
    memory-pressure relief on all 19 llama-swap text routes; only the vision
    routes and the embedding route remain on other paths). Engine changes are
    still deliberate — the llama-swap config in `personal-infratructure` is
    the source of truth; don't flip routes without a measured reason.
  - [`prefix-caching-landscape-2026-08.md`](docs/fork/prefix-caching-landscape-2026-08.md)
    — how vLLM/SGLang/LMCache/Marconi do prefix caching, why two of their three
    optimisation axes are structurally N/A on a single-user unified-memory box
    (offload tiers, cross-user sharing), and the one real gap in ours:
    recency-only eviction. Verdict is instrument-first, don't build yet.
  - [`speed-lever-ledger-2026-09.md`](docs/fork/speed-lever-ledger-2026-09.md)
    — **read before proposing any performance work**: every speed lever
    measured on this box (10 refuted, 5 shipped), the methodology lessons,
    and the watch list (mlx #4020 gated-delta kernels = the pending big win,
    taken via mlx release, never vendored). Update it with every new verdict.
  - `DESIGN-system-kv-lru.md`, `DESIGN-system-kv-ssd.md` — design docs for
    patches #13 and #16.
  - [`vision-caching.md`](docs/fork/vision-caching.md) — which caches apply on
    vision routes (pixel cache ON, media-KV deliberately OFF until phase B,
    text-KV ON), why, and the 2026-07-30 per-arch sweep verdicts.
- Consumer side (deploy config, llama-swap, model lineup) lives in the
  `personal-infratructure` repo (`mac-studio/README.md` keeps a deployed-state
  table — update its top row on each deploy).

## Conventions

- Each patch is a separate commit on `main` prefixed `patch:` (also used:
  `docs:`, `tests:`, `restore:`, `fix(scope):`, `refactor:`). Amend a patch via
  `git commit --fixup <sha>` + `git rebase -i --autosquash`, then
  `git push --force-with-lease`.
- Rebase onto `upstream/main` periodically. When upstream rewrites
  `engine/simple.py` in ways our patches supersede, reject upstream's version
  wholesale and restore wanted upstream deltas in a dedicated `restore:` commit
  (precedent: #541, #579 — see the rebase notes at the top of PATCHES.md).
- The system-KV cache stack lives in `vllm_mlx/system_kv.py` (+
  `system_kv_ssd.py`), extracted from `engine/simple.py` precisely to shrink the
  rebase conflict surface (PATCHES.md #18); the batched equivalent lives in
  `vllm_mlx/batched_system_kv.py` behind one-line scheduler delegators
  (PATCHES.md #38). Keep new cache logic in those fork-owned modules.
- Tests assert **fork semantics**, not upstream's (e.g. default admission is
  `wait`, denylist probe instead of allowlist). The full suite must stay green:
  `.venv/bin/python -m pytest tests/`.
- Upstream accepts external fork PRs again (verified 2026-07-07 — recent PRs
  are all cross-repo; the earlier collaborator restriction has lifted). Keep
  upstreaming branches ready and rebased (e.g. `fix/batched-stop-strings`,
  `fix/batched-per-request-sampling`, `feat/batched-system-kv`).
- Touch upstream-owned files (README.md, docs/ outside `docs/fork/`) as little
  as possible — every line is future rebase conflict surface.
