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
  - [`improvement-plan-2026-09-16.md`](docs/fork/improvement-plan-2026-09-16.md)
    — memory safety + caching under concurrent agent load: the 2026-09-16/17
    incident, the plan, and a **status block** kept current (shipped as
    #103–#108; what the data closed; what is open). Read before touching
    admission, relief, the SSD tier or the batched cache's eviction.
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
- **`tests/test_fork_invariants.py` pins the conditions a rebase can silently
  undo** (each test named for its PATCHES.md number). A rebase is not done
  until it is green AND re-read against upstream's diff of the functions it
  touches; a patch that adds a guarantee upstream cannot see adds its test
  there in the same commit.
- **CI `lint` is expected GREEN.** It black-checks only the lines a change
  adds (`scripts/fork/black_changed_lines.py`; ~44 files of rebase drift are
  left alone on purpose). Red means this change added drift — fix only the
  lines it prints (`python scripts/fork/black_changed_lines.py origin/main`
  locally). Never reformat a whole drifted file.
- Server-path patches get a **real-server check off the live routes**
  (`scripts/fork/e2e_*.py`: real `cli serve`, small model, spare port; run on
  the Studio from rsync'd source via `PYTHONPATH` for the prod stack + a
  hybrid model). Never manufacture a failure on a live route.
- Upstream accepts external fork PRs again (verified 2026-07-07 — recent PRs
  are all cross-repo; the earlier collaborator restriction has lifted). Keep
  upstreaming branches ready and rebased (e.g. `fix/batched-stop-strings`,
  `fix/batched-per-request-sampling`, `feat/batched-system-kv`).
- Touch upstream-owned files (README.md, docs/ outside `docs/fork/`) as little
  as possible — every line is future rebase conflict surface.
