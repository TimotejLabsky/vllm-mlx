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
    — why BatchedEngine's prefix cache gets zero hits on hybrid (attention+SSM)
    models and what a fix would take. **We deliberately run SimpleEngine +
    system-KV**; don't "helpfully" switch to `--continuous-batching`.
  - `DESIGN-system-kv-lru.md`, `DESIGN-system-kv-ssd.md` — design docs for
    patches #13 and #16.
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
  rebase conflict surface (PATCHES.md #18). Keep new system-KV logic there.
- Tests assert **fork semantics**, not upstream's (e.g. default admission is
  `wait`, denylist probe instead of allowlist). The full suite must stay green:
  `.venv/bin/python -m pytest tests/`.
- Upstream PR creation is collaborator-restricted for us; feedback goes via PR
  comments. Keep fix branches ready (e.g. `fix/admission-env-respected`).
- Touch upstream-owned files (README.md, docs/ outside `docs/fork/`) as little
  as possible — every line is future rebase conflict surface.
