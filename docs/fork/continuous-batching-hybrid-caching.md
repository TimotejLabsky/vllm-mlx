# Continuous batching × caching on hybrid models — findings & roadmap

*Status: assessment 2026-06-09; reassessed 2026-06-30; **IMPLEMENTED
2026-07-02** — item B is now built as **patch #34** (`batched_system_kv.py`,
with the Tier-1 parity series #29–#33 alongside). See
[the 2026-07-02 update](#update-2026-07-02--implemented) below. The historical
analysis is retained; the "don't port" verdict is superseded for the
*feasibility* half (built, e2e-verified bit-identical) while the *motivation*
half still governs deployment: off by default until concurrent traffic is
routine and the Studio A/B passes.*

---

## Update 2026-07-02 — implemented

The gate this doc demanded — "a standalone mlx-lm spike proving
`_merge_caches` handles a restored recurrent state in a concurrent batch" —
**PASSED** on `Qwen3.5-0.8B-8bit` (the production qwen3_5 hybrid family):
snapshot-restored mid-sequence hybrid caches merge **bit-identically**, incl.
insertion mid-flight into a decoding batch. mlx-lm 0.31.3's merge handles
this *by design*: `BatchKVCache.merge` right-justifies each row by its own
offset into the mask's `left_padding`; `ArraysCache.merge` stacks recurrent
state position-agnostically.

Item B then landed as **patch #34**: checkpoint capture at
`insert_segments` boundaries during prefill, LRU snapshot entries at request
end, LCP + `select_restore_pos`/`build_partial_restore_states` on fetch,
injected at the `request.prompt_cache` → `insert(caches=…)` seam this doc
identified. E2E on the real scheduler: exact re-send HIT at N−1, divergent
chain HIT at the nearest checkpoint, two restored requests decoding
concurrently — all byte-identical to a cache-disabled control. Enabled via
`VLLM_MLX_BATCHED_SYSTEM_KV=1`; the Tier-1 gaps found during the port
(stop strings unenforced, per-request sampling dropped, DRY no-op, lazy
realize bypass, no `--text-only` on batched) are patches #29–#33.

Work items below: **A/B/C prompt-side = done** (#34 covers exact-resend,
prefix growth, and LCP via checkpoints); **D = the merge validation passed**,
the batched SSD tier remains memory-aware-only (our `system_kv_ssd.py` is
SimpleEngine-wired — port on demand).

**Question answered:** what would it take to run BatchedEngine (`--continuous-batching`)
without forfeiting the caching we rely on — especially on hybrid (attention + SSM)
models like Qwen3-Next / Qwen3.5 / Qwen3.6, which is our entire heavy lineup?

**TL;DR:** Batched *inference* on hybrids works (the old MLX cross-thread deadlock
was fixed upstream long ago). What's broken is the *prefix cache*: it gets zero
hits on hybrids, so every request pays full prefill, and switching engines would
forfeit the 13–52× system-KV wins (PATCHES.md #4/#9/#13/#16). The root cause is
semantic, not a bug: **SSM/recurrent state cannot be rewound**, and three of the
four cache match paths require rewinding. The sound fix is checkpointing SSM
state at boundaries during prefill — which is exactly what our system-KV cache
already does at one boundary. Work items A–D below; B (port `SystemKVManager`
to the batched path) is the best ROI if we ever need batching.

---

## Update 2026-06-30 — the blocker moved

Reassessed after the v0.4.0 rebase. The "batching forfeits our cache" wall has
largely fallen — but feasibility cuts the *wrong way*, so the verdict is still
**don't port it.** The root-cause section below is unchanged and still accurate
(the match-path gates still fire); what changed is everything around it.

**What closed since 2026-06-09:**

- **Items A + C's hard core is now built** in `vllm_mlx/system_kv.py` — an
  engine-agnostic checkpoint engine: `classify_layers` (trim/ckpt/opaque),
  `append_checkpoint` (the ladder, with geometric thinning), `select_restore_pos`
  (nearest checkpoint ≤ match), `build_partial_restore_states` (slice attention
  KV to `pos`, install the recurrent checkpoint, refuse opaque). These are pure
  list-in/list-out functions, not coupled to SimpleEngine — exactly the "sound
  general fix" this doc called a 1.5–2K-line, 2–3-week upstream job.
- **Item C copy-on-store aliasing safety → upstream #576** ("Fix hybrid cache
  snapshot aliasing"): `_dequantize_cache()` deep-copies non-quantized layers on
  fetch (`memory_cache.py` ~`:586`).
- **Item C memory accounting / LRU budget → upstream #620** ("honor MLX buffer
  cache limit").
- **Item D bf16 SSD crash → upstream #563 + #605 + #612** (snapshot on producer
  thread, native QuantizedKVCache spill, preserve bf16 across quantized spill).
  The "port the MLX-native safetensors approach into `ssd_cache.py`" side quest
  is no longer needed.
- **Injection seam confirmed.** The batched fetch→schedule→insert flow has a
  clean hook: fetch at `scheduler.py:1826` parks the result on
  `request.prompt_cache`; `_schedule_waiting` reads it at `:2024`;
  `BatchGenerator.insert` at `:2080` passes `caches=[…]` into mlx-lm's
  `_merge_caches`. A restored system-KV checkpoint can be dropped in at the
  post-miss point. **Item B is now a small wiring job, not a rewrite.**

**What did *not* move:** mlx-lm is still 0.31.3 — `ArraysCache.is_trimmable()`
returns `False` and there is no `trim`. The two `has_non_trimmable` gates
(`memory_cache.py:794` supersequence, `:888` LCP) still *skip* hybrid matches
rather than fall back to a checkpoint restore. "Getting it working" = replacing
those two skips with calls into `build_partial_restore_states` + the seam above.

**Why it's still not worth porting:**

1. **The single-stream port is strictly worse than today.** BatchedEngine at
   batch=1 is a slower SimpleEngine (merge/scheduler overhead, zero concurrency
   gain). The easy version buys nothing.
2. **The version that helps — concurrent hybrid batching — rests on an
   unverified mlx-lm internal we don't own:** whether `_merge_caches` /
   `ArraysCache.merge` correctly batch a *restored mid-sequence* recurrent state
   alongside differently-positioned sequences (the `left_padding` / `lengths` /
   `make_mask` bookkeeping). That is item D's still-open "validate *concurrent*
   hybrid batching," and it lives in mlx-lm 0.31.3, not this repo.
   `ArraysCache.merge` copying into a fresh `[B, …]` array is encouraging but
   unproven for mixed positions.
3. **The ceiling is ~1.2× regardless** — the Studio measured MLX decode as *not*
   memory-bound at 4-bit/8–35B (continuous batching 1.2× aggregate, spec/MTP
   0.5–0.76×), on a workload that is effectively single-stream.

**If batching is ever needed,** the de-risking order is: (1) a standalone mlx-lm
spike proving `_merge_caches` handles a restored recurrent state in a concurrent
batch; only then (2) the item-B wiring behind a flag + a Studio A/B. Until
concurrency is routine, SimpleEngine + system-KV remains the deliberate choice.

---

## Why it fails — one root cause, three symptoms

`MemoryAwarePrefixCache` (the default batched-mode cache; the block/paged
`BlockAwarePrefixCache` is opt-in via `use_paged_cache`) has four match paths in
`fetch_nearest_cache()`. The scheduler stores entries keyed on
**prompt + completion** tokens (`vllm_mlx/scheduler.py:2346`). For hybrids,
whose cache lists mix `KVCache` (attention) with `ArraysCache` (SSM/recurrent):

| Match path | What it needs | Hybrid outcome |
|---|---|---|
| Exact (`memory_cache.py:728`) | identical key | **Never fires** — next identical prompt is *shorter* than the stored prompt+completion entry, so it falls into supersequence |
| Supersequence (`memory_cache.py:790`) | trim completion tokens off cached state | **Rejected** — duck-typed check `hasattr(lc, "offset") and hasattr(lc, "keys")` at `memory_cache.py:794–803` |
| LCP (`memory_cache.py:888–906`) — the agent pattern: same system prefix, different user message | rewind to divergence point | **Rejected** — same duck-typed check |
| Prefix (`memory_cache.py:830`) — multi-turn growth, new prompt exactly extends stored entry | nothing (no trim) | **Hybrid-safe in principle**, but in the 2026-05-09 empirical A/B even this never fired → store/fetch token-canonicalization mismatch (chat-template re-render of the assistant turn) |

The rejections are **correct, not bugs**: recurrent state after N tokens is an
aggregate hidden state, not a per-token sequence — there is no operation that
rewinds it to token M < N. mlx-lm PR #1254's approach (make `ArraysCache.trim()`
reset recurrent state) is semantically wrong; we empirically proved it crashes
the worker in `_trim_cache_offset` (see History below). It is still absent from
our pinned mlx-lm 0.31.3 anyway.

The block-based `BlockAwarePrefixCache` has partial recurrent support — a
`"latest"` storage mode that snapshots `ArraysCache` state at the stored-sequence
boundary (`vllm_mlx/prefix_cache.py:675–692`, restore at `:910–915`) — but that
snapshot is only valid for a match at exactly that boundary; it cannot serve
LCP/supersequence matches either.

## Empirical history (from `personal-infratructure/docs/llm/llm-stack.md`, 2026-05-09)

- Qwen3.6-27B-8bit under `--continuous-batching`: engine loads, tool calling
  works, output bit-identical to SimpleEngine. 4 identical 7.7K-token requests →
  `hits=0 misses=4`, 58–80 s each (vs ~17 s warm replays on SimpleEngine + system-KV).
- Installing mlx-lm PR #1254 (`ArraysCache.is_trimmable() → True` + `trim()`):
  mlx-lm-level fix verified, but vllm-mlx's own duck-typed check still rejects;
  and forcing past it crashed the worker on runs 2+ (`_trim_cache_offset` assumes
  KVCache layout).
- `--ssd-cache-dir` is wired only into BatchedEngine; on SimpleEngine it is a
  silent no-op (flag was removed from the llama-swap config).
- Concurrency was **not** tested — all A/Bs were sequential requests. Whether
  mlx-lm's `BatchGenerator` correctly merges `ArraysCache` state across
  concurrent sequences (cf. `vllm_mlx/utils/mamba_cache.py`) is unverified.

## What it would take

The only sound general fix is **checkpointing SSM state at boundaries during
prefill**: on a match, restore the nearest checkpoint ≤ the match point, trim the
attention KV to that point (which *is* safe), and re-prefill the gap. Our
system-KV grow-on-HIT already does exactly this at one boundary (the system
prompt). Generalizing, in increasing order of effort:

| | Work | Size | Gets you |
|---|---|---|---|
| **A** | Store a second, **prompt-boundary** entry (one SSM state copy at end of prefill) + fix the store/fetch key mismatch | ~200–400 lines | Exact-match re-sends and clean multi-turn prefix hits on hybrids |
| **B** | **Port `SystemKVManager` (`vllm_mlx/system_kv.py`) to the batched path** — hook the scheduler's cache fetch to supply the snapshot as the initial per-request cache before `BatchGenerator` merge. Feasible precisely because #18 extracted the manager out of `engine/simple.py` | ~400–600 lines | The 13–52× system-prefix wins *under* batching — system-KV is the one-checkpoint special case of the general fix |
| **C** | **Multi-boundary checkpoints** in the prefix cache: SSM state every K tokens; hybrid-aware `_trim_cache_offset`; copy-on-store aliasing safety (the patch-#6 lesson — `MemoryAwarePrefixCache.store()` keeps live references and `fetch` returns them directly, unsafe for mutable `ArraysCache` state lists under concurrency); memory accounting (~40–50 MB per checkpoint on Qwen3-Next-class models → needs an LRU budget) | ~1.5–2K lines, 2–3 weeks | Full LCP/supersequence hits — the general agentic case |
| **D** | Side quests: the batched SSD tier crashes on bf16 (numpy serializers; port the MLX-native safetensors approach from `vllm_mlx/system_kv_ssd.py` into `ssd_cache.py`); validate *concurrent* hybrid batching | small / unknown | Persistence + actual multi-user confidence |

> **Status 2026-06-30** (see the [update above](#update-2026-06-30--the-blocker-moved)):
> A+C's checkpoint engine is **built** in `system_kv.py`; C's aliasing/budget
> prereqs **landed upstream** (#576, #620); D's bf16 SSD crash is **fixed
> upstream** (#563/#605/#612). **B** collapses to wiring (injection seam at
> `scheduler.py:2024`). The **only** still-open item is D's *concurrent
> hybrid-batching validation* — and it's in mlx-lm internals, not this repo.

## Verdict

*Refined 2026-06-30 — see the [update above](#update-2026-06-30--the-blocker-moved).
Bullets 1–2 below are now mostly moot (the work got built/landed); bullet 3 is
the governing one and is **reinforced**, not weakened.*

- **C belongs upstream** — it touches the scheduler/cache core, and upstream PR
  creation is currently collaborator-restricted for us (we can only comment).
- **B is the piece worth owning** (1–2 days + a Mac Studio A/B) if batching is
  ever needed: it removes the "batching forfeits our cache" objection entirely.
- **Check the motivation first.** Our workload is effectively single-stream
  (voice assistant + opencode rarely overlap; the admission-control patch #15
  handles the rare overlap with wait/503). Continuous batching only pays when
  concurrent requests are routine. Until then, SimpleEngine + system-KV remains
  the deliberate choice.
