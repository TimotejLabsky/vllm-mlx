# Continuous batching × caching on hybrid models — findings & roadmap

*Status: assessment, 2026-06-09 (code as of fork tip `b98ce50`, upstream base `caa8838`, mlx-lm 0.31.3). No implementation yet.*

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

## Verdict

- **C belongs upstream** — it touches the scheduler/cache core, and upstream PR
  creation is currently collaborator-restricted for us (we can only comment).
- **B is the piece worth owning** (1–2 days + a Mac Studio A/B) if batching is
  ever needed: it removes the "batching forfeits our cache" objection entirely.
- **Check the motivation first.** Our workload is effectively single-stream
  (voice assistant + opencode rarely overlap; the admission-control patch #15
  handles the rare overlap with wait/503). Continuous batching only pays when
  concurrent requests are routine. Until then, SimpleEngine + system-KV remains
  the deliberate choice.
