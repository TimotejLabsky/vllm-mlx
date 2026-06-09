# Design: multi-slot LRU + grow-on-HIT for system-KV cache

Status: **draft, not deployed**. This document captures the plan before code is written. The investigation was triggered after patch #12 (`hybrid-probe-denylist`) exposed patch #9's single-slot limitation on opencode workloads that switch between agents (coder ↔ Plan ↔ Explore ↔ Task).

## Problem

After patch #9 + #12, the system-KV cache works on Qwen3.6-27B-4bit. Multi-turn within one agent gets the full grow-on-HIT win (~25 s cold, ~1.9 s warm). But:

- **One slot only.** opencode dispatches sub-agents (Plan/Explore/Task) with a different system prompt. Each switch replaces the active slot. Returning to the coder = cold 25 s prefill again.
- **Empirical signature in `/v1/status`:** `entry_count: 1` even when multiple agents touch the cache; `misses` increments every time the user bounces between them.

The fix is the same multi-slot LRU pattern PR #541 lands upstream — but we need to layer it under patch #9's grow-on-HIT, which #541 does NOT do.

## Goal

Run an N-slot LRU keyed by system-prefix hash. Each slot owns a snapshot, a token count, and the cached token ID list — so grow-on-HIT semantics work per slot. Capacity via env var `VLLM_MLX_SYSTEM_KV_SLOTS` (default 4 — enough for coder + 2-3 sub-agents).

## Data structure

Today (4 instance vars, single slot):

```python
self._system_kv_snapshot      # list of layer states
self._system_kv_hash          # str (md5 of system text)
self._system_kv_token_count   # int
self._system_kv_token_ids     # list[int]
```

New (1 OrderedDict + capacity + counters):

```python
self._system_kv_lru: OrderedDict[str, dict] = OrderedDict()
# value shape:
#   {"snapshot": list, "token_count": int, "token_ids": list[int]}

self._system_kv_capacity = max(1, int(
    os.environ.get("VLLM_MLX_SYSTEM_KV_SLOTS", "4")))

# counters (unchanged, plus one new):
self._system_kv_hits = 0
self._system_kv_misses = 0
self._system_kv_tokens_saved = 0
self._system_kv_evictions = 0   # NEW
```

## Helpers

Two new methods on `SimpleEngine`:

```python
def _system_kv_lookup(self, system_hash, full_tokens_list):
    """Return (snapshot_ref, token_count, token_ids) for a HIT, else (None, 0, None).

    Reads OrderedDict.get under the GIL → atomic. Returns the actual list
    references stored in the slot; caller MUST capture them at gate time
    and use the closure-local reference for the snapshot restore, never
    re-read self._system_kv_lru in the worker. This is the same TOCTOU
    contract patch #9 already documents at the gate site.

    A slot whose system_hash matches but whose token_ids is NOT a prefix
    of full_tokens_list returns MISS — caller will overwrite the slot.
    """

def _system_kv_store(self, system_hash, snapshot, token_count, token_ids):
    """Insert or replace a slot, move it to the end of the LRU, evict
    oldest if over capacity. MUST run inside _generation_lock.

    Calls mx.clear_cache() ONLY on the eviction path — patch #9 currently
    clears unconditionally on every store, which flushes the Metal
    allocator's reuse pool and costs us a few hundred ms per MISS we
    don't need to pay. PR #541's measurement.
    """
```

## Code touch points

All in `vllm_mlx/engine/simple.py`. Line numbers as of `ea40b4b` (post-patch-#12):

| Line | Today | New |
|------|-------|-----|
| 179-186 | 4 single-slot ivars + 3 counters | 1 LRU dict + capacity + 4 counters (add `_system_kv_evictions`) |
| 411-417 | `stop()` resets ivars | `stop()` clears LRU + zeroes counters |
| 1127-1135 | `stream_chat` gate: `system_hash == self._system_kv_hash and ...` | call `_system_kv_lookup()` |
| 1213-1215 | `stream_chat` MISS store: assign ivars | call `_system_kv_store()` |
| 1708-1731 | `_stream_generate_text` gate: prefix-match against `self._system_kv_token_ids` | call `_system_kv_lookup()` |
| 1769-1803 | `_stream_generate_text` HIT branch + GROW: writes back to ivars | writes back via `_system_kv_store()` to the same slot |
| 2009-2014 | `_stream_generate_text` MISS store | call `_system_kv_store()` |
| 2381-2402 | `get_stats` reports single slot | aggregate over all slots + legacy fields + new `slots: [...]` array |

Estimated diff: **~150-200 lines** changed.

## Backward compat for /v1/status and Prometheus

Today the `cache` block looks like (verified live):

```json
{
  "tokens": 13381,           // active slot
  "hash": "d232ee7cf34bf818",// active slot
  "memory_mb": 1030.9,
  "hits": 1,
  "misses": 1,
  "tokens_saved": 13381,
  "entry_count": 1,
  "current_memory_mb": 1030.9
}
```

Mac-studio's Go exporter (`mac-studio/exporter/main.go`) reads:
- `hits`, `misses`, `tokens_saved` — counter gauges
- `tokens`, `memory_mb`, `current_memory_mb`, `entry_count` — instantaneous gauges
- `hash` — informational, not used for Prom labels

To stay backward-compatible, the new `cache` block keeps every existing field with sane aggregate semantics:

```json
{
  "tokens": <max(slot.token_count) across LRU>,   // legacy — was active
  "hash": <most-recently-used slot hash>,         // legacy
  "memory_mb": <sum across LRU>,                  // legacy aggregate
  "hits": ...,
  "misses": ...,
  "tokens_saved": ...,
  "evictions": ...,                               // NEW
  "entry_count": <len(LRU)>,                      // changes semantics:
                                                  //   was 0 or 1, now 0..capacity
  "current_memory_mb": <sum across LRU>,          // legacy aggregate
  "capacity": <max slots>,                        // NEW
  "slots": [                                      // NEW
    {"hash": ..., "tokens": ..., "memory_mb": ...},
    ...
  ]
}
```

`entry_count` semantic shift is acceptable — the existing exporter gauge just emits the value, and going from `0/1` to `0..N` is meaningful. We document it as part of the patch entry in PATCHES.md.

## Concurrency model (TOCTOU)

Patch #9 already documents the rule:
- gate is read OUTSIDE the lock
- snapshot REFERENCE captured at gate time goes into a closure variable
- restore runs INSIDE the lock and uses the closure ref, never `self.<ivar>`

Same rule for the LRU:
- `_system_kv_lookup()` returns the snapshot/ids references — atomic via `dict.get`
- caller captures those into closure-local vars (`hit_snapshot`, `_cached_ids`, `_cached_len`)
- restore reads ONLY the closure vars
- `_system_kv_store()` runs inside the lock; concurrent MISS replacing or evicting the slot can't corrupt the already-captured reference

The only new failure mode vs single-slot: a concurrent MISS could **evict** a slot whose snapshot we're holding a reference to. Python keeps the underlying list alive (refcount > 0), so the restore still works — just the cache entry is gone from the LRU. Next request with the same hash will MISS and re-prefill. No correctness bug.

PR #541 ships two new pytest tests (`tests/test_simple_engine.py`) covering exactly this case + bounded-KV gating — worth porting once we land.

## Eviction & memory pressure

Each slot is ~1 GB for our typical opencode workload (13K-token system → 1030 MB metal cache, measured today). Four slots = ~4 GB. Within 64 GB Mac Studio budget alongside Qwen3.6-27B-4bit (~14 GB) and the MoE if loaded (~22 GB).

Conservative: keep default capacity at 4, document the env var, and if memory pressure grows under real opencode use, drop to 2 or 3.

## Risk + rollback

- **Risk:** the helpers must run inside `_generation_lock` for `_system_kv_store`. If we accidentally call from outside, two concurrent MISSes could corrupt LRU ordering. Defense: keep helpers tightly bracketed by the existing lock scopes (already serialized for MLX ops), and add an assert.
- **Rollback:** patch is one commit. If anything looks off, `git revert <sha>` brings back single-slot. Memory state is reset on engine restart (`stop()` clears LRU).
- **Deploy gate:** never push during an active opencode session. Watch `/v1/status` → `num_running == 0` before deploying. The reinstall + worker restart costs the model warm KV cache but no work-in-progress.

## Test plan

Local on mac-studio (between opencode sessions):

1. **Smoke single-slot:** existing 4-turn multi-turn script (`/tmp/multiturn.py`) — should produce same HIT-and-GROW trace as today. Verifies we didn't regress patch #9.
2. **Two-prefix interleave:** new script that fires alternating requests with TWO different system prompts (~4 KB each). Expected: 2 slots populated, HIT on both prefixes, no eviction at capacity=4.
3. **Eviction:** fire 5 distinct system prefixes with capacity=4. Expected: oldest one evicted (log line + counter), next request for the evicted prefix is a MISS.
4. **Stale-reference race:** synthetic test (port from PR #541) that captures a snapshot ref at gate time, races a MISS-replacement, then restores — must still see the original snapshot.

## Estimated effort

- Code: ~1.5 hours (helpers + 5 touch-point rewrites + stats aggregation).
- Test scripts: ~30 min.
- A/B + verification on mac-studio (needs opencode-idle window): ~1 hour.
- PATCHES.md + NOTICE: ~15 min.

## Open questions

1. **Stream_chat path** has its own cache block (lines 1127-1215). After patch #4 the LLM text path bypasses it (routes to `_stream_generate_text`), so the stream_chat cache only fires for MLLM+media — which doesn't use it because media changes prompt every time. **Question:** rewrite both paths for consistency, or only the path that actually matters (`_stream_generate_text`)? Recommendation: both, for symmetry and to keep `get_stats` aggregation correct regardless of which path stored a slot.
2. **Sub-agent system prompts in opencode** — verify they actually differ enough at the start for the system hash to differ. If opencode adds the sub-agent role as a *user* message and keeps the same `system`, multi-slot doesn't help and patch #9 already covers it. Worth a one-time inspection of an opencode request body for each agent type before claiming the win.
