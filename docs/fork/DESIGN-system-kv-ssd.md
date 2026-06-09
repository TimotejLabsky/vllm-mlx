# Design: SSD persistence for system-KV snapshot

Status: **research notes, not implemented**. Drafted alongside the multi-slot LRU patch as a candidate next step.

## Goal

Survive process restarts. Today the system-KV snapshot lives entirely in the SimpleEngine process. Whenever vllm-mlx exits — TTL eviction, OOM crash (now mitigated by the heavy-group fix), llama-swap model swap, manual restart — every snapshot is lost and the next request pays full cold prefill (~25 s on a 4 KB-token prompt against the 27B-4bit; ~70 s on a 13 KB-token MoE workload).

With SSD persistence, the next reload would pay disk I/O (~200 ms to read 1 GB on Apple Silicon NVMe at 5+ GB/s sequential) instead of recompute. **100× speedup vs cold prefill.**

## Building blocks already in the tree

Upstream PR #309 (`feat: SSD KV cache tiering with async promote`, merged) shipped a complete SSD tier for the `MemoryAwarePrefixCache` (BatchedEngine path). It is **not** wired into `SimpleEngine` and not used by our patch #9 snapshot. But everything we'd need is already in `vllm_mlx/ssd_cache.py`:

- `SSDCacheConfig` — paths, size cap, file perms, queue sizing.
- `SSDIndex` — SQLite-WAL metadata. `lookup_exact(tokens)`, `lookup_prefix(tokens)`, `get_lru(limit)`, `touch(tokens)`, `get_total_bytes()` — atomic.
- `KVCacheSerializer` — handles `KVCache` and `RotatingKVCache` (`.keys`, `.values`, `.offset`, and the rotating fields).
- `ArraysCacheSerializer` — handles `ArraysCache` (linear-attention layers — exactly what Qwen3.6-27B-4bit needs for its Gated DeltaNet layers).
- `get_serializer_for_layer(layer)` — duck-typed dispatch.
- `SSDCacheTier`:
  - `enqueue_spill(tokens, cache, bytes)` — async write via background writer thread.
  - `_write_entry()` — atomic temp+rename, per-layer safetensors, manifest.json.
  - Atomic crash-consistent commits, capacity-based eviction, corrupt-entry quarantine.

The two serializers cover all cache types we currently see in production on Mac Studio (KVCache, ArraysCache from hybrid attention). No new serializer work needed.

## Integration with our patch #9 + the multi-slot LRU

Conceptually:

```
                                ┌──────────────────────────┐
   stream_chat / _stream_text   │ in-memory active slot    │
              ──────────────►   │ (legacy single-slot      │
                                │  ivars in SimpleEngine)  │
                                └─────────────┬────────────┘
                                              │  demote
                                              ▼
                                ┌──────────────────────────┐
                                │ in-memory LRU bag        │
                                │ (OrderedDict, capacity=4)│
                                └─────────────┬────────────┘
                                              │  evict
                                              ▼
                                ┌──────────────────────────┐
                                │ SSD tier (per-model dir) │
                                │ index.db + safetensors   │
                                │ ~50 GB cap, LRU eviction │
                                └──────────────────────────┘
```

Three triggers:

1. **LRU bag overflow → spill to SSD** (instead of dropping).
   - When `_lru_demote_active_to_bag` evicts the oldest bag entry, it currently calls `mx.clear_cache()` and drops the snapshot. New behavior: enqueue the snapshot for async SSD write before dropping the in-memory reference. The async writer thread on `SSDCacheTier` handles I/O; the engine never blocks.

2. **Engine startup → reconcile SSD index**.
   - On `start()`, after the model loads, look up SSD entries whose `model_id` matches and bring back their *metadata* to a "ghost slot" index (we don't load them into RAM eagerly).
   - When a request comes in with `system_hash` matching a ghost, do a synchronous `_promote_from_ssd` (~200 ms) instead of a 25 s cold MISS.

3. **MISS lookup → check SSD before recompute**.
   - In the existing MISS branch, before computing tokens from scratch, check if `system_hash` (or our cached `_extended_token_ids` prefix) is in the SSD index. If yes, promote to RAM, then we're back in the HIT path.

## Key-shape question — `system_hash` vs `tokens_key`

The existing SSD tier keys by `tokens_key: tuple[int, ...]` (the actual token sequence) and uses `_tokens_hash()` for the on-disk filename. This supports `lookup_prefix(query_tokens)` which finds entries whose tokens are a *prefix* of the query — exactly the same semantics as our patch #9 grow-on-HIT.

Our patch #9 uses two keys side-by-side:
- `system_hash` — md5 of the system role TEXT (16-hex). Coarse, agent-level.
- `_extended_token_ids` — the actual cached token sequence. Used for the prefix-match validation.

For SSD persistence, the cleanest key is the **token sequence** (it's already what the SSD layer expects). `system_hash` is then just a fast first-level filter via the SQLite index (we can add a `system_hash` column for indexed lookup).

This also means the SSD tier can correctly serve **prefix matches across restarts**: if turn N-1 stored 13K tokens, and turn N's request has the first 13K identical, SSD `lookup_prefix(new_tokens)` returns the saved entry and we restore + suffix-prefill. Mirrors our in-memory grow-on-HIT exactly.

## Memory and timing budget

Disk costs (measured today, Qwen3.6-27B-4bit dense, KVCache+ArraysCache hybrid):

- 4,223-token snapshot: **439 MB** on disk (extrapolate: ~100 KB per token, dominated by KVCache fp16).
- 13,389-token MoE snapshot: **339 MB** on disk (MoE compresses better; ArraysCache layers are smaller).
- 63,045-token snapshot: **4.29 GB** on disk (the largest we observed in production).

At Apple Silicon NVMe sequential read of ~5 GB/s, restore latency:
- 439 MB: ~90 ms
- 4.29 GB: ~860 ms

That's a strict upper bound — actual restore is dominated by `safetensors.numpy.load_file` + `mx.array` conversion (probably another 100-500 ms of CPU). Total still **<1.5 s for any snapshot size we've seen**.

For comparison, cold prefill on the same prompt sizes:
- 4 K tokens on 27B-4bit dense: **25 s** (measured today)
- 13 K tokens on MoE: **70 s** (1 m 42 s observed earlier today, in mixed parallel-task workload)
- 63 K tokens on dense: would take **>5 min** based on extrapolation.

Net speedup at the upper end is ~100×–300×. The bigger the conversation history, the bigger the win.

## Disk capacity

Per-model directory: `~/.cache/vllm-mlx/ssd_cache/{model_id}/`.

With a 50 GB cap (Mac Studio internal SSD has hundreds of GB free) and 4 GB average snapshot, that's ~12 distinct system-prefix snapshots persisted at any given time. Plenty for opencode's main-agent + 3-5 sub-agents pattern, with room for tomorrow's session to start warm if we resume yesterday's conversation.

## Risks + correctness

1. **Model identity check** — must ensure SSD entries are tied to a specific model+revision. Otherwise reloading after `pip install --upgrade mlx-community/Qwen3.6-27B-4bit` (different revision) would feed garbage KV into a different model and crash or produce gibberish. Existing tier embeds model name in the path; PR #365 (`Fix garbled output from stale disk-persisted prefix cache`) already addresses staleness for the BatchedEngine path — same logic applies here.

2. **Tokenizer drift** — same model name + new tokenizer = same problem. Mitigation: store the tokenizer vocab hash in the manifest, refuse to load mismatches.

3. **Bounded-KV / RotatingKVCache** — already rejected by patch #12 (hybrid-probe-denylist). The SSD tier inherits that gate: if `_supports_system_kv_cache=False`, never spill or promote.

4. **Quantized KV** — existing SSD tier had a bug (issue #443) where `_QuantizedCacheWrapper` is missing `.state`. PR #451 fixed it but our snapshots aren't quantized (bf16) so this doesn't bite us. If we ever enable `--kv-cache-quantization` for the snapshot path we'd need to dequantize before write (already what #451 does).

5. **Concurrent process load** — two vllm-mlx processes mounting the same SSD dir would race on writes. Avoid by including the PID in the temp filename (already done by `_write_entry` via the `tmp_dir`). The SQLite WAL is already multi-process-safe for reads but not for writes.

6. **Disk-write storms during opencode bursts** — every parallel-task switch could enqueue a spill. The async writer thread + bounded spill queue (`spill_queue_size=64`) already absorbs bursts; queue-full drops are explicitly logged. The 4-slot LRU caps in-memory churn so spills only happen at the bag's tail.

## Implementation skeleton

```python
# In SimpleEngine.__init__:
self._ssd_tier: SSDCacheTier | None = None
if os.environ.get("VLLM_MLX_SSD_CACHE_DIR"):
    self._ssd_tier = SSDCacheTier(SSDCacheConfig(
        cache_dir=os.environ["VLLM_MLX_SSD_CACHE_DIR"] + "/" + safe_model_name(self._model_name),
        max_size_gb=float(os.environ.get("VLLM_MLX_SSD_MAX_GB", "50")),
    ))
    self._ssd_tier.start_writer()

# In _lru_demote_active_to_bag, before popitem:
if evicted:
    ev_hash, ev_entry = ...  # captured during popitem
    if self._ssd_tier:
        tokens = tuple(ev_entry["token_ids"])
        bytes_est = sum(... per-layer sizes ...)
        # The snapshot layers are mx.arrays — need to wrap them in a
        # cache-like object so the existing serializer dispatch works.
        # Simplest: spill the raw `cache_layers` we'd get from
        # make_prompt_cache + restore — already shaped right.
        cache_layers = _wrap_for_serializer(ev_entry["snapshot"])
        self._ssd_tier.enqueue_spill(tokens, cache_layers, bytes_est)

# In _system_kv lookup (before MISS path):
if (not cache_hit) and self._ssd_tier:
    ssd_match = self._ssd_tier.lookup_prefix(tuple(full_tokens_list))
    if ssd_match:
        # Synchronous promote — 100ms-1.5s instead of 25s recompute.
        snapshot, token_count, token_ids = _promote_from_ssd(ssd_match)
        # Put it into active slot like a HIT.
        self._lru_demote_active_to_bag()
        self._system_kv_snapshot = snapshot
        self._system_kv_hash = system_hash
        self._system_kv_token_count = token_count
        self._system_kv_token_ids = token_ids
        cache_hit = True
```

The `_wrap_for_serializer` adapter is the only new code — everything else reuses PR #309's machinery.

## Sizing the work

Compared to the LRU patch (~170 LOC for the patch alone, plus tests), SSD adds:

- `_wrap_for_serializer` adapter: ~30 LOC
- SSD tier construction in `start()`: ~15 LOC
- Spill on eviction in `_lru_demote_active_to_bag`: ~10 LOC
- Promote on MISS in both stream_chat and `_stream_generate_text`: ~25 LOC
- Reconcile on startup (optional, for "warm restart"): ~40 LOC
- Tests (multi-restart, prefix-match-across-restart, corrupted-entry): ~150 LOC

Total: ~270 LOC. Substantially more than the LRU but still tractable, and it leverages tested SSD machinery so the risk surface is narrow.

## Recommendation

Ship the multi-slot LRU first (already drafted; this branch). Land it, watch a week of opencode workloads to verify no regressions or memory pressure. **Then** add SSD persistence on top — the LRU's `_lru_demote_active_to_bag` is the natural insertion point, so adding SSD is a few-dozen-line follow-up patch with no architectural rework.

Worth doing **before** investing in SSD persistence:

- Verify the in-memory LRU is paying off (cache hit rate at agent-switch boundaries goes from ~50% to >80%).
- Confirm the `enable_thinking=False` ignore bug isn't masking other wins — if the dense and MoE are spending 30+ seconds in reasoning per turn even when asked to skip, no amount of KV caching helps that.

If we ever expand the model lineup to keep many distinct prefixes per session, or if the OOM-crash rate drops to near-zero (heavy-group fix did that) and TTL becomes the dominant restart vector, SSD goes from "nice to have" to "load-bearing for opencode flow continuity."
