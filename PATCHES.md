# Local patches in this fork

This fork carries its patches on top of [`waybarrios/vllm-mlx@caa8838`](https://github.com/waybarrios/vllm-mlx/commit/caa8838) (2026-06-09; previous pins: `015e080`, `395b13c`, `9c83c84`). Each patch is a separate commit on `main` with the prefix `patch:`. They are listed here in apply order (bottom of git log → top).

> **2026-06-09 rebase note — upstream merged PR #540 (`caa8838`) and PR #563, plus #579/#594.**
> - **`caa8838` (#540, SimpleEngine fail-fast admission) collides with our patch #15, which is the corrected port of the same PR.** Upstream merged it **with the env-var clobber bug intact** (validates `VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION`, then unconditionally re-assigns `fail_fast` — `wait` is silently impossible upstream) and with `fail_fast` as default. As with the #541 rebase below, upstream's `engine/simple.py` changes were **rejected wholesale**; ours is a strict superset (env respected, `wait` default, third MLLM-text lock site, pre-stream 503 probe, lock-wait timer). `engine/simple.py` is byte-identical to the pre-rebase tree except for one restored upstream delta (next bullet). `server.py`/`base.py` resolved trivially (our side is a superset of upstream's non-stream-only 503 translation).
> - **`53cfdb0` (#579, SpecPrefill `backbone_pct`)** — its `simple.py` hunks were re-applied in a dedicated `restore:` commit after the rebase (the wholesale rejection above would otherwise have dropped them while `cli.py`/`server.py`/`model_registry.py` threading came through, TypeError-ing at engine construction).
> - **`967d4f3` (#563, ssd-cache bf16 + producer-thread snapshot)** — taken as-is; it fixes the BatchedEngine SSD tier. It does **not** touch the three symbols our patch #16 `system_kv_ssd.py` imports from `ssd_cache.py` (`SSDIndex`, `_blob_to_tokens`, `_tokens_hash`) — verified. It also resolves patch #16's "file the bf16 serializer crash upstream" todo (their fix: f32 fallback at 2× disk; our MLX-native safetensors store keeps dtype-exact bf16 and is unaffected).
> - **`tests/test_simple_engine.py` was adapted to fork semantics** (this rebase). Upstream's #540 admission tests assumed the `fail_fast` default we rejected, the #523/#541 cache-path tests assumed the allowlist probe + exact-hash cache our #9/#12/#13 stack replaces, and one SpecPrefill test asserted the unconditional `mtp=` forward that our patch #17 guards against. Tests were adapted (or skip-marked with reasons) so the suite is green and meaningful against this fork — the previous "tests fail but aren't deployed" caveat no longer applies.

> **2026-05-29 rebase note — upstream PR #541 (multi-slot system-KV LRU) has now merged into our base.** Its `engine/simple.py` changes were **deliberately rejected** during this rebase in favor of our patch #13, which is a strict superset: #541 uses an allowlist probe (`all(isinstance(c, KVCache))`) that gates the cache off for hybrid ArraysCache models — the exact regression our patch #12 denylist probe fixes — plus exact-(hash, count) lookup and **no grow-on-HIT** (our patches #9/#13's core win). After the rebase `vllm_mlx/engine/simple.py` is byte-identical to the pre-rebase tree (verified), so the full cache stack #4/#6/#9/#12/#13 is intact. **Known divergence (resolved 2026-06-09):** `tests/test_simple_engine.py` carried upstream #541's tests, which asserted #541's cache API and failed against our `simple.py`. As of the 2026-06-09 rebase the suite has been adapted to fork semantics (see that rebase note above) and is green.

For Apache 2.0 attribution see [`NOTICE`](NOTICE). For the consumer side of these changes (how they're wired into the homelab) see [`TimotejLabsky/personal-infratructure`](https://github.com/TimotejLabsky/personal-infratructure) — particularly `mac-studio/README.md`, `mac-studio/llama-swap-config.yaml`, and the historical patch scripts in `mac-studio/patches/`.

---

## 1. `04d1c8e` — `patch: bugfixes`

**Files:** `vllm_mlx/utils/tokenizer.py`, `vllm_mlx/server.py`

**Fixes two pre-existing bugs in vllm-mlx 0.3.0:**

- **Missing return in `load_model_with_fallback`** — when `mlx_lm.load()` succeeded (no `ValueError`), the function fell through without returning the `(model, tokenizer)` tuple, causing `TypeError: cannot unpack non-iterable NoneType object` on ALL model loads.
- **Disable strict model-name validation** — upstream rejects requests where `model` field doesn't match the served HF id. Breaks llama-swap, which routes by config key, not HF model id. Patched to early-return from `_validate_model_name()`.

**Upstreaming:** clear bug fixes — both are PR-worthy.

---

## 2. `14e2eb5` — `patch: reasoning-effort`

**Files:** `vllm_mlx/api/models.py`, `vllm_mlx/server.py`

Adds the OpenAI `reasoning_effort` field to the chat completion request schema. Maps `reasoning_effort: "none"` → `chat_template_kwargs.enable_thinking=False`.

Without this, callers that need to disable thinking (e.g., Home Assistant's Extended OpenAI Conversation, opencode's `-fast` route) have no API-level way to do so on Qwen3-family models — the top-level `enable_thinking` field is parsed but doesn't actually suppress `<think>` tokens in 0.2.9+.

The patch routes `reasoning_effort="none"` through the working path. Applies to multiple call sites due to the master-branch `_resolve_chat_template_kwargs` refactor.

**Upstreaming:** good candidate — well-defined OpenAI-spec-compatible field.

---

## 3. `483b0e2` — `patch: stream-finish-reason`

**Files:** `vllm_mlx/engine/simple.py`

Sets `finish_reason="stop"` on the post-loop fallback in `stream_generate`. Without this, streams that end via natural EOS or `max_tokens` (without an explicit `chunk.finished=True`) emit a final chunk with `finish_reason: null` followed by `[DONE]`.

The AI SDK and opencode discard such streams as aborted — symptom is opencode showing "not responding" despite valid content chunks having arrived.

**Upstreaming:** PR-worthy bug fix.

---

## 4. `b3be37c` — `patch: llm-mode-kv-cache`

**Files:** `vllm_mlx/engine/simple.py`

Routes LLM-mode (non-MLLM) text requests through the same `_stream_generate_text` path that already implements system-prompt KV caching for MLLM models. Adds a `_text_route_resources()` helper that falls back to `self._model.model` / `self._model.tokenizer` when no parallel TextModel was built. Widens the gating in both `chat()` and `stream_chat()` so the system-KV snapshot is reused across requests.

**Measured: 52× speedup** on the second request with a 7.5k-token shared system prefix (51.5s → 0.99s).

Upstream PR #523 (`feat: extend system-prompt KV cache to pure-LLM stream_chat path`) added an in-tree equivalent on master, **but its probe gates the cache off for hybrid (ArraysCache) layers**. This patch is still load-bearing for Qwen3.5/3.6/Qwen3-Next models, which would otherwise lose all cache hits.

**Status:** retired for non-hybrid models on master (PR #523), but still needed for hybrid attention architectures we run.

---

## 5. `ba80be6` — `patch: system-kv-hybrid-guard`

**Files:** `vllm_mlx/engine/simple.py`

Adds `VLLM_MLX_DISABLE_SYSTEM_KV=1` env-var kill switch for the system-KV snapshot path. Originally added because hybrid-arch models drifted on snapshot replay (root cause was actually patch #6 — see below). Retained as defense in depth: set the env var per-model to bypass the snapshot mechanism entirely if a future regression appears.

Currently no model needs it. Kept for safety.

**Upstreaming:** unlikely — operational escape hatch specific to our deployment.

---

## 6. `eb18b87` — `patch: system-kv-hybrid-aliasing`

**Files:** `vllm_mlx/engine/simple.py`

Fixes the long-standing hybrid-arch snapshot drift. Root cause: `ArraysCache.state` returns `self.cache` (the same Python list), so `snapshot = [c.state for c in mc]` aliased the cache list into the snapshot. Request 1's continued `cache[i] = new_value` (which is `self.cache[i] = new_value`) overwrote the snapshot in place. Restore had the symmetric problem.

Fix: shallow-copy the list at both capture and restore sites. `KVCache` returns a tuple and is unaffected.

**Verified bit-identical replay** on Qwen3.6-35B-A3B-DWQ at T=0 with a 7.7k-token cached prefix across 6 consecutive requests (`cache_hits=5`, `tokens_saved=38735`). Makes patch #5's env-var workaround unnecessary on production hybrid models.

**Upstreaming:** strong candidate — real bug fix in the snapshot machinery that affects every hybrid-arch user.

---

## 7. `055ee3a` — `patch: system-kv-metrics`

**Files:** `vllm_mlx/engine/simple.py`, `vllm_mlx/metrics.py`

Wires the patched system-KV snapshot path into the `vllm_mlx_cache_*` Prometheus gauges. Adds `_system_kv_hits` / `_system_kv_misses` / `_system_kv_tokens_saved` engine counters, increments them in the HIT/MISS branches of `_stream_generate_text`, populates `stats["system_kv_cache"]` with the field names `metrics.py` reads via `.get()` fallbacks, and adds `system_kv_cache` to `metrics.py`'s cache-type candidate scan + label loop.

Without this, hybrid models silently run the cache (verified 7.86× speedup on a 2.5k-token prefix on Qwen3.6-35B-A3B-4bit) but report `cache_hits=0`, `cache_misses=0`, `cache_tokens_saved=0` and the `cache_type{cache_type="system_kv_cache"}` label is missing entirely.

**Upstreaming:** depends on whether upstream wants to expose hybrid-cache stats; the underlying counters are general enough to PR.

---

## 8. `97b4a6a` — `patch: incoming-reasoning-content`

**Files:** `vllm_mlx/api/models.py`, `vllm_mlx/api/utils.py`

Preserves `reasoning_content` on incoming assistant messages so the chat template can render prior-turn `<think>` blocks when callers set `chat_template_kwargs.preserve_thinking=true`.

Two-part fix:
- Adds `reasoning_content` (with `AliasChoices("reasoning_content", "reasoning")`) to the request `Message` pydantic model in `api/models.py`. Without it, Pydantic silently drops the field at request parse time.
- Propagates `reasoning_content` through `api/utils.py::extract_multimodal_content` (both the tool_calls and simple-text branches) so it survives into `apply_chat_template`.

**Root cause of cycling in opencode on Qwen3.6-35B-A3B-4bit:** the Qwen3.6 chat template strips `<think>` blocks from all assistant turns except the most recent unless `preserve_thinking=true` is set. On a 30-step coder run, the 3B-active MoE lost prior tool-call rationale by step ~5 and re-derived (often differently) the same calls until step cap. We set `preserve_thinking=true` server-side via LiteLLM `extra_body`, but it had no effect because the reasoning was already being dropped on the way in.

Matches upstream-documented fix in HF Qwen3.6-35B-A3B discussions #20 / #51 + opencode issues #24316 / #25129 / #4255.

**Verified:** prompt with a 450-token reasoning trace went from 59 → 517 tokens (full trace preserved end-to-end).

**Upstreaming:** strong candidate. Clear bug (missing field in request schema), fix matches what the community expects.

---

## 9. `68fd27e` — `patch: cache-extended-prefix`

**Files:** `vllm_mlx/engine/simple.py`

Extends the system-KV cache to cover the FULL conversation history (not just the literal `system` role) AND grows the cached prefix on every HIT.

**Background.** Pre-patch testing showed 89% "hit rate" but each hit only saved ~1800 tokens of prefill — every opencode turn re-prefilled the entire conversation history. Two underlying issues:

1. `_stream_generate_text`'s cache logic used a simple `full_prompt.find("<|im_start|>user\n")` to mark the system boundary. It picked the FIRST user message, so the cached prefix never extended past the system role.
2. The Qwen3.6 chat template appends `<think>\n` only to the gen prompt (not prior assistant turns), so naive "cache the full input including gen prompt tail" doesn't survive a turn — the gen-prompt tokens diverge from the next turn's mid-conversation tokens at that position.

**Three-part fix:**

- **(A) MISS branch:** cache the prompt up to the LAST `<|im_start|>assistant\n` marker (the gen prompt). Everything before is stable across turns; the gen-prompt suffix tokens (especially `<think>\n` added by Qwen3.6 with thinking enabled) must NOT be cached.
- **(B) Lookup:** replace `(system_hash + token_count)` equality with longest-prefix-match against the cached `_system_kv_token_ids`. If the new request's `full_tokens_list` starts with the cached tokens, HIT.
- **(C) HIT extension (grow-on-HIT):** after restoring the snapshot, prefill the new content beyond the cached prefix and re-snapshot. The cache grows incrementally with each turn.

**Validated on a 4-turn 6.5K → 8K-token test:**

| Turn | TTFT | Cache event |
|---|---|---|
| T1 (cold) | 37.3 s | MISS — full prefill |
| T2 | **3.3 s** | HIT + GROW (+424 tokens) |
| T3 | **3.4 s** | HIT + GROW (+423 tokens) |
| T4 | **4.5 s** | HIT + GROW (+629 tokens) |

**Real opencode session (audit task, 4 steps):** **86,027 tokens saved** across the session, ~70% of prefill work avoided per turn.

**Compaction handling:** when the new prompt no longer starts with the cached tokens, the prefix check fails → MISS → fresh prefill → cache re-anchored on the new conversation shape. One-time cost at compaction.

**Limits:** hybrid attention models can extend but never truncate (linear-attention layers' state is tied to the exact token sequence). Compaction events therefore can't be partially-matched.

**Upstreaming:** would need polish. The behavior change is significant — opt-in via a flag or runtime switch would be a more conservative PR.

---

## 10. `c4b2577` — `patch: status-system-kv`

**Files:** `vllm_mlx/server.py`

One-line fix: add `stats.get("system_kv_cache")` to the cache fallback chain in the `/v1/status` endpoint. Without this, models using the system-KV cache returned `cache: null` from `/v1/status`, and downstream Prometheus exporters (e.g., `mac-studio-llm-exporter`) couldn't populate cache-hit/miss/saved metrics.

**Upstreaming:** trivial, clear bug. One-line PR.

---

## 11. `30198aa` + `2a27820` — `patch: mx-compile` (+ inner-model traversal fix)

**Files:** `vllm_mlx/compile.py` (new), `vllm_mlx/cli.py`, `vllm_mlx/server.py`, `vllm_mlx/engine/batched.py`

Adds the `--compile` flag wiring `mx.compile(shapeless=True)` around the model forward pass. Off by default. New `compile.py` module with `apply_compile()` and `is_compiled()`; the engines call it after weights load.

**Ported from upstream [PR #270](https://github.com/waybarrios/vllm-mlx/pull/270) (jackneil)**, which is currently DIRTY against upstream main, so the diff was applied by hand. Skipped the docs/tests deltas since our deployment doesn't need them and they'd add churn against the next rebase.

**Why bother:** fuses elementwise Metal kernels in the model forward pass, reducing kernel launch overhead. PR #270 reported +33 % prefill on Qwen3-8B-4bit, decode unchanged (decode is memory-bandwidth-bound on Apple Silicon, not kernel-launch-bound). Effect is per-model — bigger attention shapes may see less benefit.

**Follow-up commit `2a27820`** fixes the inner-model traversal in the SimpleEngine path. The original wiring grabbed `_engine._model`, which on `SimpleEngine` is the `MLXMultimodalLM` / `MLXLanguageModel` *wrapper*, not the MLX `nn.Module`. `mx.compile` threw `'MLXMultimodalLM' object has no attribute '__call__'`; `apply_compile`'s try/except swallowed it and the server logged `compiled` when nothing was actually compiled. The fix walks down to `_engine._model.model`, falls through to bare `._model`, then `._text_model`, and re-checks `is_compiled()` so the success log only fires when the wrap took.

**Verified on mac-studio (2026-05-19):**

- Boot with `--compile`: server starts, `Model forward pass compiled with mx.compile(shapeless=True)` log line confirms wrap.

Two A/B runs on different models, same harness (warmup + 3 timed cold-prefill requests at T=0.0, max_tokens=30):

**Qwen3.5-4B-4bit, 2347-token prompt:**

| Run | --compile off | --compile on |
|-----|---------------|--------------|
| 1   | 2.603 s       | 2.454 s |
| 2   | 2.466 s       | 2.464 s |
| 3   | 2.464 s       | 2.471 s |
| avg | 2.51 s        | 2.46 s |

Decode-dominated workload (≈30 × 60 ms = 1.8 s of decode in a 2.5 s total). Within noise.

**Qwen3.6-27B-4bit (our opencode dense model), 4K-token prompt — prefill-heavy:**

| Run    | --compile off | --compile on |
|--------|---------------|--------------|
| warmup | 31.39 s       | 31.36 s |
| 1      | 30.893 s      | 30.898 s |
| 2      | 30.901 s      | 30.891 s |
| 3      | 30.886 s      | 30.888 s |
| **avg of 3** | **30.893 s** | **30.892 s** |

**Δ = 1 ms (0.003 %)**. Decode in this run is ≈ 30 tok × ~70 tok/s ≈ 0.43 s, so ~99 % of every request is prefill — yet the compile wrap provides **no measurable speedup**. System-KV snapshot was also disabled by the probe on this run (`model returned non-KVCache entries (['ArraysCache', 'KVCache'])`) so every request did full uncached prefill, isolating compile from cache effects.

**Conclusion:** PR #270's published +33 % prefill on Qwen3-8B-4bit does **not transfer** to Qwen3.6-27B-4bit. Likely reasons:

- At 4-bit on a 27B dense model, the elementwise kernel-launch overhead is no longer the bottleneck — Metal command-buffer dispatch is small relative to the per-layer matmul + dequant work.
- The hybrid attention layout (Gated DeltaNet + standard) means a chunk of forward-pass time is in op shapes that don't fuse via `mx.compile(shapeless=True)`.

Patch is left in the fork because it's non-destructive when off, the +33 % on smaller models is real per upstream, and it's the cleanest entry point for future MLX compile improvements. **Don't add `--compile` to llama-swap config for Qwen3.6-27B-4bit (or the 35B MoE without first re-measuring) — there's nothing to gain.**

**How to use:** add `--compile` to a llama-swap entry's `cmd:`. Verify it engaged by grepping the model's stdout/log for `Model forward pass compiled with mx.compile(shapeless=True)`.

**Upstreaming:** the underlying PR is the upstream candidate; our copy is just a port to keep us moving while #270 sits open. The `2a27820` inner-model traversal fix would also need to go upstream as part of the same PR (the original PR's wiring assumed `_engine._model` was already an `nn.Module`).

---

## 12. `5bdc0cb` — `patch: hybrid-probe-denylist`

**Files:** `vllm_mlx/engine/simple.py`

Reframes the SimpleEngine LLM-path system-KV probe from an **allowlist** (every cache entry must be a plain `KVCache`) to a **denylist** (no entry may be a `RotatingKVCache`).

**The bug:** patch #6 (`system-kv-hybrid-aliasing`) already made `ArraysCache` snapshot-safe by shallow-copying lists at capture and restore. But the start-of-engine probe was still using the old allowlist logic — any model that mixed `KVCache + ArraysCache` (Qwen3.6-27B-4bit, Qwen3.5-27B, every Gated DeltaNet hybrid) tripped the allowlist and got the entire snapshot path disabled. Result: opencode workloads on the 27B dense model were doing full uncached prefill on **every turn**, ~30 s of wasted prefill the cache should have saved.

**The fix:** new probe explicitly checks `isinstance(c, RotatingKVCache)` and only disables for those. `RotatingKVCache` (sliding-window — gemma3_text, olmo3, recurrent_gemma) is genuinely unsafe because `.state` aliases in-place-mutated ring buffers, so the snapshot/restore can't capture it without drift. Pure `KVCache` and `ArraysCache` are both safe under patch #6's shallow-copy semantics.

Also added an explicit `snapshot enabled` log line on the success path so a future regression surfaces as a missing log line, not a silent fallback.

**Verified on Qwen3.6-27B-4bit (2026-05-19), 4-turn multi-turn test, 4.2K-token system prompt:**

| Turn | Latency | Cache event | Cached tokens before | Prefilled new tokens |
|------|---------|-------------|---------------------|----------------------|
| T1 (cold) | **25.19 s** | MISS → store 4223-token snapshot | — | 4208 sys + 22 user |
| T2 (+50 tok history) | **1.92 s** | HIT + GROW 4223→4273 | 4223 | 57 |
| T3 (+50 tok history) | **1.91 s** | HIT + GROW 4273→4325 | 4273 | 59 |
| T4 (+60 tok history) | **1.94 s** | HIT + GROW 4325→… | 4325 | 66 |

**13× speedup on warm turns (25 s → 1.9 s).** Patch #9's grow-on-HIT semantics are now alive on the 27B dense: every turn after the first reuses the full conversation cache and only prefills the previous assistant reply + new user message + gen-prompt tail (~60 tokens vs ~4200 cold). This is the model behind opencode for us, so this is the highest-impact patch in the fork for daily use.

**Upstreaming:** strong candidate. The original allowlist was over-conservative; the denylist is the correct semantics once aliasing in `ArraysCache` is fixed (which the underlying patch in PR #523 / upstream HEAD does NOT do — that's patch #6's contribution, also a PR candidate).

---

## 13. `024f58b` — `patch: system-kv-multi-slot-lru`

**Files:** `vllm_mlx/engine/simple.py`

Extends patch #9 from a single active slot to N slots via a side-stash LRU. Capacity defaults to 4 (env var `VLLM_MLX_SYSTEM_KV_SLOTS=N` to tune; `=1` restores the single-slot behavior).

**Design (minimal-invasion):**

- Keep the legacy 4 single-slot ivars exactly as-is (`_system_kv_snapshot`, `_system_kv_hash`, `_system_kv_token_count`, `_system_kv_token_ids`). They describe the "active" slot.
- Add `_system_kv_lru: OrderedDict[hash, dict]` as a side-stash for inactive slots (capacity-1 entries max).
- Two helpers (always inside `_generation_lock`):
  - **`_lru_promote(system_hash)`** — if a matching slot is in the bag, swap it with the active slot. Now the legacy hash-equality checks and grow-on-HIT logic at the existing call sites Just Work.
  - **`_lru_demote_active_to_bag()`** — before overwriting active with a new MISS for a different hash, push the displaced active into the bag. Evicts oldest if (bag + 1) would exceed capacity-1.
- `mx.clear_cache()` only fires on the eviction path (PR #541 optimization), not the common store path.

**Touch points (3 in `stream_chat`, 3 in `_stream_generate_text`):**

- After computing `system_hash` at lookup: call `_lru_promote`.
- Before each MISS store (replacing active for a different hash): call `_lru_demote_active_to_bag`.
- Grow-on-HIT does NOT demote — it extends the same active slot in place.

**TOCTOU:** same gate-time reference-capture contract as patch #9. A concurrent MISS that evicts a slot whose snapshot ref is held in a closure can't corrupt the in-flight restore — Python refcount keeps the list alive even after dict removal.

**Stats backward compat (`/v1/status`, Prometheus exporter):**

- Legacy fields (`tokens`, `hash`) still describe the ACTIVE slot.
- Aggregate fields (`memory_mb`, `current_memory_mb`, `entry_count`) sum over active + bag. `entry_count` is now 0..capacity (was 0/1) — documented semantic shift.
- New fields: `evictions`, `capacity`, `slots: [{hash, tokens, memory_mb, active}, ...]`.

**Validated 2026-05-20 overnight on Qwen3.6-27B-4bit dense, capacity=4:**

| Phase | Pattern | Result | Time / req |
|---|---|---|---|
| 1 | 4 cold MISSes (distinct prefixes) | 4 MISSes | ~24.3 s each |
| 2 | 3 rounds × 4 prefixes, identical query | **12 HITs, 0 MISSes** | **~0.77 s each** |
| 3 | 5th distinct prefix (forces eviction) | 1 MISS, 1 eviction | ~24.4 s |
| 4 | Re-request evicted prefix | 1 cold MISS | ~24.3 s |

Total: 12 HITs / 6 MISSes / 2 evictions / 50,517 tokens saved. **~32× speedup on hits** (24.3 s → 0.77 s). Without the LRU the same sequence would be 18 MISSes — ~7 minutes of cold prefill instead of 12 HITs at ~9 seconds.

**Targets the opencode multi-agent thrash** observed live the same day on the 35B-A3B MoE: 3 distinct `system_hash` values appeared in a single session as opencode dispatched parallel sub-agents, each switch evicting the previous → cold prefill every turn. With capacity=4 default all 3 coexist.

Memory cost: each slot ~430 MB on the 27B-4bit dense (4K-token prompt), ~340 MB on the 35B-MoE. Four slots = ~1.7 GB / ~1.4 GB peak — well within Mac Studio's 64 GB envelope alongside the model weights.

**Design doc:** [`docs/fork/DESIGN-system-kv-lru.md`](docs/fork/DESIGN-system-kv-lru.md).

**Upstreaming candidate:** strong. PR #541 upstream attempts the same direction but starts from PR #523's single-slot system-prefix cache (no grow-on-HIT). This patch is the equivalent layered on patch #9. Open question whether upstream prefers the simpler per-system-prefix slot or our grow-on-HIT semantics — discussion needed.

---

## 14. `33df3be` — `patch: mllm-detect-via-hf-cache`

**Files:** `vllm_mlx/api/utils.py`

Fixes `Qwen3.5-27B-4bit-DWQ` (and any text-only Qwen3.5 variant) failing to load with `Missing 393 parameters: vision_tower.blocks.0.*` on every startup.

**Root cause:** `_try_read_config_json()` only handled local directory paths. When `is_mllm_model()` was called with the HF repo ID (which is what llama-swap passes in the launch command), the function returned `None` unconditionally and `is_mllm_model()` fell through to the legacy `MLLM_PATTERNS` substring matcher. That matcher includes `"qwen3_5"` (added upstream by PR #520 for the actual multimodal Qwen3.5 variants), so DWQ — whose checkpoint contains only text-only weights — got classified as MLLM. The MLLM loader then crashed mid-init with the missing-vision-weights error and the process exited (`exit 1`), which llama-swap surfaced as `process exited but not StateStopping`. 16 historical occurrences of this in our llama-swap log.

**Fix:** when the input looks like an HF repo ID (`owner/repo` form with a slash, not a local dir), resolve it via `huggingface_hub.try_to_load_from_cache(repo_id, filename="config.json")` so the actual config.json gets inspected. The DWQ's config has `architectures: ["Qwen3_5ForConditionalGeneration"]` but **no `vision_config` key** — so `_config_indicates_vlm()` correctly returns False, the MLLM loader is never attempted, and the text-only path loads cleanly.

Defensively wrapped in try/except so the fork keeps working if `huggingface_hub` is missing, the cache is unreachable, or the model isn't yet downloaded — falls through to the existing legacy matcher in those cases.

**Verified end-to-end (2026-05-20):**

- Before: `Loading MLLM: ... → Failed to load MLLM: Missing 393 parameters ... → exit 1`
- After: `Loading model: ... → Model loaded successfully → SimpleEngine loaded (MLLM=False) → Uvicorn running`. `/v1/status` responds normally, system-KV cache enabled with hybrid `[ArraysCache, KVCache]` types.

**Upstreaming:** clean bug fix, PR-worthy. The root issue affects every text-only quantization of an originally-multimodal model (Qwen3.5, Qwen2-VL DWQ derivatives, etc.) — upstream would benefit too.

---

## 15. `patch: simpleengine-busy-admission` — serialized-route admission control (port of upstream PR #540)

**Files:** `vllm_mlx/engine/base.py`, `vllm_mlx/engine/simple.py`, `vllm_mlx/server.py`

Ports [upstream PR #540](https://github.com/waybarrios/vllm-mlx/pull/540) (open, branch `604/simpleengine-admission-telemetry`, closes #495): adds admission control to `SimpleEngine`'s serialized MLX `_generation_lock` so a second concurrent request can fail fast with `EngineBusy` → retryable **HTTP 503** (`error=text_generation_busy`) instead of piling up behind the 120 s-bounded lock under heavy agent traffic.

**What it adds:**
- `EngineBusy(RuntimeError)` (code `text_generation_busy`) in `engine/base.py`.
- An `_acquire_generation_slot(request_id, kind)` async context manager that replaces all **three** of our `async with self._generation_lock:` entry points (`_run_blocking_serialized`, `_stream_generate_impl`, and the MLLM text-only `stream_chat` fallback — upstream only had the first two; the MLLM-text site is our fork's addition from patch #4). Tracks `_generation_waiters`, `_generation_busy_rejections`, and a best-effort `_generation_lock_holder` summary surfaced in `get_stats()["generation_lock"]` and `num_waiting`.
- Server-side: `EngineBusy` → 503 translation on the non-stream `/v1/completions`, `/v1/chat/completions`, and `/v1/messages` (Anthropic) paths, plus a pre-stream `raise_if_serialized_busy()` probe so `fail_fast` can also reject **streaming** requests with a clean 503 before the SSE headers are sent (upstream #540 only covered non-stream).

**Two deliberate divergences from upstream:**
1. **Default `wait`, not `fail_fast`.** Upstream defaults to `fail_fast`. We default to `wait` (legacy serialize-and-queue, **zero behavior change**) because OpenCode and similar agents fire a *title + main* request simultaneously (see the note at `_stream_chat_impl`), and the queue path handles that correctly — `fail_fast`-by-default would 503 one of every such pair. Opt into load shedding per-deployment with `VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION=fail_fast`.
2. **Fixed an upstream bug.** Upstream's `__init__` reads the env var, validates it, then unconditionally re-assigns `self._generation_lock_admission = "fail_fast"` — silently ignoring `wait` and the env entirely. Our port respects the configured value and only falls back to the default on an invalid string.

**Verified (2026-06-02, standalone algorithm replica — full engine import needs the Mac Studio venv):** `fail_fast` rejects the 2nd concurrent acquire with the right code + holder annotation and increments `busy_rejections`; waiter accounting returns to 0; holder clears on release; `wait` mode serializes 3 concurrent requests with no rejection. All three edited modules byte-compile.

**Upstreaming:** upstream merged #540 on 2026-06-09 (`caa8838`) **with the env-clobber bug intact** — `wait` is silently impossible upstream. Filed the same day as a [comment on #540](https://github.com/waybarrios/vllm-mlx/pull/540#issuecomment-4662810334) (env-respect fix + `wait`-default rationale + streaming-503 gap); the one-line fix + regression test sit ready on our [`fix/admission-env-respected`](https://github.com/TimotejLabsky/vllm-mlx/tree/fix/admission-env-respected) branch — upstream PR creation is currently collaborator-restricted, so open the PR if/when that lifts. Our version was kept wholesale during the rebase.

---

## 16. `patch: system-kv-ssd-persistence` — SSD-backed system-KV snapshot (survive restart)

**Files:** `vllm_mlx/system_kv_ssd.py` (new), `vllm_mlx/engine/simple.py`, `tests/test_system_kv_ssd.py` (new)

The SimpleEngine system-KV snapshot (#4/#6/#9/#12/#13) lives only in the serving process: TTL eviction, llama-swap model swap, manual restart, or OOM throws away every warm slot, and the next request pays a full cold prefill (~25 s on a 4 K-token dense prompt, ~70 s on a 13 K-token MoE workload). We bumped `Qwen3.6-27B-4bit`'s `ttl` 600→3600 purely to paper over this. This patch persists snapshots to NVMe so the next process **promotes** a stored prefix (~100 ms–1.5 s disk read) instead of recomputing it.

**Opt-in, off by default.** Enabled per-model via `VLLM_MLX_SSD_SYSTEM_KV_DIR=<base>` (+ optional `VLLM_MLX_SSD_SYSTEM_KV_GB`, default 50). Gated on the same `_supports_system_kv_cache` probe as the in-RAM cache, so RotatingKVCache (sliding-window) models never persist.

**Why a dedicated store, not `ssd_cache.SSDCacheTier` (PR #309) as the DESIGN-doc assumed:** the existing tier serializes via numpy (`np.array(layer.keys)`), which **raises** on MLX `bfloat16` (`Item size 2 ... does not match dtype B item size 1` — verified on the box). Our snapshots are unquantized bf16 KV interleaved with possibly-f32 recurrent `ArraysCache` state, so a numpy/float32 bridge can't round-trip them losslessly either. `system_kv_ssd.SystemKVSSDStore` uses **MLX-native safetensors** (`mx.save_safetensors`/`mx.load`) for dtype-exact data while **reusing the tier's tested `SSDIndex`** (SQLite, prefix-searchable) for metadata.

**How it works (all inside the serialized generation worker, so the LRU/demote contract holds):**
- *Promote (read):* at the top of the `_stream_generate_text` MISS block, before the cold prefill, `lookup_prefix(extended_tokens)` finds the longest stored entry whose tokens are a **prefix** of the request (`num_tokens <= len`), restores it whole into a fresh `make_prompt_cache`, and prefills only the delta forward. Prefix-only ⇒ **never a trim** ⇒ hybrid-safe (recurrent state restored at a boundary, exactly like the in-RAM path).
- *Spill (write-through):* on a fresh MISS store, `enqueue_spill` writes the new prefix asynchronously (background writer thread). One write per distinct system prefix at creation; the grow path does **not** re-spill (a restart promotes the stored prefix and re-grows cheaply). Write-through (not eviction-triggered, as the DESIGN doc proposed) so the resident working set survives a SIGKILL from llama-swap, not just graceful eviction.
- *Lifecycle:* store built in `start()`, drained + closed in `stop()`. Capacity-bounded LRU eviction on disk.

**Verified (2026-06-07, Mac Studio):** `tests/test_system_kv_ssd.py` — **bit-identical round-trip** of a hybrid snapshot (bf16 KV tuples + f32 recurrent `ArraysCache` lists) through disk; prefix-lookup semantics (superset query hits, shorter query never matches); async-writer round-trip; capacity eviction; corrupt-entry quarantine + de-index. All pass. Both modules `py_compile` clean.

**End-to-end A/B verified (2026-06-07, Mac Studio, real `Qwen3.6-27B-4bit`, out-of-band on :8123, MLLM-text-routed):** 3,718-token prompt.
- **Cold prefill (first request): 36–39 s.** Spills a 397 MB snapshot to disk asynchronously.
- **Kill serve (simulate restart) → relaunch → same prompt: 1.28–1.51 s** — `[system_kv_ssd] promoted ... (397 MB, 63 ms)` disk read, `restored 3711 tokens, prefilling 0 delta`. **~26–28× faster than cold.**
- **Correctness:** promoted output is **byte-identical** to cold at T=0 (`sha=483607cd…` matched); subsequent in-RAM HIT (0.70 s) confirms the promoted slot grows normally.
- Bugfix found during the A/B: the store must gate on `_is_system_kv_safe()`, **not** `_supports_system_kv_cache` — the latter is only set on the non-MLLM probe path, but Qwen3.6-27B loads `MLLM=True` and routes text through the LLM path, so gating on it skipped SSD init entirely.

**Upstreaming:** strong candidate — brings persistence to the pure-LLM path. ~~The bf16 numpy-serializer crash is worth a standalone upstream bug report regardless.~~ **Resolved upstream 2026-06-09 by PR #563 (`967d4f3`)** — they fixed the BatchedEngine tier with an f32-fallback cast (2× disk for bf16); our MLX-native safetensors store remains dtype-exact and is unaffected (verified #563 doesn't touch the `ssd_cache.py` symbols we import).

---

## 17. `patch: perf-observability-and-guards`

**Files:** `vllm_mlx/engine/simple.py`, `vllm_mlx/server.py`, `vllm_mlx/cli.py`, `vllm_mlx/bench_serve.py`, `tests/test_bench_serve.py`

A batch of low-risk observability fixes + correctness guards from a multi-agent, adversarially-verified performance study (2026-06-08/09). The study's honest top-line: the system-KV prefix cache already captured the big win (13–32×), and decode is at the 4-bit memory-bandwidth floor (`mx.compile` measured 0% — patch #11), so the real near-term opportunities are (a) making the cache *observable* and (b) closing silent-degradation traps. This patch ships those; the larger bets (native-MTP-on-SimpleEngine port, BatchedEngine-for-MoE A/B) are tracked in Future work.

**Observability fixes:**

- **`bench_serve` cache metrics read the wrong name.** `parse_metrics_text` greped `vllm_prefix_cache_{hits,misses,tokens_saved}_total` — names the server emits *nowhere*. The server exposes `vllm_mlx_cache_*` gauges (`metrics.py`). Every bench run silently recorded `hit_rate=0, tokens_saved=0` into its SQLite results even on a genuinely ~89%-hit agent workload. Fixed the three names + made `_extract` tolerate float gauge values; updated the mirrored test fixture.
- **`system_kv_cache.hit_rate` was never emitted on the production path.** `metrics.py` reads a `hit_rate` key, but the only one populated was on the MLLM `memory_aware_cache` block — absent for the non-MLLM / LLM-route path the heavy models actually use, so the `vllm_mlx_cache_hit_rate` gauge sat at a constant 0. Added the one-line field to `get_stats`'s `system_kv_cache` dict.
- **Per-request lock-wait timer.** TTFT conflated queue-wait + cold-prefill + warm-hit into one unlabeled number. Added a `perf_counter` wait timer in `_acquire_generation_slot`, exposed as `generation_lock.{wait_count,wait_avg_ms,wait_max_ms}` — the one TTFT component not derivable from hit/miss counts. `/v1/status` previously dropped the whole `generation_lock` block (and the `mllm_text_route_degraded` flag); both are now surfaced there too.
- **`VLLM_MLX_DEBUG_PROMPT_CAPTURE=1`** (server.py, OpenAI path) dumps the *untruncated* system prefix + tool-list ordering, so two consecutive opencode turns can be diffed to confirm/deny volatile content (timestamps, session ids, reordered tools) defeating the longest-prefix HIT — resolves the canonicalization findings empirically instead of by guess. (Existing `VLLM_MLX_DEBUG_MESSAGES` truncates content; the `[REQUEST]` dump is Anthropic-only.)

**Correctness guards:**

- **Loud MLLM text-route degradation.** When `build_text_model` returns `None` or raises (swallowed before), an `_is_mllm` model silently falls back to the cacheless `mlx_vlm` path — full cold prefill (~25–70 s) *every turn*, with the 13–32× system-KV win invisibly gone. Now logs a loud warning and surfaces `mllm_text_route_degraded` in `/v1/status`. `--text-only` remains the manual mitigation (forces the pure-LLM route).
- **SpecPrefill + native-MTP TypeError guard.** The SpecPrefill content-phase resume passed `mtp=use_mtp` unconditionally to `mlx_lm.stream_generate`. Verified against the deployed **mlx_lm 0.31.3**: `stream_generate` takes `**kwargs` but forwards to `generate_step`, which has *no* `**kwargs` and *no* `mtp` param — so the kwarg raises `TypeError` downstream (latent on every SpecPrefill request, not just when MTP is co-enabled). Now forwards `mtp` only when `inspect.signature(stream_generate)` actually exposes it (version-agnostic; degrades with a warning otherwise). The regular text route signals MTP via `num_draft_tokens` and never hits this site.
- **Honest MTP banner.** `cli.py` printed "MTP: enabled (native speculative decoding)" under SimpleEngine, where native mlx_lm MTP self-speculation is inert (the text route calls `stream_generate` with no `draft_model`; effective draft depth 1, matching the engine's own runtime warning). Reworded to say so.

**O(n²) hygiene:**

- The per-token stop-string scan re-scanned the entire accumulated output every token (`any(seq in accumulated_text ...)`). Bounded it to the tail window that could contain a *newly* completed stop sequence (`len(new_text)+max_stop_len-1` chars). Semantically identical (earlier matches were already caught on prior iterations); O(len(new_text)) per token. **Hygiene, not a tok/s claim** — it runs on the asyncio consumer, off the GPU critical path.

**Verified:** all five files `py_compile` clean; the `mtp` guard and the bf16/`mtp` signature facts confirmed against the installed mlx_lm 0.31.3; instrumentation validated end-to-end on an M1 Pro with a local model (hit_rate / lock-wait / prompt-capture fields populate, bench reads nonzero cache stats). Perf A/Bs that need the production 27B/35B at scale run on the Mac Studio.

**Upstreaming:** the bench-metric-name fix, the `hit_rate` field, and the SpecPrefill `mtp` TypeError guard are clean, isolated bug fixes — all PR-worthy. The debug-capture flag and lock-wait timer are general-purpose observability.

---

## 18. `refactor: system-kv-module-extraction`

**Files:** `vllm_mlx/system_kv.py` (new), `vllm_mlx/engine/simple.py`

**Pure code motion, zero behavior change.** Extracts the system-KV snapshot cache stack (patches #4/#6/#9/#12/#13 state + helpers + probe + stats, and patch #16's SSD-store lifecycle) out of `engine/simple.py` into a `SystemKVManager` class in `vllm_mlx/system_kv.py` — the same containment pattern patch #16 proved with `system_kv_ssd.py`. Motivation: `engine/simple.py` is the file upstream churns hardest; before this refactor our diff there was ~1,100 lines and every rebase conflicted in the cache regions. After: simple.py shrinks by 213 lines (3232 → 3019), the cache internals live in a fork-owned module upstream never touches, and the remaining simple.py surface is thin orchestration calls plus a stable delegation block.

**What moved:** all slot/LRU/counter state, `lru_promote`/`lru_demote_active_to_bag`, the `VLLM_MLX_DISABLE_SYSTEM_KV` kill switch, the start()-time RotatingKVCache denylist probe, SSD store start/drain/close, the stop() reset, the full `get_stats()["system_kv_cache"]` assembly, and the gate/store bookkeeping (`lookup_active`, `match_extended_prefix`, `record_hit`, `store_snapshot`, `store_extended`). Log strings byte-identical (the "snapshot enabled" canary line and friends).

**What stayed in simple.py:** the generation-interleaved code — grow-on-HIT closure, MISS prefill loop, SSD promote-from-disk restore (they drive `model(...)`/`mx.eval` mid-worker) — and the request-site log lines that format request-local values.

**Compatibility surface:** `SimpleEngine` keeps the full legacy attribute API via 13 property+setter pairs (`_system_kv_snapshot`, `_system_kv_lru`, counters, `_supports_system_kv_cache`, `_ssd_store`, …) plus one-line delegating methods (`_lru_promote`, `_lru_demote_active_to_bag`, `_is_system_kv_safe`). The TOCTOU gate-time contract is unchanged: snapshots are captured under the generation lock (`lookup_active`) and passed explicitly into worker closures.

**Verified:** full suite at exact pre-refactor baseline (2173 passed, 12 skipped, 0 failed) with **zero test edits** — the suite adapted on 2026-06-09 pins these semantics (TOCTOU, denylist probe, counters, stats shape) and was used as the behavioral oracle.

**Rebase impact:** upstream changes to simple.py's cache regions should now be rejected as before, but conflicts will be smaller and rarer; `system_kv.py` is fork-owned and conflict-free by construction.

**Upstreaming:** n/a (fork infrastructure), though the module boundary would make the eventual #6+#12 upstream PR easier to cut.

---

## 19. `patch: system-kv-partial-restore` — checkpointed partial-prefix restore (divergent chains)

**Files:** `vllm_mlx/system_kv.py`, `vllm_mlx/system_kv_ssd.py`, `vllm_mlx/ssd_cache.py`, `vllm_mlx/engine/simple.py`, `vllm_mlx/metrics.py`, `tests/test_system_kv_partial.py` (new), `tests/test_system_kv_ssd.py` (adapted)

The system-KV cache (#4/#9/#13/#16) restores only **exact prefix extensions**: when a new prompt shares the first D tokens with a cached chain but diverges after (opencode compaction, retried/edited turns, identical-request resends, interleaved sessions on one system prompt), the whole warm state is useless and the request pays a full cold prefill. This is the one workload shape where vLLM-style paged/block caches beat the snapshot design — but blocks can't hold recurrent (`ArraysCache`) state, which is why mlx-lm's own prefix cache refuses hybrids entirely (mlx-lm#980). This patch closes the gap with the llama.cpp #19408 idea adapted to our snapshot architecture: **position-indexed recurrent checkpoints + trimmable attention KV.**

**Key insight:** only recurrent layers need per-position checkpoints. Attention KV at any position p is recoverable from the final snapshot by slicing `keys[..., :p, :]` — `KVCache.state`'s setter re-derives `offset` from the shape, so a sliced assignment is position-exact. So a checkpoint is just the (fixed-size) recurrent-layer states at a chunk boundary — cheap. Pure-attention models need **no checkpoints at all**: any divergence point is restorable by slicing alone.

**How it works:**
- *Capture:* during the MISS cold-prefill loop and the grow-on-HIT loop, after each prefill chunk, shallow-copy the list-state (recurrent) layers (`capture_recurrent_states` — same rebind-not-mutate aliasing discipline patch #6 verified bit-identical) into `{"pos", "states"}` checkpoints. Bounded per slot by `VLLM_MLX_SYSTEM_KV_CHECKPOINTS` (default 8) with drop-every-other geometric thinning. The grow path captures lazily (references only) to preserve the profiled eval-only-at-end pattern.
- *Plan (gate time):* when `match_extended_prefix` fails, `plan_partial_restore` computes the divergence point D vs the active slot and hands gate-time references into the worker (same TOCTOU pattern as `lookup_active`). Floor: `VLLM_MLX_SYSTEM_KV_PARTIAL_MIN` (default 256 tokens).
- *Restore (worker MISS block):* ordered candidates by restore position — in-RAM partial, SSD full-prefix promote (#16, unchanged semantics), SSD shared-prefix partial — first clean apply wins, failures fall through to the next source and ultimately cold prefill. Restore = sliced attention KV + checkpoint recurrent state, prefill forward from there. **Never trims `ArraysCache`** — restores only at positions where recurrent state was captured (avoiding exactly the trim-path crash that killed the parked trimmable experiment). `d == donor_len` fast path: the snapshot itself is the checkpoint (identical-request resend).
- *SSD format v2:* checkpoints ride along in the entry's safetensors (`c{n}_l{i}_s{j}` keys) + meta; v1 entries read back with `checkpoints == []`. New `SSDIndex.lookup_shared_prefix` finds divergent stored entries via the existing `prefix_hash` column (first-16-token hash) + exact common-prefix from the token blob — so partial restore works **across restarts** too.
- *Observability:* `partial_hits` / `partial_tokens_saved` in `system_kv_cache` stats + new `vllm_mlx_cache_partial_{hits,tokens_saved}` Prometheus gauges; restored tokens also count in the aggregate `tokens_saved`.

**Verified (2026-06-10, M1 Pro):** unit suite `tests/test_system_kv_partial.py` (13 tests: helpers, real-KVCache trim offset semantics, aliasing, per-slot checkpoint LRU, SSD v2 round-trip bit-identical, v1 back-compat, shared-prefix lookup); full suite 2186 passed / 0 failed. End-to-end on a real served model (`Qwen3-0.6B-8bit`, T=0): divergent chains hit both live paths — `PARTIAL restore (RAM): 54 of 54 shared tokens, prefilling 16` and, after a process restart with empty RAM, `PARTIAL restore (SSD): 54 of 54 shared tokens from divergent entry, prefilling 18`. All three warm outputs (RAM partial, restored-chain continuation, SSD partial) **byte-identical** to a cache-disabled cold server. Hybrid checkpoint capture/restore is unit-tested; live hybrid A/B runs on the Mac Studio at deploy (no small hybrid model on the dev box).

**Upstreaming:** the strongest candidate yet — upstream has no partial-prefix story for the pure-LLM path and nothing at all for hybrids.

---

## 20. `patch: template-family-prefix-markers` — extend the system-KV cache beyond ChatML

**Files:** `vllm_mlx/system_kv.py`, `vllm_mlx/engine/simple.py`, `tests/test_system_kv_partial.py`

The extended-prefix cache's two prompt anchors — the system-prefix boundary (first user-turn marker) and the generation-prompt boundary (rfind of the gen marker) — were hard-coded ChatML strings (`<|im_start|>user\n` / `<|im_start|>assistant\n`). Every non-ChatML family silently bypassed the entire caching stack (#4/#9/#13/#16/#19): **measured 2026-06-10 on gemma-4-31b (warm TTFT 24.2 s == cold 23.7 s) and gemma-4-26b-a4b (3.6 s == 3.6 s)** — identical re-sent prompts paid full prefill every turn, and the SSD env was inert.

Replaced with a `TEMPLATE_MARKERS` table + `detect_template_markers()` in `system_kv.py`, detected once per request from the rendered prompt. Families and markers were **verified against each deployed model's actual `apply_chat_template` output**, not guessed:

| Family | Boundary marker | Gen marker | Models in lineup |
|---|---|---|---|
| chatml | `<\|im_start\|>user\n` | `<\|im_start\|>assistant\n` | all Qwen (unchanged — byte-identical behavior) |
| phi4 | `<\|im_start\|>user<\|im_sep\|>` | `<\|im_start\|>assistant<\|im_sep\|>` | Phi-4 reasoning/mini |
| gemma4 | `<\|turn>user\n` | `<\|turn>model\n` | gemma-4-31b, gemma-4-26b-a4b (NEW format — not Gemma 3's `<start_of_turn>`) |
| llama3 | `<\|start_header_id\|>user<\|end_header_id\|>` | `…assistant…` | Nemotron-Super-49B |
| glm4 | `<\|user\|>` | `<\|assistant\|>` | GLM-4.7-Flash |
| harmony | `<\|start\|>user<\|message\|>` | `<\|start\|>assistant` | gpt-oss-20b |
| mistral | `[INST]` | `[/INST]` | Devstral, Mistral-Small-3.2 (gen boundary = last `[/INST]`, which holds a stable position across turns) |

Unknown templates fall back to the uncached path exactly as before. The existing `prefix_valid` token-prefix validation remains the safety net for any family whose tokenizer breaks the text-prefix ⇒ token-prefix assumption.

**Verified:** marker unit tests against real rendered-template fixtures (boundary contains the system content, never leaks the user turn; gen marker anchors the final generation prompt); full suite 2187 passed / 0 failed. Live A/B on gemma-4 recorded in the infra deploy table.

**Upstreaming:** pairs with #19 — together they make the pure-LLM cache path template-portable.

---

## 21. `patch: rotating-safe-snapshots` — meta_state-aware caching + arch-agnostic MLLM text route

**Files:** `vllm_mlx/system_kv.py`, `vllm_mlx/system_kv_ssd.py`, `vllm_mlx/engine/simple.py`, `vllm_mlx/text_model_from_vlm.py`, `tests/test_system_kv_partial.py`

Completes the Gemma 4 caching fix that #20 (template markers) started. Two independent blockers remained, found by walking the live path on the box:

**(a) `build_text_model` hard-coded the Qwen3.5 TextModel class.** Feeding gemma-4's `text_config` into `qwen3_5.TextModelArgs` crashed ("float division by zero"), and the caller's None fallback silently routed all gemma text requests through the **cacheless mlx_vlm path** (caught by patch #17's loud-degradation warning). Now resolves the mlx_lm class dynamically from `text_config.model_type` (gemma4 → `mlx_lm.models.gemma4_text`); the Qwen3.5/3.6 family keeps its MTP-capable path. A weight-name **coverage guard (≥90%)** fails closed if a vlm/mlx_lm namespace mismatch would otherwise produce a silently half-loaded model. Side benefit: gemma keeps vision (no `--text-only` needed).

**(b) Snapshots only captured `.state` — RotatingKVCache needs `meta_state` too.** The sliding-window ring's indices (`keep, max_size, offset, _idx`) live in `meta_state`; state-only restore desynchronizes the ring (the original reason patch #12 denylisted these models — and a **latent text-route hazard**: that probe only ever gated `stream_chat`, so a `--text-only` Rotating model would have cached unsafely). Snapshots, checkpoints, and SSD entries (format v3) now carry per-layer `meta_state` + a **kind** classification (`trim` / `ckpt` / `opaque`) captured from the live cache classes — state shape alone cannot distinguish a Rotating tuple from a trimmable KVCache tuple:

- `trim` (plain KVCache): partial restore slices to any position, as in #19.
- `ckpt` (ArraysCache recurrent + RotatingKVCache sliding-window): restorable only at checkpoint positions, with meta applied after state (mirroring mlx-lm's own `save_prompt_cache`/`from_state` round-trip).
- `opaque` (e.g. QuantizedKVCache): whole-snapshot restore round-trips via state+meta; partial restore refuses.

v1/v2 SSD entries and pre-existing in-RAM slots load with `meta/kinds = None` and fall back to shape-based classification — correct for everything that could have produced them (Rotating models could not cache before v3).

**Verified (2026-06-10, M1 Pro):** real-RotatingKVCache round-trip test — restore past the rotation point continues **bit-identically** to the uninterrupted cache, plus a negative control documenting that state-only restore desyncs (`offset/_idx` mismatch). Full suite 2190 passed / 0 failed. Live gemma-4 A/B recorded in the infra deploy table.

**Upstreaming:** (a) is a clean bug fix — strong PR candidate. (b) extends the #19/#20 series; together they make the pure-LLM cache path cover every mlx-lm cache class except quantized partial-restore.

---

## Future work / prospects

Open upstream PRs/issues worth tracking — not yet applied here, with the reasoning:

- **[PR #541](https://github.com/waybarrios/vllm-mlx/pull/541) — multi-slot LRU for system-KV. MERGED upstream (commit `1656c15`), now in our base as of the 2026-05-29 rebase.** Its `simple.py` changes are superseded by our patch #13 (rejected during the rebase — see the rebase note at the top of this file). #541's version starts from PR #523's single-slot cache with no grow-on-HIT, and re-introduces the allowlist probe that gates off hybrid ArraysCache models (which our patch #12 denylist fixes). Our #13 is a strict superset, so we keep ours. If upstream's structure later diverges in a way worth adopting, reconciliation would mean re-layering grow-on-HIT (#9) + denylist probe (#12) + longest-prefix-match (#9) on top of upstream's OrderedDict — non-trivial; defer until there's a concrete upstream improvement to fold in.

- **SSD persistence for system-KV snapshot — IMPLEMENTED as patch #16** (see above). Design notes in [`docs/fork/DESIGN-system-kv-ssd.md`](docs/fork/DESIGN-system-kv-ssd.md); note the implementation diverges from the doc on two points the build surfaced: (1) MLX-native safetensors instead of the numpy `ssd_cache` serializers (they crash on bf16), and (2) write-through-on-store instead of spill-on-eviction (survives SIGKILL). Remaining: end-to-end real-model A/B in an idle window, then a week of metrics before considering whether to drop the `ttl` 3600 workaround.

- **[PR #233](https://github.com/waybarrios/vllm-mlx/pull/233) — TurboQuant KV cache compression (4.6×).** **CLOSED upstream without merging (checked 2026-06-09)** — dropped from tracking. If KV-compression pressure returns (more agent prefixes than the 64 GB envelope holds), re-evaluate whatever upstream's then-current approach is rather than resurrecting this branch.

- **[PR #528](https://github.com/waybarrios/vllm-mlx/pull/528) — canonicalize volatile system headers. MERGED upstream (`177f0bf`) and already in our base** (checked 2026-06-09). Strips `x-anthropic-billing-header` rotating tokens from Chat Completions + Responses system prompts. **Remaining open question is now empirical:** run one opencode session with `VLLM_MLX_DEBUG_PROMPT_CAPTURE=1` (patch #17) and diff two consecutive turns' system prefixes to confirm no *other* volatile content (timestamps, session ids, tool reordering) is defeating longest-prefix HITs.

- **[Issue #502](https://github.com/waybarrios/vllm-mlx/issues/502) — DFlash speculative decoding for Qwen 3.5 / 3.6.** External block-diffusion draft model + verification against the target. Different shape from native MTP. Not implemented yet upstream; if it lands as a distinct backend, evaluate against our 27B-4bit dense + 35B-A3B MoE workloads.

- **[Issue #508](https://github.com/waybarrios/vllm-mlx/issues/508) — adaptive idle polling.** 1 kHz step loop when scheduler is empty is wasted CPU. Not throughput-relevant on a dedicated Mac Studio, but worth picking up if a clean PR lands.

PRs evaluated and rejected for our workload:

- **PR #424** (sampling defaults + short prefix-cache reuse fix) — touches `MemoryAwarePrefixCache` (BatchedEngine paged-cache path). We use SimpleEngine for the heavy models, so the fix doesn't bite us.
- **PR #449** (O(1) tool lookup in MCPExecutor) — relevant only with many MCP tools registered. We currently expose a small set (`mcp-search` DuckDuckGo proxy); the O(N) scan is not measurable here.

---

## Maintenance

### Rebase on upstream

```bash
git fetch upstream
git rebase upstream/main          # replays our patches on top of latest upstream
git push --force-with-lease       # update our fork
```

Conflicts that come up are usually in the same files (especially `vllm_mlx/engine/simple.py` which sees a lot of upstream churn around the cache code). Resolve in favor of preserving our patch intent; rerun smoke tests after.

### When a patch lands upstream

If upstream merges an equivalent fix, drop the corresponding commit:

```bash
git rebase -i upstream/main
# Remove or comment out the `pick <sha> patch: <name>` line for the upstreamed patch
```

Update this PATCHES.md accordingly (move the entry to a "Retired" section or just delete it with a note in the commit message).

### Edit an existing patch

```bash
# Edit the relevant file(s)
git commit --fixup <patch-commit-sha>
git rebase --interactive --autosquash <patch-commit-sha>^
git push --force-with-lease
```

The fixup commit gets squashed into the original patch commit during the autosquash rebase.

### Add a new patch

Make the change, commit with the `patch:` prefix:

```bash
git commit -m "patch: <short-name>"
# Optionally update PATCHES.md and NOTICE
git push origin main
```

### Test an unmerged upstream PR

```bash
git fetch upstream pull/NNN/head:pr-NNN
git rebase pr-NNN                 # apply our patches on top of the PR's branch
# Test locally, push a side branch if you want to share / CI it
```

If the PR breaks our patches (touches the same code), the rebase will surface the conflicts. Useful for catching upstream regressions early or vetting third-party contributions before they land.
