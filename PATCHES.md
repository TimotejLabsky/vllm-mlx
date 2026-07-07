# Local patches in this fork

This fork carries its patches on top of [`waybarrios/vllm-mlx@0dd1157`](https://github.com/waybarrios/vllm-mlx/commit/0dd1157) (`v0.4.0`, 2026-06-29; previous pins: `a48c86c`, `caa8838`, `015e080`, `395b13c`, `9c83c84`). Each patch is a separate commit on `main` with the prefix `patch:`. They are listed here in apply order (bottom of git log → top).

> **2026-06-29 rebase note — rebased onto upstream `0dd1157` (`v0.4.0`, 13 commits past `a48c86c`); suite green at 2293 passed / 29 skipped / 23 deselected.** Policy this rebase: keep all fork functionality; where upstream converged on a fix we already carry, take upstream's version *unless* it would regress a fork semantic.
> - **Admission (#615) — upstream converged on our env-respecting fix; we keep our superset.** `9a75b07` (#615) fixes the env-var clobber bug the 2026-06-09 note flagged: the unconditional `self._generation_lock_admission = "fail_fast"` is now correctly indented into the invalid-input branch, so a valid `VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION=wait` survives. **The old "upstream merged with the env-clobber bug intact / `wait` is silently impossible upstream" warning (2026-06-09 note below) is now STALE.** Our patch #15 remains a strict superset — default `wait` (upstream defaults `fail_fast`) plus the third MLLM-text lock site, pre-stream 503 probe, and lock-wait timer — so per "keep our functionality" we keep ours (taking upstream wholesale would regress the `wait` default our tests assert). The rebase auto-merged upstream's now-fixed init block *alongside* ours (no textual conflict — different insertion point), producing a duplicate init block; the redundant upstream copy was removed and folded into patch #15. Net: one admission init block, `wait` default, env respected.
> - **Patch #26 (#597 cherry-pick) RETIRED.** Upstream `490cca0` (#597) is now in the base. Both our commits (`3639b7c` parse + `521f095` review follow-up) had **byte-identical change sets** to upstream's merged version (verified via sorted-hunk diff of `qwen3_xml_tool_parser.py`), so the conflict was resolved to upstream and both commits auto-collapsed/dropped during the rebase. Bare-`<function=>` parsing is now upstream's. The `## 26` section below is historical.
> - **Net-new upstream features gained** (landed clean — our patches don't touch these files): `edbb10f` (#581, gpt-oss harmony routing — new `utils/harmony_render.py` + simple.py/server.py hooks), `11b1359` (#623, gemma4 unfenced `tool_code = fn(...)` parsing), `523297c` (#625, gemma4 streaming tool marker from reasoning — refactors the inline `<tool_call>`/`<invoke` check into `_streaming_tool_markup_possible()`; coexists with our patch #27 fold-reasoning logic, verified different blocks), `f32670b` (#616, qwen3_5_mllm accepts current attention kwargs in batch patch), `addbf53` (#611, keep `tool_calls`/tool messages on MLLM chat path), `96b86c0` (#591, auto-extract audio from `video_url` on omni models).
> - **`e2d3d95` (#614, realize private lazy arrays before model leaves build thread)** — its `module.values()` build-thread `mx.eval` landed in `text_model_from_vlm.py` and **coexists with patch #21's warmup forward** (both materialize lazy arrays to dodge the cross-thread `Stream(gpu, N)` crash; ours additionally serves as a kernel-path warmup + refusal guard — complementary, both kept). Its new `test_build_text_model_realizes_private_lazy_arrays` is **skip-marked fork-hygiene**: the test's `FakeVlmModel` exposes an empty `parameters()`, so the fork's weight-name coverage guard refuses the build (0% coverage) before the realize block runs; the behavior is covered by the fork's gemma text-route tests with real weights.
> - **BatchedEngine improvements taken as-is** (we run SimpleEngine + system-KV, but adopted toward a possible future BatchedEngine investment): `201a8c2` (#620, honor MLX buffer cache limit), `28015c9` (#618, SSD cold tier on the MLLM prefix cache via `batched.py`/`mllm_scheduler.py`), `a97560d` (#612, ssd-cache preserve bf16 dtype across quantized spill). #612/#618 touch the BatchedEngine `ssd_cache.py`/MLLM tiers, **not** our MLX-native `system_kv_ssd.py` — no conflict (same separation noted for #605/#563 in earlier rebases).

> **2026-06-14 rebase note — rebased onto upstream `a48c86c` (15 commits past `caa8838`); suite green at 2224 passed / 16 skipped.**
> - **Upstream has converged onto our cache design.** Two of our patches are now upstream-equivalent and were adopted/retired rather than re-applied wholesale:
>   - **`59b43c4` (#576, "Fix hybrid cache snapshot aliasing") == our patch #6.** Upstream now ships `_snapshot_prompt_cache`/`_restore_prompt_cache` with the same shallow-copy semantics; the conflict call sites were resolved to upstream's helpers. (The active restore path for the meta_state-aware route is still patch #20's `apply_snapshot_states` — strictly more than #576.)
>   - **Upstream's `_probe_system_kv_cache_support` == our patch #12 denylist.** Verified equivalent: `RotatingKVCache` is **not** a `KVCache`/`ArraysCache` subclass, so upstream's `isinstance(c, (KVCache, ArraysCache))` allowlist enables hybrid ArraysCache and disables sliding-window — identical to our denylist. **The old "upstream allowlist gates off hybrids" warning (from the #541 notes below) is now STALE.** Patch #18's refactor routes this through `SystemKVManager.probe_snapshot_support`, which keeps the same behavior.
> - **Patch #23 (#606 cherry-pick) RETIRED.** Upstream `527f457` (#606) is now in the base. It did not auto-drop because our placement differs deliberately (after `inject_mtp_support`, before patch #21's warmup, so the warmup runs on the kernel path) — leaving a redundant double `train(False)`. The redundant upstream copy (just before `return`) was removed; the single load-bearing pre-warmup call is kept. The `## 23` section below is historical.
> - **Net-new upstream features gained** (no fork conflict): `a48c86c` (#599, Mistral/Ministral `[THINK]` reasoning parser — candidate to wire on Devstral/Mistral), `15b5bc3` (#596, chat logit bias), `7da8f2d` (#550, XLM-RoBERTa reranker heads), `714f3ac`/`a034853` (#548/#547, constrained non-stream→stream routing + JSON whitespace bound), `b27f89e` (#589, requires **mlx-vlm ≥ 0.6.2** — bump on the Studio at deploy; the 0.5.0 there is now below the floor).
> - **Other overlapping upstream commits** resolved to keep our superset: `02b631b` (#607, per-layer quant) and `90f759a` (#595, gemma4 text dispatch) overlap #20/#21 — our manual dispatch kept (upstream's `_import_text_model_classes` helper sits unused in `text_model_from_vlm.py`, a future-cleanup candidate). `b67edee` (#605, ssd-cache bf16) only touches the BatchedEngine `ssd_cache.py` tier, not our MLX-native `system_kv_ssd.py` — taken as-is.
> - **4 new upstream tests skip-marked as fork-hygiene** (`test_simple_engine.py` ×3, `test_text_model_from_vlm.py` ×1): their fixtures assume upstream internals (single-prefill cache, direct `self._model.stream_generate` route, `_supports_system_kv_cache` gating, `_import_text_model_classes` dispatch). Each skip carries a reason; the behaviors are covered by the fork's own green tests.

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

> **Status (2026-06-14 rebase): UPSTREAM-CONVERGED.** Upstream `59b43c4` (#576) now does the same shallow-copy via `_snapshot_prompt_cache`/`_restore_prompt_cache`; those helpers are used at the plain restore call sites. The meta_state-aware route (patch #20 `apply_snapshot_states`) remains the superset.

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

> **Status (2026-06-14 rebase): UPSTREAM-CONVERGED.** Upstream's `_probe_system_kv_cache_support` (`isinstance(c, (KVCache, ArraysCache))`) is functionally identical — `RotatingKVCache` is not a `KVCache`/`ArraysCache` subclass, so hybrid ArraysCache stays enabled and sliding-window stays disabled. Patch #18 routes this through `SystemKVManager.probe_snapshot_support` (same behavior). The earlier "upstream allowlist gates off hybrids" warning is stale.

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

**Spill scheduling fix (2026-06-12, defer-until-idle):** write-through spills of deep-session snapshots (an 80K-token 27B chain is ~5 GB) used to serialize + write on the writer thread WHILE the next generation ran — unified-memory spikes and I/O degraded decode and contributed to a Metal abort under tight memory. The store now takes an ``idle_check`` callable (engine passes a generation-lock probe) and the writer holds heavy work until no generation is in flight; ``close()`` flips to drain mode so shutdown never deadlocks on a busy engine. Spills land in the gaps between turns.

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

**Follow-up (2026-06-14, `SystemKVManager.stats()` idle-enabled block).** `stats()` returned `None` whenever the manager was idle (no snapshot/hits/misses), so `/v1/status` reported `cache: null` before the first request even when the system-KV path was live — the exact blind spot that previously hid `VLLM_MLX_DISABLE_SYSTEM_KV` / a degraded route (a live 35B HA backend showed `cache: null` after 5.75 h up). Now `stats()` returns a **zeroed block carrying `enabled: True`** when idle, and `None` only when the kill switch genuinely disables the path (`is_safe() == False`); the active block also carries `enabled: True`. The zeroed block has no hit/miss/tokens_saved activity, so neither the `/v1/status` cache selection nor `metrics.py`'s activity-preferring scan lets it mask an active `memory_aware_cache` (both gate on those three fields — verified). Consumed by the `mac-studio-exporter` `llm_cache_enabled` gauge. 3 new tests (idle→enabled block; kill-switch→None; active→enabled); full suite 2215 passed.

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

**Three more gotchas surfaced by the live gemma bring-up (same patch series):**
1. **Per-layer quantization overrides** — gemma-4's config carries 120 per-layer entries (8-bit MLP/router on select layers) keyed in the **VLM namespace** (`language_model.<path>`); uniform `nn.quantize` produced `[quantized_matmul] shapes incompatible`. The class predicate now returns the override dict (mlx-lm loader convention), checking both namespaces.
2. **Thread-local stream pinning** — lazy init-time module buffers (gemma's rope tables; NOT in `parameters()`, so the coverage guard can't see them) pin to the build thread's stream; the first worker-thread forward died with `There is no Stream(gpu, 1) in current thread`. `build_text_model` runs a **one-token warmup forward** on the build thread. Qwen3.5 never hit this — it computes rope per forward. **(Updated 2026-06-29 rebase.)** Upstream `e2d3d95` (#614) added an explicit module-walk `mx.eval` that realizes every array — incl. underscore-private buffers like `RoPE._freqs` — directly, which is the **authoritative lazy-array realizer** now (more reliable than hoping a forward touches every private buffer). The warmup forward is **kept** for the two jobs #614 doesn't do: (i) fail-closed guard — refuse a TextModel that can't actually run a forward, and (ii) kernel-path warmup after `train(False)`. The redundant `mx.eval(text_model.mtp.parameters())` was dropped (mtp is a registered submodule, so #614's walk covers it).
3. **Multi-eos termination** — gemma-4's `generation_config.json` declares **three** eos ids (`end_of_text`, `<turn|>`, end-of-channel); the MLLM text route handed `stream_generate` a bare HF tokenizer carrying one, so `<turn|>` leaked into content and generation ran to max_tokens. The text-route tokenizer is now pre-wrapped in mlx_lm's `TokenizerWrapper` with the full declared eos set (`wrap_tokenizer_with_eos`) — generalizing the old hard-coded Qwen3.5 `<|im_end|>` special case.

Post-fix probe (26B, served): correct short answers with `finish_reason=stop`, and the first-ever gemma cache HIT: `reusing 32 cached tokens, prefilling 3`.

**Upstreaming:** (a) is a clean bug fix — strong PR candidate. (b) extends the #19/#20 series; together they make the pure-LLM cache path cover every mlx-lm cache class except quantized partial-restore.

---

## 22. `patch: dry-sampler` — sequence-level repetition penalty (loop breaker)

**Files:** `vllm_mlx/dry_sampler.py` (new), `vllm_mlx/engine/simple.py`, `vllm_mlx/server.py`, `vllm_mlx/api/models.py`, `tests/test_dry_sampler.py` (new)

Long agentic sessions on quantized models hit **block-level repetition collapse** (Qwen3.6-27B-4bit at deep context emits the same paragraph indefinitely at T=0). Token-level penalties can't break these loops — each token is locally optimal; the loop is a sequence phenomenon. DRY (p-e-w's sampler, community-proven in koboldcpp/llama.cpp/text-generation-webui; vLLM PR #11368 died stale, so no upstream to borrow from) penalizes the token that would **extend a repeated suffix**, exponentially in the repeat length: `multiplier * base^(match_len − allowed_length)`. An 8-token loop at defaults eats a ~27-logit penalty — breaks argmax at T=0; incidental short repeats stay free.

**Implementation:**
- `DRYLogitsProcessor` rides mlx-lm's standard logits-processor chain (`processor(tokens, logits)`), appended in `_stream_generate_text` next to the existing penalty processors. Match lengths for all positions come from a **Z-array over the reversed window** — O(window) per token. That's load-bearing: the pathological repetitive context is exactly where naive per-occurrence scans degrade to O(window²).
- **Sequence breakers** (default `\n : " *`) cap matches at structural boundaries and are never penalized themselves (they're the escape hatch). This makes tool-call JSON near-immune — content between quotes/colons rarely clears `allowed_length`.
- Config: request fields `dry_multiplier/base/allowed_length/range/sequence_breakers` (OpenAI + completions APIs) override per-model `VLLM_MLX_DRY_*` env defaults; off unless a multiplier > 0 arrives from either. Exponent capped at 18 (fp16-safe).
- Interaction: any custom logits processor **disables MTP** for the request (existing engine rule) — don't enable DRY defaults on the MTP heavies casually.

**Verified:** 10 unit tests including a randomized cross-check against an independent O(n²) reference implementation (breaker semantics included), fp16 overflow guard, env/request precedence. Full suite 2200 passed / 0 failed.

**Deployment intent:** enabled via the LiteLLM route for `Qwen3.6-27B-4bit` (`extra_body`: multiplier 0.8, base 1.75, allowed_length 16, range 2048).

**Production incident (2026-06-12) → GENERATION-ONLY matching.** The first live session with whole-context matching corrupted repeated shell tool-calls: re-running a command from earlier in the conversation is a long verbatim repeat with **no sequence breakers inside bash text**, so the exponential penalty forced mid-command divergence (`...llama-swap-config.yaml | head -20` came out as `...llama-swap-defaultconfig.yaml | headdefault-20`). Fix (deliberate divergence from the original DRY): the processor records the prompt length on its first invocation and **matches only within the current generation** — re-emitting prompt content (prior commands, filenames, history) is legitimate agentic behavior, while the loops DRY exists to break repeat within one generation by definition. Deployment `allowed_length` also raised 3 → 16: within-generation echoes of identifiers/code lines up to ~16 tokens are normal in coding output; collapse loops repeat 50+-token blocks and still get a decisive penalty. Regression-tested (prompt-region repeats unpenalized; generation-internal repeats caught).

**Upstreaming:** vLLM rejected/preferred not to merge DRY upstream; for vllm-mlx it's a natural fit (single-stream serving is where loops hurt most). PR-worthy.

---

## 23. `fix(text-model-from-vlm): eval() the derived TextModel` — cherry-pick of upstream #606

> **Status (2026-06-14 rebase): RETIRED.** Upstream `527f457` (#606) is now in the base. Our cherry-pick did not auto-drop (different placement: ours is before patch #21's warmup so the warmup runs on the kernel path). The redundant upstream `train(False)` before `return` was removed (commit `6d69dd0`); the single load-bearing pre-warmup call is kept. Section retained for history.

**Files:** `vllm_mlx/text_model_from_vlm.py`

Cherry-picks upstream [`527f457` (#606)](https://github.com/waybarrios/vllm-mlx/commit/527f457) ahead of our next rebase. `build_text_model` constructs a fresh mlx_lm TextModel for the MLLM→text route but left it in mlx's default `training=True`. Hybrid gated-delta layers (Qwen3.5/3.6 linear attention) select their compute path with `use_kernel = not self.training`, so **every gated-delta forward on our Qwen3.6-27B-4bit and 35B-A3B text routes ran the slow Python `for t in range(T)` recurrence instead of the Metal kernel** — a context-scaling prefill penalty plus a decode hit on every token of the primary coding workload AND the always-loaded HA voice model. `mlx_lm.load` / `mlx_vlm.load` both eval() their models, so the regular routes never hit this; this VLM→text path was the one that didn't.

One line — `text_model.train(False)` — but placed AFTER `inject_mtp_support` (so it recurses into the injected `mtp` submodule) and BEFORE patch #21's warmup forward (so the warmup materializes lazy buffers on the *actual* kernel compute path), vs upstream's placement just before `return`. Numerically identical output; only the compute path changes.

**Upstream-measured (Qwen3.6-35B-A3B 4-bit, M-series):** prefill ~2k 2.78→0.47 s, ~5k 6.92→1.13 s (≈6×); decode 116.8→133.8 tok/s (+15%). **Pending local A/B on the Studio 27B** (cold prefill tok/s, decode tok/s, T=0 warm-vs-cold byte-identity).

**Status:** TEMPORARY cherry-pick — **retire on the next rebase past upstream `527f457`** (the rebase will collide on this one line; drop ours). Unaffected: gpt-oss/gemma/GLM/Phi (no gated-delta); Qwen3-Next-80B / Qwen3-Coder-Next (loaded via `mlx_lm.load`, already eval()'d).

---

## 24. `patch: system-kv-ram-budget` — cap the resident slot set (memory-abort guard)

**Files:** `vllm_mlx/system_kv.py`, `tests/test_system_kv_partial.py`

The system-KV slot snapshots are the only **unbounded** RAM term in the serving process: a grown deep-context slot is multi-GB (a ~80K-token 27B chain ≈ 5 GB), so the default 4 slots can hold ~14 GB beside ~15 GB of weights on the 64 GB box — implicated in the live jetsam / Metal-abort cluster on the 27B coding route.

Adds `VLLM_MLX_SYSTEM_KV_RAM_MB` (MB; **0/unset = unlimited, exactly the prior unbounded behavior** — a no-op until enabled per-model). When set, a new `enforce_ram_budget()` runs at every store site (inside `_generation_lock`) and evicts LRU-bag entries until the resident set (active + bag + checkpoints) fits:

- **Never evicts the active slot** — the in-flight request needs it. The budget caps the BAG overhang (slots that exist only to skip a future cold prefill). If the active slot alone exceeds the budget, it is kept with a warning.
- **SSD-spilled bag entries first** (re-acquiring them is a ~1.3 s promote, not a ~25-39 s cold prefill), then oldest-first within each class. A per-slot `spilled` flag (set from the `enqueue_spill` accept / promote result) and cached `bytes` ride into the bag dict on demote.
- `mx.clear_cache()` fires once after eviction (same discipline as the capacity-eviction path).

TOCTOU contract unchanged from patch #13: a worker holding an evicted snapshot ref keeps it alive by refcount after the dict drop. The `_entry_bytes` / `_ckpt_bytes` accounting helpers were hoisted from `stats()` to module scope (`entry_bytes` / `ckpt_bytes`) for reuse — pure code motion, byte-identical numbers.

**Verified:** 4 new unit tests (no-op at budget 0; bag-evicted / active-kept under budget; spilled-first ordering; oversized-active retained). Full suite 2212 passed.

**Deployment intent:** `VLLM_MLX_SYSTEM_KV_RAM_MB=6144` on the Qwen3.6-27B-4bit route ONLY, after correlating slot-RAM with the crash timestamps; then watch `llm_backend_crashes_total` for two weeks.

**Upstreaming:** plausible — an unbounded snapshot set is a general risk on memory-constrained single-box deployments; gated behind an env so the default is unchanged.

---

## 25. `patch: ssd-tier-hardening` — real-disk-byte cap, startup reconcile, queued-spill byte cap

**Files:** `vllm_mlx/system_kv_ssd.py`, `vllm_mlx/ssd_cache.py`, `tests/test_system_kv_ssd.py`

Three protections for the SSD persistence tier (patches #16/#19), all confirmed against the live 27B dir:

- **Index the actual on-disk size, not the snapshot's in-RAM `nbytes`.** The safetensors file also holds the flattened partial-restore checkpoint tensors, which `nbytes` excludes — so the cap **under-counted** (~37% live: 23.3 GB on disk vs 16.9 GB indexed) and never fired. `_write_entry` now indexes `disk_bytes` (`os.path.getsize`).
- **Startup reconcile** (`_reconcile`, runs once in `start_writer`): removes stale `*.tmp` dirs from interrupted writes (a SIGKILL mid multi-GB serialize leaves them); reconciles the SQLite index against the on-disk `data/` dirs (drops rows whose dir is gone, deletes orphan dirs); backfills `memory_bytes` from the real file size for rows written before the byte-cap fix. New `SSDIndex.update_memory_bytes` does a targeted UPDATE that **preserves LRU timestamps**. After reconcile, `_enforce_capacity` evicts down to the now-honest cap.
- **Queued-spill byte cap** (`max_queued_gb`, default 12 GB): the spill queue holds references to flattened (still-resident) mx arrays, so a defer-until-idle backlog of deep-context snapshots pins unified memory — the residual vector behind the Metal-abort class the 2026-06-12 defer-until-idle change otherwise addressed. A spill that would push queued bytes over the cap is dropped (a sub-prefix is usually already on disk; the loss costs at most one re-grow); an **empty queue always admits one** spill so huge prefixes still persist. `_queued_bytes` is tracked under the existing lock and released when the writer pops an item.
- **`close()` join 10 s → 4 s**: bounded below llama-swap's ~5 s SIGTERM→SIGKILL grace so the index commit reliably runs before a kill. The resident working set is already on disk via idle-window write-through, so the drain is a best-effort tail; interrupted writes are recovered by the next startup reconcile.

**Verified:** 4 new unit tests (disk-byte accounting incl. checkpoints; queue byte cap drop + empty-admits-oversized; reconcile orphan-sweep + byte backfill). Full suite 2212 passed.

**Upstreaming:** the disk-byte accounting and reconcile are general SSD-tier robustness; `ssd_cache.update_memory_bytes` is a small additive index method (PR-friendly).

---

## 26. `fix(qwen3-xml): parse bare <function=> without <tool_call> wrapper` — cherry-pick of upstream #597 — **RETIRED (in base as of `0dd1157`)**

> **RETIRED on the 2026-06-29 rebase.** Upstream `490cca0` (#597) merged into the base; our cherry-pick (`3639b7c` + `521f095`) had a byte-identical change set and auto-collapsed during the rebase. Section kept for history.


**Files:** `vllm_mlx/tool_parsers/qwen3_xml_tool_parser.py`

Cherry-picks upstream open PR [#597](https://github.com/waybarrios/vllm-mlx/pull/597) (commits `55f2296` + `03164f2`) — the backport of vLLM #26345, broadened. Under load (large system prompt + many tools) **Qwen3-Coder drops the outer `<tool_call></tool_call>` wrapper and emits only the inner `<function=NAME><parameter=…>…</function>` block.** The wrapper-gated `StreamingXMLToolCallParser` leaked these as raw text → `<function=Agent>` printed as prose followed by a hung 0-token subagent. Three coordinated changes: (1) buffer-wait on partial `<function=`/`<parameter=` tags (avoids text leak + expat "not well-formed" crash); (2) deferred commitment on bare `<function=Name>` — held until `<parameter=` opens or `</function>` closes, rolled back to content if prose char-data arrives (keeps `test_streaming_plain_text_is_not_misclassified` green); (3) implicit-wrapper close on `</function>` so sequential bare calls get independent ids.

**Affects only the `qwen3_xml`/`qwen3.5`/`qwen3_coder` parser** (registered in `qwen3_xml_tool_parser.py`) — in our lineup that's **`Qwen3-Coder-Next-4bit`** (`--tool-call-parser qwen3_coder`). The hermes-parser Qwen3.6/3.5 routes (the opencode primary) are a different parser and unaffected.

**Verified:** PR test plan 52 passed (incl. new bare-function + prose-safety streaming tests); full suite 2239 passed / 16 skipped; direct parse demo confirmed bare `<function=>` → `tools_called=True` and prose `<function=` stays content.

**Status:** TEMPORARY cherry-pick — **retire on the next rebase past upstream `#597`** (the rebase will collide on `qwen3_xml_tool_parser.py`; drop ours).

---

## 27. `f7499a9` — `patch: run-reasoning-parser-when-thinking-disabled`

**Files:** `vllm_mlx/server.py` (`stream_chat_completion` gate + `_extract_reasoning_and_tool_calls`)

When `enable_thinking=False`, the server **skipped the reasoning parser entirely** on both chat-completion paths (`stream_chat_completion` gated on `... and not _thinking_disabled(...)`; non-stream via `allow_reasoning=not _thinking_disabled(...)` in `_extract_reasoning_and_tool_calls`). That assumes thinking-off output carries no reasoning-protocol markers — **false for gemma-4.**

**Root cause:** gemma-4's chat template *prefills* `<|channel>thought\n<channel|>` into the prompt when thinking is disabled (its native skip-thinking mechanism, template lines ~263-264). That prefill is echoed into the completion. With the parser skipped, the raw `<|channel>thought\n<channel|>` markers leaked into `content`. Home Assistant (hass_local_openai_llm) rejected the reply on the post-tool-result turn: *"Last content in chat log is not an AssistantContent … model not returning a valid response."* Qwen MoE/dense were unaffected — their thinking-off path emits no markers, so skipping the parser was harmless there.

**Fix:** always run the reasoning parser when one is configured; when thinking is disabled, **fold any reasoning output back into content** (do not surface a separate reasoning stream). The `gemma4` reasoning parser already strips the empty-thought prefill correctly — unit-verified that both `extract_reasoning()` and token-by-token `extract_reasoning_streaming()` return clean content for `<|channel>thought\n<channel|>ANSWER`.

> **2026-06-17 follow-up fix (commit `80bb05b`): fold, don't drop.** The first cut *discarded* reasoning when thinking was off. That broke Qwen in the STREAMING path: the `qwen3` parser routes markerless thinking-off output (no `<think>` tags) to `reasoning`, so discarding it emptied the response entirely (`ha-qwen-moe` / `ha-llm` returned `''`). gemma was unaffected because its prefill markers delimit the answer into `content`. The fix folds `reasoning` into `content` instead of dropping it (in both `stream_chat_completion` and `_extract_reasoning_and_tool_calls`), recovering Qwen markerless text while remaining a no-op for gemma. Lesson: thinking-off output is not always markerless — the parser's content/reasoning split is model-specific, so never assume "thinking off ⇒ no reasoning field".

**Safe across the lineup:** for models whose thinking-off output has no markers (Qwen `<think>` family, plain models), `extract_reasoning` returns the content unchanged, so running it is a no-op on content; only the now-suppressed reasoning differs. This makes gemma-4 (and any channel/prefill-style template) usable with thinking on **and** off — e.g. the HA `ha-gemma` route.

**Upstreaming:** candidate — the skip-when-disabled optimization is incorrect for any template that prefills a reasoning frame; running the parser and dropping reasoning is the correct general behavior.

---

## 28. `fix(llm): realize lazy init-time arrays on the load thread (gpt-oss serialized-route stream crash)`

**Files:** `vllm_mlx/models/llm.py` (`MLXLanguageModel.load`), `vllm_mlx/models/mllm.py` (`MLXMultimodalLM.load`)

**Symptom:** every `/v1/chat/completions` to **gpt-oss-20b-MXFP4-Q4** returned HTTP 500. Traceback bottomed out in mlx-lm's `generate_step` at `mx.eval([c.state for c in prompt_cache])` with `RuntimeError: There is no Stream(gpu, 1) in current thread`. Surfaced while deploying the `0dd1157`/v0.4.0 rebase, but **pre-existing** — independent of that rebase (worker-thread generation code byte-identical) and independent of upstream #581 harmony rendering (crashed identically on the no-harmony fallback path and with `VLLM_MLX_DISABLE_SYSTEM_KV=1`).

**Root cause:** MLX lazy graphs are tagged to the stream of the thread that *recorded* them, and (mlx-lm >= 0.31) generation runs on a SimpleEngine serialized **worker thread** (`_run_blocking_serialized` → `to_thread`), not the load thread. gpt-oss's attention **`sinks = mx.zeros((num_heads,))`** (`mlx_lm/models/gpt_oss.py`) is an *init-time* array created on the load thread and left lazy; the first worker-thread forward evaluates it off its home thread and dies. Qwen3.5/3.6 have no such init-time buffer (rope is computed per forward), so they never hit it — which is why only gpt-oss broke. Reproduced in isolation: a worker-thread `stream_generate` crashes before any realize and **succeeds after** `mx.eval`-ing the model's module arrays on the load thread.

This is the same class of bug as the gemma-4 RoPE crash — fixed for the VLM→text route by patch #21's warmup + upstream #614's module-walk realize in `build_text_model`. Plain `mlx_lm.load` models (gpt-oss et al.) never went through that path and so were unprotected.

**Fix:** after `load_model_with_fallback`, walk `self.model.modules()` and `mx.eval` every `mx.array` value — realizing all init-time arrays on the load thread before any worker forward. Mirrors the #614 block. Harmless no-op for models whose arrays are already realized (verified: full suite 2293 passed / 29 skipped unchanged; Qwen unaffected).

**Defensive parity (same patch):** `MLXMultimodalLM.load()` (the mlx-vlm path) gets the identical realize. Its multimodal generation runs on the *same* `to_thread` serialized worker (`_run_blocking_serialized`), so a future mlx-vlm model with a lazy init buffer would crash the same way. Current VLMs don't trip it — verified live that the guard doesn't regress them: GLM-4.6V vision (`"A yellow circle."` on a synthetic probe) and the VLM text route both return `finish_reason=stop`. The guard keeps both load paths symmetric with the LLM path. (The MLLM *text* route was already covered separately by `build_text_model`'s #614/#21 realize.)

**Verified live on the Studio:** gpt-oss-20b returns clean completions on both the fallback path and the `#581` harmony-rendering path (`openai-harmony` installed) — `finish_reason=stop`. Investigated and **rejected** a `mlx_streams.py` change: with mlx-lm >= 0.31 `generation_stream` is already a cross-thread-safe `ThreadLocalStream`, so `bind_generation_streams` was not the culprit; the realize is the whole fix.

**Upstreaming:** candidate — mlx-lm's own `generate` realizes nothing at load time, so any embedding host that generates off the load thread is exposed; a load-time realize (or lazy-array warmup) belongs upstream.

---

## 29. `patch: batched-lazy-realize` — realize lazy init-time arrays on the batched LLM load path

**Files:** `vllm_mlx/lazy_realize.py` (new), `vllm_mlx/engine/batched.py`, `vllm_mlx/models/llm.py`, `vllm_mlx/models/mllm.py`, `tests/test_lazy_realize.py` (new)

First of the **BatchedEngine parity series (#29–#33)** toward parallel-request serving. The gate that unblocked the series: a 2026-07-02 spike on `Qwen3.5-0.8B-8bit` (same qwen3_5 hybrid family as the production 27B/35B) proved mlx-lm 0.31.3's `BatchGenerator` merges a **snapshot-restored mid-sequence hybrid cache** (attention KV at offset p>0 + recurrent `ArraysCache` state, round-tripped through `classify_layers`/`apply_snapshot_states`) **bit-identically** into a concurrent batch — including insertion mid-flight into an already-decoding batch (9/9 rows identical to single-stream references, T=0). That closes the `continuous-batching-hybrid-caching.md` doc's last open item (D, concurrent recurrent-merge validation).

**This patch:** BatchedEngine's LLM path loads via `load_model_with_fallback` on the event-loop thread (the issue-#407 inline load) but steps the model on the engine-core executor thread. Patch #28's realize only covered `MLXLanguageModel`/`MLXMultimodalLM`, both bypassed here — so a gpt-oss-style lazy init-time array (attention `sinks`) makes the **first batched step** die with "There is no Stream(gpu, N) in current thread", and the engine limps through `engine_core`'s stream-error self-heal (`_is_stream_thread_error` → model-thread stepping + cache recovery) instead of starting clean.

Fix: extract #28's module-walk `mx.eval` into fork-owned `vllm_mlx/lazy_realize.py::realize_module_arrays` and call it in `_prepare_llm_model` immediately after load (before `apply_compile`). `models/llm.py` / `models/mllm.py` now delegate to the helper — pure code motion there, and it shrinks our diff in those upstream-owned files (same containment rationale as #18).

**Verified:** 3 new tests (cross-thread evaluability of a private lazy buffer after the direct call; no-`modules()` no-op; `_prepare_llm_model` wiring end-to-end with a monkeypatched loader). Engine suites green.

**Upstreaming:** pairs with #28's candidate — the batched LLM load path is upstream's own gap; the self-heal in `engine_core` masks rather than prevents it.

---

## 30. `patch: batched-text-only` — thread `--text-only` into BatchedEngine

**Files:** `vllm_mlx/engine/batched.py`, `vllm_mlx/server.py`, `tests/test_batched_engine_mllm_config.py`

BatchedEngine parity series (#29–#33). `force_text_only` previously reached only the SimpleEngine constructor (`server.py`); under `--continuous-batching` the flag was silently ignored and any checkpoint that classifies MLLM (vision weights present — Qwen3.6-27B, gemma-4) routed **all** requests, text included, through `MLLMScheduler`/`MLLMBatchGenerator`. BatchedEngine has no per-request text split and no `build_text_model` route, so that path forfeits the LLM scheduler (worker-thread stepping, the prefix cache the #33 system-KV port targets, per-sequence `caches=` injection).

Fix: `BatchedEngine.__init__` takes `force_text_only` with SimpleEngine's exact precedence — `(force_mllm or is_mllm_model(...)) and not force_text_only` — and `server.py` forwards it in the batched branch. With the flag, the model loads via `load_model_with_fallback` (mlx_lm `strict=False` discards `vision_tower.*`) and text serves on the LLM scheduler.

**Verified:** routing unit test (auto-detect / text-only / text-only-beats-force-mllm / force-mllm). Suite file green.

**Upstreaming:** trivial parity fix, PR-worthy alongside the flag's existing SimpleEngine documentation.

---

## 31. `patch: batched-dry-sampler` — DRY rides the batched per-sequence processor chain

**Files:** `vllm_mlx/engine/batched.py`, `tests/test_batched_dry.py` (new)

BatchedEngine parity series (#29–#33). The `dry_*` request fields (patch #22) reached `BatchedEngine.chat`/`stream_chat` in `**kwargs` and **silently vanished** — never popped into `SamplingParams`, never forwarded to either scheduler. mlx-lm's `BatchGenerator.insert()` already takes per-sequence `logits_processors` and applies each row's chain to that row's logit slice, and the LLM scheduler already threads `sampling_params.logits_processors` through (`scheduler.py` `_schedule_waiting`); the MLLM scheduler carries per-request processors too. Only the pop-and-build was missing.

Fix: `_merge_dry_processor(kwargs)` in `batched.py` pops the `dry_*` kwargs + external `logits_processors` once, before the MLLM/LLM branch in `generate()`/`stream_generate()` (which `chat`/`stream_chat` delegate to), builds `DRYLogitsProcessor` via the same `build_dry_processor` env-default resolution as SimpleEngine, and passes the merged list to both branches. One fresh processor instance per request = per-sequence state isolation in the batch.

Two behavior notes: (a) the scheduler passes no `all_tokens` into `insert()`, so the batched token context contains only *generated* tokens — DRY's generation-only matching (the 2026-06-12 incident fix) holds by construction; (b) the batched MTP patches (`patches/qwen3_next_mtp.py`/`qwen3_5_mtp.py`) never run logits processors during draft acceptance (verified by grep), so DRY + `--enable-mtp` on this engine would be silently bypassed — same "don't combine" guidance as SimpleEngine, now noted in the helper docstring.

**Verified:** 6 new tests pinning all four entry paths (LLM/MLLM × generate/stream) plus off-by-default and external-processor coexistence. DRY + engine suites green.

**Upstreaming:** rides with #22 if that ever goes; the helper is fork-shaped (env defaults).

---

## 32. `patch: batched-stop-strings` — enforce stop strings under `--continuous-batching`

**Files:** `vllm_mlx/stop_strings.py` (new), `vllm_mlx/engine/batched.py`, `tests/test_batched_stop_strings.py` (new)

BatchedEngine parity series (#29–#33). The batched schedulers only honor stop **token ids** (`sampling_params.stop_token_ids` → `BatchGenerator` stop machinery); `sampling_params.stop` — where the server puts API stop params AND tool-parser stop strings (`server.py` merges both into `chat_kwargs["stop"]`) — was **never read** on either batched path. Under `--continuous-batching`, stop strings silently ran through to `max_tokens`.

Fix: fork-owned `vllm_mlx/stop_strings.py` with two helpers, wired at the engine layer in `batched.py` (all four paths: LLM/MLLM × generate/stream):

- `StopStringScanner` (streaming): bounded-tail scan — patch #17's discipline, O(len(new_text)) per chunk, `max(len(stop))-1` chars carried across chunk boundaries. On a hit the final chunk is **cut at the earliest match start** and the underlying request is aborted via `abort_request` (frees the batch slot instead of decoding to `max_tokens`). This is deliberately *stricter* than SimpleEngine, which finishes at chunk granularity and can leak stop text into the last chunk — the cut matches OpenAI semantics (stop sequence not returned).
- `truncate_at_stop` (non-stream): earliest-match truncation of the complete text. Runs **before** `clean_output_text` — stop strings are frequently special tokens (`<|im_end|>`) that cleaning strips, which would blind the scan (caught by the test suite during development).

Residual gap (documented, not a regression): text preceding a stop match that was already emitted in an *earlier* chunk cannot be unsent; stop markers are normally single tokens so the window is one chunk.

**Verified:** 9 new tests — scanner unit level (within-chunk / spanning-chunks / single-char / inactive), earliest-of-multiple truncation, and wiring level (stream cut + abort called + post-stop chunk suppressed; passthrough when no stop; non-stream truncation; both LLM and MLLM branches). Batched suites green.

**Upstreaming:** strong candidate — upstream's batched path has the same hole; the helper module is dependency-free.

---

## 33. `patch: batched-per-request-sampling` — per-request samplers on the batched LLM path

**Files:** `vllm_mlx/scheduler.py`, `tests/test_batched_per_request_sampling.py` (new)

BatchedEngine parity series (#29–#33). The LLM scheduler built ONE `make_sampler(temp, top_p, min_p)` per `BatchGenerator` from the **first** request's params: `top_k` was dropped entirely, `presence_penalty` never became a processor, and a request whose params differed from the running batch either sampled with stale settings (mid-batch) or forced a generator recreation (idle). mlx-lm's `BatchGenerator.insert()` accepts per-sequence `samplers=` and applies them row-wise (`generate.py` `_step` loops rows, falling back to the shared sampler only where a row has none).

Fix, in `_schedule_waiting`: build the request's own `make_sampler(temp, top_p, min_p, top_k)` and pass `samplers=[sampler]` on insert; convert `presence_penalty` to a `make_logits_processors(presence_penalty=…)` entry in the same per-request chain that already carried `repetition_penalty`. `_ensure_batch_generator`'s recreate-when-idle dance is kept (it still refreshes the fallback sampler + stop tokens, and a test pins the close-on-replace discipline), but its mid-batch warning is rewritten — per-request samplers now apply immediately regardless.

**Residual gap (documented):** per-request `stop_token_ids` still bake into the generator at creation (both before and after this patch — recreation only triggers on sampler-param changes). A per-sequence `SequenceStateMachine` on insert would close it; deferred since stop ids are constant per deployment (tokenizer EOS + parser). Note the per-row sampler loop in mlx-lm is Python-level — negligible at our `max_num_seqs`, worth knowing at 32+.

**Verified:** 3 new tests (sampler kwargs incl. top_k captured + `samplers=` on insert; presence_penalty→processor; empty-chain default preserved). Scheduler suites green.

**Upstreaming:** strong candidate — upstream's batched path drops `top_k`/`presence_penalty` today.

---

## 34. `patch: batched-system-kv` — hybrid-safe checkpoint prefix cache for the batched LLM scheduler

**Files:** `vllm_mlx/batched_system_kv.py` (new), `vllm_mlx/scheduler.py`, `tests/test_batched_system_kv.py` (new)

The **item-B port** from `docs/fork/continuous-batching-hybrid-caching.md`, closing the "batching forfeits our cache" objection. The default batched prefix cache (`MemoryAwarePrefixCache`) gets **zero hits on hybrids** — its supersequence/LCP paths need rewind, recurrent `ArraysCache` state can't rewind, and the `has_non_trimmable` gates correctly skip — so every Qwen3.5/3.6-class request under `--continuous-batching` paid full prefill (measured 2026-05-09: 4 identical 7.7K-token requests → `hits=0 misses=4`).

**Design** — reuses the engine-agnostic checkpoint engine (`system_kv.py`, #19/#21) unchanged:

- *Capture:* `_schedule_waiting` inserts via `insert_segments` with the prompt split at `VLLM_MLX_BATCHED_KV_CKPT_INTERVAL` (default 2048) boundaries; the generator stops exactly there. `step()` watches `end_of_segment` prompt responses and `_capture_hybrid_checkpoints` extracts the row (`BatchGenerator.extract_cache`, same executor thread), keeps only the recurrent-layer states (`capture_checkpoint_states`, eval'd off the batch views), and appends to the request's ladder (`append_checkpoint`, geometric thinning). mlx-lm's own last-token split means every insert also yields a free checkpoint at prompt-boundary−1 — exactly what an identical re-send needs.
- *Store:* on finish, the full snapshot (state+meta+`classify_layers` kinds) + the ladder become an LRU entry (`VLLM_MLX_SYSTEM_KV_SLOTS`=4, `VLLM_MLX_SYSTEM_KV_RAM_MB` budget, same envs as the SimpleEngine stack). **Identical-token chains replace their older entry with ladders merged** — otherwise an exact re-send (which prefills only the kickoff token) stores a duplicate whose one checkpoint sits above the usable cap and shadows the rich original in the LCP scan (found by the e2e concurrent scenario, not by unit tests).
- *Fetch (`add_request`):* token LCP over entries → `select_restore_pos` (nearest checkpoint ≤ divergence; attention KV slices to any position; `d == donor_len` fast path for pure extensions) → `build_partial_restore_states` → fresh `make_prompt_cache` + `apply_snapshot_states` → `request.prompt_cache`/`remaining_tokens`, flowing into `BatchGenerator.insert(caches=…)` per the seam. **A hit seeds the new request's ladder with the donor's checkpoints ≤ restore-pos** — the restored request continues the same chain, so its eventual entry stays divergence-restorable.
- *Off by default*, enabled per-model via `VLLM_MLX_BATCHED_SYSTEM_KV=1`; when active it **replaces** the memory-aware cache on this scheduler (double-storing wastes RAM; the replaced paths were hybrid-dead anyway). Stats surface under the existing `system_kv_cache` key — `/v1/status` + Prometheus pick them up with zero changes (#7/#17 plumbing).

**Verified (2026-07-02, M1 Pro, real `Qwen3.5-0.8B-8bit` — same qwen3_5 hybrid family as the production 27B/35B):**
- *Gate spike* (recorded under #29): snapshot-restored mid-sequence hybrid caches merge **bit-identically** into concurrent batches in mlx-lm 0.31.3, incl. mid-flight insertion (9/9 rows, T=0).
- *End-to-end against the real Scheduler+BatchGenerator:* cold MISS→store; identical re-send **HIT at N−1** (599/600 saved); divergent chain (shared 450) **partial HIT** at the 256-token checkpoint; **two restored requests decoding concurrently in one batch** — all outputs byte-identical to a cache-disabled control run.
- 20 unit/wiring tests (real `KVCache`+`ArraysCache` slice/apply semantics; ladder inheritance; duplicate-chain merge; LRU eviction; floor; pure-attention any-position restore; cross-thread-realized fetch; scheduler init/fetch/insert_segments/capture/store/stats hooks). Full suite green.
- **Studio A/B finding (2026-07-02, real 27B):** the first live run exposed a cross-thread hazard the single-threaded dev e2e could not see — `fetch()` (event-loop thread) sliced trim-layer KV **lazily**; the first executor-thread step evaluated the slices and hit the MLX stream/thread mismatch (patch #28's crash class), which `engine_core` self-healed by switching to model-thread stepping while the request **silently re-prefilled cold** (32 s "warm" re-send despite a logged restore; divergent restores then worked only because stepping had moved to the model thread). Fix folded in: `fetch()` now `mx.eval`s the restored states on its own thread before handing them over — concrete buffers cross threads safely, and the restored copy detaches from the donor snapshot. Fourth instance of this class in the fork (#21/#28/#29/#34): **any MLX arrays created on one thread and consumed on another must be realized on the creating thread.**
- **Second live finding (gpt-oss batched smoke, same day):** sliding-window models (gpt-oss, gemma text) carry `RotatingKVCache` — ckpt-class like recurrent layers but with **tuple** states, and `capture_segment`'s realize loop only unpacked **lists** (deltanet `ArraysCache`). Rotating checkpoint states stayed lazy on the executor stream; `fetch`'s cross-thread eval then raised "no Stream(gpu, 2)" — streaming requests died pre-first-token, non-stream 500'd, and only on the long-prompt shape (short prompts stayed under `partial_min`, so fetch never ran). One-line fix (`isinstance(st, (list, tuple))`, matching `store()`); regression test with real `RotatingKVCache` round-trip + cross-thread eval.

**Not in v1 (deliberate):** no SSD tier (the batched `ssd_cache.py` cold tier stays memory-aware-only; our `system_kv_ssd.py` store is SimpleEngine-wired), no grow-on-HIT re-store of unfinished chains (entries only at request end), completion-region checkpoints (ladder covers prefill positions; divergence inside a prior completion restores at the last prompt-side checkpoint). Each is an incremental follow-up if the Studio A/B shows the gap matters.

**Deployment intent:** Studio A/B behind the env on the 27B route with `--continuous-batching --text-only` (#30) before any llama-swap change; SimpleEngine + system-KV remains the default until concurrent traffic is routine ([[speculative-decoding-dead-on-mseries]] measured the ~1.2× aggregate ceiling — this port buys *non-blocking concurrency*, not throughput).

**Upstreaming:** the strongest batched-path candidate — upstream has no hybrid prefix-cache story at all; the module is self-contained and the scheduler hooks are ~60 lines.

---

## 35. `patch: batched-kv-prompt-boundary-store` — abort resilience for the batched cache

**Files:** `vllm_mlx/batched_system_kv.py`, `vllm_mlx/scheduler.py`, `tests/test_batched_system_kv.py`, `tests/test_batched_system_kv_threading.py`

Patch #34 stored entries only at request **end** — an aborted request (client disconnects and agent cancels are routine in opencode) lost its entire prompt prefill: the pending ladder was discarded and no entry existed. Now the scheduler's `end_of_prompt` capture (where the row's cache is already being extracted for the boundary checkpoint) also calls `store_prompt_boundary`: the full prompt-position snapshot becomes an entry **mid-request**, so a cancel after prefill still leaves a warm prefix.

Mechanics: the boundary store *copies* the pending ladder (generation continues; the final `store` still owns it). Entry insertion is generalized to **prefix subsumption**: a new entry absorbs every existing entry whose tokens are a (proper or equal) prefix of its chain — ladders merge in (same token chain ⇒ checkpoint states valid), subsumed entries drop. That covers both the old identical-re-send dedup case and the boundary entry being replaced by its own finished chain (steady state stays one entry per chain). Guard: skipped when the request added `< partial_min` new tokens beyond its restored prefix (a near-full re-send would only duplicate the donor). Runs on the executor thread — the realize contract from #34's live findings holds by construction, and the cross-thread harness pins it.

**Verified:** 4 new unit tests (ladder kept + absorbed-on-finish; aborted chain served; below-floor skip; wiring at `end_of_prompt` incl. not-called on mid-segment responses) + a threaded lifecycle test (boundary store on the stream-bound executor, abort, fetch on main, consume on executor). Suite green. New `boundary_stores` counter in `system_kv_cache` stats.

**Upstreaming:** rides with #34.

---

## 36. `patch: batched-kv-ssd-tier` — the batched cache survives restarts and model swaps

**Files:** `vllm_mlx/batched_system_kv.py`, `vllm_mlx/scheduler.py`, `tests/test_batched_system_kv.py`, `tests/test_batched_system_kv_threading.py`

Patch #34's cache was RAM-only: every llama-swap TTL eviction, model swap, or restart threw away all warm entries and the next request paid a full cold prefill (~32 s on the 27B) — the exact gap patch #16 closed for SimpleEngine (26–28× faster recovery). This wires the **same store module** (`system_kv_ssd.SystemKVSSDStore` — MLX-native safetensors, dtype-exact bf16, SQLite prefix index, checkpoint format v3, defer-until-idle writer, reconcile-on-start, #25 hardening) into `BatchedSystemKV`:

- *Spill:* write-through on every entry insert — full chains AND #35's prompt-boundary entries (abort resilience now holds **across restarts**). The post-subsumption entry spills, so the richest merged ladder is what persists.
- *Promote:* two-stage, mirroring the scheduler's existing `ssd_pending` pattern so **blob I/O never touches the event loop** — `check_ssd` in `add_request` is index-only (SQLite + tiny json; full-prefix via `lookup_prefix`, divergent via `lookup_shared`), then `_try_promote_hybrid_ssd_pending` in `_schedule_waiting` (executor thread) does `read_entry` → RAM entry → normal `fetch` restore. `read_entry` realizes loaded arrays on the calling thread, so the cross-thread realize contract holds by construction — pinned by a threaded lifecycle test (spill on executor → restart → promote on executor → fetch on main → consume on executor).
- *Config:* same envs as SimpleEngine (`VLLM_MLX_SSD_SYSTEM_KV_DIR`/`_GB`) — one llama-swap env block serves either engine. Per-model subdir is `batched-<slug>` (tokenizer-derived), deliberately distinct from SimpleEngine's model-name subdir: the store is single-writer and the two engines must never share a directory. `idle_check` = scheduler-running probe (heavy spills wait for gaps between generations, the 2026-06-12 lesson). Writer closes on scheduler `reset()`/`deep_reset()` (shutdown path only — cache-error recovery doesn't reset, verified).

**Verified:** 4 SSD unit tests (restart round-trip bit-exact incl. recurrent state; divergent shared-prefix promote restoring at a checkpoint; boundary-entry restart resilience; env-off no-op), 3 scheduler wiring tests (miss→`ssd_pending` marker; promote→restore fields; failed-promote→cold fallback with no fetch), 1 threaded lifecycle test. Full suite 2356 passed / 0 failed.

**Upstreaming:** rides with #34/#35; the store module itself is unchanged.

---

## 37. `patch: batched-kv-grow-on-hit` — segmented snapshots kill the per-turn O(context) store copy

**Files:** `vllm_mlx/batched_system_kv.py`, `tests/test_batched_system_kv.py`, `tests/test_batched_system_kv_threading.py`

The last economic gap vs SimpleEngine: its grow-on-HIT re-snapshot is reference-rebinding (~free), while #34 stored a **full materialized copy of the whole chain at every request end** — multi-GB per turn at deep context (an 80K-token 27B chain ≈ 5 GB of memcpy + transient double-residency per turn, plus the same again for the boundary store and the SSD re-serialize).

**Design — segment lists.** Trim-layer (plain KV) entry state becomes a list of `(keys, values)` segments; ckpt-class layers (recurrent/rotating — fixed-size) copy whole as before. `fetch` records the donor entry per request (`_restore_source`); `store`/`store_prompt_boundary` then build the new entry as **donor segments by reference + one O(delta) evaluated slice** of the finished row (`_build_snapshot`). The boundary→final sequence cascades the linkage (the boundary entry absorbs the donor and becomes the final store's donor), so the original cold arrays flow through a whole session untouched — #35's prefix subsumption becomes literal segment reuse. Restore assembles `[:pos]` with whole-segment references + one lazy concat (`_slice_segments`), evaluated on the fetch thread as before — same single materialization the unsegmented slice paid, and the pure-extension single-segment case now passes donor arrays through zero-copy. Segment lists consolidate to one array past 16 pieces (one O(chain) concat per ~16 turns, amortized). **Grown entries skip the SSD re-spill** — SimpleEngine's policy (#16): a restart promotes the stored prefix and re-grows cheaply.

Bounds honesty: a *divergent* grow from a single-segment (cold) donor slices that segment at the divergence — an O(common-prefix) copy once; thereafter the chain is segmented and subsequent grows are O(delta). Budget accounting counts shared segments in full for every holder (conservative overcount → earlier eviction, the safe direction). Token-identical prefixes guarantee state validity for reuse (KV at position i is a function of the token prefix only), which also keeps the grow correct when a restored insert had to retry cacheless.

**Verified:** 4 new unit tests (donor arrays shared **by identity** + O(delta) tail segment; divergent grow reuses exactly the common prefix with restore values equal to a straight-stored control across the segment seam; boundary→final cascade preserves the original array through two grows; grown entries don't re-spill) + a threaded lifecycle test (grow on the executor, seam restore consumed cross-thread). Real-model e2e (0.8B) byte-identical through re-send/divergent/concurrent grow paths. Full suite 2362 passed / 0 failed. New `grown_stores` stats counter.

**Upstreaming:** rides with the #34–#36 series.

---

## 38. `refactor: scheduler-seam extraction` + upstreaming branches + gemma verification (hygiene round)

**Files:** `vllm_mlx/scheduler.py`, `vllm_mlx/batched_system_kv.py` (extraction); branches `fix/batched-stop-strings`, `fix/batched-per-request-sampling`, `feat/batched-system-kv` (upstreaming prep)

- **Scheduler-seam extraction (`0e7b298`)** — the #18 containment pattern applied to `scheduler.py`: the bodies of the fork's system-KV hooks (init, add_request fetch + SSD probe, ssd-pending promote, checkpoint capture + boundary store, cleanup store, segmented insert) moved into fork-owned `batched_system_kv.py`; scheduler keeps one-line delegators. **−139 net lines** from upstream's churniest batched file. Pure code motion: full suite green with zero test edits (the #18 oracle).
- **Upstreaming branches cut and pushed**, based on `upstream/main` (`0dd1157`), each with its feature tests passing against upstream code, ready the day PR access opens:
  - [`fix/batched-stop-strings`](https://github.com/TimotejLabsky/vllm-mlx/tree/fix/batched-stop-strings) — #32, 11 tests, zero fork leakage (verified no DRY/#31 references).
  - [`fix/batched-per-request-sampling`](https://github.com/TimotejLabsky/vllm-mlx/tree/fix/batched-per-request-sampling) — #33, 3 tests, fork-internal comments reworded.
  - [`feat/batched-system-kv`](https://github.com/TimotejLabsky/vllm-mlx/tree/feat/batched-system-kv) — the #34–#37 series as one assembled feature (checkpoint engine `system_kv.py`, SSD store `system_kv_ssd.py`, `ssd_cache.py` additive index methods, `batched_system_kv.py`, stream harness, thin scheduler seam) — **80 tests green on upstream+seam**.
- **Gemma sliding-window verified under batched** (closing the last unverified architecture): mlx-lm `gemma4_text.make_cache` uses `RotatingKVCache(keep=0)` — the merge-supported variant — and the live Studio smoke on `gemma-4-26B-A4B-it-qat-4bit` (`--continuous-batching --text-only` + cache env) passed: warm TTFT 82 ms, 66 tok/s, both concurrent requests restored, grow-on-HIT active, zero errors.

---

## 39. `patch: batched-flip-enablement` — co-batching guard, queue-cap 503, observability completion

**Files:** `vllm_mlx/batched_system_kv.py`, `vllm_mlx/scheduler.py`, `vllm_mlx/engine/batched.py`, `vllm_mlx/metrics.py`, `vllm_mlx/server.py`, `tests/test_batched_flip_enablement.py` (new)

The three safety rails identified for flipping a production route to `--continuous-batching` (batch-of-guards patch, #17 precedent). All inert by default.

- **Length-aware co-batching guard** (`VLLM_MLX_BATCHED_PAD_WASTE_MB`, 0 = off). mlx-lm's `BatchKVCache.merge` right-justifies every row to the longest chain in the batch, so admitting a short request beside an 80K-token chain transiently allocates ~5 GB of padded KV at 27B scale — on the box with the jetsam history. `should_defer_cobatch` (fork-owned seam) estimates waste = Σ(L_max−L_i) × bytes/token — the per-token footprint *learned from the newest cache entry* (≈200 KB/token on the 27B-4bit), so no per-model config — and defers admission (`appendleft` + break, FCFS preserved) while over budget: brief queueing instead of a memory spike. Inert on a cold cache (first request runs solo anyway); logs once per deferred request.
- **Queue-cap overload shedding** (`VLLM_MLX_BATCHED_MAX_QUEUE`, 0 = unbounded/upstream). `Scheduler.add_request` raises `EngineBusy` past the cap → the server's existing translation returns retryable 503s on non-stream paths, and a new `BatchedEngine.raise_if_serialized_busy` lets the server's **pre-stream probe** (the same `getattr` seam SimpleEngine's #15 admission uses) reject streaming requests before SSE headers. `queue_cap`/`queue_rejections` in scheduler stats.
- **Observability completion:** `grown_stores` / `boundary_stores` / `ssd_promotes` become Prometheus gauges (`vllm_mlx_cache_*`, additive `metrics.py` block per the #7 precedent); `/v1/status` now carries the top-level `engine_type` bench-serve's `auto_detect_runtime` reads (both engines emit it in `get_stats`, the payload dropped it); the cache stats block carries `type: batched_system_kv` for the bench's cache label. Fixes the empty `Runtime: engine= cache=` provenance line in stored bench results.

**Verified:** 11 tests after the day-one incident fix (guard: inert-by-default / cold-cache / no-running, defers extreme mixes + logs once, admits similar lengths, scheduler FCFS defer wiring; cap: reject-over-capacity + counter, unbounded default, pre-stream probe raise/no-op **called with the server's positional request_id** + a signature-contract test over both engines). Live 503 + gauge checks in the deploy revalidation.

> **Day-one canary incident (2026-07-02, fixed same day, folded into this patch):** the pre-stream probe shipped as `raise_if_serialized_busy(self)` but the server calls `probe(request_id)` (SimpleEngine's signature) — the TypeError became a **500 on every streaming request** while non-stream worked, so the first real opencode session against the flipped route failed entirely. Two process lessons encoded: (1) the probe's calling convention is a cross-engine CONTRACT — now pinned by a signature test over both engines, and the unit tests call it the way the server does; (2) the #39 revalidation had actually hit this (the A/B's streaming scenario failed) but the result was read by grepping for expected lines instead of checking the script's exit code — the failure was invisible. Revalidations now assert exit codes.

**Deployment intent (the canary flip config):** `--continuous-batching --text-only --max-num-seqs 3` + `VLLM_MLX_BATCHED_SYSTEM_KV=1`, `VLLM_MLX_SYSTEM_KV_RAM_MB=6144`, `VLLM_MLX_BATCHED_PAD_WASTE_MB=4096`, `VLLM_MLX_BATCHED_MAX_QUEUE=8`, SSD envs as deployed.

**Upstreaming:** the queue cap and the status `engine_type` field are clean candidates; the guard is fork-shaped (depends on the #34 cache for its bytes/token estimate).

---

## 40. `patch: batched-dynamic-concurrency` — seats float on a KV-byte budget, not a fixed count

**Files:** `vllm_mlx/batched_system_kv.py`, `vllm_mlx/metrics.py`, `tests/test_batched_flip_enablement.py`

`--max-num-seqs` is a crude proxy for the real constraint: three deep-context sequences are ~15 GB of padded KV while eight short ones are ~2 GB. This extends #39's admission gate into three independent, env-gated checks (any one defers; solo requests are never deferred, so progress is guaranteed; all inert by default):

1. **Pad waste** (#39, unchanged): Σ(L_max−L_i) × bytes/token vs `VLLM_MLX_BATCHED_PAD_WASTE_MB` — the mixing penalty.
2. **Total padded-KV budget** (`VLLM_MLX_BATCHED_KV_BUDGET_MB`): (B+1) × L_max × bytes/token vs the budget — **the dynamic max-num-seqs**. Effective concurrency floats on the live request mix: deep contexts serialize themselves, short ones batch up to the hard `--max-num-seqs` cap (which becomes a generous upper bound, e.g. 8, instead of the safety limit).
3. **Memory watermark** (`VLLM_MLX_BATCHED_MEM_WATERMARK_PCT`): ground-truth backstop — defer while `mx.get_active_memory()` exceeds the percentage of the device's recommended working set. Works on a cold cache (no bytes/token needed) and catches pressure the estimates can't see (spill queue backlog, cache entries, other processes' unified-memory share).

Checks 1–2 use the bytes/token footprint learned from the newest cache entry (≈200 KB/token on the 27B-4bit). New `admission_deferrals` counter + `vllm_mlx_cache_admission_deferrals` gauge for the soak.

**Verified:** 6 new tests (budget defers over-total / floats seats upward on short contexts / inert by default; watermark defers above and admits below, cold-cache capable; counter + stats). Full suite 2375 passed / 0 failed. Live validation in the deploy revalidation (budget sized to force 2-seat serialization of 4 concurrent 5.4K requests, then floating back up).

**Canary flip config (updated):** `--continuous-batching --text-only --max-num-seqs 8` + `VLLM_MLX_BATCHED_SYSTEM_KV=1`, `VLLM_MLX_SYSTEM_KV_RAM_MB=6144`, `VLLM_MLX_BATCHED_KV_BUDGET_MB=8192`, `VLLM_MLX_BATCHED_PAD_WASTE_MB=4096`, `VLLM_MLX_BATCHED_MEM_WATERMARK_PCT=85`, `VLLM_MLX_BATCHED_MAX_QUEUE=8`, SSD envs as deployed.

**Upstreaming:** rides with the #34-series branch; the watermark check is the most upstream-general piece.

---

## 41. `patch: embedding-truncation-from-config` — cherry-pick of upstream #626

**Files:** `vllm_mlx/utils/truncation.py` (new), `vllm_mlx/embedding.py`, `vllm_mlx/rerank.py`, `vllm_mlx/utils/__init__.py`, `tests/test_truncation.py` (new), `tests/test_embeddings.py`, `tests/test_rerank.py`

Cherry-picks upstream open PR [#626](https://github.com/waybarrios/vllm-mlx/pull/626) (brandy975). `EmbeddingEngine.embed()` hard-coded `max_length=512` with `truncation=True` — **every embedding request for Qwen3-Embedding-4B (native 32k) was silently truncated at 512 tokens**, degrading all RAG embeddings on the exact `--embedding-model` path we deploy. The PR adds `resolve_max_length(config, tokenizer, *, cap, default)`: `config.max_position_embeddings` → `tokenizer.model_max_length` → default 512, clamped to `cap=8192`, with bool/int/≤0/HF-1e30-sentinel guards. Same fix wired into the reranker (which we don't currently serve — harmless).

**Deployment note:** the effective embedding window becomes **8192, not 32k** (the PR's cap). Chunks between 8k and 32k tokens still truncate; irrelevant for our RAG chunk sizes. `padding=True` pads to the longest actual sequence in the batch, so short inputs cost nothing extra.

**Conflict surface:** zero — `embedding.py`/`rerank.py`/`utils/__init__.py` carried no fork patches (verified byte-identical to upstream's "before" side).

**Status:** TEMPORARY cherry-pick — **retire on the next rebase past upstream #626** if/when it merges (MERGEABLE, no reviews yet as of 2026-07-07).

---

## 42. `fix(mistral-parser): parse the [ARGS]-marker tool format` — cherry-pick of upstream #631

**Files:** `vllm_mlx/tool_parsers/mistral_tool_parser.py`, `tests/test_tool_parsers.py`

Cherry-picks upstream open PR [#631](https://github.com/waybarrios/vllm-mlx/pull/631) (mabaeyens). Dec-2025 Mistral tokenizers (Ministral 3, **Devstral Small 2** — which we serve with `--tool-call-parser mistral`) emit `[TOOL_CALLS]name[ARGS]{json}`. The parser had no `[ARGS]` awareness: non-streaming parsed the function name as `get_weather[ARGS]` (matches no tool → call dropped); streaming re-classified deltas by leading punctuation and appended JSON fragments to the name. **Devstral Small 2 tool calling was fully broken on our tree.** The fix adds an `[ARGS]`-gated branch (older `[TOOL_CALLS]name{...}` and JSON-array formats untouched) plus persistent `_args_started`/`_name_buffer` streaming state with a `reset()` override chaining `super().reset()` (framework calls `reset()` at stream start — no cross-request leak). Mistral-Small-3.2-2506 predates the marker and is unaffected.

**Known pre-existing limitation (not introduced here):** the streamed tool-call `id` may be omitted on the first delta when the name spans multiple deltas before `[ARGS]`.

**Conflict surface:** zero — no fork patches touch `mistral_tool_parser.py`.

**Status:** TEMPORARY cherry-pick — **retire on the next rebase past upstream #631** (opened 2026-07-06, MERGEABLE, no reviews yet).

---

## 43. `fix(gpt-oss): plumb harmony tool calls through to response` — cherry-pick of upstream #562

**Files:** `vllm_mlx/server.py`, `vllm_mlx/tool_parsers/harmony_tool_parser.py`, `vllm_mlx/api/utils.py`, `tests/test_harmony_parsers.py`, `tests/test_server.py`

Cherry-picks upstream open PR [#562](https://github.com/waybarrios/vllm-mlx/pull/562) (CBribiescas, head `98d4f83` incl. the owner-review fix). **gpt-oss-20b tool calling was broken on our tree** — `--tool-call-parser harmony` returned no `tool_calls`; arguments came back glued into `content` with `finish_reason: "stop"`. Two independent layers, both fixed:

1. **server.py `_extract_reasoning_and_tool_calls`:** when harmony reasoning was extracted but there was no `<|channel|>final` block (i.e. *every* gpt-oss tool call — the model jumps from analysis straight to `commentary to=functions.*`), the tool-parser input was blanked to `""`. Now preserves raw `output_text` for the parser, gated on `request.tools` (without tools the parser is skipped and raw harmony tokens would leak into content — the owner's flagged regression, fixed in `98d4f83`).
2. **harmony_tool_parser.py:** `_COMMENTARY_BLOCK_PATTERN` hard-required a `<|call|>` terminator, which is consumed as a stop sequence and never appears in `output.text`. Now also matches at EOS (named `terminator` group; `matched_at_eos` drops silently-truncated JSON instead of emitting raw-args garbage).

Plus a defensive `clean_output_text` bypass for commentary tool blocks in `api/utils.py` (load-bearing on the streaming-Responses/Anthropic paths).

**Hand-merge note:** the server.py hunk collided with patch #27's `if not allow_reasoning:` fold block (inserted right after the replaced line). Resolution: PR's tools-gated assignment first, #27's fold kept after it — they compose (when raw output is preserved, the fold's empty-content condition doesn't fire; with tools absent, #27 semantics unchanged).

**Conflict surface:** only that one server.py region; `harmony_tool_parser.py`/`api/utils.py` carried no fork patches.

**Status:** TEMPORARY cherry-pick — **retire on the next rebase past upstream #562** (MERGEABLE; collaborator-approved, owner re-review pending; PR self-closed/reopened once for staleness, so upstream timing is uncertain).

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
