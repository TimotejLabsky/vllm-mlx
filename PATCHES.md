# Local patches in this fork

This fork carries 11 patches on top of [`waybarrios/vllm-mlx@7e30484`](https://github.com/waybarrios/vllm-mlx/commit/7e304840). Each patch is a separate commit on `main` with the prefix `patch:`. They are listed here in apply order (bottom of git log → top).

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

## Future work / prospects

Open upstream PRs/issues worth tracking — not yet applied here, with the reasoning:

- **[PR #541](https://github.com/waybarrios/vllm-mlx/pull/541) — multi-slot LRU for system-KV.** Replaces the single-slot snapshot (PR #523) with a 4-slot LRU keyed by system-prefix hash. Directly addresses the opencode main-agent / sub-agent thrash pattern. Touches the same code as our patch #9 (`cache-extended-prefix`) but is **incompatible by construction** — they take opposite design directions (LRU over many short prefixes vs. one grow-on-HIT slot with the full conversation). Adopting #541 cleanly would mean rewriting patch #9 to thread grow-on-HIT through the LRU values. Plan: rebase #541 in as a replacement for #9, port grow-on-HIT onto the LRU values, real-load test with two opencode agents alternating before keeping. Substantial — deferred until a session needs it.

- **[PR #233](https://github.com/waybarrios/vllm-mlx/pull/233) — TurboQuant KV cache compression (4.6×).** Compresses prefix cache via PolarQuant (Hadamard rotation + Lloyd-Max codebook). Lets us cache more agent prefixes inside the same 64 GB envelope. Blocked on upstream MLX-LM PR #1067 landing. Big patch (~46K LOC across 100 files) — wait for upstream maturity.

- **[PR #528](https://github.com/waybarrios/vllm-mlx/pull/528) — canonicalize volatile system headers.** Strips `x-anthropic-billing-header` (with rotating `cch=<hash>` tokens) from Chat Completions + Responses system prompts. Reported 13.7× TTFT improvement on Gemma 4 26B with 60K-token prompts when the client is Claude Code. **Open question:** does opencode (which talks OpenAI Chat Completions through LiteLLM) inject any volatile content into its system prompts? If yes, this gives us free cache hits we're missing today. Worth a one-session inspection of the actual on-wire prompt before adopting.

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
