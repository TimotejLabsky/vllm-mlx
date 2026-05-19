# Local patches in this fork

This fork carries 10 patches on top of [`waybarrios/vllm-mlx@7e30484`](https://github.com/waybarrios/vllm-mlx/commit/7e304840). Each patch is a separate commit on `main` with the prefix `patch:`. They are listed here in apply order (bottom of git log → top).

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
