# Apple-Silicon engine field survey — 2026-08-18

> **Status note (added 2026-09-03, when this doc was finally committed):**
> this is a historical snapshot, written 2026-08-18 at fork `5c16039` and left
> untracked until now. Most of its GO items have since shipped — read it for
> the *reasoning*, not the to-do list:
>
> - Item 0 (deploy #73/#74/#75): done 2026-08-18 (`373cca0`).
> - P0.5 (instrumented session): superseded by patch #85 durable verdict
>   counters (2026-08-31); day-2 read = 50 evictions / 200 reuses /
>   **0 evict_to_reuse** → item 3(a)'s single-entry-collapse hypothesis is
>   trending REFUTED as harmful; final read due ~2026-09-07.
> - Item 2 (message-boundary checkpoints): shipped as patch #88.
> - Item 10 (DRY holes): fork half shipped as #86; then 2026-09-01 found DRY
>   itself corrupts tool JSON (#93 + the LiteLLM fix) and 2026-09-02 measured
>   DRY as a 3.7% decode tax — the pending call is now *dropping* DRY on
>   4-bit, not widening it.
> - Item 11 (silent empty turns): shipped as #87; the 2026-09-01 diagnosis
>   went further (engine `inference_requests_total{result="error"}` is ground
>   truth; #87 misses raise-before-first-token).
> - Item 8: the stop-vs-grammar half shipped as #89; the per-token
>   `tolist()`/O(n²) suffix-decode micro-fix in the llguidance processor is
>   still open.
> - Item 1 Phase A (Qwen3.8 deep-context ladder): still open.
> - MTP/DFlash sections: superseded by the 2026-09-02 lever-5 measurements
>   (DFlash2 = first >1x on M1: code 1.14x, but chat 0.82x — still don't
>   build).
> - Current canonical state:
>   [`speed-lever-ledger-2026-09.md`](speed-lever-ledger-2026-09.md).

Scoped-plan companion to [`improvement-roadmap-2026-08.md`](improvement-roadmap-2026-08.md)
(2026-08-17, 4-track ecosystem research). That round surveyed *upstreams*
(MLX/mlx-lm, llama.cpp/GGML, Mac serving stacks, big-backend serving layers).
This round surveys *peer engines* — llama.cpp server, oMLX (`jundot/omlx`),
vMLX, vllm-metal, LM Studio `mlx-engine` — and asks, item by item, whether
what they shipped is something this fork lacks.

Fork state at survey time: `5c16039`, patch #75, upstream base `5021350`
(v0.4.1). Serving box: Mac Studio M1 Ultra 64 GB, single user, llama-swap
`exclusive: true` heavy group. Nothing below is implemented — this is the
approve-item-by-item list.

**Deploy status (checked against `origin/main` of `personal-infratructure`):
patches #73/#74/#75 are NOT deployed.** The deployed-state table's top row is
still `2026-08-11 | c4f6a2a` (patch #72); no 08-18 row and no infra PR #326
exist in `origin/main`, contradicting a memory note that claims otherwise.
Everything in P0.5 that reads patch-#74 metrics is blocked until that deploy
lands — **deploy first**.

**Primary route is `Qwen3.8-27B-8bit`, not 3.6.** Infra PR #322 (2026-08-17)
added the Qwen3.8-27B 8-bit and 4-bit routes and flipped the opencode default
off 3.6. Same `qwen3_5` arch class, so the hermes/qwen3 parsers and the whole
system-KV hybrid stack carry over unchanged; rails were **cloned** from the 3.6
route (`VLLM_MLX_SYSTEM_KV_RAM_MB=6144`,
`VLLM_MLX_MAX_PROMPT_TOKENS=122880` — `mac-studio/llama-swap-config.yaml:424,430`)
and the route comment records the open caveat: *"deep-context ladder NOT yet
re-measured on 3.8 — treat >96K as unproven."* That caveat is exactly what
P0.5 and item 1 Phase A close.

**Every external claim in the brief was checked against its primary source.**
All verified except one, which is materially wrong and is corrected in
[Closed items](#closed-items-corroboration-and-one-correction).

---

## Verdict table

| # | Item | Verdict | Effort | Risk |
|---|------|---------|--------|------|
| 0 | **Deploy patches #73/#74/#75** | **GO — blocks everything** | one install-only deploy | low |
| — | **P0.5 (new): one instrumented production session** | **GO — right after the deploy** | ~0 (config + read gauges) | none |
| 1 | Long-context decode ladder vs llama.cpp | **GO, downscoped** | 1 idle window (MLX arm), +1 day (llama.cpp arm) | medium (llama.cpp arch support) |
| 2 | Checkpoints at user-message boundaries | **GO — the one clear build** | ~150 LOC + tests, 1 session | low–medium |
| 3 | Quadratic → linear mixed-CacheList storage | **DROP as a build** (we don't have the bug) — but two real residuals surfaced, folded into P0.5 | — | — |
| 4 | Prefix reuse across tool-adjacent messages | **DEFER** — mechanism already present, fire-rate unverified | measurement only | none |
| 5 | int4-KV with fused Metal attention | **DEFER (watch)** with a named trigger | very high if built in-fork | very high |
| 6 | Bounded SSD sidecars for GDN state | **DROP** | — | — |
| 7 | Repeat-prefill wall in (137K, 148K] | **CLOSED BY CONFIG** — item 5 is not the structural fix | — | — |
| 8 | Grammar-decode per-token sync | **GO, low priority, re-scoped** — does *not* touch the tool-call path | ~40 LOC | low |
| 9 | Decode fairness during concurrent prefill | **DEFER** — quantify first; scenario is rarer than assumed | measurement only | none |
| 10 | DRY coverage holes | **GO** — two independent fixes, one config, one fork bug | ~10 LOC + 4 config lines | very low |
| 11 | Silent empty turns | **GO** — observability only | ~30 LOC + 1 alert | very low |
| — | Upstream patch #19 | **GO in principle**, retarget to mlx-lm #980 | Tim's call on PRs | — |

---

## P0.5 (new) — the measurement that actually gates the rest

The brief nominated item 1 as the gate. Grounding the other items surfaced a
**cheaper gate that answers three of them at once**, needs no code, no model
probing, and no idle window. Recommend running it before anything else.

One production opencode session on `Qwen3.8-27B-8bit` with
`VLLM_MLX_DEBUG_PROMPT_CAPTURE=1` (`vllm_mlx/server.py:5502`), then read:

- `/v1/status` → `system_kv_cache.{entry_count, memory_mb, max_memory_mb,
  memory_utilization, evictions, hits, misses, partial_hits, tokens_saved,
  partial_tokens_saved, grown_stores, boundary_stores}`
  (`vllm_mlx/batched_system_kv.py:909-948`)
- the patch-#74 histograms, especially
  `rate(vllm_mlx_cache_evict_to_reuse_gap_seconds_count[1d])` and
  `histogram_quantile` over `idle_before_evict` vs `reuse_gap`
- the captured prompts, diffed pairwise across a tool-call turn

What it settles:

1. **The RAM-budget hypothesis (item 3's real residual).** See below — if
   `entry_count` sits pinned at 1 with `evictions` climbing, the batched cache
   is structurally single-entry at production depth and several of the fork's
   divergent-chain features cannot fire at all.
2. **Item 4** — whether tool turns restore at the prompt boundary as designed.
3. **Item 1's MLX arm** — decode t/s vs depth comes out of the same session for
   free, and decides whether the expensive llama.cpp arm is worth funding.

**Blocked on the #73/#74/#75 deploy** — readings 1 and 3 work on the current
build, but reading 2 (the histograms) does not exist on the deployed `c4f6a2a`.

It also closes a question PATCHES.md has carried open since the PR #528 entry
("run one opencode session with `VLLM_MLX_DEBUG_PROMPT_CAPTURE=1` and diff two
consecutive turns' system prefixes").

---

## P0 — 1. Long-context decode gap vs llama.cpp — **GO, downscoped**

**Claim verified.** [mlx-lm #763](https://github.com/ml-explore/mlx-lm/issues/763),
"Token Generation Performance on long context consistently 50% lower than
Llama.cpp", opened 2026-01-15, still open. M3 Ultra 256 GB / 80-core GPU,
Minimax-m2.1-4bit. 30K ctx: MLX 25 t/s vs llama.cpp+FA 32 t/s. 146K ctx:
**5.95 vs 12.12 t/s**.

**Does the fork inherit it?** Structurally, most likely yes — but far less than
2x, and we may already have the data.

- The gap is in the *attention kernel at depth*, not in cache management. The
  fork replaces cache **management** (`batched_system_kv.py`), never
  `mx.fast.scaled_dot_product_attention`. BatchedEngine decodes through
  mlx-lm's `BatchGenerator`, so the same SDPA path applies.
- But `Qwen3.6-27B` is **hybrid** (Gated DeltaNet + attention — the whole
  reason patches #34–#40 exist). Only a subset of layers carry a growing KV;
  the recurrent layers are O(1)-state and flat in depth. A hybrid backbone
  should degrade materially *less* than the full-attention model in #763.
- Existing fork data supports that: the 2026-07-09 ladder measured 27B-8bit (on 3.6)
  decode **15.4 → 11.9 t/s** from shallow to 112K — a 1.29x falloff, versus
  #763's 4.2x from 30K to 146K.

**Scoped plan.**

*Phase A (cheap, do inside P0.5).* Re-derive the decode-vs-depth curve at ~8K /
32K / 90K on `Qwen3.8-27B-8bit`, T=0, fixed 200-token output, warm-cache state
held constant across rungs. If the falloff stays ≲1.5x, the #763 pathology does
not reproduce on our arch and the llama.cpp arm is **not funded**.

*Phase B (only if Phase A shows ≳1.5x).* llama.cpp arm, same box, idle window:
`-fa` on, Q8_0 GGUF of the same model, same three depths, same output length.

**Prerequisites and risks for Phase B, in order:**

1. **Arch support is the real gate** — a Q8_0 GGUF of Qwen3.6-27B must exist
   *and* llama.cpp must support its hybrid GDN layers. Verify before
   downloading ~29 GB.
2. `exclusive: true` means the two engines cannot be co-resident; the ladder is
   two serial passes with a swap between, so idle-window cost roughly doubles.
3. Confounder: llama.cpp's own hybrid checkpoint invalidation is broken
   ([llama.cpp #24055](https://github.com/ggml-org/llama.cpp/issues/24055),
   open) — pin both arms to *decode* rate at a fixed depth and ignore TTFT, or
   the caches make the numbers meaningless.

**Honest limit on the payoff:** even a confirmed 2x gap is not fork-fixable.
The fix lives in mlx core (track
[mlx #2955](https://github.com/ml-explore/mlx/issues/2955), FlashAttention /
PagedAttention integration). The value of the measurement is expectation-setting
and evidence for item 5 — not a patch.

**Run it on `Qwen3.8-27B-8bit`** (`mac-studio/llama-swap-config.yaml:419`) —
the current opencode default since infra PR #322. This does double duty: the
route ships with `>96K` explicitly marked unproven because its deep-context
ladder was never re-measured after the 3.6 → 3.8 flip, so Phase A closes a
standing gap regardless of what it says about #763.

---

## P1 — 2. Checkpoints at user-message boundaries — **GO (highest-value build)**

**Claims verified.**
[llama.cpp #24176](https://github.com/ggml-org/llama.cpp/pull/24176) merged
2026-06-23: checkpoints at the start of **every** user message; **tool
responses are treated as user messages**; `checkpoint_min_step` default raised
256 → 8192; the last user message's checkpoint bypasses min-step. It scans for
boundaries **directly on tokens**, dropping an earlier byte-offset translation
pass. [llama.cpp #25472](https://github.com/ggml-org/llama.cpp/pull/25472)
merged 2026-07-12: evicts checkpoints falling within min-step of an earlier
one, tracking task IDs so the penultimate checkpoint is not destroyed by the
final one; a follow-up makes the near-prompt-end checkpoint unconditional.

**What we do today.**

- `BatchedSystemKV.split_segments` (`vllm_mlx/batched_system_kv.py:288`) splits
  the prompt into **uniform** `VLLM_MLX_BATCHED_KV_CKPT_INTERVAL` = 2048 chunks.
- Capacity `VLLM_MLX_SYSTEM_KV_CHECKPOINTS` = 8 (`batched_system_kv.py:204`).
- Thinning is **drop-every-other geometric** (`system_kv.py:309-323`) —
  content-blind, so it discards message-aligned positions indiscriminately.
- Floor `VLLM_MLX_SYSTEM_KV_PARTIAL_MIN` = 256 (`batched_system_kv.py:205`).

**Why this is cheap for us specifically.** The seam already has everything:

- `insert_segmented` (`batched_system_kv.py:1116`) receives the `request`, and
  `Request.prompt` holds the **rendered prompt string** (`vllm_mlx/request.py:97`);
  tokenization happens later in `Scheduler.add_request`
  (`vllm_mlx/scheduler.py:1936-1955`).
- Patch #20 already ships `TEMPLATE_MARKERS` + `detect_template_markers`
  (`vllm_mlx/system_kv.py:157-185`) with verified user/gen markers for all
  seven deployed template families. Following llama.cpp's lesson, resolve
  boundaries on the **token** stream (the markers are special tokens with
  stable ids) rather than translating byte offsets.

**Proposed change, two parts.**

1. *Placement.* `split_segments` splits at message boundaries **in addition
   to**, not instead of, the 2048 interval — boundaries are added, and
   `ckpt_interval` stays a hard upper bound on segment length. Apply a
   min-step (start ~2048, evaluate 8192) so a burst of short turns doesn't
   shred the ladder, and always keep the boundary nearest the prompt end.
2. *Thinning.* Replace drop-every-other in `append_checkpoint` with min-step
   eviction that prefers keeping boundary-aligned positions and never evicts
   the last two.

**Risk (low–medium), and the mitigation is the same in both cases.** Segment
boundaries also set prefill chunk size, and the peak-memory profile that
patches #48/#53 rails are tuned against depends on it. A message-aligned split
could otherwise emit one enormous segment (a 60K-token tool result). Keeping
`ckpt_interval` as an upper bound removes that failure mode and also protects
item 9 (fairness) from regressing.

**Acceptance.** T=0 byte-identical output vs cold on a ≥2K-token prompt (the
standing gate), plus a unit test asserting the restore position lands on the
message boundary rather than the next lower 2048-multiple, plus a
divergent-chain test at ≥2K.

---

## P1 — 3. Quadratic → linear mixed-CacheList storage — **DROP as a build; two residuals found**

**Claim verified verbatim.** oMLX 0.6.0: "The affected Inkling workload dropped
from 282.7 GB at 84K tokens to 15.3 GB at 94K tokens while preserving
byte-identical restores"; legacy mixed-CacheList SSD blocks are invalidated
automatically.

**We do not have this bug, and we avoided it deliberately.** Two independent
reasons:

1. Patch #19's founding insight (PATCHES.md §19): only **ckpt-class** layers
   (recurrent `ArraysCache`, `RotatingKVCache`) get per-position checkpoints,
   and those are **fixed-size**. Attention KV at any position is recovered by
   slicing the single final snapshot. So the per-checkpoint cost never scales
   with position — which is exactly the quadratic term oMLX removed.
2. Patch #37 already replaced the per-turn O(chain) materialised copy with
   **segment lists** — donor segments by reference plus one O(delta) evaluated
   slice (`batched_system_kv.py:355-410`).

**But the audit surfaced two residuals worth fixing cheaply.**

**(a) Byte accounting double-counts shared segments — likely collapsing the
cache to one entry at production depth.**
`_entry_nbytes` (`batched_system_kv.py:84-97`) sums every segment of an entry,
and `_enforce_budgets_locked` (`:649-660`) evicts while
`sum(e["bytes"] for e in self._entries.values()) > budget`. Segments shared by
reference between a donor and its grown successor are therefore counted **once
per holder**. PATCHES.md §37 documents this as a deliberate conservative
overcount ("earlier eviction, the safe direction") — but the deployed numbers
suggest it is not a small overcount at our depths:

- `VLLM_MLX_SYSTEM_KV_RAM_MB=6144` on the opencode route — cloned unchanged
  from 3.6 onto the current 3.8 route
  (`mac-studio/llama-swap-config.yaml:424`)
- measured ≈200 KB/token (PATCHES.md §39/§40) ⇒ the budget is worth roughly
  **30K tokens**
- production prompts are **90–137K**

If that holds, the budget loop is permanently in its terminal state, with the
`len(self._entries) > 1` guard as the only thing preventing total eviction —
i.e. **exactly one entry survives**. Every feature that needs two resident
chains (divergent-chain partial restore across sessions, interleaved-session
reuse) then cannot fire *by construction*, while hit-rate looks healthy because
the single surviving chain keeps growing.

This is a **hypothesis, not a finding** — it is decided by `entry_count`,
`memory_utilization` and `evictions` in the P0.5 session, plus patch #74's
`evict_to_reuse_gap` (traffic there = the policy discarded something still
needed; PATCHES.md §74 explicitly nominates that metric as the verdict).
If confirmed, the fix is small: account shared segments once (identity set per
budget pass), and/or raise `VLLM_MLX_SYSTEM_KV_RAM_MB` on the deep routes.

**(b) `_SEGMENT_CONSOLIDATE_AT = 16` (`batched_system_kv.py:158`)** forces an
O(chain) concat every ~16 grows. Amortised that is linear, but it is a
multi-GB transient at 90K+ landing in the exact regime the #48/#53 rails
police. Consider a byte-denominated trigger instead of a piece count. Low
priority; note it, don't build it yet.

---

## P1 — 4. Prefix reuse across tool-adjacent messages — **DEFER (measure in P0.5)**

**Claim verified, but it is not the mechanism the brief assumed.** oMLX 0.6.1
(2026-08-17) reads: "Claude Code system reminders that follow tool results can
remain in place when the model template safely supports them, avoiding repeated
large prefills". That is a **template/history-rendering** fix on the client
side — not an engine cache-matching fix.

**What our path actually does.** LCP runs over raw token ids, so a tool turn is
a pure extension *provided the re-rendered history is byte-stable*. It usually
is not, and the reason is thinking, not tools: the KV chain from turn N contains
the model's generated `<think>` tokens, while turn N+1's re-rendered prompt
drops them. The chain therefore diverges at **the start of turn N's assistant
response** — which is precisely where patch #35's `store_prompt_boundary` places
an entry, and PATCHES.md §34 notes mlx-lm's last-token split yields a free
checkpoint at prompt-boundary−1. So the tool loop should restore ≈100% of the
prior prompt and re-prefill only the small assistant+tool-result tail.

**The mechanism is present; what is unverified is whether it fires.** No build
until P0.5's captured prompts and `partial_hits` / `tokens_saved` counters say
otherwise. If the diffs show something *else* volatile in the middle of the
history (timestamps, session ids, tool reordering, injected reminders), that is
a real finding and this item reopens with a concrete target.

---

## P1 — 5. int4-KV with fused Metal attention — **DEFER (watch), named trigger**

**Claims verified.**
[arXiv 2604.16957](https://arxiv.org/abs/2604.16957) "Open-TQ-Metal: Fused
Compressed-Domain Attention for Long-Context LLM Inference on Apple Silicon"
(2026-04-18): int4 KV quantised on the fly with attention computed directly on
the compressed representation; across 330 experiments on Gemma-4-31B and
Llama-3.1-70B, **48x attention speedup at 128K** over dequantise-then-attend,
KV **40 GB → 12.5 GB** (3.2x), identical top-1 predictions vs FP16.

[mlx #3404](https://github.com/ml-explore/mlx/issues/3404) (quantized KV in
`scaled_dot_product_attention`), opened 2026-04-12, is **CLOSED** — the closing
comment was not recoverable from the page, so **do not assume it landed**.
Installed mlx 0.31.2 on the dev box exposes only
`mx.fast.scaled_dot_product_attention` with no quantized variant; verify on the
fleet's mlx 0.32 before treating this as available.

**Fork state.** `kv_bits` / `kv_cache_quantization*` exist but live entirely
inside `MemoryAwarePrefixCache` (`vllm_mlx/memory_cache.py:1017-1024`,
`vllm_mlx/scheduler.py:1361-1370`) — and that is **storage-side** quantization
of cached entries, not compute-side quantized attention. Our routes *replace*
that cache with `BatchedSystemKV` (PATCHES.md §34), so these flags are inert on
every production route. This confirms the standing "kv-quant flags are inert
hygiene" verdict, and clarifies it: they were never the same thing as item 5.

**Why not build it in-fork.** The blast radius is the whole cache stack. Every
snapshot invariant assumes plain float arrays: `_segments_upto` /
`_slice_segments` (`batched_system_kv.py:101-155`),
`build_partial_restore_states` (`system_kv.py`), and the SSD safetensors
checkpoint format v3 would each need a quantized variant, and the T=0
byte-identical gate would have to be renegotiated (the paper reports identical
top-1, not identical bytes).

**Trigger to reopen:** mlx core ships a quantized SDPA path. Track mlx #3404
and #2955. Also relevant: item 5 does *not* solve item 7 — see below.

---

## P2 — 6. Bounded SSD sidecars for GDN recurrent state — **DROP**

**Claim verified.** oMLX 0.5.8.dev3 / 0.6.0: "GDN recurrent state now uses
bounded SSD sidecars by default… keeps recurrent state separate from ordinary
KV storage, with RHT-INT16 reducing storage by 1.93x versus FP32", restored in
full precision.

**Comparison.** Ours persists recurrent state *inside* the entry's safetensors
(checkpoint format v3, `c{n}_l{i}_s{j}` keys — PATCHES.md §19/§36), bounded by
the ≤8-checkpoint ladder and `VLLM_MLX_SSD_SYSTEM_KV_GB=20`.

**Verdict: nothing to gain.** Their 1.93x applies to the recurrent term, which
in our layout is already the *small* term — fixed-size per checkpoint, at most
8 checkpoints per entry. The dominant SSD cost is attention KV, which their
change does not touch. Their separation-of-lifetime argument is real but buys
us nothing on a single-user box where entries are already bounded per model
subdir. Recorded here so it is not re-litigated.

---

## P2 — 7. The repeat-prefill wall in (137K, 148K] — **CLOSED BY CONFIG**

**Verified locally.** `mac-studio/README.md:112` records crash #24
(`Python-2026-07-19-125808.ips`): 148,105-token cold prefill PASS, then the
identical repeat SIGABRTs with `kIOGPUCommandBufferCallbackErrorOutOfMemory` in
a completion handler, with patch #53 relief already armed. 137K-twice is proven
clean. Caps are already aligned everywhere — litellm `max_input_tokens` and
opencode `context` at 137000, `VLLM_MLX_MAX_PROMPT_TOKENS=138000` on both 45 GB
routes (`mac-studio/llama-swap-config.yaml:201,268`).

**Is item 5 the structural fix? No — partially at best.** int4-KV lowers the
*steady-state* KV baseline and buys headroom, but the crash is a **transient**
allocation failure during prefill, in a regime where relief cannot evict
weights. Lowering the baseline moves the wall; it does not remove the failure
mode.

**The structural lever is bounding the prefill transient, and it is unexplored:**
`chunked_prefill_tokens` defaults to **0 = disabled** (`vllm_mlx/scheduler.py:119`),
so our only chunking is the 2048-token `insert_segments` split. Upstream #648's
chunked prefill was assessed "inert + defective" in the 2026-07-28 review, but
it is now **in the base** after the v0.4.1 rebase — worth a re-read before
anyone spends money here.

**Verdict:** no work. The 137K cap is the correct answer. Reopen only if a
route genuinely needs >137K, and then start with chunked prefill, not KV
quantization.

---

## P2 — 8. Grammar-constrained decoding per-token sync — **GO, low priority, re-scoped**

**Claim verified.** oMLX 0.5.8.dev3: "Removed a per-token host sync from
grammar-constrained decoding. Token acceptance is deferred to the top of the
next step with bit-identical output."

**We have the equivalent stall — and something worse.**
`LLGuidanceJSONSchemaLogitsProcessor.__call__`
(`vllm_mlx/constrained/llguidance_schema_processor.py:169-206`) does, per token:

- `_token_list(tokens)` → `tokens.tolist()` (`:28-34`) — a **host sync** every
  step, the exact thing oMLX removed;
- `self._tokenizer.decode(suffix, …)` (`:191-195`) — re-decodes the **entire**
  generated suffix every token, i.e. **O(n²) CPU** across a generation, purely
  to count trailing JSON whitespace.

**But the brief's premise does not hold for us: this never touches the
tool-call path.** Constrained processors attach only via `response_format`
(`_prepare_json_logits_processor`, `vllm_mlx/server.py:461-519`, wired at
`:778-787`). Tool calls are template-driven and parsed after the fact
(hermes / mistral / qwen3-xml parsers) with no grammar attached. And
`response_format` is barely used in the lineup — that was the stated reason
#636 was deferred in the first place (`mac-studio/README.md:112`).

**Scope:** two mechanical, local fixes — track the suffix incrementally instead
of `tolist()`-ing the whole token array; keep a rolling decoded tail instead of
re-decoding. Both bit-identical, ~40 LOC. **Do it when next touching #73** for
the strict-tool-argument-schemas follow-up already listed in the 08-17 roadmap;
not worth a session of its own.

---

## P2 — 9. Decode fairness during concurrent prefill — **DEFER (quantify first)**

**Claim verified.** oMLX 0.6.0.dev1 / 0.6.0: "Prefill now yields GPU time to
active decodes, including streams running in another engine, and sizes
contended chunks by a target stall time instead of a fixed token count" —
decode throughput during concurrent prefill improved **1.6x to 43x**.

**Our current behaviour, precisely.** Prefill *is* already chunked, at a
**fixed** 2048-token granularity, via `insert_segmented`
(`batched_system_kv.py:1116`) → `split_segments` (`:288`); the generator returns
between segments. What we lack is stall-time-targeted sizing and any explicit
decode priority. Upstream's own chunked prefill is off
(`chunked_prefill_tokens = 0`, `vllm_mlx/scheduler.py:119`).

**The motivating scenario is rarer than assumed.** The heavy group is
`exclusive: true`, and the HA voice routes point at *different* models
(`ha-llm` → Qwen3.6-27B-4bit, `ha-gptoss`, `ha-qwen-moe` →
`kubernetes/litellm/config.yaml:254,304,323`). A voice turn arriving during an
opencode prefill triggers a **model swap**, not co-batching — the two only
contend when the voice route resolves to the same loaded model.

**Measure before deciding.** Two-request probe: a deep prefill plus a short
request, measuring the short request's TTFT and inter-token latency. If the
stall is bounded by roughly one segment, there is nothing to fix. Reuse
`mac-studio/benchmark/benchmark.py`.

**Interaction with item 2:** message-aligned segments can be *larger* than 2048
and would make fairness worse. This is the second reason item 2 must keep
`ckpt_interval` as an upper bound.

---

## P3 — 10. DRY coverage holes — **GO (two independent fixes)**

**(a) Config gap — confirmed, and narrower than the brief implies.** DRY is
carried on all four dense-27B routes, including both new Qwen3.8 ones
(`kubernetes/litellm/config.yaml:102, 134, 157, 178`). The gap is the
**35B-A3B family only**: `Qwen3.6-35B-A3B-4bit` (`:204`),
`Qwen3.6-35B-A3B-4bit-fast` (`:226`) and `ha-qwen-moe` (`:323`) each have an
`extra_body` block with **no `dry_*` keys** — exactly the routes that produced
block-repetition runaways to the 32768 cap.

Mirroring the 4-line block is the obvious fix. **Recommended instead:** set the
engine-side `VLLM_MLX_DRY_*` env defaults on those llama-swap routes
(the per-model default path the server explicitly defers to —
`vllm_mlx/server.py:796-802`), and keep the litellm block as a per-route
override. Env defaults cover *every* client including direct llama-swap
requests, which the gateway block structurally cannot — the same residual
recorded in `fleet-batched-flip`.

**(b) Fork bug — confirmed, and it is not just `/v1/completions` being
unhooked.** `CompletionRequest` **declares** the DRY fields
(`vllm_mlx/api/models.py:316`), so callers get a 200 and reasonably assume they
took effect — but neither completion handler forwards them:

- non-streaming `create_completion` → `generate_kwargs`
  (`vllm_mlx/server.py:5297-5310`)
- `stream_completion` → `generate_kwargs` (`vllm_mlx/server.py:6595-6614`)

Both omit every `dry_*` key, unlike chat (`:796-802`) and Anthropic
(`:942-948`). A declared-but-ignored parameter is worse than an absent one.
~10 LOC + one test per path.

---

## P3 — 11. Silent empty turns — **GO (observability only)**

Two layers of masking, both outside the fork: llama-swap returns a dead upstream
stream as HTTP 200 with an empty completion, and LiteLLM converts upstream
`finish_reason: "error"` into `"stop"` with `content: None`. Don't try to make
llama-swap propagate — **detect at both ends**.

Smallest change, in cost order:

1. **Fork side — 90% already shipped.** Patch #74 added
   `vllm_mlx_finish_reasons_total{endpoint,finish_reason}` via an optional
   `finish_reason` on `InferenceTracker.finish` (PATCHES.md §74), and the
   `tracker.finish(result="error")` call sites already exist
   (`vllm_mlx/server.py:4287,4299,4310,4589,4598,4647,4656,5341,5584,6015`).
   Add one counter for the case the gateway erases:
   `vllm_mlx_empty_completions_total`, incremented when a request finishes with
   `completion_tokens == 0`. This is ground truth from the only layer that
   knows.
2. **LiteLLM side.** A `post_call` callback counting responses where
   `finish_reason == "stop"` with empty content and zero completion tokens —
   catches the case where the fork never saw the request at all (llama-swap
   wedge, the 2026-07-14 shape).
3. **Alert.** `increase(vllm_mlx_empty_completions_total[15m]) > 0`.

---

## Also worth considering — upstreaming patch #19

**Agreed, and the evidence got stronger.**
[llama.cpp #25913](https://github.com/ggml-org/llama.cpp/issues/25913) (opened
2026-07-20, **open**): `/slots` save/restore reports success on hybrid/recurrent
models but delivers nothing — a 14,906-token prefix "restores" in 0.12 s and
then recomputes all 14,914 tokens, against a 75.9 s original prefill. Root
cause: the save file never contains context checkpoints, so
`slot.prompt.checkpoints` is empty after restore. In-memory caching works
(97.9% hit rate); only the disk path is broken. Meanwhile
[mlx-lm #980](https://github.com/ml-explore/mlx-lm/issues/980) (hybrid prefix
caching) is still open.

**Retarget suggestion.** The 08-17 roadmap already nominated **mlx-lm #980** as
the upstreaming opening, and that is the better target than a vllm-mlx PR:
`system_kv.py` is engine-agnostic by construction (patch #18), #980 is the
acknowledged gap, and llama.cpp #25913 supplies the "the whole ecosystem lacks
this" framing plus a number. PRs remain Tim's call.

---

## Closed items — corroboration, and one correction

### MTP / speculative decoding — stays closed, corroboration verified

- [oMLX #1311](https://github.com/jundot/omlx/issues/1311) (2026-05-19, **open**),
  M1 Max 64 GB — our hardware class. Warm 32K, 200-token decode: PARO baseline
  **31.3 t/s**; Qwen 27B oQ6-mtp **7.4 t/s (−76%)** at 70.3% accept with
  sampling at 66% of decode wall time; 35B-A3B oQ4-mtp **25.5 t/s (−18%)** at
  73.8% accept, sampling 60%. Title is explicit: "compute-bound verify, not the
  sampler".
- [llama.cpp #23752](https://github.com/ggml-org/llama.cpp/issues/23752)
  (2026-05-27, closed), M1 Max 32 GB, Qwen3.5-9B-MTP Q4_K_M: baseline 25.3 t/s
  → 22.4 (`n_max=0`, −11%) → 21.9 (`n_max=2`, −13%) → 19.3 (`n_max=6`, −24%).
  *Minor correction:* the brief quoted "−11% to −28%"; the measured range is
  **−11% to −24%**. Doesn't change the verdict.

Third independent confirmation of `speculative-decoding-dead-on-mseries`.
Permanently closed for this hardware.

### DFlash — **the premise is wrong; the item's status should change for a different reason**

The brief flagged this as the one genuinely new finding. It doesn't survive
checking.

[oMLX #1441](https://github.com/jundot/omlx/issues/1441) (2026-05-27) is
**CLOSED**, and its associated
[PR #1768](https://github.com/jundot/omlx/pull/1768) — "fix(dflash): report
prefix-cache hits as cached_tokens", **merged 2026-06-10** — records that the
underlying performance problem was already resolved and what remained was a
**reporting** gap: with DFlash enabled the API returned `cached_tokens: 0`
while the cache was in fact working. The fix maps `PrefixCacheFlow.hit_tokens`
to `cached_tokens`.

So "DFlash breaks the KV prefix cache outright" is a **retracted** claim, not
evidence against upstream issue #502.

This does **not** make DFlash a GO. It stays parked — but on the honest reason:
it is an external block-diffusion draft model requiring verification against the
target, and M-series draft-model economics are the same ones that killed MTP
three times over. It is also still unimplemented upstream. Record the corrected
reasoning so the next reviewer doesn't cite a withdrawn cache-breakage claim.

*(Caveat worth noting: #1441's original report did describe real 45–85 s
prefills at 100K that vanished on disabling DFlash. The merged PR asserts the
perf half was fixed separately. If DFlash ever becomes interesting, re-verify
rather than trusting either statement.)*

### Switching engines — stays closed

llama.cpp #25913 verified as stated above. Combined with the `exclusive: true`
heavy group restarting the process on every swap, oMLX having no DRY and no
Prometheus, LM Studio's `/tmp`-only disk cache cleared on unload, and
vllm-metal having no disk KV tier — the fork remains the only engine with both
DRY and restart-surviving hybrid KV. No change.

---

## Recommended sequence

0. **Deploy #73/#74/#75.** They are committed and not deployed; P0.5's second
   reading is patch-#74 metrics, so nothing below starts until this lands.
1. **P0.5** — one instrumented opencode session. Zero code. Settles item 3's
   RAM-budget hypothesis, item 4, and item 1 Phase A.
2. **Item 2** — message-boundary checkpoints. The one clear build.
3. **Item 10** — both halves (4 config lines / env block + ~10 LOC fork fix).
4. **Item 11** — `vllm_mlx_empty_completions_total` + LiteLLM callback + alert.
5. **Item 8** — bundled with the #73 strict-tool-schema follow-up.
6. **Items 1 Phase B, 5, 9** — funded only if step 1 says so.
7. **Items 3-as-build, 6, 7** — dropped / closed by config; reasoning recorded
   above.
