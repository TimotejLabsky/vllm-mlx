# Improvement research — 2026-08-30

Delta pass on top of [`improvement-roadmap-2026-08.md`](improvement-roadmap-2026-08.md)
(08-17, upstream ecosystems) and `engine-field-survey-2026-08-18.md` (peer
engines, item-by-item; still untracked in the working tree). This round asks
two questions: *what did the 08-18 survey leave unbuilt*, and *what changed
externally in 08-10 → 08-30* — MLX core/mlx-lm/mlx-vlm, vLLM/SGLang/llama.cpp,
oMLX/vllm-metal/LM Studio, and waybarrios upstream.

Fork state: `origin/main` = `cb2dcff` (patch #80), upstream base `22efb47`,
Studio on mlx 0.32.1 + mlx-lm `0.32.0@9acef5f` (patch #78 pin), mlx-vlm 0.6.14.
Every external claim below was checked against its primary source by the
research agents unless marked *(snippet)*; the two P0 items were re-verified
directly against the PR/issue pages and the pinned `mlx_lm` tree in `.venv`.

---

## TL;DR — the ranked list

| # | Item | Verdict | Effort | Why now |
|---|------|---------|--------|---------|
| P0-a | **Lift the mlx-lm pin to `11a6ce7` (mlx-lm #1632)** — it is the fix for our `[metal::malloc] Resource limit (499000)` crash | **GO** | ~60 LOC in `classify_layers` + tests | the "ceiling" and a production crash are the same commit |
| P0-b | ~~Frozen sampling on the worker thread~~ (mlx #4234) — **RESOLVED BY 0.32.1**: fix `ce30733` is in v0.32.1 (deployed 08-22) but NOT v0.32.0 (fleet at #72 time) | 10-min sanity check only | 2 requests | retroactively explains #72's "byte-identical at T=1.0"; no build |
| P1-a | **Session-referenced eviction** (SGLang #29173 policy + vLLM #48048 `session_id` header) | GO, gated on #74 histograms | ~150 LOC | the field's answer to the one gap the landscape doc named |
| P1-b | Message-boundary checkpoints (08-18 survey item 2) | still GO, unbuilt | ~150 LOC | llama.cpp #24176/#25472 lineage; unchanged verdict |
| P1-c | `usage.prompt_tokens_details.cached_tokens` (upstream PR #732) | adopt the API, re-implement over our stats | ~40 LOC | free observability on the client side; rebase hazard if taken as a diff |
| P1-d | Grammar / thinking / stop ordering audit (vLLM #44993/#49227, llama.cpp #26252) | GO, audit-shaped | ½ day | our llguidance (#73), thinking processor (#79) and DRY/rep-stop (#77) now stack — nobody has checked the order |
| P2-a | int8 SSM-state checkpoints (SGLang #30626) | measure fidelity first | 1 session | 2–4x checkpoint capacity per RAM budget on Qwen3.8 |
| P2-b | ~~mlx 0.32.2 kernels~~ — **DEPLOYED + MEASURED NULL 2026-08-31** (infra PR #398): 0.32.2 + mlx-vlm 0.6.17 live, all gates green; #4077's GQA-8 gain is **−0.14% at n=6 on M1 Ultra** (claimed +3.5–9.5% is M5-only; 27B-4bit is GQA-6 and cannot even engage the kernel; only 5 fleet models are GQA-8). `force_fused` (#4185) needs fork code — written-up follow-up, not built | — | closed by measurement |
| P2-c | Upstream PR #745 (exit before MLX teardown segfault), issue #746 (`engine_steps_executed` on MLLM) | adopt | tiny | matches the 08-17 "stale process squats the port" fingerprint |
| P2-d | 08-18 survey items 10/11 (DRY holes, `vllm_mlx_empty_completions_total`) | still GO, unbuilt | ~40 LOC | unchanged |
| watch | oMLX 0.6.4 boundary-diagnostic reason codes; LM Studio #367 static prefill-step autofit; vllm-metal 0.4.0 channel; upstream PRs #740/#742 (rebase landmines) | — | — | — |
| closed | DFlash/MTP (4th confirmation), n-gram/prompt-lookup (unmeasured anywhere on Metal, low prior), SnapKV-style eviction (nothing shipped), FlashAttention issue #2955 (closed Jan, no activity) | — | — | — |

---

## P0-a — the pin ceiling is the crash fix

**What we knew.** Patch #78 pins mlx-lm at `9acef5f` and names `11a6ce7`
(mlx-lm #1632) as the CEILING because it changes `ArraysCache.state` from a
bare list to `(cache, left_padding, lengths)`, which `classify_layers` would
read as `opaque` and silently disable hybrid prefix caching. Patch #80 made
crossing it loud. Separately, `qwen38-looping-investigation.md` §"Operational
gotchas" records `[metal::malloc] Resource limit (499000) exceeded` at ~21,000
generation steps with active memory at only 29.4 GB, called it a buffer-handle
ceiling, and moved on.

**What #1632 actually is** (verified on the PR page, merged 2026-08-22 by
zcbenz): "unbounded `ArraysCache` metadata graph during decode — lazy
subtraction on `left_padding`/`lengths` accumulates 1,280 graph edges per
metadata array per `advance()` step until Metal reports
`Resource limit (499000) exceeded`". The fix puts the metadata arrays into
`state` so they get evaluated with the rest of the cache each step.
Post-merge test: 50,000-step synthetic decode, zero residual edges.

So: the commit we pinned *below* is the fix for the crash we hit *above* the
thinking budget. Every hybrid route (Qwen3.8 4/8-bit, Qwen3.6, Coder-Next) is
exposed on long generations; the 6144 thinking budget being "unverified on the
hardened build — three attempts died on the Metal buffer-handle ceiling" is
this bug, not a budget bug. oMLX independently shipped the same fix as
"periodic materialisation of recurrent state to avoid Metal resource-count
exhaustion" (oMLX #3227, 0.6.3).

**Build.** In `system_kv.py::classify_layers` (and the batched twin), accept
the 3-tuple: element 0 is the recurrent list we checkpoint today; elements 1–2
are per-row metadata that must be **snapshotted and restored alongside**, not
dropped (they are what the fix keeps bounded — restoring a snapshot without
them re-opens the leak). Keep the #80 tripwire but flip its sense: warn on the
*old* bare-list shape once the pin moves. Gate: the standing T=0 byte-identical
warm-vs-cold test on ≥2K prompts, the partial-restore suite, and a new
long-decode test (≥25K generated tokens on a hybrid model, assert no growth in
`mx.metal.get_active_memory` *buffer count* — see P2 gauge below).

**Scope check (verified 08-30):** `9acef5f...main` is only **20 commits**, and
`11a6ce7` is the 4th — the lift is small and enumerable. The 3-tuple change is
confirmed at the diff level: `state` returns `(cache, left_padding, lengths)`
(Nones encoded as empty arrays), setter unpacks symmetrically — exactly the
shape #80's tripwire fires on, and the metadata is what our snapshots must
carry.

**Also in the `9acef5f..11a6ce7+` window, take deliberately:** #1772 (see
P1-d), #1790 (deepseek graph growth, same class), #1600 (`apply_min_p`
TypeError with `min_tokens_to_keep>1`), #1775 (`generation_stream` context in
`generate_step` — SimpleEngine's twin of our patch #28; cites vllm-mlx).
Nothing in the window touches `BatchGenerator` signatures, sampling APIs, or
adds a Qwen3.8/GLM/gpt-oss architecture change.

**Buffer-count gauge.** The 08-17 roadmap's Tier-2 item 6 asked for it; this
is the concrete reason. Expose the Metal buffer count (not bytes) in
`/v1/status` and `/metrics`; alert on monotonic growth per generated token.
It would have caught #1632 in soak instead of at 21K steps in a probe.

## P0-b — frozen worker-thread sampling: RESOLVED by the 0.32.1 already deployed

mlx #4234 (opened 08-13, **closed**): on mlx **0.32.0**, RNG state became
thread-local and a *compiled* function containing `mx.random.categorical` run
on a worker thread returns **bit-identical results on every call** (fine on
0.31.1). The fix is mlx PR #3828 "Fix captured random state in compile",
merge `ce30733` — verified via the GitHub compare API to be **in v0.32.1 but
NOT in v0.32.0** (v0.32.0 is 6 commits short of it).

Timeline against our fleet:

- 2026-08-03 rebase dragged **mlx 0.32.0** fleet-wide (via mlx-vlm 0.6.8).
- 2026-08-11, patch #72's write-up recorded the fingerprint without a cause:
  "batched sampling is reproducible (identical requests at `temperature=1.0`
  return byte-identical output; `temperature=0.0` differs)". **That was the
  #4234 bug, live in production.** It also means every degenerate Qwen3.8
  turn in that window repeated exactly on retry instead of resampling out —
  part of why the looping investigation's retries never helped.
- 2026-08-22, the patch-#78 pin deploy moved the Studio to **mlx/mlx-metal
  0.32.1**, which contains `ce30733`. The bug left production then, as a
  side effect nobody noticed.

**Remaining action (10 min, non-build):** two identical requests at
`temperature=1.0, top_p=1.0` on the current build should now differ —
confirming the fix is live — and a line in PATCHES.md #72/#78 recording the
retroactive explanation, so "batched sampling is reproducible" is not cited
as a fork property again (it was a bug, and it's gone).

---

## P1 — policy and API items

### P1-a Session-referenced eviction (SGLang #29173 + vLLM #48048)

- SGLang (merged 08-02): `session_id` per request; on completion the reusable
  KV is registered under the session; eviction drains **unreferenced** nodes
  first, referenced ones only if space is still short; SWA/Mamba parts use
  LRU partitions around a locked sentinel. Admitted gap: never-closed sessions
  pin forever — TTL is future work.
- vLLM (merged 08-03): `session_id` as body field / `X-Session-ID` header /
  `vllm_xargs`, threaded to `Request`, *no policy yet*; ARC (frequency+recency)
  added to the offload-tier policy factory (#49114).
- **Fork shape.** We have 1–3 concurrent agent sessions, so we can be blunter
  than either: `X-Session-ID` (opencode/LiteLLM can stamp one; llama-swap
  passes headers) → tag on the `batched_system_kv` entry → eviction prefers
  entries whose session is untouched for > N minutes → hard TTL, no
  `/close_session`. This is precisely the "recency-only eviction" gap named in
  `prefix-caching-landscape-2026-08.md`, and the landscape verdict still
  holds: **build only if the #74 `evict_to_reuse_gap` histogram shows
  traffic** after a week of the 08-22 deploy. Read it before anything else in
  this tier. If it is flat, this item stays closed and the field's work is
  corroboration only (4th independent validation of the hybrid checkpoint
  design: SGLang Mamba radix default, vllm-metal #584).

### P1-b Message-boundary checkpoints — still the one clear build from 08-18

Unchanged verdict; nothing landed. llama.cpp #24176/#25472 remains the
reference (checkpoint at every user message incl. tool results, min-step
8192, dedupe within min-step, keep the near-prompt-end one). The 08-18 survey's
plan (add boundaries *in addition to* the 2048 interval, keep the interval as
an upper bound so #48/#53 rails don't move) stands.

### P1-c `cached_tokens` in `usage` (upstream PR #732, open)

Adds `usage.prompt_tokens_details.cached_tokens` across all engines by
touching `engine/simple.py`, `batched.py`, `scheduler.py`, `server.py`,
`request.py` — all fork-owned conflict surface. Take the **API shape**, source
the number from our own cache-stats (we already know tokens saved per
request), and reject the diff at rebase. oMLX had the same reporting gap
(#1768). Cheap, and it gives opencode/LiteLLM-side dashboards a real number.

### P1-d Ordering audit: grammar × thinking × stop × rep-stop

Since 08-17 four logits-time mechanisms stack on a batched request: llguidance
(#73), the thinking phase machine (#79), DRY (per-request sampling), and
repetition-stop (#77, consumer side). The field shipped three ordering rules
in this window that we have not checked against that stack:

1. **Grammar engages only after the reasoning block** (vLLM #44993). Does our
   llguidance processor start masking inside `<think>`? If so, JSON-mode
   requests on Qwen3.8 either kill thinking or corrupt the JSON.
2. **User `stop` strings are masked while the grammar is un-terminated**
   (vLLM #49227/#50595). Our stop-string matcher (#7/#22) can truncate a
   half-open object; #77 repetition-stop could as well (JSON with repeated
   keys is periodic).
3. **`<tool_call>` counts as a think terminator for Qwen3** (llama.cpp #26252).
   Our `ThinkingAwareLogitsProcessor` ends on `</think>` only
   (`thinking_processor.py:96-105`); a model that goes straight to
   `<tool_call>` from THINK is the same failure class as the #79 bug.
4. **`continue_final_message` / trailing assistant turn with `tool_calls`**
   (llama.cpp #27626): dropped tool calls silently. Audit the harmony (#75)
   and Jinja renderers for the same shape.

Also from mlx-lm #1772 (08-25): `GenerationBatch.filter` uses `[[]] * n` for
empty processor slots — one shared list object — so a constrained request
after a plain one can lose its processors. The fork's
`_sanitize_batch_generator_logits_processors` (`scheduler.py:49-67`)
normalises with `or []` (fresh lists) and looks like it already covers this;
add a regression test "plain request, then JSON-schema request, on one
BatchGenerator" so the pin lift can't reopen it.

`reasoning_effort:"none"` → `enable_thinking=False` (llama.cpp #26045) is
already in #76 (`server.py:882-885`) — no action.

---

## P2 — measure-when-ready

- **int8 SSM-state checkpoints** (SGLang #30626, v0.5.17): recurrent
  checkpoints stored int8 in the Mamba radix cache to raise capacity. Our
  DeltaNet states are the expensive part of every `ckpt`-class snapshot;
  at ≈200 KB/token measured, 2–4x more checkpoints per `SYSTEM_KV_RAM_MB`
  is real. Needs a fidelity gate (SGLang reports none): T=0 byte-identical
  restore on the 2K gate, then a needle run at the cap. Note oMLX went the
  *other* way — 0.6.2 reverted a lossy GDN sidecar to exact fp32 because it
  changed greedy output (#2775). Our tests would catch that; theirs didn't.
- **mlx 0.32.2 kernels** — `force_fused` SDPA (#4185: 44.2→40.1 GB peak at 64K
  on 35B) and GQA-8 two-pass decode (#4077, extended to GQA-12/16 in #4380 on
  main: +7–18% at 8K–131K, M5 numbers). First upstream movement on mlx-lm
  #763 (long-context decode). Measure on the 27B-8bit ladder when the pin
  moves; **bump mlx-vlm to ≥0.6.17 in the same step** (0.6.15 and 0.6.17 carry
  mlx-0.32.1/0.32.2 compat fixes; 0.6.8 was never exercised on them) and
  re-run `scripts/fork/` vision sweep. Re-run the #48 staged 94K crash recipe
  first — residency sets were restructured.
- **Upstream small adopts:** PR #745 (`os._exit` before MLX thread-local
  teardown segfault on shutdown — plausibly the stale-process-squats-port
  fingerprint from 08-17), issue #746 (`engine_steps_executed` never populated
  on MLLM routes — 5-line fix to our #74 counters), PR #729 / issue #711
  (`--prefill-step-size` dropped under continuous batching — check ours is
  set via SchedulerConfig, not CLI).
- **LM Studio #367** (08-21): static prefill-step shrink computed from the
  working-set ceiling, vs our reactive #48 relief. One ladder comparison on a
  45 GB route decides whether a static floor is worth adding as a pre-flight.
- **08-18 survey items 10/11**: DRY coverage holes (4 config lines + ~10 LOC)
  and `vllm_mlx_empty_completions_total` + alert — unchanged, unbuilt.

---

## Watch / rebase hazards

- **Upstream `main` is still `22efb47`** (nothing merged since 08-26). Open
  PRs that will collide when they land: **#740** (5 fixes across scheduler,
  every reasoning parser, ssd_cache — #27/#77/system_kv_ssd territory; the
  stream-interval token-loss fix is real but we run interval 1), **#742**
  (schema-aware streamed tool args in `server.py`), **#732** (P1-c),
  **#709** (RotatingKVCache allowed in system-KV probe — our denylist owns
  this; only matters if Gemma-4 is served on SimpleEngine). PR #737/#736
  independently confirms our 07-28 finding that upstream's chunked-prefill
  and prompt-cache-save monkey-patches are inert on ≥0.31 mlx-lm.
- **oMLX 0.6.2–0.6.4**: hot-cache write-through, crash-safe SSD writes,
  incremental boundary snapshots (64 GB clean-prefill ceiling 43K → 96–100K —
  they caught up to where our segmented snapshots were in July), and a
  **boundary-diagnostics reason taxonomy** (#3249: capture / SSD-fallback /
  restore / fail-closed store-skip / re-prefill reason). The taxonomy is a
  good model for labelling our #74 histograms; nothing else to take.
- **vllm-metal**: now under the vLLM org, 0.4.0-dev channel, hybrid GDN prefix
  caching landed (#584, 08-10; 2.4x TTFT on shared prefix but 0.38x cold until
  #634). Still align-mode (one checkpoint per block), no disk tier,
  Python-3.12-only, torch in the loop; torch sampling corrupted ~11% of
  completions at T>0 until pinned to CPU (#622/#629). Not a migration target;
  remains the exit path if waybarrios freezes.
- **mlx-lm PR #1778 (open, 08-24) is the SECOND ceiling.** "Make `state` of
  cache return full state": padded KV caches return raw
  keys/values/**offset** in `state` (scalars serialized, `meta_state`
  removed) so restored caches stay padded for the optimized SDPA kernels.
  That breaks `classify_layers`' *other* branch — "2-tuple of arrays = plain
  KV = trim" — and our attention-slice restore logic, on every model, not
  just hybrids. When lifting the pin past `11a6ce7`, extend the #80 tripwire
  to also warn on a KV-cache state that is not a 2-tuple, so #1778 landing
  upstream announces itself the same way #1632 would have.
- **mlx-lm issue #1798 / PR #1799 (open, 08-28, no maintainer response):**
  claims `gated_delta_update` omits a `1/sqrt(head_v_dim)` readout scale that
  llama.cpp's GGML reference applies unconditionally, "output ~11x too
  large", and the PR patches **all four callers incl. `qwen3_5`** — our
  production arch, which does call `gated_delta_update`
  (`models/qwen3_5.py:17,184` in the pin). Our models are coherent, so the
  likeliest resolution is that the HF reference folds the scale elsewhere
  (the issue author admits not verifying it) and the PR is wrong for
  qwen3_5 — but if it merges it lands in our forward pass and would change
  T=0 outputs. **Do not take blind at any future pin lift; the byte-identical
  gate is the tripwire.** If maintainers instead confirm the bug, it becomes
  a quality lever for every GDN route and possibly relevant to the residual
  paraphrase tic. Watch the thread.
- **mlx-lm issue #1807** (08-30): `mlx_lm.server` on 0.31.3 grows unboundedly
  past its prompt-cache cap on **Qwen3.8-27B-4bit, M1 Ultra** under multi-day
  serving. Upstream's server, not ours — but same model class and silicon;
  corroborates keeping the #48/#53 rails and the buffer-count gauge (P0-a)
  rather than trusting byte gauges alone.
- **mlx-lm #1776** (08-23): `BatchKVCache.filter()` drives `_idx` negative
  after `fetch_nearest_cache → deepcopy → trim_prompt_cache` when a
  sliding-window model shares the process. llama-swap gives us one model per
  process so the trigger can't fire; the unguarded decrement (`cache.py:548,
  1007, 1323` in the pin) is in code our batched path shares — add an
  assertion in the restore path rather than trusting the topology.
- **mlx #4265** (closed, no fix): `quantized_matmul` doesn't amortise weight
  reads at M=2–8 (4-bit scales linearly to M=8, 27B sees 3.65x at M=8). This
  is the kernel-level reason for every measured verdict in
  `speculative-decoding-dead-on-mseries` and the ~1.2x batching ceiling.
  Nothing to build; cite it instead of re-measuring.

## Closed, with the new corroboration

- **MTP / DFlash / draft speculation**: oMLX DFlash 2 + Lightning MTP report
  2.3–2.6x — on M3 Ultra only; vllm-metal spec-decode still net-negative
  (#482 open); mlx #4265 explains why. Fourth confirmation; stays closed on
  M1 Ultra.
- **n-gram / prompt-lookup / suffix-tree speculation**: llama.cpp ships
  several (`--suffix-decoding` etc.) with no published Metal numbers anywhere;
  same M=2–8 verify cost applies. Low prior; a 2-hour measurement on agent
  transcripts is the only way to close it for good, not a build.
- **Token-eviction KV compression (SnapKV/H2O)**: nothing shipped in any
  production server this window.
- **Long-context llama.cpp arm of survey item 1**: no new M1/M2 Ultra data at
  30–150K appeared; our July ladders remain the only numbers for that regime.

## P1-d audit results (run 2026-08-30, same day)

The ordering audit ran against the four field rules. Verdicts, with the
runtime checks that remain:

1. **Grammar vs thinking — BUG (conditional).**
   `_attach_response_format_logits_processor` (`server.py:730-746`) forces
   `enable_thinking=False` instead of gating the grammar behind `</think>`;
   the `ThinkingAwareLogitsProcessor(inner=...)` composition that would gate
   it is only ever built in tests. Coherent for Qwen-style templates that
   honour the flag; broken for **harmony rendering**, which takes no
   `enable_thinking` at all — on a gpt-oss route the grammar masks harmony
   control tokens from token 0. The #79 prompt-token convention is SAFE on
   both schema processors (they self-learn the boundary on first call).
   *Runtime check:* JSON-schema request against the live gpt-oss route.
2. **Stop strings / repetition-stop vs un-terminated grammar — BUG.**
   `truncate_at_stop` / `StopStringScanner` and the #77 repetition-stop all
   fire with zero grammar coordination: `stop=["\n\n"]` + pretty-printed
   schema truncates mid-object; non-streaming is saved by the 422 re-parse,
   streaming has already sent the broken JSON. Fix model = vLLM #49227:
   mask the stop while the matcher is not accepting; repetition-stop inside
   a grammar should finish as `length`, never `stop`.
   **FIXED 2026-08-31 as patch #89** (`vllm_mlx/grammar_guard.py`): both
   schema processors publish `is_accepting()`; the four batched stop sites,
   the SimpleEngine stop-check and the rep-stop branch consult it, and the
   rep-stop downgrades to `finish_reason="length"` inside an open value.
   Suppressions are counted in
   `vllm_mlx_grammar_stop_suppressions_total{source}`. Live probe still
   owed at deploy (JSON schema + `stop=["\n\n"]` on an llguidance route).
3. **Think-terminator set — BUG, latent.** `</think>` is the ONLY THINK
   exit (`server.py:540-541`); a THINK → `<tool_call>` transition stays in
   THINKING, and an armed thinking budget's `_force_transition` then masks
   everything except `</think>` — **injecting `</think>` mid-tool-call**.
   Unarmed by default; exposure = whether any route sets a thinking budget
   in llama-swap (check `personal-infratructure`). Fix model = llama.cpp
   #26252: register tool-start markers as THINK terminators and suppress
   the forced transition while the tail matches a partial tool-start.
4. **Trailing assistant turn — the llama.cpp #27626 mechanism is N/A**
   (no prefill-continuation here), **but the same corruption existed via
   `_normalize_messages`**: the same-role merge dropped `tool_calls`
   entirely. Reproduced live, **fixed as patch #83** (structural merge).

## Recommended sequence

1. **P0-b sanity check** (10 min): confirm T=1.0 sampling varies on the
   current build; add the retroactive note to PATCHES.md.
2. **P0-a**: lift the pin to `11a6ce7`+ (20 commits total), extend
   `classify_layers` for the 3-tuple, widen the #80 tripwire to KV-shape
   (PR #1778 is the next ceiling), skip/park #1799 explicitly, add the
   buffer-count gauge and the ≥25K-token decode test. Unblocks the
   6144-thinking-budget verification and mlx 0.32.2 kernel trials.
3. Read the #74 histograms after a week of soak → decides P1-a.
4. P1-d audit, P1-c `cached_tokens`, P1-b boundary checkpoints, in that order
   (audit is cheapest and most likely to find a live bug).
5. P2 items as ladder windows allow.
