# Improvement plan — memory safety and caching under concurrent agentic load (2026-09-16)

*Research + plan only. Nothing here has been built, deployed or probed. Fork
base for every citation: `origin/main` = `5edda49` (Studio runs `a2901ae`,
same code). Companion to
[`speed-lever-ledger-2026-09.md`](speed-lever-ledger-2026-09.md) (what NOT to
re-propose) and
[`prefix-caching-landscape-2026-08.md`](prefix-caching-landscape-2026-08.md)
(whose "instrument first" gate is now answered — see D).*

**Workload this plan is for:** idealplace opencode CI reviewers (5 per PR,
REVIEW_PARALLELISM 2–4, 2 runner slots) + Tim's interactive opencode + HA voice,
all through LiteLLM → HAProxy (`llm_parallel maxconn 4`) → llama-swap v255 →
`Qwen3.8-27B-4bit` on BatchedEngine (hybrid GDN: 48 linear + 16 attention
layers). Agent turns are 18–66K prompt tokens, one step at a time.

---

## 0. What the evidence actually says (read this first)

The handoff's open-issue list was written before today's route log was mined.
Three of its premises change:

### 0.1 The 14:47–15:52Z OOM storm was NOT cross-process. It was a leak in our recovery path.

- The 35B's own log shows it resident **14:29:27–14:32:17Z** (3 min, zero 27B
  OOMs in that window) and again **16:02–17:08Z** (no 27B traffic in that
  hour). During the storm (16:47–17:52 local = 14:47–15:52Z) **no second model
  process existed.**
- The 27B's own post-recovery baseline ratchets monotonically across the storm,
  at `running=1 waiting=1`:
  `55.9 → 57.2 → 63.0 → 65.6 → 67.1 → 68.4 → 69.0 → 70.8 → 71.2 GB`,
  while `#48` relief logs "evicted 1 snapshot entry" nine times and **recovers
  nothing** (swap on the box is 0 — those are MLX-accounted allocations the OS
  was compressing).
- The 17:08Z "end" of the incident is the 19:08 local **config reload that
  restarted the 27B process.** A fresh process ran the same CI shape 19:08+ with
  0 OOMs.

**Root cause (code-verified):** `Scheduler._recover_from_generation_error`
(`scheduler.py:3306-3343`) aborts every running row but never calls
`hybrid_kv.discard_pending(rid)` — the normal abort path does
(`scheduler.py:2346`). The `except` branch then `break`s (`:3462-3471`), so
`_cleanup_finished` (`:3427`, which reaches `store_finished` →
`discard_pending`, `batched_system_kv.py:1281-1283`) never runs for those rows.
Each aborted row leaks its `_pending` ladder: up to 8 checkpoints × 146.8 MiB
of materialised deltanet state (`capture_segment`, `batched_system_kv.py:410-455`,
`mx.eval`'d into independent buffers). Worse, a row that HIT installs
`self._pending[rid] = list(inherited)` (`:919`) — dicts that reference the
**donor entry's** checkpoint arrays — so a leaked ladder pins an evicted
donor's state and relief evictions free nothing. 138 rows were aborted in the
storm (90×1 + 10×2 + 7×4 recoveries); ~1 GB/row explains the ~40 GB of
unexplained active memory. The 15 `deque mutated during iteration` crashes
(0.2) triggered the same leaking recovery.

Consequences for the rest of the plan: the "solo 4–5K requests OOM at 70 GB"
observation, the "relief fires too late / lower the watermark" suggestion, and
the 17:30 retry livelock are all downstream of this one bug. **Fix it first,
then re-measure before touching any threshold.**

### 0.1b (added 2026-09-17) The larger holder: the SSD spill backlog, ~12 GB

Checking the deployed state the next morning: **bag 0 entries / 0 MB,
running 1, waiting 0 — active 52–54 GB**, in a process with only 5 OOM
recoveries. The recovery leak cannot explain that. `/v1/status → cache.ssd`
read `spill_count 4, spill_drops 30`: the SSD writer defers every write until
the engine is idle, idle on the batched path is `not scheduler.running`, and a
CI-loaded route never is. The backlog hit 11.5 GB 23 minutes after spawn and
sat at the hardcoded 12 GB cap for three hours — outside the bag's accounting
and untouched by relief (the queue owns its references), so relief wiped the
*useful* cache 61 times instead (`pressure_cache_clears` 3,864 in 67K steps).
A third, smaller pin: the writer's loop locals keep the last-written snapshot
alive until the next spill arrives. All three, plus 0.2, are **patch #103**
(PATCHES.md) — P0-1 below, widened.

### 0.2 A second fork bug killed the batch generator 15 times today

```
scheduler.py:2386 _schedule_waiting → batched_system_kv.py:1176 promote_ssd_pending
    for request in scheduler.waiting:          # executor thread
RuntimeError: deque mutated during iteration   # add_request appends on the event loop
```

Each hit runs the (leaking) recovery and aborts every in-flight reviewer.

### 0.3 Every failure today is served as HTTP 200

2266 responses in the post-deploy window, **100 % status 200**. 382 streams
started, 366 produced a first token → **16 empty 200 bodies (4.2 %)**; 105
Metal OOMs + 15 deque crashes → `finish_reason:"error"` inside a 200. Code
path: recovery appends `RequestOutput(finished=True, finish_reason="error")`
(`scheduler.py:3463-3470`) → `server.py:7550/7751` stamps it into a normal
chunk → `[DONE]`; `#91`'s `_stream_error_chunk` is never reached (it lives on
the `except` branch, `server.py:7818`); `result_label` stays `"success"` so
`stream_aborts_total` is **not** incremented (`metrics.py:530-537`), and a
mid-stream OOM after N tokens is metered as a successful request. Non-streaming:
`ChatCompletionResponse(content=None, finish_reason="error")`, HTTP 200
(`server.py:6155-6186`). opencode retries only on status ≥ 500 / a
`429|5xx|overloaded|service unavailable` message regex and never checks for
empty output; LiteLLM (`num_retries: 0` in our config) rewrites a clean EOF to
`[DONE]`. Nobody upstream of the engine can tell a dead turn from an empty one.

### 0.4 Other measured facts used below

- Prompt lengths (n=387): 4–16K 178 · 16–32K 100 · 32–64K 92 · 64–96K 15,
  max 79K. Running/waiting: max running = **4** (never the `--max-num-seqs 8`
  cap); 185 defers (132 `padded-KV total > 4096`, 39 watermark, 14 pad-waste).
- TTFT (n=366): p50 **72.6 s**, p90 **393 s**, p99 794 s (queue wait isn't
  logged separately). opencode's stream timeout is 300 s.
- Cache: 82 % of scheduled requests got a restore, but **55 % restored at
  exactly 3324 tokens** (the shared system prompt; `lcp=3330` = the reviewer
  persona diverges 6 tokens later). Mean restore depth 10.9K vs mean prompt
  22.8K → the median request re-prefills ~19K tokens it has already seen.
  Store: 216 attempts, 197 stored (the 19 refusals log no reason).
- **#85 durable verdict** (`timing-verdict.json`, since 2026-08-28):
  `evictions 470, reuses 975, evict_to_reuse_events 117`. One in four
  evictions was followed by a request that would have hit the evicted entry.
  The landscape doc's gate ("if it stays near zero, close the item") is
  answered: **funded.**
- SSD tier: 11 entries, 19.05 GB of a 20 GB cap — **full**, churns within one
  working day. Disk copies are ~2.4× the RAM footprint (a 5240-token entry =
  1267 MB on disk vs ~105 KB/tok in RAM). Promote latency 59 ms–5.2 s.
  Eviction never spills (grown entries are simply gone,
  `batched_system_kv.py:1023-1030`).
- Deployed route env (infra `origin/main`): `KV_BUDGET_MB=4096`,
  `BPT_FLOOR_KB=64`, `PAD_WASTE_MB=4096`, `MEM_WATERMARK_PCT=85`,
  `MAX_QUEUE=8`, `MAX_PROMPT_TOKENS=170000`, RAM floor 8192 / dynamic / reserve
  8192 / max 20480, SSD 20 GB, `ttl 600`. Slots = default 4.
- Buffer count per bag entry ≈ 0.9–2.2K mx arrays (16 attn layers × segments
  ×2 + 48 GDN × 2–4 + ladder ≤ 8 × 48 × 2–4). 4–8 slots ≈ 4–18K buffers vs the
  499000 limit — **slots are not the 499000 risk**; the per-token
  `ArraysCache.advance` leak (mlx-lm #1845, open) is, and it is independent of
  our cache.
- Where the guards read memory: everything routes through
  `PressureManager.watermark_status()` (`memory_pressure.py:56-70`) — admission
  gate 3, `#101` budget, store-overshoot, relief exit, MLLM branch. `psutil` is
  already a prod dep; `host_statistics64` via ctypes and IOKit
  `IOAccelerator → PerformanceStatistics["Alloc system memory"]` both work on
  this box (verified, no new deps). The solo hole is one line:
  `if not scheduler.running: return False` (`batched_system_kv.py:1444`).

---

## 1. Prioritised plan

Scoring: impact (what it stops/gains today) × confidence × 1/effort. "Live
verify" steps are all idle-window or read-only; none send inference probes
while CI or Tim is on the box.

| # | Item | Impact | Risk | Effort | Verdict |
|---|---|---|---|---|---|
| **P0-1** | Recovery leak + deque crash (bug fixes) | Removes today's OOM ratchet and 15 generator kills | Low | ~30 lines + tests | **Build first** |
| **P0-2** | Honest failure signalling (503 / error frame / counters) | Ends silent empty turns; makes retries deliberate | Low–med | ~1 day | **Build second** |
| **P1-1** | Cache retention for concurrent chains (spill-on-evict, session-affine protection, utility-per-byte) | Biggest throughput lever: ~19K tokens of re-prefill per median request | Med | 2–3 days, staged | **Build, staged** |
| **P1-2** | Retune concurrency on a clean process (KV_BUDGET, slots, SSD size) | Recovers seats lost to leak-era tuning | Low | config + ladder | After P0 |
| **P2-1** | Solo deep-prefill guard | Closes a real hole with no incident on a clean process yet | Low | ~60 lines | Build, after P0 measurements |
| **P2-2** | Cross-process memory guard (config first, then device-wide signal) | Real blindness, no attributable incident | Med | config: 0 code; engine: ~80 lines | Config now (Tim's trade-off), engine later |
| **P2-3** | Fork-invariant guard tests for rebases | Prevents the next #683 | Low | 1 day | Build alongside P0 |
| **P3** | Preemption / chunk-budget scheduling | Marginal on a 1-user box; mlx-lm already alternates decode + prefill chunk | High | Weeks | **Do not build now** |

### P0-1 — Fix the recovery leak and the deque crash

**Evidence:** §0.1, §0.2.

**Approach (fork-owned files only):**
1. `_recover_from_generation_error`: for every aborted id call
   `self.hybrid_kv.discard_pending(rid)` (mirror `_do_abort_request`,
   `scheduler.py:2344-2346`) and drop `request._extracted_cache`,
   `request.prompt_cache`. Also discard pending for rows that are in
   `waiting` with a restored `prompt_cache` only if they're being dropped (they
   aren't — leave them).
2. `promote_ssd_pending`: iterate `list(scheduler.waiting)`. Audit the two
   other `_schedule_waiting` hooks (`capture_checkpoints`, `fetch`) for the
   same cross-thread iteration.
3. Observability so this can never hide again: `stats()["pending_ladders"]`
   (= `len(self._pending)`) and `pending_ladder_bytes` (sum of
   `state_arrays` nbytes) → `/v1/status` + a `vllm_mlx_cache_pending_*` gauge
   (additive `metrics.py` block, #7 precedent). Log the store-refusal *reason*
   (the 19 `stored=False` today are opaque).

**Verify:**
- Unit: a scheduler test that seeds `_pending` for two running rows, raises
  from `step()`, and asserts `_pending`/`_base_pos`/`_restore_source` are
  empty afterwards; a test that `promote_ssd_pending` survives a concurrent
  `waiting.append`. Full suite green.
- Local e2e (laptop, `Qwen3.5-0.8B-8bit`, the family already used for the #34
  merge spike): monkeypatch `BatchGenerator.next` to raise once mid-prefill;
  assert `mx.get_active_memory()` returns to baseline and `pending_ladders==0`.
- Live (idle window deploy): watch `[generation_error_recovery]` lines — the
  next `[Metal memory] active=` after each must be ≤ the pre-request baseline.
  `pending_ladders` must read 0 whenever `running=0`.

**Don't:** lower `MEM_WATERMARK_PCT` or `MAX_PROMPT_TOKENS` on the strength of
the storm's numbers — they were measured on a leaking process. Re-ladder after
this lands.

### P0-2 — Honest failure signalling

**Evidence:** §0.3. Also: streaming `PromptTooLong` is raised inside the SSE
body (`engine/batched.py:1202-1206`) → 200 + `#91` error chunk, a 400-class
condition as a 200 stream; out-of-band aborts never enqueue a terminal output
(`scheduler.py:2290-2350`), so the SSE stalls on heartbeats until the
disconnect guard (`server.py:5426-5442`).

**Approach:**
1. Give recovery outputs an `error_kind` (`"oom_recovery"`), like the MLLM
   scheduler's `prompt_too_long` (`mllm_scheduler.py:1320-1336`). In the
   server: before first byte → **HTTP 503 + `Retry-After`** (retried by
   LiteLLM's RetryPolicy, AI SDK and opencode alike; prefer 503 over 429, which
   carries rate-limit cooldown semantics); after headers are flushed → the
   vLLM/LiteLLM error frame `data: {"error": {"message": "Service Unavailable:
   generation aborted (oom_recovery)", "type": "server_error", "code": 503}}`
   then `[DONE]` (opencode's retry regex matches `503|service unavailable`).
   Non-streaming → 503.
2. Make the pre-stream `EngineBusy` probe also cover `PromptTooLong` (400
   before headers).
3. `_do_abort_request` enqueues a terminal output so out-of-band cancels close
   the stream with `[DONE]`.
4. Metrics: count recovery-aborted requests under
   `stream_aborts_total{result="error", phase=...}` (today they're
   `"success"`), and add `generation_recoveries_total{kind}`.
5. `Retry-After` = a small constant (e.g. 15 s) — the point is to make the
   client back off *and* to keep the retry loop visible in
   `inference_requests_total{result="error"}`.

**Verify:** unit tests for each path (status, frame shape, `[DONE]`,
counters). Live: read-only — after deploy the 200-only histogram must show
503s during any recovery, and LiteLLM's log must show the 503 propagating (no
probe needed; CI traffic will exercise it). Confirm in opencode's sqlite that
a failed reviewer turn is recorded as an error, not an empty assistant
message.

**Don't:** retry inside the engine (the request that OOMed would OOM again —
the existing recovery comment is right); don't emit `finish_reason:"error"`
with a 200 (that is the current silent shape).

### P1-1 — Cache retention for many concurrent chains

**Evidence:** §0.4 — 55 % of hits stop at the system prompt, ~19K tokens
re-prefilled per median request, 117 evict-to-reuse events, SSD full at 11
entries, grown entries never spill. Under CI the `#101` budget clamps to the
8 GB floor = 2–3 deep chains while 5–8 are live; LRU then evicts the chain that
is *about to be extended*.

**Approach, staged so each step is measurable on its own:**
1. **Spill-on-evict for grown entries** (fork-owned `batched_system_kv.py`):
   consolidate segments and write the blob on eviction (today only non-grown
   entries write through at store). Turns an eviction from "re-prefill 19K
   tokens" (~15–40 s) into "promote 0.06–5 s". Needs the SSD cap raised (disk
   is cheap; check free space) and the **2.4× disk inflation investigated**
   first — likely full-precision ladder copies; a 20 GB tier should hold ~4×
   what it does.
2. **Session-affine soft protection** (SGLang's session radix cache idea): an
   entry whose chain has a running or waiting request, or finished within the
   last N minutes, is evicted after all unprotected entries. Key: no header
   exists (only `Authorization` is read, `server.py:2114`); use the LCP itself
   — a waiting request's LCP ≥ `partial_min` against an entry marks it
   protected. Zero client changes; opencode chains map onto it directly.
3. **Utility-per-byte eviction tiebreak** (Marconi; the landscape doc's ~40
   lines): within the unprotected class, evict ascending
   `recompute_seconds(token_count) / bytes`, spilled entries first (already
   the case). Fit `recompute_seconds` from the route's own prefill rate
   (`prompt_tokens / prefill_seconds` is in the logs).
4. Minor: a boundary checkpoint at the first message boundary after the system
   prompt (the `lcp=4040 → restore 3324` pattern) is thinned by
   `boundary_min_step 2048`; exempting the first post-system boundary saves
   ~700 tokens per hit — cheap but small; do last.

**Verify:** the #85 counters are the gate — `evict_to_reuse_events / evictions`
must fall from 0.25; restore-depth histogram (`restore at X/Y`) must shift off
3324; TTFT p50/p90 from the log. All read-only after an idle-window deploy.
Byte-identity gate (T=0 warm vs cold) on a local hybrid model for the
spill-on-evict path.

**Don't:** build a radix tree, tiered offload, CacheBlend, block-hash caching
(all rejected in the landscape doc); don't raise `SLOTS` as the fix — under CI
the RAM floor, not slots, is what binds, and idle sessions already grow into
`RAM_MAX_MB`.

### P1-2 — Retune concurrency on a clean process

**Evidence:** `KV_BUDGET_MB=4096` and the reserve were set on a day whose
numbers were leak-contaminated after ~12:00 local. The one *clean* 4-row OOM
(10:44Z, first OOM of its process, 55.9 → 62.6 GB in a 16K prefill step) is
real: 4-wide deep is too much. 2-wide at ~23K is proven safe (peak 48.6 GB).

**Approach:** after P0-1 lands, a read-only ladder from CI traffic (no probes):
compare peak-per-step against seats; if peaks stay < 52 GB at 2 seats for a
day, try `KV_BUDGET_MB=6144` in an idle window and re-read. Reconsider the
`RAM_RESERVE_MB=8192` once the +6.7 GB step transient is re-measured clean.
Consumer side (Tim's call, infra repo): REVIEW_PARALLELISM should match
effective seats (2), and opencode's 300 s stream timeout sits *below* TTFT p90
(393 s) — either raise it for the CI profile or accept that deep reviewers
queue.

**Don't:** re-test `--prefill-step-size` (ledger #6: NULL ≥ 2048, 8192 costs
+14 GB), KV-cache quantisation (REFUTED, worsens with context), or "continuous
batching for speed" (ledger #2, ~1.2× ceiling).

### P2-1 — Solo deep-prefill guard

**Evidence:** `should_defer_cobatch` returns `False` when nothing is running
(`batched_system_kv.py:1444`); the only solo protection is the static
`MAX_PROMPT_TOKENS`. A single 2048-token prefill step realises the whole
64-layer chunk graph in one `mx.eval` (mlx-lm `generate.py:1245`) — the +6.7 GB
transient. **No clean-process solo OOM has been observed** at ≤ 79K; this is
defence, not a fix.

**Approach (small, reuses the seam):** before inserting a solo request,
project `active + tokens_to_prefill × bpt + transient_step + bag_floor` against
the watermark. If over: run relief first (evict bag, clear cache), re-check;
if still over, **reject 503 + `Retry-After`** (P0-2 shape) rather than admit —
and if the projection can never fit even with an empty bag, 400
`prompt_too_long`. `transient_step` = measured once per (model, step size)
from the log's peak−active jump (vLLM's profile-run idea), stored as an env
with a conservative default.

**Verify:** unit tests; live read-only: no `[generation_error_recovery]` with
`running=1`.

**Don't:** shrink the prefill step to shrink the transient (ledger #6 — it's
NULL for speed and the transient is already bounded at 2048).

### P2-2 — Cross-process memory guard

**Evidence:** every guard is per-process (§0.4). The 35B (22 GB peak) plus
the 27B (55 GB peak under CI) exceed the 60 GiB wired limit if they ever
overlap under load — but **today's storm was not that** (§0.1), and the two
real overlaps today carried no traffic. llama-swap v255 has no memory-aware
scheduling (`internal/hw` is reporting-only, #1004 unanswered), but its
router defers a swap that would evict a busy process until in-flight hits
zero, and `exclusive`/matrix sets control what may co-reside.

**Approach:**
1. **Config (0 code, Tim's trade-off):** either (a) move the 35B out of the
   non-exclusive `ha` group into a set that cannot co-reside with the heavy
   27B routes — then a voice request during CI *waits* for in-flight to drain
   (possibly minutes) instead of loading beside it; or (b) keep voice
   responsive and accept the overlap risk with (2) as the backstop. `hooks.on_startup.preload`
   re-loads `ha` on every config reload, so any grouping change must keep that
   consistent.
2. **Engine (later):** a device-wide reader behind `PressureManager` —
   IOKit `IOAccelerator PerformanceStatistics["Alloc system memory"]` via
   ctypes (verified working here; ~ms, cache per step) — and
   `threshold_bytes()` derived from `wired_limit − other_processes_alloc`
   instead of a fixed % of our own ceiling. The relief *trigger* stays
   `get_peak_memory()` (there is no device-wide peak). Add a startup check that
   logs the device-wide figure so a co-resident model is visible in our log.

**Verify:** engine part is unit-testable with a fake reader; live read-only:
the startup line during a known 35B overlap.

**Don't:** build model-level memory arbitration into the fork (that is
llama-swap's/oMLX's job — single-process ownership is what makes per-process
accounting sufficient); don't poll `ioreg` as a subprocess per step.

### P2-3 — Fork-invariant guard tests (rebase safety)

**Evidence:** #683 silently broke the finish store for a month; today's two
bugs and the 200-on-error shape have **zero** tests. Specifically untested:
the `#100` gate condition itself (`scheduler.py:3084-3087`, only reached
indirectly); `_recover_from_generation_error` (no test at all); eviction never
spills grown entries; `promote_ssd` inserting under a full bag evicts a live
chain; fetch-vs-store lock races; the client-visible shape of a recovery.

**Approach:** one module, `tests/test_fork_invariants.py`, each test named
for the PATCHES.md number it pins, asserting the *condition* (e.g. "hybrid
extraction happens regardless of `_prompt_output_entry_is_useless`", "the
non-hybrid path still honours upstream's gate", "recovery discards pending
ladders", "recovery output is a 503/error frame, never a 200 with
`finish_reason:error`", "`promote_ssd_pending` tolerates a mutating deque").
Add a PATCHES.md maintenance rule: a rebase is not done until this module is
green *and* its assertions were re-read against upstream's diff.

### P3 — What not to build (and why)

- **Row preemption / swap-out:** `RequestStatus.PREEMPTED` exists but nothing
  sets it; a hybrid row's partial KV is not in its ladder (only ckpt-class
  states are), so recompute-preemption needs a partial store first. High
  effort, and on a 1-user box the honest 503 + client retry (P0-2) plus cache
  retention (P1-1) buys the same outcome.
- **Per-step chunked-prefill token budget (vLLM `max_num_batched_tokens`):**
  mlx-lm's `_next` already decodes first and prefills one 2048-token chunk per
  step; our admission gate *is* the budget. mlx-lm closed every memory-aware
  admission PR (#1541/#1542/#1835) — nothing arrives from upstream, and we
  don't vendor mlx-lm.
- **Speed levers:** none apply — see the ledger (spec decode, batching-as-speed,
  KV-quant, prefill step, prewarm, GDN kernels, proj fusion, burst decode all
  refuted).
- **Raising the watermark or the bag above measured ladders** — the 2026-07-09
  and 2026-09-16 (10:44Z) crashes are the reference recipes.

---

## 2. Suggested order and gates

1. **P0-1 + P2-3 (tests for it)** → deploy in an idle window → 24 h read-only
   soak: post-recovery baselines flat, `pending_ladders` 0 at idle, 0 deque
   crashes.
2. **P0-2** → deploy → read-only: 503s appear in place of empty 200s; opencode
   records errors.
3. **P1-2 ladder** on the clean process (read-only), then the config retune.
4. **P1-1** in stages 1→3, each with the #85 ratio as the gate.
5. **P2-1**, **P2-2** (config part is Tim's call any time; engine part after
   P1-1).

Everything above is fork-owned code (`scheduler.py` recovery/hooks,
`batched_system_kv.py`, `memory_pressure.py`, `server.py` error paths) — no
upstream-owned file needs to move, and nothing touches `engine/simple.py`.
