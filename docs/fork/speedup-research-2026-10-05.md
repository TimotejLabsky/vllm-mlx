# Speedup research — what other engines do, 2026-10-05

Research round against fork main `493513c` (mlx 0.32.2, mlx-lm `f4f3b57`-era
pin, mlx-vlm 0.7.2). Four tracks: the MLX ecosystem since 2026-09-06,
llama.cpp and other local engines, datacenter serving engines (vLLM, SGLang,
Dynamo, LMCache), and a read-only audit of the fork's own hot path.

**Nothing here is measured on the Studio.** Every item is a candidate with a
kill gate, not a verdict; verdicts go into
[`speed-lever-ledger-2026-09.md`](speed-lever-ledger-2026-09.md) once
measured. Levers the ledger already refuted (speculative decode, KV-quant,
GDN kernels, projection fusion, decode burst, prefill step size ≥ 2048) were
excluded up front and are not re-proposed.

## Summary

The ledger's conclusion still holds: decode is inside mlx's Metal kernels,
and agent-turn wall time is queue wait plus re-prefill. The field's new ideas
line up with that — the largest candidates are **scheduling order**, **what
the SSD tier holds**, and **prompt stability across turns**, not kernels.
Kernel-side, there are three cheap A/Bs and two pin bumps.

| Tier | Item | Moves | Cost to test |
|---|---|---|---|
| A — zero-code probes | A1 wired-memory decay after ~2 s idle | TTFT after every pause | ~1 h on the Studio |
| | A2 Metal command-buffer limits (MoE routes) | MoE decode, claimed +5–8 % | 15-min env A/B |
| | A3 dequantize-then-matmul for prefill chunks | dense cold prefill, claimed +31 % | 5-min kernel repro, then ~½ day |
| | A4 think-stripping re-prefill (divergence log) | re-prefill per user turn | ~30 lines of logging |
| B — scheduler | B1 shortest-prefill-first admission with aging | queue wait, TTFT tail | 60–100 lines + stress A/B |
| | B2 decode-first / stall-bounded prefill chunks | stream stalls, 300 s timeouts | ~20–40 lines + A/B |
| | B3 re-match at admission + in-flight prefix dedup | duplicate cold prefills | ~100 lines |
| | B4 idle-time prefill of the next-turn prefix | first turn after a user message | ~200 lines, after A4 |
| C — cache/SSD | C1 delta / content-addressed SSD spill | deep-chain cache reach | new on-disk format |
| | C2 thin checkpoints only at capacity | short shared prompts | < 20 lines |
| D — pin bumps | D1 mlx 0.32.3 (released) | D256 prefill memory, MoE prefill | routine bump |
| | D2 mlx 0.32.4 (unreleased) | long-context decode on D256 | wait for release |
| E — hygiene | per-request dead work, spawn time, 499000 fix | 0.3–1 s per request, s per swap | trivial each |

## A. Zero-code probes (do these first)

### A1. Wired-memory residency decays after ~2 s of GPU idle

- **What:** mlx makes one standing `requestResidency()` and never renews it;
  macOS un-wires weights ~2 s after the last GPU work and the next request
  pays to re-wire them. [mlx #4609](https://github.com/ml-explore/mlx/issues/4609)
  (open, filed 2026-10-02).
- **Evidence:** issue reports 342 ms vs 63 ms first compute on a 4 GiB array
  after 12 s idle (M2 Air / M4 Pro). oMLX added a keep-warm ticker for it
  ([omlx #3974](https://github.com/jundot/omlx/pull/3974): +996 ms after 2 s
  idle on a 156 GB model, M5 Ultra). llama.cpp shipped a residency heartbeat
  ([llama.cpp #17766](https://github.com/ggml-org/llama.cpp/pull/17766), M2
  Ultra). Reproduced on the laptop this round (M1 Pro, macOS 27.0.1, mlx
  0.32.2): wired 6.9 → 2.4 GiB after 12 s idle, first compute ~150 ms vs
  ~25 ms warm, ≈ 30 ms per GiB touched.
- **Why it matters here:** every agent turn follows a tool-execution pause.
  Extrapolated (not measured) 0.5–1.5 s per post-idle turn on 15–45 GB
  routes; proportionally largest on voice and short warm turns.
- **Caveat:** a tiny-GPU-op heartbeat did **not** keep memory wired in the
  laptop repro (the issue says it does). The reliable fixes are
  `sudo sysctl iogpu.disable_wired_collector=1` (untested, needs sudo) or
  re-requesting residency.
- **Test:** run the issue's repro on the Studio; TTFT back-to-back vs after
  10 s idle on a live-shaped route. **Kill:** < 100 ms difference at fleet
  model sizes. If real: one sysctl line in the existing
  `com.local.gpu-wired-limit` LaunchDaemon (infra repo).

### A2. Metal command-buffer limits on MoE routes

- **What:** mlx commits a command buffer every 50 ops or 50 MB of distinct
  inputs on Ultra chips, and routed `gather_qmm` binds the whole expert array,
  forcing many extra commits per step.
  [mlx #4521](https://github.com/ml-explore/mlx/issues/4521),
  [mlx #4562](https://github.com/ml-explore/mlx/pull/4562),
  [omlx #4039](https://github.com/jundot/omlx/pull/4039).
- **Claim:** `MLX_MAX_OPS_PER_BUFFER=1000 MLX_MAX_MB_PER_BUFFER=400` →
  +7.7 % decode short context, +5 % at 20K (M3 Ultra, GLM MoE, mlx 0.32.2),
  outputs bit-identical. **Dense is null** in the same issue (27B-8bit
  52.8 → 53.0 ms/token).
- **Risk:** #4562 reports a much larger MB limit made a 30K prefill hold
  +88 GB instead of +8 GB. Stay near 400 MB and re-run the prefill peak check.
- **Test:** two env lines on a spare-port 35B-A3B, T=0 SHA gate + decode A/B +
  prefill peak. **Kill:** < 2 % or any peak-memory growth.

### A3. Dequantize-then-matmul for prefill-sized chunks

- **What:** at ≥ ~512 rows, `x @ mx.dequantize(w).T` beats
  `mx.quantized_matmul`.
  [mlx #4621](https://github.com/ml-explore/mlx/issues/4621) (open, filed
  2026-10-04, single unreviewed report).
- **Claim:** M2 Max, mlx 0.32.3: 2048 rows 29.8 → 21.8 ms; model-level
  Qwen3.8-27B-4bit prefill at 8K 137 → 179 tok/s (+31 %), peak unchanged.
  Slower below ~512 rows, so it must be size-gated.
- **Why it matters:** the ledger says dense prefill is matmul-bound and that
  hybrid prefill levers must attack attention or the quantized matmuls — this
  is the first lead that does. Does not cover `gather_qmm` (MoE experts).
- **Risks:** prefill numerics change → T=0 shift expected; a transient
  dequantized matrix per call; KV from the ≥ 512-row path differs slightly
  from the decode path.
- **Test:** the issue's 30-line script at 27B shapes on the Studio first.
  **Kill:** < 1.15× at the kernel level. Only then a fork-owned
  `QuantizedLinear` subclass (~½ day) and a real cold-prefill A/B.

### A4. Think-stripping re-prefills the previous tool loop

- **What:** the Qwen3-family template keeps `<think>` only for assistant
  turns after the last user message. With `preserve_thinking:false` (27B
  routes), each new user message re-renders the whole previous tool loop
  without reasoning, so the prompt diverges at that loop's first assistant
  turn and the loop is re-prefilled. Verified on the Qwen3.8 HF template by
  one track and on the cached Qwen3 template by two others; **frequency and
  depth on live traffic are unmeasured**, and it depends on whether the
  client round-trips `reasoning_content`.
- **What others do:** llama.cpp now defaults `preserve_reasoning` on
  ([#28174](https://github.com/ggml-org/llama.cpp/pull/28174)) and logs the
  divergence point
  ([#27600](https://github.com/ggml-org/llama.cpp/pull/27600)); Dynamo
  measured a mutated thinking prefix at 1.9× latency
  ([blog](https://developer.nvidia.com/blog/streaming-tokens-and-tools-multi-turn-agentic-harness-support-in-nvidia-dynamo/)).
- **Test:** env-gated divergence log in `batched_system_kv.py` — on a partial
  hit, log the decoded window around the LCP end and the gap between restore
  position and true divergence (~30 lines; `VLLM_MLX_DEBUG_MESSAGES=1`
  already exists for a one-off look). The gap histogram decides B4, the
  `preserve_thinking` config call, and suffix reuse (see Watch).
- Headless single-user-message runs (CI reviews) are unaffected.

## B. Scheduler (where the queue-wait data points)

Fork facts, verified in code this round:

- Admission is strict FCFS with head-of-line blocking: `_schedule_waiting`
  pops the head and, when `should_defer_cobatch` defers it, does
  `appendleft` + `break` (`vllm_mlx/scheduler.py:2433-2445`).
  `Request.priority` and `SchedulingPolicy.PRIORITY` exist and are unused.
- mlx-lm's `BatchGenerator._next` runs one decode step, then one prefill chunk
  of up to 2048 tokens per prefilling row. Derived from ledger rates (not
  measured): running streams on the 27B get one token per ~10–13 s while one
  row prefills, ~25 s with two; ~1.2 s on 35B-A3B.

### B1. Shortest-prefill-first admission with aging

- **What others do:** SGLang ships `SHORTEST_PREFILL_FIRST` and `HRRN`
  (rank by waited / uncached tokens)
  ([schedule_policy.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py));
  vLLM lets short prompts pass long prefills
  ([#10235](https://github.com/vllm-project/vllm/pull/10235): p90 TTFT ~−50 %
  on a mixed load, A100).
- **Why it should transfer:** prefill is effectively a serial resource here,
  and the 09-17 data has the signature — median turn prefilled 5.1K tokens
  (~15 s) but median TTFT was 137 s, with requests waiting in 133 of 191
  samples. This reopens the 2026-08 "needs a deep queue" dismissal.
- **Port:** scan the waiting queue (≤ 8) instead of popping the head; rank by
  uncached tokens (already computed at enqueue) with an aging term so a 60K
  cold request cannot starve; admit the first that passes the existing gates.
  Optional first sort key = request priority from a header set per LiteLLM
  route (interactive > CI). Tie-break toward RAM-hot continuing sessions
  (the transferable half of Continuum / ThunderAgent,
  [2511.02230](https://arxiv.org/abs/2511.02230),
  [2602.13692](https://arxiv.org/abs/2602.13692)).
- **Cost:** 60–100 lines in upstream-owned `scheduler.py` (keep the policy in
  a fork-owned delegator) + invariant tests. **Test:**
  `scripts/fork/stress_deep_chains.py` with mixed depths. **Kill:** no
  improvement in median TTFT or turns/min, or any cold-request starvation.
- Skip preemption: vLLM's has a live KV-corruption bug
  ([#59907](https://github.com/vllm-project/vllm/issues/59907)) and recurrent
  state makes it harder.

### B2. Decode-first, stall-bounded prefill

- **What others do:** vLLM chunked prefill prioritises decode with a per-step
  token budget ([docs](https://docs.vllm.ai/en/latest/configuration/optimization/));
  SGLang caps the active chunk; oMLX added prefill/decode fairness
  ([omlx #3487](https://github.com/jundot/omlx/pull/3487)).
- **Port:** `prefill_step_size` is read per `_next` call, so a fork hook can
  shrink the chunk (or run K decode steps per chunk) while any row is
  decoding; and/or serialise prefills (`prefill_batch_size=1`) with B1's
  ordering so two cold rows finish at T/2 and T instead of both at T.
- **Unknown:** sub-2048 chunk efficiency (ledger lever 6 only measured
  ≥ 2048) and solo-vs-co-prefill tok/s. Measure those first.
- **What it buys:** latency shape — rows with a few hundred tokens left
  finish instead of holding a seat for a whole 60K prefill; fewer turns past
  opencode's 300 s timeout. Total GPU work is conserved.

### B3. Re-match at admission, and in-flight prefix dedup

- **Gap (from code):** matching happens only at enqueue
  (`scheduler.py:2210-2211`); a request that missed is never re-matched at
  admission (`batched_system_kv.py:1657-1658`), so a follower deferred behind
  the KV budget still prefills cold after the leader's entry exists. The
  prompt-boundary store is solo-only (`batched_system_kv.py:1777`).
- **What others do:** SGLang in-batch prefix caching deprioritises requests
  that share a long uncached prefix with another waiting request so one
  pioneer prefills it; PEEK generalises this
  ([2607.02525](https://arxiv.org/abs/2607.02525)).
- **Port:** (a) re-peek at admission for enqueue-time misses — small and
  fork-owned; (b) hold a candidate whose LCP with a still-prefilling row
  exceeds its own cached tokens until that row's boundary store lands.
- **Value:** minutes per duplicate when it happens (30K on the 27B ≈ 2.5–3
  min), but only on cold shared prefixes / sub-agent fan-out. Do (a) anyway;
  (b) only if stacked fan-out load returns.

### B4. Idle-time prefill of the next-turn prefix

- **What others do:** Dynamo's `speculative_prefill` — after a response
  finishes, re-render the history with the assistant turn appended and send a
  `max_tokens=1` request to warm the cache
  ([agent hints](https://docs.nvidia.com/dynamo/v1.1.1/user-guides/agents/agent-hints)).
  No published gain for the feature itself.
- **Port:** on `finish_reason=stop` with nothing running or waiting, enqueue
  an internal lowest-priority request for the stripped-thinking re-render up
  to the next user slot; abort the moment a real request arrives. ~200 lines
  + e2e.
- **Gate:** only worth building if A4 shows the re-prefill gap is routinely
  large. Zero value on a saturated queue. Alternative with no code:
  `preserve_thinking:true`, at the context-growth cost the infra config
  already records.

## C. Cache / SSD tier

### C1. Delta or content-addressed SSD spill

- **Gap (from code):** each SSD entry is one safetensors file holding the
  full KV from token 0 plus every checkpoint
  (`system_kv_ssd.py:689-754`); grown entries skip the spill
  (`batched_system_kv.py:739-740`, `792-793`), so the on-disk copy of a
  live chain stays at its first full spill; a chain whose donor was evicted
  re-serialises 4–5 GB of mostly identical bytes. Under load every deep hit
  arrives via SSD (ledger), so this decides how much is re-prefilled.
- **What others do:** ExLlamaV3 keys KV pages and recurrent checkpoints by a
  hash of the token chain
  ([exllamav3 #336](https://github.com/turboderp-org/exllamav3/issues/336));
  LMCache stores fixed token chunks keyed by a parent-chained hash
  ([docs](https://docs.lmcache.ai/kv_cache/local_storage.html)); llama.cpp is
  moving checkpoints to refcounted immutable buffers
  ([#27451](https://github.com/ggml-org/llama.cpp/pull/27451)). The field has
  converged on hash-chain keys; none publishes a dedup-specific gain.
- **Sizing (derived, not measured):** a delta spill is ~64–320 MB per turn
  plus new checkpoints vs ~4–5 GB today; the 9 GB spill queue would hold
  dozens of turns instead of two 60K entries.
- **Cost:** fork-owned (`system_kv_ssd.py`, `batched_system_kv.py`) but a new
  on-disk format with parent pointers, refcounted eviction and chained
  promote — and the SSD index is pinned at v1, so a version bump needs a
  migration or it wipes spills on deploy. Largest item in this document;
  only justified if deep concurrent chains are a recurring load.
- Lossy checkpoint codecs (int16 Hadamard,
  [llama.cpp #27211](https://github.com/ggml-org/llama.cpp/issues/27211))
  flip 18.6 % of tokens on 35B-A3B — dedup first, never the codec on MoE.

### C2. Thin checkpoints only once the list is at capacity

- llama.cpp fixed spacing-eviction deleting the recent checkpoint on short
  prompts by applying the spacing rule only at the cap
  ([#28302](https://github.com/ggml-org/llama.cpp/pull/28302), ~10 lines).
  Same shape as the open follow-up "first post-system message boundary is
  thinned → short (~1K) shared system prompts miss entirely". Under 20 lines
  in `batched_system_kv.py` plus an invariant test.

## D. Pin bumps

- **D1 — mlx 0.32.3 (released 2026-09-29).**
  [#4505](https://github.com/ml-explore/mlx/pull/4505) head-dim-256 SDPA on
  pre-M5: 16K prompt peak 46.9 → 35.5 GB, prefill +8 % (M4 Pro, 27B nvfp4);
  the dispatch is narrow (causal, D=256, query = key length ≥ 2048 — i.e. the
  first chunk only at 2048-token chunks), so the real question is whether it
  reopens *large* first chunks, which lever 6 killed on memory.
  [#4572](https://github.com/ml-explore/mlx/pull/4572) gather_qmm tile
  scheduling 1.2–1.3× on the op (MoE prefill; non-M5 gain unverified). Also
  carries #4020 (already measured ≤ 3 %) and #4431. Expect the T=0 shift the
  ledger predicts; re-run the #48 crash recipe.
- **D2 — mlx 0.32.4 (unreleased).**
  [#4596](https://github.com/ml-explore/mlx/pull/4596) unrolls the generic
  two-pass decode attention kernel: 1.04–1.66× on the op (M1 Pro, D128, no
  mask). Our Qwen3.5-family models are head-dim 256, and the GQA-specialised
  kernel accepts only 64/128 — so our long-context decode runs exactly this
  generic kernel. Estimate (not measured) ~5–8 % on 27B near 100K. This also
  suggests ledger lever 4's NULL was structural for every D256 model.
  [#4516](https://github.com/ml-explore/mlx/pull/4516) (fast qmv for outputs
  not divisible by 8) matters for qwen4_exp / REAP-288 only.
- **mlx-lm main:** no decode or prefill lever in ~70 commits past the pin; a
  bump crosses #1778 (cache-state refactor, merged 2026-09-09). No speed
  reason to bump.

## E. Hygiene (small, certain)

- **Dead work per request:** `_compute_prefix_boundary`
  (`engine/batched.py:1404-1463`) does two renders + two encodes on the event
  loop, and its result is only read by a legacy chunked-prefill branch that
  mlx-lm 0.32's `BatchGenerator` never takes. Measured on the laptop: encode
  136 ms per 97K tokens → ~0.28 s of event-loop block per 100K request.
- **Python-speed prefix matching:** `common_prefix_len`
  (`system_kv.py:295-301`) is a token-by-token loop run up to four times per
  request across all bag entries, plus `os.path.commonprefix` over SSD index
  rows; derived 30–150 ms typical, ~0.7 s worst case, on the generation
  worker. A numpy/buffer LCP in fork-owned code is ~30 lines.
- **Spawn:** `import vllm_mlx.server` 2.2–2.7 s, of which ~0.6 s is torch
  pulled in via transformers on text-only routes; a Hub round-trip per spawn
  unless offline; a full-weights zero-scan for a log line on checkpoints with
  a vision config (`utils/tokenizer.py:232-239`, cost unmeasured).
- **Thinking-budget processor:** `tokens.tolist()` every step on the main
  Qwen3.8 routes (`constrained/thinking_processor.py:243-245`) — the same
  class as the measured 3.7 % DRY tax; unmeasured.
- **Reliability, not speed:** the 499000 crash has a candidate fix —
  [mlx-lm #1911](https://github.com/ml-explore/mlx-lm/pull/1911). The naive
  version costs 15–25 % decode; evaluating only the non-KV (GDN/conv) cache
  states per step is reported to fix it with no regression. A better
  candidate than the unarmed #84 clamp.

## Audit side-findings (not speed)

- Since upstream #707, `add_request` (tokenization + cache lookup) runs on
  the generation worker, but `BatchedSystemKV`'s docstring and comments
  (`batched_system_kv.py:197-203`, `1146-1155`) still describe an event-loop
  contract; 400/503 responses can wait behind an in-flight prefill chunk.
- RAM miss + SSD hit sets `ssd_pending`, and `promote_ssd_pending` then fully
  materialises the restore for a request that is still waiting
  (`batched_system_kv.py:1717-1736`) — the case #106's lazy restore was meant
  to remove, on the path the stress runs say all deep hits take. Worth a
  memory-safety look.

## Watch, not build

- **Suffix cache reuse for hybrids** — reuse KV of text that survives a
  mid-prompt edit (re-rotate RoPE for attention layers, carry the GDN state
  forward). [paper](https://arxiv.org/abs/2609.37725),
  [llama.cpp #29918](https://github.com/ggml-org/llama.cpp/pull/29918)
  (unreviewed). Targets our exact model class, but it is approximate, breaks
  the T=0 gate by construction, and its baseline lacks message-boundary
  checkpoints. Revisit only if A4 shows routine mid-history divergence of
  tens of K tokens.
- **mlx-lm #1922** — fused 4-bit matmul for 4–8 batched decode rows, +22.8 %
  aggregate claimed on an unstated (likely small) model
  ([PR](https://github.com/ml-explore/mlx-lm/pull/1922)). Dense 4-bit only;
  the fleet rarely holds ≥ 4 decoding rows on a dense route.
- **oMLX #3853** — fused GDN "prework" kernel, +6 % single-stream on M1 Max
  ([PR](https://github.com/jundot/omlx/pull/3853)); batch-1 FP16 only, does
  not engage on batched decode.
- **vllm-metal D256 GQA decode kernels**
  ([#715](https://github.com/vllm-project/vllm-metal/pull/715)) — not
  portable, but evidence that D256 decode attention is under-served; the
  ledger has no 35B decode-vs-depth ladder.
- **GPU QoS starvation between co-resident Metal processes on macOS 27**
  ([report](https://github.com/Layr-Labs/d-inference/issues/1116)) — single
  unreplicated report; relevant only when voice and a heavy route decode
  together.

## Checked, nothing to take

Tokenization/template caching (93K tokens: render 0.012 s, encode ~0.1 s —
only the duplicate calls in E matter); llguidance fast-forward tokens (real,
but `response_format` is barely used and tool calls are unconstrained);
early stop on tool-call completion (≈ 1 token, breaks parallel calls);
session/TTL-aware eviction (closed by the 2.9 % evict-to-reuse data);
tiered offload, KV-aware routing, prefill/decode disaggregation (multi-node);
llama.cpp `--cache-ram` / `--cache-disk` (equivalent to ours); lazy/FP8 KV
quant and all speculative-decode enablers (refuted class); mlx-qsdpa and ZMLX
(dormant, still unreplicated); LM Studio mlx-engine, Ollama, koboldcpp,
TabbyAPI, mistral.rs, MLC (same architecture or behind); oMLX 0.7.0's
remaining additions (M5/ANE-specific).

## Suggested order

1. A1, A2, A3 kernel repro — an afternoon on a spare port, no code.
2. A4 divergence log + E's dead-work removal and LCP — small patches.
3. B1 (+ priority key), then B2 after measuring sub-2048 chunk efficiency.
4. C2, B3(a).
5. D1 with the next routine pin bump; D2 when 0.32.4 ships.
6. C1 and B4 only if their gates (recurring deep concurrent chains; A4 gap
   size) say so.
