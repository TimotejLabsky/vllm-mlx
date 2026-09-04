# Speed-lever ledger — consolidated 2026-09-03

Canonical list of every decode/prefill speed lever this fork has measured on
the Mac Studio (M1 Ultra, 64 GB), so future improvement passes stop
re-proposing dead ones. Companion to PATCHES.md (which records what *shipped*);
this records what was **tested and why it lives or died**. Update this file
whenever a lever gets a measured verdict.

Fleet state when written: fork `afc580c` (#96 MoE fusion armed on 7 MoE
routes), mlx 0.32.2, mlx-lm pinned `f4f3b57`, mlx-vlm 0.6.17, all 19 text
routes on BatchedEngine + batched system-KV.

## Where the time actually goes

After the 2026-06→09 campaign, decode time on this box is almost entirely
inside mlx's Metal kernels:

- **Dense 27B-class**: bound by 4-bit g64 `quantized_matmul` gemv kernel
  efficiency (~130–140 GB/s effective of ~800 peak — measured 2026-09-03).
  Only upstream mlx kernel work moves this.
- **MoE 3B-active class**: was dispatch-bound in the expert MLP → fixed by
  #96 (+12.8%). Remaining dispatch (plain qmm projections) measured too
  cheap to matter (see #97 below).
- **Hybrid GDN layers** (qwen3_5 / qwen3_next): bound by the gated-delta
  forward → mlx PR #4020 is the pending 1.3–2.2× fix (see watch list).
- **Host/Python side**: audited 2026-09-02, no leak; the scheduler already
  runs the correct `mx.async_eval` pipeline with exactly one sync per step
  and on-GPU sampling. Only real CPU-in-loop cost is DRY (3.7% decode tax).

## Shipped positives (details in PATCHES.md)

| Lever | Result |
|---|---|
| #96 MoE gate/up fusion (one `gather_qmm` per expert MLP) | **+12.8%** live on 35B-A3B (69.4→78.3 tok/s), T=0 byte-identical, armed on 7 MoE routes |
| Batched system-KV + SSD tier (#29–#40, #13/#16) | warm TTFT collapse (e.g. 911s→12s at 91K), restart 31.7s→1.89s |
| upstream #606 `train(False)` cherry-pick | ~6× prefill (2026-06) |
| mlx-lm pin lift to `f4f3b57` (fp32 co-batch promotion fix) | removed whole-batch fp32 KV promotion when cold+warm co-batch |
| memory-pressure relief (#48, #53) | not a speedup — removes the crash ceiling that throttled long-context use |

## Refuted levers (do NOT re-propose without new evidence)

| # | Lever | Verdict | When / where measured |
|---|---|---|---|
| 1 | Speculative decode / MTP | 0.5–0.76× on chat; 2026-09-02 re-test: DFlash2 is the first >1× on M1 (code 1.14×, math 1.30×) but chat 0.82× + integration cost → do not build. Flips positive only on M5-class silicon | 2026-06-24, re-measured 2026-09-02 |
| 2 | Continuous batching as a *speed* lever | ~1.2× aggregate ceiling (MLX decode not memory-bound at 4-bit/8–35B). Deployed fleet-wide for capacity/multiplexing, not speed | 2026-06-24 |
| 3 | KV-quant 4-bit | quality/speed loss at fleet shapes. Caveat: verdict predates fused-kernel approaches; see mlx-qsdpa in the watch list before ever re-opening | 2026-06 |
| 4 | GQA-8 two-pass attention (mlx #4077) | NULL on M1 Ultra (−0.14% at n=6); 27B-4bit is GQA-6 and can't even exercise it | 2026-08-31 |
| 5 | Thunderbolt clustering for speed | decode latency-bound, dual TB4 changes nothing; RDMA is TB5-only. (Capacity path via weighted PP remains a separate, viable experiment) | 2026-06-13 / 07-13 |
| 6 | Prefill step-size tuning | NULL ≥2048; 8192 costs +14 GB transient. (Found+fixed #95: the flag never reached batched routes) | 2026-09-02 |
| 7 | Page-cache prewarm before swap | NULL (cold-vs-warm spawn Δ0.6s; weight I/O ≈2.4s/15GB on this NVMe — swap pain is spawn+init+KV, not weights) | 2026-09-02 |
| 8 | Host-side Python overhead trimming | no leak found (SSE 0%, REPDETECT free); only DRY = 3.7% decode tax → config call, not code | 2026-09-02 |
| 9 | mlx-lm PR #1824 fork-pin cherry-pick (`BatchKVCache` `.item()` → `tolist()`) | NULL at fleet scale: 123.1 vs 123.1 agg tok/s, 77.9 vs 77.9 single (35B-A3B, prod env, 6 staggered concurrent, SHAs identical). Upstream's +11.6% is a 0.5B-at-2370-tok/s effect; our per-retirement syncs ≈0.5%. Take it for free in the next routine pin bump | 2026-09-03 shadow A/B |
| 10 | Plain-`quantized_matmul` same-input projection fusion (prototype "#97": GDN in_proj 4→1 + shared-expert gate/up + router gates; retro-covers dense QKV and dense MLP gate/up) | REFUTED twice: (a) perf FLAT (77.5 vs 77.5 tok/s with 110 groups fused — plain qmm dispatch is too cheap; #96's win was a `gather_qmm` property, not general); (b) **breaks the T=0 byte-identical gate** — the fused 12352-row matmul selects a different Metal kernel variant than separate 8192/4096/32/32 calls → accumulation-order rounding → stable-but-different output. Dense MLP gate/up additionally NULL in isolation (21504-row kernels are pure weight-bandwidth). Branch deleted; suite-green design recipe preserved in session memory (rebuild ~30 min if mlx ever unifies qmm variants) | 2026-09-03 live A/B |

### Methodology lessons (paid for, don't re-learn)

- **Small-shape unit-test equality does not clear the live gate.** Kernel-variant
  selection is shape-dependent; exact equality of a fused op must be re-proven
  at production shapes via the live T=0 SHA gate.
- **Microbench fusion wins only reproduce in serial dependency chains** —
  independent-op timing shows zero because Metal overlaps independent kernels
  (it even fails to reproduce the shipped #96 win).
- **Isolated µs don't transfer linearly to live steps** — #96 amplified ~3–4×
  live, #97 amplified to zero. Always confirm with the end-to-end A/B.
- **A/B throughput claims from tiny models don't scale down-column** — an
  overhead that is 11.6% at 2370 tok/s is 0.5% at 78 tok/s (lever 9).

### 2026-09-04 — cross-engine benchmark round (see engine-benchmarks-2026-09.md)

- **oMLX 0.6.4 measured on this box**: vendored GDN Metal kernels
  (`custom_kernels/qwen35_prefill/gdn.py`) + burst/MTP decode. Natural-prompt
  T=0 usage-counted A/B on 35B-A3B: **88.9 vs our 78.7 tok/s = +13 %
  single-stream** — a live, on-box floor estimate for what the mlx #4020 class
  of kernels is worth at short context (its 1.3–2.2× was the isolated GDN
  forward, not end-to-end). Its 2× showing on the serving benchmark is a
  workload artifact (below). Prefill par, TTFT worse, warm-prefix cache far
  weaker than ours. Verdict: no action for us beyond the existing
  #4020-at-next-mlx-release plan; oMLX is the engine to re-check after that
  lands.
- **New methodology lesson: random-token serving benchmarks structurally
  favor speculative decoders** — degenerate loopy outputs give the draft path
  near-free acceptances (oMLX showed 132 tok/s there vs 88.9 real). Any
  speculative engine must be cross-checked with a natural-prompt
  usage-counted run before believing its serving numbers.
- **Fork-batched vs upstream-batched at identical kernels**: +13.7 % MoE
  serial (fusion #96 reproducing), +9 % MoE / +17 % dense at 4-way, TTFT
  better everywhere; upstream warm-prefix = cold on hybrid models re-confirmed
  on 2026-09-04 main (32.6 s vs our 0.15 s at 7.2K on 27B-8bit).
- **MLX vs llama.cpp at matched ~4-bit (27B dense)**: +33 % decode for us,
  prefill par. Ollama trails llama-server on every metric (same engine, more
  wrapper). No lever here.

## Watch list / open items

- **mlx PR #4020 — gated-delta Metal kernels: DOWNGRADED 2026-09-04 from
  "big pending win" to routine pin-bump.** Tested pre-release on this box via
  the PR's own CI wheel (head `c7e1a2a`, merge-ref build) + a 3-line
  env-gated dispatch shim in mlx-lm's `gated_delta.py`:
  - **Semantics verified**: `mx.fast.gated_delta_update(q,k,v,g,beta,
    initial_state,mask)` matches mlx-lm's existing kernel bit-exactly at
    T=1 and to fp32 epsilon at prefill shapes, straight pass-through (no
    q-scaling — the model scales, same contract mlx-lm #1823 documents).
    Integration when it lands is trivial.
  - **Speedup on our stack ≈ nil**: kernel-vs-kernel inside the same wheel
    (both paths share identical host overhead) at Qwen3.6-35B GDN shapes
    (Hk16/Hv32/D128, M1 Ultra): decode T=1 **1.12×** wall, prefill T=4096
    **0.98×** (parity). The old kernel is ~1–2 % of a release-build decode
    step, so end-to-end expect **≤ ~3 % decode, nothing on prefill**. The
    PR's 1.3–2.2× headline is against a baseline we don't have — mlx-lm's
    own JIT gated-delta kernels (which our pin ships) already sit within
    ~12 % of the fused kernels on this hardware. Corollary: oMLX's +13 %
    is mostly its burst-decode pipeline, not GDN kernels.
  - **T=0 outputs shift** (argmax-tie cascade from 1e-8 prefill drift;
    diverged at token ~100–150 on the 35B). The byte-identity gate WILL
    flag the pin bump that picks this up — expected, not a bug.
  - **CI artifact wheels are debug-slow** (25 vs 77.5 tok/s on identical
    setup) — usable for semantics and same-wheel kernel ratios, never for
    absolute end-to-end numbers. (No Metal toolchain on the Studio or the
    laptop — CLT only, no Xcode — so building release wheels locally is
    blocked; the CI artifact route is the pre-release test path.)
  Still take it at the next mlx release (free ~1–2 %, upstream-maintained),
  just not as an event. Test residue: `~/bench-2026-09-04/{gdn-env,ctl-env}`
  on the Studio, harness `gdn_test.py`. Related draft: mlx #4409.
- **DRY-off-on-4bit** — pending config call (worth the measured 3.7% on that
  route; recommendation on record 2026-09-02).
- **mlx-qsdpa** (fused quantized-KV SDPA, claims 1.56–1.71× at 64–128K on
  M2 Ultra): the only untested idea left. n=1 unreplicated repo — if ever
  re-opened, first step is a half-day standalone kernel replication on this
  box (kill: <1.2× at 128K), only then discuss integration.
- **Requant split** (27B at ~5–6bpw DWQ; 6-bit RTN vs DWQ-4bit-MoE) — a
  quality/speed trade from the 2026-08 research, still unexplored.
- **`--compile` flag is a proven no-op** (instance-level `__call__` wrap is
  never invoked through `model(x)`; verified empirically 2026-09-03) — delete
  or document in a future hygiene patch. The useful granularity is already
  `mx.compile`d inside mlx-lm's model code (39 sites).
- Tiny hoist candidate: `gemma4_text.py` builds `mx.array(cache.offset)` per
  layer per step.
- Upstream watch (unchanged): mlx-lm #1821 (499000 leak site), #1778 (cache
  state refactor — tripwire armed), mlx #4431 GQA batch-offset fix (merged,
  unreleased), mlx-vlm 0.7.0 stable (re-run the arch sweep before taking).
- mlx-lm #1799 (gated-delta readout scale): CLOSED upstream as incorrect
  (2026-09) — the landmine defused itself; a plain pin bump can no longer
  pick it up.
