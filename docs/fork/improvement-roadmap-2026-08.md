# Improvement roadmap — 2026-08-17 ecosystem research

Four parallel surveys (MLX/mlx-lm upstream, llama.cpp/GGML on Metal, Mac-native
serving stacks, big-backend serving layer), cross-checked against the fork's
measured verdicts (PATCHES.md, docs/fork/, memory). This file records what
survived; the refuted/rejected list at the bottom is as load-bearing as the
roadmap.

**Status 2026-08-18** — executed so far (all pushed on fork `8d2bbf8`, **not
yet deployed** to the Studio):

- Tier 1 item 1 → **DONE**, patch #73 (llguidance fail-closed structured
  output, upstream PR #636 adapted). Open follow-up: strict tool-argument
  schemas.
- Tier 1 item 2 → **DONE**, patch #74 (eviction-timing histograms + tombstone
  gap, counter mirrors, finish-reason counter, `/health/ready`). The verdict
  now waits on a post-deploy soak:
  `rate(vllm_mlx_cache_evict_to_reuse_gap_seconds_count[1d])` staying at zero
  closes the eviction question. The live Metal buffer-count gauge turned out
  impossible (MLX exposes only the ~499k ceiling) — recorded as a limitation
  in PATCHES.md #74.
- Tier 1 item 3 (requant split) → discussed, **awaiting go**; agreed canary =
  one 27B-8bit route to 6-bit RTN with KLD + needle-at-cap validation.
- Also done en route (2026-08-17): rebase onto upstream `5021350` (v0.4.1) —
  which consumed several watch items: upstream's #683 non-trimmable scheduler
  cache and #648 chunked prefill are now in the base (batched-side; our
  system-KV remains the active cache), and PR #636 became patch #73.

## Standing validations

- **Hybrid-safe caching: third independent validation.** SGLang shipped a
  unified radix cache (2026-08-11) with exactly the `batched_system_kv.py`
  design — per-component reuse semantics, Mamba checkpoint frontiers,
  copy-before-mutate. Meanwhile mlx-lm's hybrid prefix caching is still broken
  ([mlx-lm #980](https://github.com/ml-explore/mlx-lm/issues/980), open) and
  llama.cpp's hybrid checkpoint invalidation is still broken
  ([llama.cpp #24055](https://github.com/ggml-org/llama.cpp/issues/24055),
  open). #980 is an upstreaming opening for a `system_kv` port to mlx-lm
  (Tim's call on PRs).
- **The MLX bet won.** Ollama moved macOS inference to MLX (0.19, stable
  v0.30; Qwen3.5-35B-A3B +57% prefill / +93% decode vs their llama.cpp
  wrapper); Apple co-designed the M5 GPU neural accelerators around MLX's
  compute patterns.
- **Spec decode stays dead on this hardware — now with external confirmation.**
  llama.cpp MTP on M1 Max: net loss at every configuration (−11% at even 100%
  acceptance; [#23752](https://github.com/ggml-org/llama.cpp/issues/23752)) —
  same physics as our 0.5–0.76x measurement. It flips positive only on
  M5-generation silicon (tensor-core verification: MTP +75% dense 27B) or with
  co-trained MTP heads + Ollama's acceptance auto-tuner (+90%, M5-assisted).
  Re-evaluate on hardware upgrade, not before. An M5-gen Studio attacks our
  worst number directly: 16-minute cold TTFT at 112K (M5 ≈ 3.3–4x prefill).

## Tier 1

1. ✅ **DONE 2026-08-18, patch #73.** **Structured output via llguidance** — the one big-backend capability the
   fork lacks entirely; three of four surveys converged on it. Current
   `constrained/json_schema_processor.py` is lm-format-enforcer: maintenance
   mode, slowest of the field (llguidance ~50µs/token CPU; xgrammar <40µs),
   incomplete schema coverage, and fails OPEN. Upstream PR
   [#636](https://github.com/waybarrios/vllm-mlx/pull/636) is the right base:
   request-local llguidance matcher, MLX token masking, padded-vocab handling,
   schema-aware EOS, fail-closed. Extension worth adding after: strict
   tool-argument schemas (mistral.rs precedent) — constrain tool-call args
   JSON during decode; malformed tool calls currently waste whole generations.
   → **fork patch #73.**
2. ✅ **DONE 2026-08-18, patch #74** (verdict pends post-deploy soak). **Instrumentation package** (fulfils the instrument-first verdict of
   `prefix-caching-landscape-2026-08.md`): convert cache gauges to monotonic
   counters (vLLM v1 metric shapes), add `idle_before_evict` and `reuse_gap`
   histograms — the two distributions that decide empirically whether
   recency-only eviction ever hurts on this box. Add TTFT / queue / prefill /
   decode histograms and a finish-reason counter. Two extras from the
   research: a **Metal buffer-count gauge** (the ~499k resource limit is a
   count of live Metal buffers, not bytes — byte-denominated relief (#48)
   cannot see that crash class; [mlx-lm
   #1332](https://github.com/ml-explore/mlx-lm/issues/1332)) and a
   `/health/ready` probe that runs a real 1-token forward pass (vLLM
   convention; would catch the llama-swap-wedge failure class).
3. ⏳ **AWAITING GO** (canary agreed: one 27B-8bit route → 6-bit RTN). **Requant split** (refines the 2026-07 "27B at 5–6bpw DWQ" lever — DWQ's
   own docs say distillation isn't worth it above 4-bit):
   - 27B-8bit routes → **6-bit RTN or ~5.5bpw `mlx_lm.dynamic_quant`**:
     measured KLD 0.029 vs bf16 on a 27B dense ("imperceptible"; 8-bit is
     0.014, 4-bit 0.113), frees ~6–8GB of weights = context/relief headroom.
   - DWQ where it measurably shines — the 4-bit MoE routes, above all
     **Qwen3-Coder-Next-80B** (DWQ protects router tensors; +34% vs
     mixed-precision on a 35B-A3B MoE). 80B DWQ run needs the 8-bit-teacher +
     `--max-seq-length 512` tricks; verify 63GB feasibility first.
   - Validate any requant with KLD (`mlx-kld`) + needle-at-cap + ladder
     re-measure, per the vLLM FP8-KV lesson (128K recall 91%→13% — validate at
     max context, never at 4K).

## Tier 2

4. **MoE gate/up projection fusion** ([mlx-lm
   #956](https://github.com/ml-explore/mlx-lm/issues/956)): one `gather_qmm`
   per MoE layer instead of two; +8.6% decode measured (Qwen3-30B-A3B), +5%
   (gpt-oss-120B). Dispatch-overhead reduction, not a bandwidth trick — no
   conflict with the refuted decode levers. Implementable as load-time weight
   concat at sanitize level.
5. **Streaming tool-parse fallback ladder** (Ollama design): template-derived
   prefix detection → tool-shaped bare-JSON fallback → return-as-text on
   mismatch. Retires the per-family parser bug class (#27/#47/#72).
6. **Aliasing + per-step-concat audit** of the snapshot stores: mlx
   [#3689](https://github.com/ml-explore/mlx/issues/3689)'s "use-after-free"
   root-caused to application-side buffer aliasing in a prefix-cache restore —
   the class we hit twice (#69, upstream #642). Also audit for per-step
   `mx.concatenate` chains without eval (buffer-count leak;
   Falcon-Mamba precedent, mlx-lm #1656).
7. **8-bit KV-quant needle-at-cap experiment** (experiment, not build): rerun
   the 9/9 needle harness with `kv_bits=8` on one 45GB route at its cap.
   4-bit KV is disqualified (quality collapse reports + 92% Metal decode
   regression at 64K in llama.cpp). Upstream has no batched quantized KV and
   TurboQuant guards out SSM/hybrids — a build would be dense-routes-only.
8. **API conformance pass**: reject-don't-ignore unsupported OpenAI params
   (`n>1`, `logprobs`, `seed`); test the Anthropic endpoint with
   Claude-Code-shaped traffic (documented emulation pitfalls elsewhere:
   images dropped in the adapter, tool_use round-trip, `count_tokens`, SSE
   event framing).

## Tier 3 — watch / conditional

- **vllm-project/vllm-metal**: official vLLM plugin, MLX compute backend under
  vLLM's engine (daily-active Aug 2026; paged varlen Metal kernel; Qwen3.8
  hybrid SDPA+GDN landing). Benchmark against our stack; the credible exit
  path if waybarrios upstream freezes again; mine its kernels regardless.
- **Pre-flight fit computation** (llama.cpp `--fit` concept): compute
  seats/ceilings by virtual allocation instead of hand-measured ladders.
- **Tool-call-pending eviction bit** (Continuum, arXiv 2511.02230):
  `finish_reason=tool_calls` → short-TTL protection so relief prefers other
  victims. The #74 instrumentation now measures the need: check whether
  evict-to-reuse events cluster behind finish_reason=tool_calls bursts
  before building.
- ~~Upstream PR #636~~ — consumed: adapted as patch #73.
- **Next mlx / mlx-lm releases** (we are at the released tips today): take mlx
  promptly (fp32-dequant + quantized-matmul numerics fixes) but re-run the
  patch-#48 staged 94K crash recipe first — the residency-set restructure
  (mlx #4211) may shift `get_peak_memory` behavior. For mlx-lm, diff
  `generation/batch*`, `cache.py`, `server.py` against our subclass points
  before taking it (0.31.0 was yanked for batch-cache cross-contamination).
- **Open-TQ-Metal** (arXiv 2604.16957): compressed-domain int4-KV attention
  shaders; claims 48x attention @128K, bit-identical outputs, code released;
  single-author, unverified. Strongest concrete KV-quant artifact — attacks
  the 45GB-class OOM wall. Verify before believing anything.
- **MLX Thunderbolt-RDMA** (WWDC26/macOS 26): weakens the "RDMA is TB5-only"
  premise of the clustering verdict. Capacity path only; docked-only; still
  not a speed play.
- **mmap fail-soft capacity route**: llama.cpp pages over-limit MoE weights
  from SSD (degraded but alive; MLX wires everything). One llama-swap route
  for GLM-4.5-Air/gpt-oss-120b-class models — no fork work, an alternative to
  the TB4 capacity experiment.
- **Uniform thinking-effort parameter**: per-request `think: bool|effort`
  mapped per family (Qwen `enable_thinking`, gpt-oss harmony effort, GLM).

## Rejected, with reasons (do not re-research without new evidence)

- **KV pruning** (H2O/SnapKV line): zero production adoption anywhere;
  CodeComp (arXiv 2604.10235) shows attention-score eviction breaks on source
  code — our primary workload.
- **CacheBlend/UCM non-prefix reuse**: structurally incompatible with SSM
  layers (no partial-state rollback), exact-quality risk on code. Watch only.
- **Session-aware eviction** (SGLang's upgrade): measured gains at 64–128
  concurrent clients; collapses to ≈LRU at concurrency 1.
- **mistral.rs as a platform**: Metal MoE dispatch measured ~280x slower than
  llama.cpp (issue #2032).
- **LM Studio engine internals**: behind this fork (no batched constrained
  generation, no batched spec decode, stock mlx-lm caching).
- **Ollama/LM Studio model scheduling**: llama-swap remains superior for this
  deployment (per-route ceilings, groups, health-gated swaps).
- **4-bit KV quant**: quality collapse on small models (PPL>500 reports) and
  Metal decode collapse at long context.
