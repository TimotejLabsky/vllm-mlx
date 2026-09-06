# Engine comparison benchmarks — 2026-09-04

Cross-engine serving benchmark on the production Mac Studio (M1 Ultra, 64 GB),
comparing this fork against upstream vllm-mlx, raw mlx-lm, oMLX, llama.cpp and
Ollama on one production MoE and one production dense model. Run while the
fleet was quiesced (llama-swap models unloaded, `voice-reload.sh` bench-guarded);
every cell was watchdogged for llama-swap contamination — all cells clean.

## Setup

**Engines** (all serving OpenAI-compatible `/v1/chat/completions` on a spare
port):

| engine | build | notes |
|---|---|---|
| `mlx_lm.server` | mlx-lm 0.32.0 (git `f4f3b57`) | the raw library the MLX engines sit on |
| upstream-simple | waybarrios/vllm-mlx `f2d3ad3` (2026-09-04 main) | pinned to the same mlx 0.32.2 / mlx-lm `f4f3b57` / transformers 5.14.1 as the fork, so deltas are engine code, not kernels |
| upstream-batched | same, `--continuous-batching` | upstream has dropped `--text-only` (text is now the default; `--mllm` forces vision) |
| fork-simple | deployed fork build (tip `afc580c`, v0.4.1 base) | `VLLM_MLX_SYSTEM_KV_RAM_MB=6144` |
| fork-batched | same | production route env: batched system-KV + rails, MoE gate-up fusion (#96) on the MoE |
| oMLX | jundot/omlx 0.6.4 (source install) | its own mlx-lm 0.31.3 pin, as shipped; mlx 0.32.2 |
| llama.cpp | brew `llama-server` build 10566 | `-ngl 99 -c 65536 -np 4 -fa on --jinja --cache-reuse 256` (production route minus DRY, ctx sized to match) |
| Ollama | brew, local GGUF imported | `num_ctx 16384`, `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_FLASH_ATTENTION=1` |

**Models** (both production routes, both hybrid attention+GDN):

- MoE: `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ` (~19 GB, 3B active, 8 seats)
- dense: `mlx-community/Qwen3.8-27B-8bit` (~28 GB, 4 seats)
- quant-matched dense trio: `Qwen3.8-27B-4bit` (MLX) vs `Qwen3.8-27B-UD-Q4_K_XL.gguf`
  (llama.cpp and Ollama, same GGUF file)

**Measurement**: vLLM `benchmark_serving.py` (v0.6.6 standalone, openai-chat
backend; patched for SSE line-buffering, `reasoning` deltas, and to send
`max_tokens`) — random dataset, 1024-token inputs / 256-token outputs, seed 42,
T=0; one pass at concurrency 1 (6 prompts) and one at concurrency 4
(12 prompts). Plus custom probes: load time (spawn → first completion),
~7.2K-token cold prefill (cache-busted ×2), warm-prefix TTFT (identical
~7.2K-token prompt re-sent after a store), peak RSS summed over the server's
process group. Client ran on the same box. Harness:
`scripts/fork/engine_bench/`, raw data:
`data/engine-benchmarks-2026-09-results.jsonl` (per-cell JSON incl. p99s).

## Results — MoE (Qwen3.6-35B-A3B-4bit-DWQ)

| engine | load s | c1 out tok/s | c1 TTFT ms | TPOT ms | c4 out tok/s | prefill 7.2K tok/s | warm TTFT s | peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| mlx_lm.server | 4.9 | 50.4 | 825 | 16.7 | 101.4 | ~1450 | 0.19 | 20548 |
| upstream-simple | 7.2 | 58.6 | 876 | 13.7 | 58.3 | 1476 | 4.86 | 20507 |
| upstream-batched | 7.6 | 57.7 | 773 | 14.4 | 120.5 | 1511 | 4.72 | 20419 |
| fork-simple | 6.0 | 59.2 | 912 | 13.4 | 59.3 | ~1464 | 4.87 | 20442 |
| **fork-batched (prod)** | 6.4 | **65.6** | **659** | **12.7** | **131.5** | 1512 | **0.06** | 8982 |
| oMLX | 6.0 | 132.1* | 995 | 5.6* | 204.9* | 1483 | 2.32 | 9015 |

## Results — dense (Qwen3.8-27B-8bit)

| engine | load s | c1 out tok/s | c1 TTFT ms | TPOT ms | c4 out tok/s | prefill 7.2K tok/s | warm TTFT s | peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| mlx_lm.server | 5.4 | 13.5 | 4354 | 57.1 | 20.7 | 217 | 0.28 | 28074 |
| upstream-simple | 10.2 | 14.1 | 5023 | 51.6 | 14.0 | 221 | 32.4 | 28099 |
| upstream-batched | 8.7 | 14.0 | 4959 | 52.5 | 21.6 | 219 | 32.6 | 28083 |
| fork-simple | 8.3 | 14.2 | 4924 | 51.5 | 14.0 | 221 | 32.3 | 28805 |
| **fork-batched (prod)** | 9.2 | **14.5** | **4235** | 52.4 | **25.2** | 218 | **0.15** | 28088 |
| oMLX | 9.4 | 26.1* | 5117 | 29.5* | 28.6* | 219 | 14.4 | 29591 |

## Results — quant-matched dense trio (~4-bit Qwen3.8-27B)

| engine | load s | c1 out tok/s | c1 TTFT ms | TPOT ms | c4 out tok/s | prefill 7.2K tok/s | warm TTFT s | peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| **fork-batched (MLX 4bit)** | 8.0 | **20.0** | 4226 | **33.6** | **26.6** | 217 | 0.11 | 15099 |
| llama.cpp (Q4_K_XL) | 9.4 | 15.0 | 4055 | 48.9 | 22.8 | 224 | 0.14 | 32583 |
| Ollama (same GGUF) | 9.6 | 14.4 | 4635 | 49.2 | 16.7 | 196 | 0.16 | 27634 |

## Findings

1. **Fork vs upstream (same kernels, engine code isolated):** fork-batched
   beats upstream-batched on every serving metric — MoE c1 +13.7 % (the #96
   gate-up fusion, consistent with its +12.8 % deploy A/B), MoE c4 +9 %,
   dense c4 +17 %, TTFT better across the board.
2. **Upstream's prefix cache is still zero-hit on hybrid models** — on
   2026-09-04 upstream main, warm-repeat TTFT equals cold on both engines and
   both models (4.7 s MoE, 32.6 s dense = a full re-prefill of 7.2K tokens).
   The fork's system-KV restores in 0.06–0.15 s: **~75× (MoE) and ~215×
   (dense) faster warm TTFT at 7.2K context.** This re-confirms
   `continuous-batching-hybrid-caching.md` against current upstream.
3. **fork-simple's warm miss is by design, not a bug**: the SimpleEngine
   system-KV caches the *system prefix* (text before the first user-turn
   marker), and the probe carries its document in the user message. Agent
   traffic with fat system prompts is its target shape. The batched cache
   (#88 message-boundary checkpoints, grow-on-HIT) caches deep prefixes
   regardless and hits.
4. **`mlx_lm.server` is a decent single-user baseline**: its single-slot
   prompt cache works (0.19–0.28 s warm), and it pipelines concurrent
   requests to ~2× serial throughput — but it is multi-model-on-demand with
   no admission control, no memory rails, and the `model` field of a request
   can trigger an arbitrary HF download.
5. **oMLX is the real rival** (\* = inflated, see below). Its standard-bench
   decode looks 1.8–2.6× — but the random-token prompts produce degenerate,
   loopy outputs that its speculative burst decode exploits. On a natural
   prompt at T=0, usage-counted: **oMLX 88.9 tok/s vs fork-batched 78.7 tok/s
   = +13 %** single-stream on the MoE (identical output text). It ships
   vendored custom Metal kernels for the Qwen3.5+ gated-delta family
   (`custom_kernels/qwen35_prefill/gdn.py`, `decode_fast/`) — the mlx #4020
   class of win, taken by vendoring (which this fork deliberately refuses).
   Where it loses to the fork: TTFT (worse at both scales), warm-prefix cache
   (2.3 s / 14.4 s vs our 0.06 / 0.15 s), and it ignored `max_tokens` on the
   streaming path during the A/B. Methodology lesson: **random-prompt serving
   benchmarks structurally favor speculative decoders** — never compare a
   speculative engine on them without a natural-prompt cross-check.
6. **MLX beats llama.cpp on this box at matched ~4-bit**: +33 % decode
   (20.0 vs 15.0 tok/s), prefill par (217 vs 224), warm-prefix par
   (`--cache-reuse` works). Ollama = llama.cpp minus 4–27 % (same engine,
   more wrapper), with the worst concurrency scaling of the field.
7. **Peak RSS is an approximation** — fork-batched-moe and oMLX-moe read
   ~9 GB against ~19 GB of weights (mmap-backed pages evicted under
   pressure-relief don't count against RSS the way wired allocations do).
   Use the exporter's memory gauges for real capacity planning, not these.

## Caveats

- Single box, warm page cache, client on the same machine, n=6/12 requests
  per pass — good for ranking, not for publication-grade percentiles.
- oMLX runs its own mlx-lm 0.31.3 (as shipped); all other MLX engines share
  the fork's exact kernel set.
- llama.cpp/Ollama chat-template handling differs from the MLX engines
  (`--jinja` vs Ollama's GGUF-embedded template); speed rows only.
- Output lengths are capped (`max_tokens` 256) but not forced (`ignore_eos`
  off); at T=0 on seeded prompts all engines generated to the cap.

## Rerunning

The Studio bench dir (`~/bench-2026-09-04/`: venvs, patched vLLM client,
server logs, Ollama model store) was **cleaned up 2026-09-06** (~21 GB).
The full harness lives in this repo under `scripts/fork/engine_bench/`
(orchestrator, A/B probes, GDN/KV-quant/burst trial scripts); rebuilding
the venvs takes ~15 min following the Setup table above. The vLLM client
needs the two compat patches described under Measurement. Before a rerun:
`mkdir -p ~/bench-2026-09-04 && touch ~/bench-2026-09-04/BENCH_ACTIVE`
(pauses the HA auto-reload daemon — its bench-guard reads that flag), and
remove the flag afterwards. The declined decode-burst prototype survives as
branch `exp/decode-burst` on `/Users/ai/vllm-mlx-src`.
