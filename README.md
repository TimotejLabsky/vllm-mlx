# vllm-mlx (TimotejLabsky fork)

> ⚠️ **This is a personal fork** of [`waybarrios/vllm-mlx`](https://github.com/waybarrios/vllm-mlx),
> maintained for a single home-lab Mac Studio that serves a small model fleet
> behind [llama-swap](https://github.com/mostlygeek/llama-swap).
>
> For general use you almost certainly want
> [upstream](https://github.com/waybarrios/vllm-mlx), not this fork.

## Fork vs upstream, measured (2026-11-19)

Same box (M1 Ultra, 64 GB), same mlx / mlx-lm / mlx-vlm, same models, prompts
and sampling. **Fork** = `main` `b98dc212` with each llama-swap route's
production command and env; **upstream** = stock
[`3c228769`](https://github.com/waybarrios/vllm-mlx/commit/3c228769) (the commit
this fork is rebased on) with the same command, minus the flags it does not
have. One server at a time on a spare port, fresh SSD cache dir, streamed,
client-side timing. Harness:
[`scripts/fork/bench_fork_vs_upstream.py`](scripts/fork/bench_fork_vs_upstream.py);
per-run numbers and method:
[`docs/fork/fork-vs-upstream-2026-11-19.md`](docs/fork/fork-vs-upstream-2026-11-19.md).

| route (model) | measurement | upstream | **fork** | change |
|---|---|---|---|---|
| **Qwen3.8-27B-8bit** | decode, short prompt | 15.1 tok/s | **24.4 tok/s** | **+61 %** |
| | decode at ~12K context | 14.7 tok/s | **22.8 tok/s** | **+55 %** |
| | first turn, 4.7K-token system + tools (cold) | 21.4 s | 21.9 s | same |
| | tool-loop follow-up turn (5.9-10.8K), TTFT, median of 5 | 5.6 s | 6.1 s | same per token³ |
| | 6-turn tool session, wall | 108.1 s | **99.3 s** | **−8 %** |
| | new user turn after the tool loop (12.1K), TTFT | 53.5 s | **35.1 s** | **−34 %** |
| | next tool turn after a server restart, TTFT | 37.9 s | **19.7 s** | **−48 %**² |
| **Qwen3.6-35B-A3B-4bit** | decode, short prompt | 54.7 tok/s | **87.8 tok/s** | **+60 %** |
| | 4-turn tool session, wall | 18.2 s | **12.8 s** | **−30 %** |
| | follow-up turn, TTFT | 0.94 s | 0.97 s | same |
| **Qwen3-VL-30B-A3B-8bit** | decode, image + text | 33.7 tok/s | **57.7 tok/s** | **+71 %** |
| | follow-up turn about the same image, TTFT | 0.60 s | **0.35 s** | **−42 %** |

² The restarted upstream server restored nothing and prefilled all 8.2K
tokens. The fork restored the session's opening prompt (4,721 tokens) from SSD and
prefilled the rest; a longer session also re-spills every 16K tokens of growth
(#164), so it would restore more.

³ The fork's turns had more to prefill (1,160-1,276 tokens against
upstream's 1,148-1,159): with sampling, its model answered in text on three
turns, and the harness sent that text back with the next call. At equal size
the gap is 1 % (1,160 tokens: 5.68 s against 5.62 s); the server log puts the
fork's TTFT within 0.1 s of its prefill time.

Where the difference comes from, and where there is none:

- **Decode.** Upstream has no `--text-only`, so it serves these
  multimodal-format Qwen checkpoints through the mlx-vlm path; the fork serves
  text through mlx-lm's `BatchedEngine` with its hybrid-safe cache. On top of
  that the 27B-8bit decodes with an external MTP drafter head (#160/#161,
  speculative sampling #167), and the Qwen3-VL route no longer makes a host↔GPU
  sync in every attention layer of every token (#168: 48 syncs per token in
  mlx-vlm's `qwen3_vl_moe`).
- **Prefill is the same.** Cold first turns match within noise: the fork does
  not make the GPU faster at prefill.
- **Shallow follow-up turns are not faster.** Upstream's prefix cache now also
  hits a simple tool-loop continuation on these hybrid models, and both
  engines then prefill the new tokens at the same ~205 tok/s on the 27B³. The fork's cache pulls ahead
  where the prompt does not simply extend the previous one: measured above, a
  new user turn after a tool loop and a route reload; not measured in this
  run (each has its own real-server check in [`PATCHES.md`](PATCHES.md)), a
  template that re-renders the model's own output differently (#139, #166),
  conversations interleaved with other clients (#163, #165), and deep
  sessions near the memory ceiling, where it sheds cache instead of crashing
  (#48, #103-#108, #150).

## What this fork is

A **patch stack**, not a feature branch: 156 numbered `patch:` commits (plus
their tests and docs) on top of upstream
[`3c228769`](https://github.com/waybarrios/vllm-mlx/commit/3c228769), each
documented in [`PATCHES.md`](PATCHES.md) with its rationale and measurements.
`main` is periodically rebased onto `waybarrios/main`; generally useful fixes
go upstream, and several have been merged there and retired from the stack.

It is tuned for one box: one GPU memory pool, ~20 models swapped in and out by
llama-swap, and long multi-turn agent sessions hitting the same
prompt prefixes over and over. That shapes every patch: **decode fast, cache
the prefix aggressively, never OOM the GPU, fail loudly instead of hanging.**

## What's included

**Faster decode**
- **MTP drafter on the cached text path** (#160, #161, #167). The
  Qwen3.8-27B-8bit decodes with an external MTP head that proposes the next
  token; the target verifies it in the same forward pass, and a partial accept
  restores the recurrent state recorded during that pass instead of re-running
  it. At the route's T=1.0 sampling the draft is *sampled* from the head and
  accepted with min(1, p/q), so output is distributed exactly as plain
  sampling; at T=0 output and stored cache entries are byte-identical to plain
  decoding. +20-37 % decode in controlled tests, +8-20 % on live agent traffic.
- **Vision decode without per-layer host syncs** (#168). Qwen3-VL-30B decodes
  34 → 58 tok/s, T=0 output identical.
- **Text-only serving for multimodal-format checkpoints** (`--text-only`,
  BatchedEngine for the whole text fleet since 2026-07-09): the Qwen3.5-3.8
  checkpoints decode on mlx-lm's text path instead of mlx-vlm's (most of the
  35B-A3B's +60 % above).

**A prefix cache built for agents on hybrid models** (`system_kv.py`,
`batched_system_kv.py`)
- Works on hybrid attention + recurrent models (Qwen3.5/3.6/3.8, Qwen3-Next):
  checkpoint/snapshot entries instead of hashable KV blocks, multi-slot LRU,
  grow-on-hit so a warm chain extends instead of re-prefilling, partial
  restore for divergent branches, and segmented stores (O(new tokens) per turn).
- **SSD tier** so caches survive restarts and model swaps (restart 31.7 s →
  1.9 s in the original measurement), now also for long sessions that grew
  turn by turn (#164), with small one-off prompts kept off it so they cannot
  evict agent sessions (#169).
- **Repairs prompts that a client re-renders differently from what the model
  wrote**: whitespace (#139), tool-call arguments in another order (#166), and
  the record of what the model generated survives its cache entry being
  evicted (#163).
- **Agent turns go first**: after a long turn, the same conversation's next
  turn is admitted before background jobs that queued meanwhile (#165).
- Image follow-ups are served from the cache, keyed on the image's hash
  (#142: GLM-4.6V 1.85 → 0.19 s TTFT).

**Memory safety on a 64 GB box**
- Peak-memory relief that sheds cache before the GPU ceiling, admission that
  projects a co-batch's real memory peak, lazy restores, and a solo-prefill
  guard that prices the drafter too (#48, #103-#108, #150, #162). Deep
  concurrent agent chains that used to OOM now finish (5/5 deep follow-ups
  served from cache, 123 s instead of 1,150 s).
- Long generations on hybrid models no longer die at ~10.5K tokens (#125), and
  prompts past a route's measured envelope get an honest
  `400 prompt_too_long` instead of a crash.

**Correct agent behaviour**
- Per-request `reasoning_effort` mapped onto each model's own template
  vocabulary, a thinking budget that binds, reasoning-parser fixes, tool-parser
  fixes (gpt-oss harmony, qwen3_coder, GLM), structured output that fails
  closed (llguidance), and a stop for degenerate repetition loops.
- Engine-side aborts surface as retryable 503s instead of HTTP 200 with an
  empty turn.

**Operations**
- A route's chat template can come from a file in the deploy config
  (`VLLM_MLX_CHAT_TEMPLATE_FILE`, #169), so a model re-download cannot swap it.
- `/v1/status` and Prometheus gauges for cache hits, evictions, memory
  pressure, admission and drafter acceptance; a `GET /` for llama-swap probes.

Everything new is off by default behind an env flag or CLI option, and the
llama-swap config enables it per route.

## How it compares to other engines (measured 2026-09-04)

Cross-engine serving benchmark on the production box (M1 Ultra 64 GB), vLLM
`benchmark_serving` (1024-in/256-out, T=0) plus load/warm-cache/prefill probes.
Full method, tables and caveats:
[`docs/fork/engine-benchmarks-2026-09.md`](docs/fork/engine-benchmarks-2026-09.md).

**MoE — Qwen3.6-35B-A3B-4bit (production route config per engine):**

| engine | decode tok/s (serial) | 4-way agg tok/s | TTFT ms | warm 7.2K-prefix TTFT |
|---|---|---|---|---|
| **this fork, batched** | **65.6** | **131.5** | **659** | **0.06 s** |
| upstream `f2d3ad3`, batched | 57.7 | 120.5 | 773 | 4.7 s (= cold) |
| raw `mlx_lm.server` | 50.4 | 101.4 | 825 | 0.19 s |
| oMLX 0.6.4 | 132.1¹ | 204.9¹ | 995 | 2.3 s |

**Dense — Qwen3.8-27B-8bit:**

| engine | decode tok/s (serial) | 4-way agg tok/s | TTFT ms | warm 7.2K-prefix TTFT |
|---|---|---|---|---|
| **this fork, batched** | **14.5** | **25.2** | **4235** | **0.15 s** |
| upstream `f2d3ad3`, batched | 14.0 | 21.6 | 4959 | 32.6 s (= cold) |
| raw `mlx_lm.server` | 13.5 | 20.7 | 4354 | 0.28 s |
| oMLX 0.6.4 | 26.1¹ | 28.6¹ | 5117 | 14.4 s |

At matched ~4-bit quant on the dense model, the fork decodes **+33 % over
llama.cpp** (20.0 vs 15.0 tok/s) with prefill at par, and Ollama trails
llama.cpp on every metric.

¹ oMLX's speculative burst decode dominates the random-token benchmark
workload; on a natural prompt at T=0 it is **+13 %** over fork-batched
(88.9 vs 78.7 tok/s), with worse TTFT and a far weaker warm-prefix cache. It
gets its decode edge from vendored gated-delta Metal kernels (the mlx #4020
class of win this fork waits to take via an mlx release rather than vendor).

## Recent changes

> **2026-11-19 — agent sessions survive route reloads; the 27B's template is
> version-controlled** (PATCHES.md #169): the 27B-8bit's SSD tier had filled
> with a background job's one-off 2K-token prompts (~600 MB each) and
> evicted every agent session, so scheduled jobs started cold after each
> reload (41-69 s TTFT). Chains under 4K tokens now stay RAM-only and the tier
> grew to 40 GB. The route's custom chat template loads from the deploy config
> (`VLLM_MLX_CHAT_TEMPLATE_FILE`) after a model re-download had silently
> replaced it; the copy moves the reasoning-effort line after the tool block
> so low/medium/xhigh share the cached tools prefix.
>
> **2026-11-18 — Qwen3-VL decode +70 %, speculative sampling, tool-order
> repair** (PATCHES.md #166-#168): mlx-vlm's `qwen3_vl_moe` attention made 48
> host syncs per decoded token; removed (34.2 → 58.2 tok/s, T=0 identical).
> The 27B's drafter now speculates by sampling at T > 0 (acceptance
> 0.78 → 0.82, +3.7 % decode over exact matching, +2-12 % on live traffic).
> A tool call sent back with its arguments reordered restores from cache.
>
> **2026-11-16/17 — MTP drafter on the cached text path, long-session fixes**
> (PATCHES.md #159-#165): the 27B-8bit decodes with an external MTP head
> (+20-37 % controlled, T=0 byte-identical); generated-span records survive
> eviction (#163), grown sessions spill to SSD (#164), and an agent's next
> turn goes before background work queued during its last one (#165). The
> 27B-8bit also got the memory guards it lacked (it had OOMed 3x at 109K).
>
> **2026-10-14 — image follow-ups served from the prefix cache** (PATCHES.md
> #142, vision phase B): image requests are keyed on the image's SHA-256
> instead of its placeholder tokens, so a follow-up turn or re-ask about the
> same image restores its KV and skips the vision tower (GLM-4.6V 1.85 s →
> 0.19 s, Devstral 4.0 s → 0.27 s TTFT), while a different image behind an
> identical prompt still misses. On for glm4v / gemma4 / mistral3 / qwen3_5 /
> qwen3_vl_moe (Qwen3-VL-30B 0.88 s → 0.20 s).
>
> **2026-10-05 — long generations stop crashing; MoE decode +6–21 %**
> (PATCHES.md #125–#127): hybrid models no longer die with `[metal::malloc]
> Resource limit (499000)` after ~10.5K generated tokens — only the recurrent
> cache states are evaluated each step, so it costs no decode speed (a 12K
> completion on the 27B now finishes instead of returning 503). A short shared
> system prompt (under 2,048 tokens) is now reused across conversations on
> hybrid models, and a queued request's SSD hit no longer builds its restore
> while it waits. Config-only, from the same round: Metal command-buffer
> limits on the MoE routes (`MLX_MAX_OPS_PER_BUFFER=1000`,
> `MLX_MAX_MB_PER_BUFFER=200`: +6–21 % decode, T=0 byte-identical) and
> `iogpu.disable_wired_collector=1` (idle GPU memory stays wired: ~0.45 s off
> every turn after a pause). Numbers: `docs/fork/speed-lever-ledger-2026-09.md`.
>
> **2026-09-21 — deep concurrent agent chains, stress-tested on the live 27B**
> (PATCHES.md #106–#108): admission now projects the real memory peak before
> co-batching (and makes what doesn't fit *wait*, never rejects); a queued
> request's cache restore is built when it is admitted instead of pinning
> 4–5 GB while it waits; its entry is pinned until then, comes back from the
> SSD tier if RAM lost it, and an entry whose spill is still queued is no
> longer evicted (that frees nothing). The pattern that OOMed three times on
> 09-17 at 63 GB now peaks at 51 GB with **5 of 5 deep follow-ups served from
> cache — 123 s instead of 1,150 s**. Reusable harnesses:
> `scripts/fork/e2e_lazy_restore.py`, `scripts/fork/e2e_recovery_signalling.py`.
>
> **2026-09-17 — memory the cache doesn't count, and honest failures**
> (PATCHES.md #103–#105): a never-idle SSD writer pinned ~12 GB outside every
> guard and OOM recovery leaked checkpoint ladders — the 27B route sat at its
> watermark with an *empty* cache for a day. Fixed, and made visible
> (`pending_ladders`, `ssd.queued_bytes`). An engine-side abort is now a
> retryable **503 / 503-shaped stream frame** instead of HTTP 200 with an
> empty turn. Solo prefills get an admission check too. CI `lint` now checks
> only the lines a change adds and is expected green
> (`scripts/fork/black_changed_lines.py`); `tests/test_fork_invariants.py`
> pins the conditions a rebase can silently undo.
>
> **2026-09-16 — hybrid chains cache again under concurrency** (PATCHES.md
> #100–#102): an upstream gate had silently disabled the batched hybrid
> cache's only concurrent store path since the v0.4.1 rebase; plus a dynamic
> cache budget and an admission-pricing floor that survives engine recovery.
>
> **2026-09-06 (2) — the MLLM path gets the repetition rail** (PATCHES.md #98):
> REPDETECT was silently ignored on the multimodal scheduler — found when the
> new 180B route looped forever on an unknown entity at T=0. Consumer-side
> hook mirrors the text scheduler's; verified live cutting the loop at 800
> tokens. Also corrected: the thinking budget was never inert on this path.
>
> **2026-09-06 — Qwen3.8-Flash-Next (qwen4_exp) vendored** (PATCHES.md #97):
> first 180B-class checkpoint served from this box — REAP-288 at ~39 GB
> resident via the SSD-backed n-gram PLE table. Live smoke green through the
> batched engine, and tool calls work via the fork's `qwen3_coder` parser —
> which oMLX's own serving of this model cannot do.
>
> **2026-09-04 — cross-engine benchmark round** (see table above): fork-batched
> leads every MLX serving configuration measured on this box; upstream's
> prefix cache re-confirmed zero-hit on hybrid models on current main; oMLX
> identified as the engine to watch (vendored GDN kernels, +13 % real decode).
>
> **2026-08-27 — the mlx-lm ceiling now has a real tripwire** (PATCHES.md #78
> correction, #80). The prefix cache identifies recurrent layers by the *shape*
> of their state; a later mlx-lm changes that shape, at which point every hybrid
> model silently loses partial restore. The guard was documented but had never
> been committed — it now exists, matches over the class MRO, and fires once
> per process.
>
> **2026-08-23 — the thinking phase machine no longer walks the prompt**
> (PATCHES.md #79). `ThinkingAwareLogitsProcessor` assumed it only saw generated
> tokens, but mlx-lm hands logits processors the full sequence. Any prompt
> replaying an earlier `<think>…</think>` span — i.e. every multi-turn agent
> conversation — drove it to CONTENT before the first generated token, where it
> masks `</think>` to `-inf`, leaving the model unable to close its own think
> block. Reasoning then landed in `content`, the budget never engaged, and turns
> ran to `max_tokens`.
>
> **2026-08-22 — mlx-lm pinned to a git commit** (PATCHES.md #78) for two
> qwen3_coder tool-parser fixes and a GLM tool-name fix. The pin is also a
> **ceiling**: a later mlx-lm commit changes `ArraysCache.state` to a 3-tuple,
> which the system-KV stack would classify as opaque and silently stop caching
> hybrid models. A one-shot `logger.error` makes crossing it loud.
>
> **2026-08-22 — repetition-detection stop** (PATCHES.md #77): degenerate exact
> repetition cycles are ended at the scheduler's stop-check rather than burning
> to `max_tokens`. Default-off, `VLLM_MLX_REPDETECT=1` to arm.
>
> **2026-08-19 — per-request reasoning effort** (PATCHES.md #76): the OpenAI
> `reasoning_effort` parameter (and Responses `reasoning.effort`) reaches the
> chat template instead of being dropped for every value except `none`. Values
> resolve **exact vocabulary match → route floor → nearest neighbour → drop**,
> so an unsupported level degrades to the operator's configured floor rather
> than to the template default — and never reaches `raise_exception` as a 500.
>
> **2026-08-18 — fail-closed structured output** (PATCHES.md #73, upstream
> PR #636 adapted): strict `json_schema` decodes under a request-local
> llguidance token mask (schema-aware EOS, fail-closed on setup errors),
> `json_object` requires an object root, and streaming structured requests
> validate server-side before HTTP 200.
>
> **2026-07-28 — hardened batched vision serving** (PATCHES.md #54–#67):
> image-safe prefix caching, per-row MRoPE correctness for glm4v/qwen3_vl
> families (real-model byte-compare gates), memory-pressure relief with
> vision-encode bracketing, queue cap / prompt ceiling / media limits with
> honest 400s, and stats/Prometheus parity.

## Fork documentation

- [`PATCHES.md`](PATCHES.md) — **single source of truth**: every patch with
  rationale, measurements, rebase history, and upstreaming status.
- [`docs/fork/`](docs/fork/) — design docs and investigations, e.g.
  [`continuous-batching-hybrid-caching.md`](docs/fork/continuous-batching-hybrid-caching.md)
  (why the stock batched prefix cache gets zero hits on hybrid models, and how
  the fork's own batched cache fixed it).
- [`NOTICE`](NOTICE) — Apache License 2.0 attribution.

Everything below this line is upstream's README.

---

**Continuous batching + OpenAI + Anthropic APIs in one server. Native Apple Silicon inference.**

**Read this in other languages:** [English](README.md) · [Español](README.es.md) · [Français](README.fr.md) · [中文](README.zh.md)

[![PyPI version](https://img.shields.io/pypi/v/vllm-mlx.svg)](https://pypi.org/project/vllm-mlx/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-mlx)](https://pepy.tech/projects/vllm-mlx)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)
[![GitHub stars](https://img.shields.io/github/stars/waybarrios/vllm-mlx.svg?style=social)](https://github.com/waybarrios/vllm-mlx)

---

## What is vllm-mlx?

A vLLM-style inference server for Apple Silicon Macs. Unlike `Ollama` or `mlx-lm` used directly, it ships **continuous batching, paged KV cache, prefix caching, and SSD-tiered cache**, and exposes **both OpenAI `/v1/*` and Anthropic `/v1/messages`** from a single process. Run LLMs, vision models, audio, and embeddings on Metal with unified memory, no conversion step.

## Quick start

New users: follow the [isolated release install](docs/getting-started/installation.md#install-a-release)
and [first-response check](docs/getting-started/quickstart.md#first-response).
The example below assumes that environment is activated. Initial model
download/loading time depends on the machine and connection.

The version below is this walkthrough's pinned reference release, not an
automatically updated latest version.

```bash
python -m pip install 'vllm-mlx==0.4.1'
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --host 127.0.0.1 --port 8000
```

After the first response, restart with `--continuous-batching` to try
[continuous batching and its cache options](docs/guides/continuous-batching.md).
The minimal command above uses the default engine without that flag.

**OpenAI SDK:**

Install `openai` in your Python client environment first; it is not installed
by the server package. See the [client setup](docs/getting-started/quickstart.md#option-1-openai-compatible-server).

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(model="mlx-community/Llama-3.2-3B-Instruct-4bit", messages=[{"role": "user", "content": "Hi!"}])
print(r.choices[0].message.content)
```

**Anthropic SDK / Claude Code:**

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

## Coding CLI compatibility

**Validated with OpenCode, pi, Codex, Claude Code, GitHub Copilot CLI, Cline CLI,
and OpenClaw's embedded agent.** All seven completed a streamed tool interaction
and an exact file edit in one local run with Qwen3.8-27B-4bit on September 19,
2026. Results apply to the tested client versions and settings.

See the [validated CLI matrix and setup guide](docs/guides/client-acceptance.md#validated-cli-versions)
for versions, API transports, reproduction commands, and coverage limits.

## Features

### APIs
- **OpenAI-compatible**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/responses`
- **Anthropic-compatible**: `/v1/messages` (streaming, tool use, system prompts)
- **MCP Tool Calling**: 19 parsers (OpenAI, Anthropic, Gemini, Qwen, DeepSeek, Gemma, and more)
- **Structured output**: JSON Schema via `response_format` (lm-format-enforcer)

### Throughput & memory
- **Continuous batching**: high throughput for concurrent requests
- **Paged KV cache**: memory-efficient with prefix sharing
- **SSD-tiered KV cache**: spill prefix cache to disk for long-context agents (`--ssd-cache-dir`)
- **Warm prompts**: preload popular prefixes at startup (`--warm-prompts`) for 1.3-2.25x TTFT
- **Prefix cache**: trie-based, shared across requests

### Multimodal
- **Text + image + video + audio** from one server
- Vision models: Gemma 3, Gemma 4, Qwen3-VL, Pixtral, Llama vision
- **Audio input** in chat (`audio_url` content blocks)
- **Native TTS**: 11 voices, 15+ languages (Kokoro, Chatterbox, VibeVoice, VoxCPM)
- **STT**: Whisper family with RTF up to 197x on M4 Max

### Reasoning & advanced
- **Reasoning extraction**: Qwen3 and DeepSeek-R1 parsers (`--reasoning-parser`)
- **MoE expert reduction**: `--moe-top-k` for +7-16% on Qwen3-30B-A3B
- **Speculative decoding**: `--enable-mtp` for supported models
- **Sparse prefill**: attention-based `--specprefill` for supported model/draft combinations

### Observability
- **Prometheus metrics**: `/metrics` endpoint with `--enable-metrics`
- **Built-in benchmarker**: `vllm-mlx bench-serve` for prompt sweeps with CSV/JSON output

### Native GPU acceleration
- Apple Silicon only (M1, M2, M3, M4, M5) with Metal kernels via MLX
- Unified memory, no model conversion

## Performance

**LLM decode (M4 Max, 128 GB, greedy, single stream):**

| Model | Tok/s | Memory |
|-------|------:|-------:|
| Qwen3-0.6B-8bit | 417.9 | 0.7 GB |
| Llama-3.2-3B-Instruct-4bit | 205.6 | 1.8 GB |
| Qwen3-30B-A3B-4bit | 127.7 | ~18 GB |

**Audio speech-to-text (M4 Max, RTF = real-time factor):**

| Model | RTF | Use case |
|-------|----:|----------|
| whisper-tiny | 197x | Real-time / low latency |
| whisper-large-v3-turbo | 55x | Quality + speed |
| whisper-large-v3 | 24x | Highest accuracy |

See [docs/benchmarks/](docs/benchmarks/) for continuous-batching results, KV-cache quantization (4-bit / 8-bit / fp16), and MoE top-k sweeps.

## Examples

### Anthropic API (Claude Code, OpenCode)

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --port 8000
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

### Reasoning models (Qwen3, DeepSeek-R1)

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

```python
r = client.chat.completions.create(
    model="mlx-community/Qwen3-8B-4bit",
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
)
print("Thinking:", r.choices[0].message.reasoning)
print("Answer:",   r.choices[0].message.content)
```

### Multimodal (image + text)

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

```python
r = client.chat.completions.create(
    model="mlx-community/Qwen3-VL-4B-Instruct-3bit",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
    ]}],
)
```

### Structured output (JSON Schema)

With the text server from the quick start:

```python
r = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "List 3 colors."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "schema": {"type": "object", "properties": {"colors": {"type": "array", "items": {"type": "string"}}}}
        },
    },
)
```

### Reranking (`/v1/rerank`)

```bash
curl http://localhost:8000/v1/rerank -H 'Content-Type: application/json' -d '{
  "model": "default",
  "query": "apple silicon inference",
  "documents": ["MLX is Apples framework", "Metal kernels on M-series", "CUDA on NVIDIA"]
}'
```

The built-in MLX reranker forward path supports standard BERT/XLM-RoBERTa
sequence-classification weights with `gelu`, `gelu_new`/`gelu_fast`, `relu`, or
`silu`/`swish` `hidden_act` values. Other activations fail explicitly so custom
reranker architectures can add a dedicated adapter instead of silently using the
wrong activation.

### Embeddings

```bash
vllm-mlx serve <llm-model> --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

```python
emb = client.embeddings.create(model="mlx-community/all-MiniLM-L6-v2-4bit", input=["Hello", "World"])
```

### Audio (TTS / STT)

```bash
pip install vllm-mlx[audio]
brew install espeak-ng        # macOS, needed for non-English TTS

python examples/tts_example.py "Hello, how are you?" --play
python examples/tts_multilingual.py "Hola mundo" --lang es --play
```

### Built-in benchmarking

```bash
vllm-mlx bench-serve --url http://localhost:8000 --concurrency 5 --prompts prompts.txt --output results.csv

# Product-style workload with quality checks and metrics deltas
vllm-mlx bench-serve --url http://localhost:8000 --workload workload.json --repetitions 5 --output results.json

# Append workload rows into SQLite for longitudinal comparisons
vllm-mlx bench-serve --url http://localhost:8000 --workload workload.json --repetitions 5 --format sqlite --output bench.db
```

### Model acquisition and conversion

```bash
# Inspect repo metadata, file sizes, config, and rough fit before downloading weights
vllm-mlx model inspect mlx-community/Llama-3.2-3B-Instruct-4bit

# Acquire with resumable Hugging Face transfer and write a local artifact manifest
vllm-mlx model acquire mlx-community/Llama-3.2-3B-Instruct-4bit --target-dir ./models/llama-3b-4bit

# Wrap mlx-lm conversion and record the exact recipe in the converted artifact
vllm-mlx model convert meta-llama/Llama-3.2-3B-Instruct --output ./models/llama-3b-mlx-q4 --quantize --q-bits 4 --q-group-size 64 --q-mode affine
```

### Prometheus metrics

```bash
vllm-mlx serve <model> --enable-metrics
curl http://localhost:8000/metrics
```

## Installation

Use the [release walkthrough](docs/getting-started/installation.md#install-a-release)
for an isolated environment and a pinned server version. If you already manage
CLI tools with uv, its isolated equivalent is:

```bash
uv tool install 'vllm-mlx==0.4.1'
```

Keep [development checkouts](docs/getting-started/installation.md#development-checkout)
separate. See the [Installation Guide](docs/getting-started/installation.md) for
optional extras and [Audio Guide](docs/guides/audio.md) for audio setup.

## Documentation

Browse the complete documentation at [vllm-mlx.is-a.dev](https://vllm-mlx.is-a.dev/).

- **Getting started**: [Installation](docs/getting-started/installation.md) · [Quick Start](docs/getting-started/quickstart.md)
- **Servers & APIs**: [OpenAI server](docs/guides/server.md) · [Anthropic Messages API](docs/guides/server.md#anthropic-messages-api) · [Python API](docs/guides/python-api.md)
- **Client compatibility**: [Run coding-client acceptance checks](docs/guides/client-acceptance.md)
- **Features**: [Multimodal](docs/guides/multimodal.md) · [Audio](docs/guides/audio.md) · [Embeddings](docs/guides/embeddings.md) · [Reasoning](docs/guides/reasoning.md) · [MCP & Tool Calling](docs/guides/mcp-tools.md) · [Tool Parsers](docs/guides/tool-calling.md)
- **Performance**: [Continuous Batching](docs/guides/continuous-batching.md) · [Multi-Model Serving](docs/guides/model-registry.md) · [Warm Prompts](docs/guides/warm-prompts.md) · [MoE Top-K](docs/guides/moe-top-k.md)
- **Reference**: [CLI](docs/reference/cli.md) · [Models](docs/reference/models.md) · [Configuration](docs/reference/configuration.md)
- **Benchmarks**: [LLM](docs/benchmarks/llm.md) · [Image](docs/benchmarks/image.md) · [Video](docs/benchmarks/video.md) · [Audio](docs/benchmarks/audio.md)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           vllm-mlx Server                               │
│   OpenAI /v1/*  ·  Anthropic /v1/messages  ·  /v1/rerank  ·  /metrics   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Continuous batching · Paged KV cache · Prefix cache · SSD tiering      │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
        ┌─────────────┬────────────┴────────────┬─────────────┐
        ▼             ▼                         ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│    mlx-lm     │ │   mlx-vlm     │ │   mlx-audio   │ │mlx-embeddings │
│    (LLMs)     │ │  (Vision)     │ │  (TTS + STT)  │ │ (Embeddings)  │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MLX · Metal kernels · Unified memory                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Contributing

Bug fixes, perf work, docs, and benchmarks on different Apple Silicon chips all welcome. See the [Contributing Guide](docs/development/contributing.md).

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{vllm_mlx2025,
  author = {Barrios, Wayner},
  title  = {vllm-mlx: Apple Silicon MLX Backend for vLLM},
  year   = {2025},
  url    = {https://github.com/waybarrios/vllm-mlx},
  note   = {Native GPU-accelerated LLM and vision-language model inference on Apple Silicon}
}
```

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx). Apple's ML framework.
- [mlx-lm](https://github.com/ml-explore/mlx-lm). LLM inference library.
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Vision-language models.
- [mlx-audio](https://github.com/Blaizzy/mlx-audio). Text-to-Speech and Speech-to-Text.
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings). Text embeddings.
- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX). Community fork of vllm-mlx.
- [vLLM](https://github.com/vllm-project/vllm). High-throughput LLM serving. vllm-mlx is inspired by vLLM and adopts its continuous-batching and paged KV-cache design for Apple Silicon via MLX.

## Star history

[![Star History Chart](https://star-history.dera.page/svg?repos=waybarrios/vllm-mlx&type=Date)](https://star-history.dera.page/#waybarrios/vllm-mlx&Date)

---

**If vllm-mlx helped you, please star the repo. It helps more Apple Silicon devs find it.**
