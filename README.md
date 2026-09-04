# vllm-mlx (TimotejLabsky fork)

> ⚠️ **This is a personal fork** of [`waybarrios/vllm-mlx`](https://github.com/waybarrios/vllm-mlx),
> maintained for a single home-lab Mac Studio that serves a small model fleet
> behind [llama-swap](https://github.com/mostlygeek/llama-swap).
>
> For general use you almost certainly want
> [upstream](https://github.com/waybarrios/vllm-mlx), not this fork.

## How it compares (measured 2026-09-04)

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

## What this fork is

A **patch stack**, not a feature branch. `main` carries ~137 local patches on top
of upstream [`22efb47`](https://github.com/waybarrios/vllm-mlx/commit/22efb47);
each is a separate commit prefixed `patch:`, and the branch is periodically
rebased onto `waybarrios/main` to pick up upstream changes. Fixes that are
generally useful get cherry-picked from upstream PRs or prepared as upstreaming
branches; several have since merged upstream and been retired from the stack.
The rest are home-lab-specific and stay here.

The deployment it is tuned for is one box, one GPU memory pool, many models
swapped in and out on demand, and long multi-turn agent sessions (Claude Code,
coding agents, chat bots) hitting the same prompt prefixes over and over. That
shapes every patch: **cache the prefix aggressively, never OOM the GPU, fail
loudly instead of hanging.**

## What the patches actually add

**Hybrid-safe prefix caching ("system-KV").** Upstream's prefix cache gets zero
hits on hybrid attention+SSM models (Qwen3-Next, Qwen3.5/3.6/3.8 — most of the
lineup here) because their `ArraysCache` state can't be hashed or sliced the way
an attention KV cache can. The fork replaces it with a checkpoint/snapshot cache
that works on both: multi-slot LRU across concurrent conversations, grow-on-HIT
so a warm chain extends instead of re-prefilling, partial restore for divergent
branches, a RAM budget on the resident slot set, and an SSD tier so caches
survive restarts and model swaps. Lives in fork-owned modules —
`vllm_mlx/system_kv.py`, `system_kv_ssd.py`, `batched_system_kv.py`.

**BatchedEngine made usable for this fleet.** The batched (continuous-batching)
path got the same hybrid-safe cache plus the pieces it was missing: `--text-only`
support, stop-string enforcement, per-request sampling parameters, the DRY
sampler, segmented snapshots (O(delta) stores instead of an O(context) copy per
turn), an SSD cold tier, and dynamic concurrency where batch seats float on a
measured KV-byte budget instead of a fixed count. **Since 2026-07-09 the entire
text fleet runs BatchedEngine**; vision routes stay on the mlx-vlm path and
embeddings on their own route.

**Memory-pressure survival.** Deep-context prefill on a 27B model would ramp GPU
memory into the hard ceiling and crash mid-request. The fork watches peak memory
per step, sheds cache slots and clears the MLX buffer cache before the ceiling
rather than after, and rejects prompts past a per-route measured token envelope
with a non-retryable `400 prompt_too_long` instead of dying.

**Correct reasoning and tool-call behaviour for agent clients.** Per-request
`reasoning_effort` normalized against each model's own chat-template vocabulary;
a thinking-token budget that actually binds; reasoning-parser correctness when
thinking is disabled; a detector that ends degenerate repetition loops at the
scheduler; and several tool-parser fixes. Agent replay shapes (an assistant turn
carrying `tool_calls` but no reasoning) are a first-class test case here, because
that is what broke in practice.

**Reliability and operational fixes.** Admission control on the serialized route
(503 rather than a silent lock wait), lazy MLX array realization on the load
thread (the cross-thread stream crash that took out gpt-oss and Gemma text
routes), a `GET /` route for llama-swap's preload probe, and extra `/v1/status`
+ Prometheus gauges for cache hit rate, memory pressure, eviction timing, and
admission.

## Recent changes

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

## Quick start (30 seconds)

```bash
pip install vllm-mlx
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

**OpenAI SDK:**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(model="default", messages=[{"role": "user", "content": "Hi!"}])
print(r.choices[0].message.content)
```

**Anthropic SDK / Claude Code:**

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

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
- **Reasoning extraction**: Qwen3, DeepSeek-R1, DeepSeek-V4 (`--reasoning-parser`)
- **MoE expert reduction**: `--moe-top-k` for +7-16% on Qwen3-30B-A3B
- **Speculative decoding**: `--mtp` for Qwen3-Next
- **Sparse prefill**: attention-based `--spec-prefill` for TTFT reduction

### Observability
- **Prometheus metrics**: `/metrics` endpoint with `--metrics`
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
    model="default",
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
    model="default",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
    ]}],
)
```

### Structured output (JSON Schema)

```python
r = client.chat.completions.create(
    model="default",
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
vllm-mlx serve <model> --metrics
curl http://localhost:8000/metrics
```

## Installation

**Using uv (recommended):**

```bash
uv tool install vllm-mlx                 # CLI, system-wide
# or in a project
uv pip install vllm-mlx
```

**Using pip:**

```bash
pip install vllm-mlx

# Audio extras
pip install vllm-mlx[audio]
brew install espeak-ng
python -m spacy download en_core_web_sm
```

**From source:**

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

See [Installation Guide](docs/getting-started/installation.md) for full options.

## Documentation

Browse the complete documentation at [vllm-mlx.is-a.dev](https://vllm-mlx.is-a.dev/).

- **Getting started**: [Installation](docs/getting-started/installation.md) · [Quick Start](docs/getting-started/quickstart.md)
- **Servers & APIs**: [OpenAI server](docs/guides/server.md) · [Anthropic Messages API](docs/guides/server.md#anthropic-messages-api) · [Python API](docs/guides/python-api.md)
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
