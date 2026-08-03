# Vision-route caching semantics (BatchedEngine MLLM branch)

*Status: 2026-07-30, fork `9cf1e90`. Both VLM routes (GLM-4.6V-Flash-4bit,
Qwen3-VL-30B-A3B-8bit) run BatchedEngine in production; `vlm-server.py` is
retired. This doc answers one question: **which caches apply to a vision
route, which are deliberately off, and why.** Patch-level detail lives in
PATCHES.md #54–#69.*

## The cache matrix

| Cache | What it saves | Media requests | Text requests on the route | Live evidence (Studio) |
|---|---|---|---|---|
| **Pixel cache** (`vision_embedding_cache.py`, level 1) | Image preprocessing: decode, resize, normalize, tokenize-with-placeholders | **ON** — keyed on sha256(image content) + prompt | n/a | `pixel_cache_hits=1` on both routes; same-image re-ask 0.4 s |
| **Prefix-KV cache** (`memory_cache.py` `MemoryAwarePrefixCache`, RAM + SSD tier, namespace `mllm-v2`) | The prefill itself (KV reuse) | **OFF by design** — neither store nor fetch (patch #56, phase A) | **ON** | GLM route, 2.2 K-token text prompt repeated: 5.4 s → **1.0 s, hit, `tokens_saved=2189`** |
| **Encoding cache** (`vision_embedding_cache.py`, level 2) | Full vision-tower forward output | Present in the module but not exercised by the batched serve path (`encoding_cache_hits/misses` stay 0) | n/a | all status snapshots show 0/0 |
| **Batched system-KV** (`batched_system_kv.py`, #29–#40) | Text-fleet KV bag + SSD | *Not used by the MLLM branch at all* | *ditto* | text fleet (19 routes) unchanged |

Rails (pressure relief incl. the atomic-vision-encode bracket, queue-cap 503,
prompt ceiling 400, media limits 400/413, admission seats, stats parity) are
all active on both branches — they're rails, not caches; see PATCHES.md
#60–#66.

## Why media-KV reuse is off (and must stay off until phase B)

The prefix cache is keyed on **token ids alone**. An image enters the prompt
as placeholder tokens that do not encode pixel content, so two requests with
identical prompts but different images produce **identical keys** — and would
serve each other's KV. This wasn't hypothetical: the pre-#56 store had no
media check and the fetch guard skipped exact matches (cross-image aliasing,
one of the three live bugs the vision series fixed). Patch #69 was the same
disease in a smaller cache: the pixel cache aliased the request's
`extra_kwargs` dict and every HIT replayed the model without its processor
kwargs (fatal on mistral3, silent elsewhere).

The rule of thumb this repo follows: **a cache keyed on less than what
determines the output is a correctness bug, not an optimization.**

Practical cost of phase A: a repeated *image* conversation re-prefills its KV.
The pixel cache still absorbs the preprocessing, and vision prompts are short;
no measured pain so far.

## Phase B — when to build it

Designed (plan P19 + PATCHES.md "Future work"): prepend a fingerprint of the
sha256 media hash (already computed by the pixel cache, currently discarded)
to the token key — prefix, because `MemoryAwarePrefixCache` LCP-matches — and
persist the row's rope-delta (#57) on the entry. Bump the SSD namespace on
landing.

**Trigger:** `memory_aware_cache` hits pinned at 0 on a vision route while the
same image repeats (multi-turn-over-one-image traffic). One-shot
different-image traffic gains nothing — don't build it for that.

## Arch coverage (2026-07-30 fleet sweep)

Every vision-capable arch family in the fleet passed the 14-check e2e smoke
on BatchedEngine (spare-port runs on the Studio):

| Arch | Tested with | Verdict |
|---|---|---|
| glm4v | GLM-4.6V-Flash-4bit | production since 07-29 |
| qwen3_vl_moe | Qwen3-VL-30B-A3B-8bit | production since 07-30 |
| qwen3_5 | Qwen3.5-4B | 14/14 |
| qwen3_5_moe | Ornith-1.0-35B | 14/14 |
| gemma4 | gemma-4-26B-A4B | 14/14 (overthinks tiny `max_tokens`; artifact) |
| mistral3 | Devstral-Small-2-24B | 14/14 after #68 (`mask` positional) + #69 |

Two conversions have **stripped vision towers** and can never serve vision
regardless of engine: `Qwen3.6-35B-A3B-4bit-DWQ`, `Mistral-Small-3.2-24B`
(0 vision weight keys in the safetensors index).

Mixed `--text-only` routes are *not* flipped: serving vision there today would
trade away the text stack (system-KV etc.). That's phase 2 `--vision-split`
(plan P15–P19), de-risked by this sweep.

## Known residuals

- MLLM preprocessing failures surface as HTTP 200 + `finish_reason:"error"` +
  `content:null` — should be a proper 4xx/5xx (noted in PATCHES.md #68).
- Pre-stream prompt-ceiling estimate for streams (both branches).
- Audio is not forwarded into `chat_kwargs` by the server; audio-bearing
  requests on text routes get an honest 400 (#65).
