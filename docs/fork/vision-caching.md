# Vision-route caching semantics (BatchedEngine MLLM branch)

*Status: 2026-10-14 — phase B (#142) built, see below. Earlier: 2026-08-10 (arch coverage re-swept after the 08-03 dependency bump;
cache semantics unchanged since 2026-07-30, fork `9cf1e90`). Both VLM routes
(GLM-4.6V-Flash-4bit,
Qwen3-VL-30B-A3B-8bit) run BatchedEngine in production; `vlm-server.py` is
retired. This doc answers one question: **which caches apply to a vision
route, which are deliberately off, and why.** Patch-level detail lives in
PATCHES.md #54–#69.*

## The cache matrix

| Cache | What it saves | Media requests | Text requests on the route | Live evidence (Studio) |
|---|---|---|---|---|
| **Pixel cache** (`vision_embedding_cache.py`, level 1) | Image preprocessing: decode, resize, normalize, tokenize-with-placeholders | **ON** — keyed on sha256(image content) + prompt | n/a | `pixel_cache_hits=1` on both routes; same-image re-ask 0.4 s |
| **Prefix-KV cache** (`memory_cache.py` `MemoryAwarePrefixCache`, RAM + SSD tier, namespace `mllm-v2`) | The prefill itself (KV reuse) | **ON since #142 (phase B)** for glm4v/gemma4/mistral3/qwen3_5, keyed on image content; phase A (off) for other arches, video, audio, URL images | **ON** | GLM route, 2.2 K-token text prompt repeated: 5.4 s → **1.0 s, hit, `tokens_saved=2189`** |
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

## Phase B — BUILT 2026-10-14 (PATCHES.md #142)

Built as a *substitution*, not the planned prefix: every image placeholder
token is replaced, **in the cache key only**, by negative values derived from
that image's full SHA-256 (`vllm_mlx/mllm_media_key.py`). That keeps the text
before the image shared, diverges at the image's first placeholder when the
image differs, and can never equal a real token id. The k-th placeholder run
is the k-th image (Qwen/GLM/gemma); split layouts (Pixtral rows) use one
order-sensitive digest of all images. `http(s)` URL images, video and audio
stay phase A.

What the original plan got wrong, checked in code:
- **No rope-delta to persist.** Since upstream #744 every request derives its
  mRoPE state from the full prompt + `image_grid_thw` at preprocessing
  (`_derive_request_rope_deltas`), and restored continuations pass the sliced
  `position_ids`; all mRoPE families here expose `get_rope_index`.
- **No SSD namespace bump.** Text keys are unchanged and fingerprints are
  negative, so `mllm-v2` entries stay valid both ways (a rollback's text-only
  lookups can only share an image entry's text prefix, whose KV is
  image-independent).

A hit requires every image to lie inside the cached prefix (the
remaining-ids placeholder guard), so a follow-up skips the vision tower
entirely. Persisted prefix-cache entries reload on restart, so a re-asked
image survives one.

**Measured on the Studio (e2e `scripts/fork/e2e_media_prefix_cache.py`, ON
vs `VLLM_MLX_MLLM_MEDIA_PREFIX_CACHE=0`, T=0, isolated HOME):**

| Arch (model) | Same-image follow-up TTFT | Exact re-ask | Different image, identical ids | Restored == cold |
|---|---|---|---|---|
| glm4v (GLM-4.6V-Flash) | 1.83 → **0.19 s** | 1.88 → 0.05 s | 0 cached | 8/8 (+ e2e 7/7) |
| mistral3 (Devstral-24B) | 4.00 → **0.27 s** | 3.97 → 0.08 s | 0 cached | 7/7 |
| gemma4 (26B-A4B) | 0.69 → **0.17 s** | 0.73 → 0.05 s | 0 cached | 8/8 (one e2e near-tie) |
| qwen3_5 (Qwen3.5-4B) | template re-renders turn 1 → no follow-up hit | 0.80 → 0.03 s | 0 cached | 7/7 |
| qwen3_vl_moe (Qwen3-VL-30B) | 0.88 → **0.20 s** | 0.88 → 0.13 s | 0 cached | **6/8** (text-only control 8/8) |

Default ON for glm4v, gemma4, mistral3, qwen3_5, and since 2026-10-14 (Tim's call) **qwen3_vl_moe**. It was opt-in at first
(`VLLM_MLX_MLLM_MEDIA_PREFIX_ARCHS`): its two divergences are late (~15
tokens) and coherent, consistent with MoE near-ties, but its text-only
control was 8/8, so "exact" is not proven. Next check before enabling: a
larger image sample, or an in-process logits comparison of a restored vs cold
continuation. Kill switch for everything: `VLLM_MLX_MLLM_MEDIA_PREFIX_CACHE=0`.

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

## Re-sweep after the 2026-08-03 dependency bump (2026-08-10)

The 08-03 rebase dragged `mlx-vlm` 0.6.3 → 0.6.8 with `mlx` 0.32 and
`transformers` 5.5.4 → 5.14.1 fleet-wide, and the arch sweep was never re-run —
it stood as the open residual. Re-run 2026-08-10 against the **deployed**
site-packages (fork `c7da98d`), spare port 8096, one model at a time on
BatchedEngine:

| Arch | Tested with | Verdict |
|---|---|---|
| qwen3_5 | Qwen3.5-4B-4bit | **10/10** |
| glm4v | GLM-4.6V-Flash-4bit | **10/10** (production route) |
| mistral3 | Devstral-Small-2-24B-2512-4bit | **10/10** (#68/#69 still hold) |
| gemma4 | gemma-4-26B-A4B-it-qat-4bit | **10/10** |
| qwen3_vl_moe | Qwen3-VL-30B-A3B-Instruct-8bit | **10/10** (production route) |
| qwen3_5_moe | — | **not re-tested** — see below |

**No regression from the dependency bump on any testable family**, both
production vision routes included. The residual is closed except for one family.

**`qwen3_5_moe` cannot be tested on this box.** The 07-30 sweep used
Ornith-1.0-35B, which is no longer in the HF cache, and the only cached
`qwen3_5_moe` checkpoint (`Qwen3.6-35B-A3B-4bit-DWQ`) is one of the
stripped-tower conversions above — `vision_config` absent. Re-testing this
family requires re-downloading a vision-capable `qwen3_5_moe` checkpoint. No
route serves vision on this arch today, so this is a coverage gap, not a live
risk.

**About the checks.** The original 14-check `fleetsmoke` script is gone from the
Studio (only a stale `.pyc` under `~/Library/Caches` remains), so this is a
reconstructed 10-check e2e over HTTP, weighted toward the vectors this fork
actually cares about rather than breadth:

1. `server_up` · 2. `single_image_red` · 3. `second_image_blue_not_aliased`
· 4. `resend_red_after_blue` · 5. `divergent_size_green` · 6. `multi_image`
· 7. `concurrent_cobatch` · 8. `text_only_on_vision_route`
· 9. `streaming_vision` · 10. `status_vision_gauges`

Checks 2–5 identify solid-colour images by name, which only passes if the vision
tower genuinely sees pixels. The red → blue → red ordering (3, 4) is the direct
probe for **cross-image cache aliasing** — the known open gap from the batched
vision gap analysis; a second, different image answered from the first's cached
embedding fails check 3. Check 5 uses a differently-*shaped* image (64×160 vs
96×96) so its MRoPE delta differs, and check 7 co-batches two divergent-size
images concurrently — the #57 vectors.

**Harness note worth keeping:** the first run scored 5/10 on *every* model
because `max_tokens=24` plus default thinking meant the budget was spent
entirely inside `<think>` and the answer never arrived — the same artifact the
07-30 sweep logged against gemma4. These are reasoning models; a vision smoke
must send `chat_template_kwargs={"enable_thinking": false}` **and** leave real
token headroom, and must match on `content` with reasoning stripped so a colour
mentioned mid-thought isn't scored as the answer. Scoring reasoning text as the
answer is how a vision smoke lies to you in both directions.

Mixed `--text-only` routes are *not* flipped: serving vision there today would
trade away the text stack (system-KV etc.). That's phase 2 `--vision-split`
(plan P15–P19), de-risked by this sweep.

## Known residuals

- MLLM preprocessing failures surface as HTTP 200 + `finish_reason:"error"` +
  `content:null` — should be a proper 4xx/5xx (noted in PATCHES.md #68).
- Pre-stream prompt-ceiling estimate for streams (both branches).
- Audio is not forwarded into `chat_kwargs` by the server; audio-bearing
  requests on text routes get an honest 400 (#65).
