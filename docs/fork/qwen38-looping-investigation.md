# Qwen3.8 agentic looping — the two-day hunt (2026-08-21 → 2026-08-23)

**Verdict:** the thinking phase machine was walking the *prompt*. Fixed in
[`PATCHES.md`](../../PATCHES.md) **#79**, merged and deployed 2026-08-23.

This document exists mostly for the **dead ends**. Three separate mechanisms
were proposed, believed, and in two cases acted on — one of them all the way
through a production deploy — before any of them was checked against the actual
artifact. Each died in under five minutes once someone looked. The failure mode
is worth more than the fix.

---

## The symptom

Qwen3.8-27B degenerated in agentic sessions (opencode, Claude Code): reasoning
text appearing as answer content, thinking budgets that appeared not to bind,
turns burning to `max_tokens` at 20+ minutes, paraphrase spirals, and —
diagnostically the strangest part — **hallucinated scaffold vocabulary**
(`<next_thinking>` tags, stray `";}` fragments) that exists nowhere in the
template, the fork, or mlx-lm.

Single-turn prompts were always fine. Only multi-turn agent sessions broke.

---

## Dead end 1 — "mlx-lm corrupts the weights at load"

**The story.** mlx-lm ≤0.31.3's `Qwen3_5.TextModel.sanitize` applied the MLX
`gamma + 1` RMSNorm conversion whenever MTP weights were present:

```python
should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
```

If a checkpoint shipped MTP weights *and* was already converted, every load
shifted twice. Upstream `4eeaf20` (mlx-lm#1623) narrows the trigger. The model
would still produce fluent text, which elegantly explained why this looked like
a sampling problem. It was reproduced upstream, and there had been no mlx-lm
release since April, so the fix was only reachable by pinning a git commit.

**Why it was believed.** It explained the symptom shape, it had an upstream
issue, an upstream fix, and an independent reproduction. Two sessions accepted
it and shipped a production deploy on it.

**How it died.** Check the checkpoints actually being served:

```
mlx-community/Qwen3.8-27B-{4bit,8bit}: 2180 weight keys, ZERO matching "mtp"
                                       conv1d.weight shape [10240, 4, 1] (last dim 1)
```

So `has_mtp_weights=False` and `has_unsanitized_conv1d=False`; both the pre- and
post-fix predicates evaluate `False`. Proven at tensor level rather than by
reading code — calling the real `TextModel.sanitize` from `4eeaf20^` and
`9acef5f` in one process against 40 real tensors per checkpoint gives
**bit-identical output**, and the returned norm means (~0.91–0.94) sit correctly
near 1.0 where a double shift would read ~1.9. Every cached qwen3_5-family model
reports `has_mtp_weights=False`. The fix is **inert fleet-wide**.

**Cheap technique worth reusing:** `sanitize` only reads
`self.args.tie_word_embeddings`, so it can be called unbound with a
`SimpleNamespace` stand-in — no 27B model instantiation, no touching the
deployed venv. Load the other version by rewriting its `from .` imports to
`from mlx_lm.models.` and `importlib`-ing the file.

**Lesson.** An upstream commit message describing a bug is not evidence that
*your* artifacts exhibit it. Check weight keys and tensor shapes of the thing
you actually serve before attributing a fix to it.

---

## Dead end 2 — "`prompt_has_think_tag` is a guess and it's wrong"

**The story.** The server passes `prompt_has_think_tag=bool(enable_thinking)`
into the thinking processor — a guess, not an observation of the rendered
prompt. A history whose assistant turn carries `tool_calls` but no
`reasoning_content` (exactly what agent clients replay, since they strip
reasoning) plausibly renders *without* a trailing `<think>`, so the guess would
be wrong precisely where it hurts.

**Why it was believed.** It named a real code smell, and it matched a real,
minimal, reproducible precondition someone had already isolated.

**How it died.** Render all three shapes through the real tokenizer with the
route's kwargs:

```
A simple user turn                   -> endswith <think> : True
B tool_calls, NO reasoning_content   -> endswith <think> : True
C tool_calls, WITH reasoning_content -> endswith <think> : True
```

Every shape already ends in `<think>`, so the flag was correct in all three
cases and deriving it from the render returns `True` every time — a no-op. A
complete patch with 10 passing unit tests and a green 2904-test suite was
written and then **discarded**.

**Lesson.** A code smell adjacent to a bug is not the bug. "This looks wrong"
and "this is what is wrong" need separate evidence.

---

## Dead end 3 — "it's the prefix cache / cold-vs-warm"

**The story.** A clean, deterministic, four-minute reproduction: cold requests
misclassify, warm requests are correct, and the invariant is exact —
**cold `content` length == warm `reasoning_content` length**, byte for byte,
every time, with identical `prompt_tokens`.

**Why it was believed.** That invariant is about as clean as evidence gets, and
it correctly retired a competing theory (the other session's "intermittent /
sampling-dependent" reading was cache warmth varying between runs).

**How it died.** It was a *symptom* of the real cause, not the cause. Warm hits
differed only because the cached path fed a different token sequence into the
same broken walk. It was, however, the reproduction that made instrumentation
possible — so it earned its keep even while pointing at the wrong layer.

**Lesson.** A reliable reproduction and a correct diagnosis are different
achievements. Getting the first does not license skipping the second.

---

## The actual cause

`ThinkingAwareLogitsProcessor` assumed its `tokens` argument contained only
**generated** tokens. On the batched path it does not: mlx-lm's
`BatchGenerator.prompt()` folds prompt tokens into the sequence handed to
processors — *"Add the tokens to the self.tokens so they represent the tokens
contained in the KV Cache."*

Instrumented on Qwen3.8-27B-4bit with a 448-token tool-replay prompt:

```
[dbg-sync] FIRST sync: n_tokens=448  end_token_in_seq=[103, 181, 217]
                                     start_token_in_seq=[95, 180, 216, 446]
                                     phase_before=Phase.THINKING
[dbg-sync] after sync: processed=448 phase=Phase.CONTENT thinking_tokens=103
```

The prompt replays earlier `<think>…</think>` spans. The machine counted 103
prompt tokens as thinking, hit a prior `</think>` at index 103, and reached
`CONTENT` **before a single token was generated**. `CONTENT` masks
`<think>`/`</think>` to `-inf` — so the model was *physically forbidden from
closing its own think block*.

Every symptom falls out of that one line:

| symptom | mechanism |
|---|---|
| reasoning in `content`, `reasoning_content` empty | the parser needs a closer **in the output** to split; the closer was masked |
| thinking budget "inert" | already in `CONTENT`, so it never counted a generated token |
| runs to `max_tokens` | nothing can end a think span that cannot close |
| `<next_thinking>` scaffold hallucination | the model reaching for a closer it is forbidden to emit, improvising training-time structure |
| single-turn prompts fine | **by luck** — no prior `</think>` in the prompt to trip on |

**Fix.** Walk generated tokens only. Two call conventions exist and both are
handled from an observable signal:

| path | first call | handling |
|---|---|---|
| batched / mlx-lm | full sequence (prompt folded in) | first call **is** the prompt — skip it |
| MLLM scheduler | empty, then `output_tokens` only | empty call records `prompt_len = 0` — skip nothing |

A second bug was found by re-reading the patch as a reviewer:
`_restore_snapshot` indexes snapshots by absolute `processed_len`, which is
wrong once the prompt is skipped (snapshot 0 now sits at the prompt boundary).
Fixed by indexing on generated position, plus a clamp so a rewind can never
re-enter the prompt and re-run the original bug.

---

## Method notes

- **Instrument before patching.** Three plausible mechanisms survived review and
  died on contact with the artifact. The thing that actually worked was logging
  the phase machine's state at the *first* sync — the whole answer was one line.
- **Log at the boundary, not per token.** Per-token logging showed six tokens of
  noise. One line summarising state after the prompt sync showed everything.
- **Prod-safe instrumentation:** clone the fork to `/tmp` on the Studio, add
  env-gated logging, run the route on a spare port via `PYTHONPATH`. The
  installed package is never touched. Backend logs go nowhere under llama-swap,
  so a manually-run route with captured stderr is the only way to see engine
  internals.
- **`usage.prompt_tokens` is a free fingerprint** of which chat-template branch
  executed — it ruled out "different prompt" in seconds, repeatedly.
- **A reproduction is not a diagnosis** (dead end 3).
- **CI catches what local testing cannot.** A `mlx-lm>=0.32.0` floor looked
  correct and passed every local test, but 0.32.0 exists only as a git commit —
  no PyPI release since April — so the package became uninstallable. Only CI saw
  it. Related: this fork's CI is **pre-existing red on `lint`** (black drift over
  33 files), so the signal to watch is "red *beyond* lint", not "green".
- **`git rerere` caches mistakes as happily as fixes.** A resolution recorded
  during one rebase attempt still contained a stray `>>>>>>>` marker; rerere
  replayed it on the next attempt, and the file was **not** listed as unmerged.
  Only a whole-tree marker scan caught it. `git rerere forget <path>` purges it.

## Operational gotchas found along the way

- **`anyio` is a production dependency** (fastapi/starlette need it) even though
  the fork also lists it under dev extras. Uninstalling it alongside pytest
  broke **every new route spawn** with `ImportError: cannot import name
  'ObjectReceiveStream' from 'anyio.abc'` — at import time, before any model
  code, so it presents as a model failure. Already-running routes survive
  (module in memory), so `/running` looks healthy while voice is down.
- **llama-swap `/unload` is a GET, not a POST.** `curl -X POST .../unload`
  silently no-ops and makes a model look stuck.
- **`voice-reload.sh` deliberately yields** (exit 0, does nothing) while any
  non-HA model is resident — unload the heavy model first.
- **`[metal::malloc] Resource limit (499000) exceeded` is a buffer-HANDLE
  ceiling, not an OOM.** Seen at ~21,000 generation steps with active memory at
  only 29.4 GB. No byte gauge shows it. A genuine `kIOGPUCommandBufferCallback
  ErrorOutOfMemory` can follow *downstream* of it once the process is degraded —
  the byte-OOM is the second error, not the cause. A ~16K-token single-turn
  probe is therefore not a reliable test on this box.
- **The laptop dev venv is not representative of the Studio.** Prod runs
  mlx-vlm 0.6.14 / transformers 5.14.1 / Python 3.11; the laptop had 0.6.2 /
  5.10.2 / 3.12. A laptop-green suite is not a faithful proxy — run the suite on
  the box (the 5 `test_mcp_security` failures there are just missing Node).

## Still open

- **DRY was an amplifier, not the cause.** Removing it eliminated runaway
  paraphrase spirals, but a low-level paraphrase tic persisted with DRY fully
  off, riding the same mis-primed think continuation. Now that #79 has landed
  the A/B is worth re-running: some of DRY's apparent benefit may have been this
  defect all along. Current config: DRY off on the 8-bit route, retained on
  4-bit as a control, `VLLM_MLX_REPDETECT=1` armed on both.
- **Thinking budget at the 6144 limit** is unverified on the hardened build —
  three attempts died on the Metal buffer-handle ceiling above. Bounded budgets
  (128/256/1024) bind exactly, and the hardening touches only a rollback path,
  so this is a narrow gap rather than an open question.
- **`black` drift over 33 files** keeps CI's lint job red. Worth doing as its own
  deliberate commit; it was kept out of unrelated PRs because it touches
  upstream-owned files and every line is future rebase conflict surface.
