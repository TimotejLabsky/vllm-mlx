# Prefix caching: how the other engines do it, and what we should copy

*Research round 2026-08-03. Companion to
[`continuous-batching-hybrid-caching.md`](continuous-batching-hybrid-caching.md)
(why we built our own cache) — this doc asks the follow-up question: **now that
it exists, is it behind the state of the art, and where?***

**TL;DR.** For our model lineup our cache is *ahead* of vLLM's and SGLang's,
because both are block/token-granular designs that hybrid (attention+SSM) models
break, and they are still working through that; we solved it in 2026-07. Two of
the three axes the industry optimises on are **structurally irrelevant to a
single-user Mac Studio** — memory-hierarchy offload (unified memory deletes it)
and cross-user sharing (we have ~1 user). The third axis — *what to keep when
the cache exceeds the memory cap* — is exactly our situation and is precisely
where our design is weakest: **eviction is recency-only**. That is the one real
gap. It is also not obviously worth fixing yet; the recommendation is to
instrument before building. No code changes are proposed by this doc.

---

## 1. What we have

`system_kv.py` (SimpleEngine) and `batched_system_kv.py` (BatchedEngine) are a
**snapshot + checkpoint** cache, not a block cache:

- **Key:** a `system_hash` over the template-detected system prefix, plus
  longest-common-prefix matching over the full token list
  (`match_extended_prefix`, `plan_partial_restore`).
- **Unit:** a whole cache *state* per entry (`snapshot`), plus a ladder of
  mid-prefill `checkpoints` captured every `VLLM_MLX_BATCHED_KV_CKPT_INTERVAL`
  tokens (default 2048), bounded to `VLLM_MLX_SYSTEM_KV_CHECKPOINTS` (default 8)
  by geometric thinning (`checkpoints[::2]`).
- **Restore:** `select_restore_pos` picks the highest checkpoint ≤ the
  divergence point for checkpoint-class (recurrent / sliding-window) layers, and
  slices freely for trimmable attention layers.
- **Capacity:** LRU (`OrderedDict` insertion order) under a RAM budget, then an
  SSD tier (`system_kv_ssd.py`, patches #16/#25/#36/#49), plus memory-pressure
  relief (#48/#53/#60).

The design constraint that produced all of this: **recurrent state cannot be
rewound.** You cannot slice an SSM state back to token *i* the way you can trim
an attention K/V tensor. So the only way to serve a prefix of length *i* is to
have *captured* the state at *i*. That single fact is what separates our design
from everyone else's, and it is why the industry's designs do not port.

---

## 2. The landscape

### vLLM — Automatic Prefix Caching (hash-chained blocks)

Blocks of fixed size are hashed as `hash(parent_hash, block_tokens, extras)`
(SHA-256 by default as of v0.11; xxHash available), into a global hash table
with refcounted physical blocks and **LRU eviction** over a doubly-linked free
queue. Freed blocks re-enter the queue *in reverse order*, on the reasoning that
a request's last block hashes more tokens and is less likely to be reused —
a small, cheap value heuristic layered on top of recency. Only **full** blocks
are cached, so partial-block overlap yields nothing.

**On hybrids it is in worse shape than ours.** Prefix caching for
Mamba/SSM/GDN models is an [open tracking
issue](https://github.com/vllm-project/vllm/issues/26201) with work still
outstanding on kernel fusion, TP>1, block-size constraints, and default
enablement. The mode these models actually use, `mamba_cache_mode="align"`,
retains **one** Mamba state checkpoint per request — at the last block boundary
before the prompt ends — which produces two live bugs that are the direct
analogue of what our checkpoint ladder exists to prevent:

- [#45238](https://github.com/vllm-project/vllm/issues/45238) — hit rate
  silently drops to **0%** when that single checkpoint lands in request-unique
  tokens rather than the shared prefix.
- [#40696](https://github.com/vllm-project/vllm/issues/40696) — on Qwen3.5 the
  attention block size is forced to 528 to align with the Mamba page size, so
  **any prompt under 528 tokens gets 0% hit rate**.

Our design has neither failure mode: we keep up to 8 checkpoints, not 1, and our
matching is token-granular LCP rather than block-aligned, so short prompts and
unaligned divergence points both work. **Qwen3.5 is our production lineup.**

### SGLang — RadixAttention

A radix tree over cached prefixes with LRU eviction, plus **cache-aware
scheduling**: the scheduler prioritises the request with the longest matched
prefix, approximating a DFS order over the tree to maximise hit rate. Reported
hit rates 50–99%, and 75–95% on multi-turn agent workloads with a shared system
prompt.

The radix tree's advantage over our linear LCP scan is *many-way branching
across many concurrent sessions* — it shares interior nodes between unrelated
requests. Cache-aware scheduling's advantage requires **a queue with several
waiting requests to reorder**.

### LMCache / llm-d / Mooncake — tiered offload

KV blocks migrate GPU HBM → CPU DRAM → local NVMe → remote (Redis, S3,
Mooncake). Reported 3.7–6.8× lower TTFT, up to 15× throughput on chat and RAG.
This is the most heavily engineered area in the field right now.

### Marconi — prefix caching for hybrid LLMs

The one paper aimed squarely at our problem
([arXiv 2411.19379](https://arxiv.org/pdf/2411.19379)). Two contributions:

1. **Admission** — don't checkpoint uniformly; place checkpoints where request
   chains actually *branch*, because those are the only positions a future
   divergent request can restore from.
2. **Eviction** — score entries by **recompute cost (FLOPs) and reuse
   likelihood**, not recency alone. A deep prefix is far more expensive to
   regenerate than a shallow one, so LRU throws away the wrong entries.

### CacheBlend — non-prefix reuse

Reuses KV blocks at *arbitrary* positions (not just as a prefix) by selectively
recomputing a small fraction of tokens to repair cross-attention. Aimed at RAG,
where the same document chunk appears at varying offsets.

---

## 3. Which of this actually applies to us

The industry optimises caching along three axes. Our deployment nullifies two of
them outright.

| Axis | Motivated by | Our situation | Verdict |
|---|---|---|---|
| **Memory-hierarchy offload** (LMCache, Mooncake, llm-d) | GPU HBM is small and separated from host DRAM by PCIe | **Unified memory.** There is no HBM→DRAM copy to optimise — the "tiers" are the same physical RAM | **Structurally N/A.** The only meaningful tier is RAM→SSD, and we already have it (#16/#25/#36/#49) |
| **Cross-user prefix sharing** (RadixAttention tree, block dedup, cache-aware scheduling) | Many concurrent tenants sharing system prompts | ~1 user; concurrent traffic is not routine; measured batching ceiling ~1.2× aggregate | **Near-zero value.** Nothing to reorder in a 1-deep queue; nothing to dedup between tenants |
| **What to keep under a hard cap** (Marconi admission + eviction) | Cache working set exceeds capacity | **Exactly us.** ~60 GB wired wall, multi-GB entries, one entry can be a meaningful fraction of the budget | **This is where to invest** |

This is worth stating plainly because it inverts the intuition: most of the
impressive numbers in KV-cache papers come from solving a memory-topology
problem that Apple Silicon does not have. Copying that work would be effort
spent on a non-problem.

---

## 4. The one real gap: eviction is recency-only

`enforce_ram_budget` (`system_kv.py:672`) and `_enforce_budgets_locked`
(`batched_system_kv.py:639`) both evict in `OrderedDict` insertion order — pure
recency. Credit where due: there **is** one value-aware term already — eviction
runs in two passes, **SSD-spilled entries first**, on the sound reasoning that
re-acquiring a spilled entry is a ~1.3 s promote versus a ~25–39 s cold prefill.
That is a genuine cost-aware decision and it is in the right direction.

The gap is *within* each pass, where ordering is recency and nothing else. Two
entries are treated identically whether one saves 2K tokens of prefill or 100K.
At our measured prefill rates that is the difference between evicting something
worth a couple of seconds and something worth **~16 minutes** of recompute. The
information needed to do better is already tracked on every entry (`bytes`,
`token_count`) and patch #52 already measures ground-truth bytes/token.

A minimal version needs no ML and no FLOPs model:

```
value_density ≈ estimated_recompute_seconds(token_count) / bytes
evict ascending value_density, decayed by staleness, spilled-class first
```

`estimated_recompute_seconds` can be a fitted curve over the ladder measurements
we already have per route (`fleet-batched-flip`, `context-envelope-27b`).

**But do not build this yet.** Whether it ever changes a decision depends on
facts we have not measured: how often the budget actually binds, how many
entries are resident when it does, and whether the LRU victim differs from the
value victim. With few entries and one user, the two policies may pick the same
entry nearly always. The fork's own precedent is to measure first — the #48
crash fix only worked because the failure was reproduced first, and the
"spec decoding is a decode lever" belief died on contact with measurement.

**Proposed first step (cheap, no behaviour change):** log on every eviction the
victim's `token_count` and `bytes`, and count subsequent misses whose LCP would
have hit an entry evicted in the last N minutes. If that counter stays near
zero, the budget is not binding and value-based eviction is dead on arrival —
close the item. If it is non-trivial, the fix is ~40 lines in two functions.

---

## 5. Second gap: checkpoint placement is stride-based, not divergence-aware

Checkpoints land every 2048 tokens (`VLLM_MLX_BATCHED_KV_CKPT_INTERVAL`),
capped at 8 and thinned geometrically. At 100K context that is an effective
spacing of ~12.5K tokens, so a divergent request restores from up to ~12.5K
tokens *before* where it actually diverged, and re-prefills the difference.

Marconi's admission insight applies: the useful checkpoint positions are the
ones where chains actually branch, and for chat/agent traffic that position is
highly predictable — **the end of the rendered conversation prefix, where the
new turn begins**. We already detect template boundaries
(`detect_template_markers`), so the machinery exists to force a checkpoint there
in addition to the stride.

**Important qualification that shrinks this a lot:** the *common* multi-turn
case — appending a turn to an unchanged history — does **not** go through the
checkpoint path at all. `select_restore_pos` special-cases `d == donor_len` and
restores the whole snapshot directly. Checkpoints only matter for genuinely
*divergent* chains: edits, retries, regenerations, and history trimming by the
client. So this only pays off on branchy traffic — which coding agents do
produce (the [CacheWise](https://arxiv.org/pdf/2606.16824) workload study finds
agent traffic branches and that plain LRU is a poor fit for it), but our own
`needle-test` runs showed the warm path collapsing 911 s → 12 s on the
append case that already works.

Verdict: **lower priority than eviction**, and gated on the same instrumentation
— log the gap between the chosen restore position and the true divergence point.
If that gap is usually ~0, the stride is fine.

---

## 6. Explicitly rejected

- **Block-hash prefix caching (vLLM-style).** Fundamentally mismatched to
  recurrent state; vLLM is actively fighting the exact failure modes our
  checkpoint ladder avoids. Adopting it would be a downgrade for our lineup.
- **Radix tree replacing the LCP scan.** Its win is interior-node sharing across
  many concurrent sessions. With a handful of LRU entries, a linear scan is not
  the bottleneck.
- **Cache-aware scheduling (SGLang).** Needs a deep queue to reorder. Revisit
  only if concurrent traffic ever becomes routine — at which point it is the
  best idea in this doc, because it is architecture-agnostic and so works on
  hybrids where block tricks do not.
- **LMCache / tiered offload.** Unified memory removes the tier it optimises.
- **CacheBlend / non-prefix reuse.** Selective recomputation trades accuracy for
  TTFT; heavy to implement, and our traffic is conversational (prefix-shaped),
  not RAG-chunk-shaped. Revisit only if a document-RAG route appears.
- **KV-cache quantisation.** Unchanged from prior reviews: still a *watch* item,
  not a build item.

---

## 7. Summary

| Item | Verdict |
|---|---|
| Our hybrid handling vs vLLM/SGLang | **We are ahead** — keep, and it is worth an upstream write-up |
| Value-based eviction (Marconi) | **Only real gap.** Instrument first, then ~40 lines |
| Divergence-aware checkpoints | Real but narrower; gate on the same instrumentation |
| Cache-aware scheduling | Correct idea, wrong deployment — revisit if concurrency arrives |
| Tiered offload, radix tree, CacheBlend, block hashing | Rejected for this deployment |

**Sources:**
[vLLM APC design](https://docs.vllm.ai/en/stable/design/prefix_caching/) ·
[vLLM hybrid prefix-caching tracker #26201](https://github.com/vllm-project/vllm/issues/26201) ·
[vLLM #45238](https://github.com/vllm-project/vllm/issues/45238) ·
[vLLM #40696](https://github.com/vllm-project/vllm/issues/40696) ·
[SGLang / RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) ·
[Marconi](https://arxiv.org/pdf/2411.19379) ·
[CacheWise](https://arxiv.org/pdf/2606.16824) ·
[LMCache](https://github.com/lmcache/lmcache)
