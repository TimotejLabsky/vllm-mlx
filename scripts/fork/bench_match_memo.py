"""Issue #109 follow-up: time one request's add_request + solo-verdict +
admission sequence on BatchedSystemKV with a bag of fake deep chains.

Usage: bench_match_memo.py ENTRIES CHAIN_TOKENS PROMPT_TOKENS [REPEATS]

Setup (mirrors the #109 audit): real mlx-lm cache classes, 8 layers
(6 KVCache + 2 ArraysCache), KV (1, 2, N, 64) bf16 shared by every entry
(arrays are immutable, so sharing does not change any host path).
ENTRIES chains of CHAIN_TOKENS share a 20K system prefix then diverge; each
has a checkpoint every 8K and a generated span from CHAIN-5000. All are
spilled to an SSD tier (writer drained before timing). The prompt is
chain 0's first PROMPT-1000 tokens + 1000 new ones (divergent from every
entry). Production-style flags: LAZY_RESTORE, RECONCILE_GENERATED,
DIVERGENCE_LOG, SSD tier.
"""

import os
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace as NS

ENTRIES, CHAIN, PROMPT = (int(a) for a in sys.argv[1:4])
REPEATS = int(sys.argv[4]) if len(sys.argv) > 4 else 7

tmp = tempfile.mkdtemp(prefix="bench109-")
os.environ.update(
    {
        "VLLM_MLX_BATCHED_SYSTEM_KV": "1",
        "VLLM_MLX_SYSTEM_KV_SLOTS": str(ENTRIES + 2),
        "VLLM_MLX_BATCHED_LAZY_RESTORE": "1",
        "VLLM_MLX_BATCHED_RECONCILE_GENERATED": "1",
        "VLLM_MLX_DIVERGENCE_LOG": "1",
        "VLLM_MLX_SSD_SYSTEM_KV_DIR": tmp,
    }
)

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import ArraysCache, KVCache  # noqa: E402

import vllm_mlx.batched_system_kv as bkv  # noqa: E402


class Model:
    def make_cache(self):
        return [KVCache() if i % 4 else ArraysCache(size=2) for i in range(8)]


class Tok:
    name_or_path = "bench/model"

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def cache_list(n):
    out = []
    k = mx.zeros((1, 2, n, 64), dtype=mx.bfloat16)
    for i in range(8):
        if i % 4:
            c = KVCache()
            c.update_and_fetch(k, k)
        else:
            c = ArraysCache(size=2)
            c[0] = mx.zeros((1, 4))
            c[1] = mx.zeros((1, 2, 3))
        out.append(c)
    mx.eval([a for c in out for a in (c.state if isinstance(c.state, (list, tuple)) else [c.state]) if a is not None])
    return out


kv = bkv.BatchedSystemKV(Model(), tokenizer=Tok())
big = cache_list(CHAIN)
system = list(range(10_000, 30_000))
chains = []
for e in range(ENTRIES):
    chain = system + [200_000 + e * 1_000_000 + t for t in range(CHAIN - len(system))]
    chains.append(chain)
    rid = f"seed-{e}"
    kv.note_scheduled(rid, 0)
    for pos in range(8192, CHAIN, 8192):
        kv.capture_segment(rid, pos, big)
    kv.store(rid, chain, big, gen_start=CHAIN - 5000)
# drain the SSD writer
deadline = time.time() + 300
while kv._ssd.get_stats().get("entry_count", 0) < ENTRIES and time.time() < deadline:
    time.sleep(0.2)
assert kv._ssd.get_stats().get("entry_count", 0) == ENTRIES, kv._ssd.get_stats()
assert kv.stats()["entry_count"] == ENTRIES, kv.stats()["entry_count"]

calls = {"lcp": 0, "lcp_s": 0.0, "ssd": 0, "ssd_s": 0.0}
_orig_lcp = bkv.common_prefix_len


def counting_lcp(a, b):
    t = time.perf_counter()
    try:
        return _orig_lcp(a, b)
    finally:
        calls["lcp"] += 1
        calls["lcp_s"] += time.perf_counter() - t


bkv.common_prefix_len = counting_lcp
ssd = kv._ssd
for name in ("lookup_prefix", "lookup_shared"):
    orig = getattr(ssd, name)

    def wrap(*a, _o=orig, **k):
        t = time.perf_counter()
        try:
            return _o(*a, **k)
        finally:
            calls["ssd"] += 1
            calls["ssd_s"] += time.perf_counter() - t

    setattr(ssd, name, wrap)

new = [900_000_000 + t for t in range(1000)]
totals, add_ms, adm_ms, snaps = [], [], [], []
for r in range(REPEATS):
    for k in calls:
        calls[k] = 0 if isinstance(calls[k], int) else 0.0
    prompt = chains[0][: PROMPT - 1000] + new
    req = NS(
        request_id=f"bench-{r}",
        prompt_token_ids=prompt,
        num_prompt_tokens=len(prompt),
        cache_hit_type=None,
        cached_tokens=0,
        prompt_cache=None,
        remaining_tokens=None,
    )
    sched = NS(hybrid_kv=kv, running=[], config=NS(prefill_step_size=2048))
    t0 = time.perf_counter()
    bkv.fetch_for_request(kv, req)
    t1 = time.perf_counter()
    bkv.solo_prefill_verdict(sched, req)
    bkv.materialize_pending_restore(sched, req)
    t2 = time.perf_counter()
    kv.discard_pending(req.request_id)
    totals.append((t2 - t0) * 1e3)
    add_ms.append((t1 - t0) * 1e3)
    adm_ms.append((t2 - t1) * 1e3)
    snaps.append(dict(calls, hit=req.cache_hit_type, cached=req.cached_tokens))

last = snaps[-1]
print(
    f"entries={ENTRIES} chain={CHAIN} prompt={PROMPT} n={REPEATS} "
    f"total_med={statistics.median(totals):.1f}ms "
    f"(add {statistics.median(add_ms):.1f} / admission {statistics.median(adm_ms):.1f}) "
    f"min={min(totals):.1f} lcp_calls={last['lcp']} lcp_ms={last['lcp_s'] * 1e3:.1f} "
    f"ssd_probes={last['ssd']} ssd_ms={last['ssd_s'] * 1e3:.1f} "
    f"hit={last['hit']} cached={last['cached']}"
)
kv.close()
