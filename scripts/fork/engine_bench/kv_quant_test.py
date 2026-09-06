#!/usr/bin/env python3
"""KV-cache quantization A/B at long context (mlx-lm built-in kv_bits).

For each kv config (fp16 / 8-bit / 4-bit) x context length: prefill a
deterministic ~N-token prompt, then decode 128 tokens at T=0. Reports
prefill tok/s, decode tok/s, peak memory, and the first output chars
(same-text sanity across configs).
"""
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.8-27B-8bit"
CTXS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                         else ["32000", "64000"])]

model, tokenizer = load(MODEL)

_WORDS = ("system memory cache latency throughput kernel scheduler batch "
          "tensor gradient attention token stream buffer socket protocol "
          "packet routing storage index shard replica quorum consensus log "
          "compaction snapshot checkpoint recovery migration allocator page "
          ).split()

def build_ids(n_tokens):
    x = 7
    words = []
    for i in range(int(n_tokens * 0.9)):
        x = (x * 1103515245 + 12345) % (2 ** 31)
        words.append(_WORDS[x % len(_WORDS)])
    doc = " ".join(words)
    ids = tokenizer.encode("Document:\n" + doc)
    ids = ids[:n_tokens - 24]
    ids += tokenizer.encode(
        "\n\nSummarize the above document in one sentence.")
    return ids

sampler = make_sampler(temp=0.0)
prompts = {c: build_ids(c) for c in CTXS}

for kv_bits in (None, 8, 4):
    for ctx in CTXS:
        kwargs = {}
        tag = "fp16"
        if kv_bits:
            kwargs = dict(kv_bits=kv_bits, kv_group_size=64,
                          quantized_kv_start=1024)
            tag = f"{kv_bits}bit"
        mx.clear_cache()
        mx.reset_peak_memory()
        ids = prompts[ctx]
        text = []
        ptps = dtps = None
        t0 = time.perf_counter()
        try:
            for r in stream_generate(model, tokenizer, ids, max_tokens=128,
                                     sampler=sampler, **kwargs):
                text.append(r.text)
                ptps = r.prompt_tps
                dtps = r.generation_tps
        except Exception as e:
            print(f"{tag} ctx={ctx}: FAILED {type(e).__name__}: {e}",
                  flush=True)
            continue
        dt = time.perf_counter() - t0
        peak = mx.get_peak_memory() / 1e9
        head = "".join(text)[:60].replace("\n", " ")
        print(f"{tag} ctx={ctx}: prompt_tokens={len(ids)} "
              f"prefill={ptps:.0f} tok/s decode={dtps:.2f} tok/s "
              f"peak={peak:.1f}GB total={dt:.0f}s head={head!r}", flush=True)
