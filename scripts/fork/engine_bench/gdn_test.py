#!/usr/bin/env python3
"""Bench-only evaluation of mlx PR #4020 (mx.fast.gated_delta_update).

Phase 1: numeric allclose — mlx-lm's gated_delta_kernel vs the new op, at the
         real model head configs, T=1 (decode) and T=64 (prefill), plus a
         q-scaling contract probe.
Phase 2: model-level T=0 token-identity gate + decode/prefill timing A/B on
         Qwen3.6-35B-A3B-4bit-DWQ (GDN_FAST=0/1 toggled per call via env).

Requires the gdn-env: mlx @ PR head c7e1a2a + pinned mlx-lm f4f3b57 with the
env-gated shim applied to models/gated_delta.py.
"""
import os
import sys
import time

import mlx.core as mx

def phase1():
    from mlx_lm.models import gated_delta as gd
    print("has op:", hasattr(mx.fast, "gated_delta_update"))
    for (Hk, Hv, Dk, Dv) in [(16, 32, 128, 128), (8, 16, 128, 128),
                             (12, 24, 128, 128)]:
        for T in (1, 64):
            B = 1
            mx.random.seed(7)
            q = mx.random.normal(shape=(B, T, Hk, Dk)) * 0.1
            k = mx.random.normal(shape=(B, T, Hk, Dk))
            k = k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
            v = mx.random.normal(shape=(B, T, Hv, Dv)) * 0.1
            g = mx.random.uniform(shape=(B, T, Hv), low=0.5, high=0.999)
            beta = mx.sigmoid(mx.random.normal(shape=(B, T, Hv)))
            st = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
            try:
                o_ref, h_ref = gd.gated_delta_kernel(q, k, v, g, beta, st)
                mx.eval(o_ref, h_ref)
            except Exception as e:
                print(f"  ref kernel failed Hk={Hk} Hv={Hv} T={T}: {e}")
                continue
            try:
                o_new, h_new = mx.fast.gated_delta_update(
                    q, k, v, g, beta, initial_state=st)
                mx.eval(o_new, h_new)
            except Exception as e:
                print(f"  NEW OP failed Hk={Hk} Hv={Hv} T={T}: {e}")
                continue
            do = mx.abs(o_ref - o_new).max().item()
            dh = mx.abs(h_ref - h_new).max().item()
            scale = Dk ** -0.5
            o_scaled, _ = mx.fast.gated_delta_update(
                q * scale, k, v, g, beta, initial_state=st)
            do_scaled = mx.abs(o_ref - o_scaled).max().item()
            print(f"  Hk={Hk} Hv={Hv} T={T}: max|d_out|={do:.3e} "
                  f"max|d_state|={dh:.3e} (with q*Dk^-0.5: {do_scaled:.3e})")

def phase2(model_repo, n=3, tokens=300):
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = load(model_repo)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content":
          "Explain how TCP congestion control works, covering slow start, "
          "congestion avoidance, fast retransmit and fast recovery."}],
        add_generation_prompt=True)
    words = ("system memory cache latency throughput kernel scheduler batch "
             "tensor gradient attention token stream buffer socket protocol "
             ).split()
    doc = " ".join(words[i % len(words)] for i in range(6000))
    long_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content":
          "Document:\n" + doc + "\nSummarize the document in one sentence "
          "then explain TCP slow start."}],
        add_generation_prompt=True)
    sampler = make_sampler(temp=0.0)

    def run(p, max_tokens, tag):
        t0 = time.perf_counter()
        ids = []
        prefill_tps = decode_tps = None
        for r in stream_generate(model, tokenizer, p, max_tokens=max_tokens,
                                 sampler=sampler):
            ids.append(r.token)
            prefill_tps = r.prompt_tps
            decode_tps = r.generation_tps
        dt = time.perf_counter() - t0
        print(f"    {tag}: {len(ids)} tok in {dt:.2f}s "
              f"prompt_tps={prefill_tps:.0f} gen_tps={decode_tps:.1f}")
        return ids, decode_tps, prefill_tps

    print("  warmup"); os.environ["GDN_FAST"] = "0"
    run(prompt, 64, "warmup")
    results = {}
    for mode in ("0", "1", "0", "1"):
        os.environ["GDN_FAST"] = mode
        key = "fast" if mode == "1" else "base"
        ids, dtps, ptps = run(prompt, tokens, f"short GDN_FAST={mode}")
        results.setdefault(("short", key), []).append((ids, dtps, ptps))
    for mode in ("0", "1"):
        os.environ["GDN_FAST"] = mode
        key = "fast" if mode == "1" else "base"
        ids, dtps, ptps = run(long_prompt, 128, f"long GDN_FAST={mode}")
        results.setdefault(("long", key), []).append((ids, dtps, ptps))
    for ctx in ("short", "long"):
        b = results[(ctx, "base")][0][0]
        f = results[(ctx, "fast")][0][0]
        ident = b == f
        print(f"  {ctx}: T=0 tokens identical: {ident}"
              + ("" if ident else
                 f"  (diverge at {next(i for i,(x,y) in enumerate(zip(b,f)) if x!=y)})"))

if __name__ == "__main__":
    if sys.argv[1] == "phase1":
        phase1()
    else:
        phase2(sys.argv[2] if len(sys.argv) > 2
               else "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ")
