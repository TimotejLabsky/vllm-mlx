#!/usr/bin/env python
"""Kernel probe for ledger lever #42 at 8-bit: dequantize-then-matmul vs
quantized_matmul at prefill-chunk shapes (Qwen3.8-27B, group 64).

#42 (2026-10-05) measured the 27B-4bit and found +3-4 % of prefill, below
the 1.15x gate. The 27B-8bit route was never probed. Random weights of the
real shapes; per projection: qmm, dequantize alone, dequantize + bf16 mm.
Weighted by how often each projection runs in one forward of a chunk.

    python scripts/fork/probe_dequant_prefill.py [M] [bits]
"""

import statistics
import sys
import time

import mlx.core as mx

M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
BITS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
GROUP = 64
H = 5120
# (name, in, out, layers that run it) - Qwen3.8-27B: 64 layers, every 4th
# full attention (24 q heads x 256 with output gate, 4 kv heads), the rest
# GatedDeltaNet (16 k heads x 128, 48 v heads x 128).
SHAPES = [
    ("mlp.gate", H, 17408, 64),
    ("mlp.up", H, 17408, 64),
    ("mlp.down", 17408, H, 64),
    ("attn.q+gate", H, 24 * 256 * 2, 16),
    ("attn.k", H, 4 * 256, 16),
    ("attn.v", H, 4 * 256, 16),
    ("attn.o", 24 * 256, H, 16),
    ("gdn.qkv", H, 2 * 16 * 128 + 48 * 128, 48),
    ("gdn.z", H, 48 * 128, 48),
    ("gdn.out", 48 * 128, H, 48),
]


def timeit(fn, n=15):
    for _ in range(3):
        mx.eval(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def main():
    x = mx.random.normal((1, M, H)).astype(mx.bfloat16)
    tot_q = tot_d = tot_deq = 0.0
    print(f"M={M} bits={BITS} group={GROUP} mlx={mx.__version__}")
    print(f"{'proj':<12}{'n':>4}{'qmm ms':>9}{'deq ms':>9}{'deq+mm':>9}{'ratio':>7}")
    for name, din, dout, n in SHAPES:
        w = (mx.random.normal((dout, din)) * 0.02).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=GROUP, bits=BITS)
        xi = x if din == H else mx.random.normal((1, M, din)).astype(mx.bfloat16)
        mx.eval(wq, s, b, xi)
        del w
        q = timeit(lambda: mx.quantized_matmul(xi, wq, s, b, transpose=True,
                                               group_size=GROUP, bits=BITS))  # fmt: skip
        dq = timeit(lambda: mx.dequantize(wq, s, b, group_size=GROUP, bits=BITS))
        d = timeit(lambda: xi @ mx.dequantize(wq, s, b, group_size=GROUP, bits=BITS).T)
        tot_q += q * n
        tot_d += d * n
        tot_deq += dq * n
        print(f"{name:<12}{n:>4}{q:>9.2f}{dq:>9.2f}{d:>9.2f}{q / d:>7.2f}")
    print(f"per chunk (all layers): qmm {tot_q / 1000:.2f} s, deq+mm {tot_d / 1000:.2f} s "
          f"(dequant alone {tot_deq / 1000:.2f} s), speedup {tot_q / tot_d:.3f}x, "
          f"saves {(tot_q - tot_d) / 1000:.2f} s per {M}-token chunk")  # fmt: skip


if __name__ == "__main__":
    main()
