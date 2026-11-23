# SPDX-License-Identifier: Apache-2.0
"""Opt-in MoE single-token decode kernels (fork patch #179, issue #111).

Two Metal kernels from open mlx-lm PRs, ported fork-side and armed ONLY by
``VLLM_MLX_MOE_DECODE_KERNELS=1`` (default OFF):

- mlx-lm #2054 (pierreablin @ ``04ecae9``, ``kernels/softmax_topk.py``): the
  router softmax + top-k (+ renormalisation) in one kernel.
- mlx-lm #2053 (pierreablin @ ``8467a54``, ``kernels/switch_glu.py``): one
  token's expert MLP (gate/up GLU, then down) and the routing-weighted sum
  over its experts in two kernels.

The Metal source below is upstream's (Copyright (c) 2026 Apple Inc., MIT
License, as mlx-lm). The one edit is the gate/up kernel's row addressing:
two template ints, ``ES`` (rows per expert in the tensor) and ``GO`` (the gate
rows' offset inside an expert), let the same kernel read either upstream's
separate gate/up tensors (``ES=N, GO=0`` -- the same arithmetic as upstream)
or patch #96's fused ``[up; gate]`` tensor (``ES=2N, GO=N``). So with #96
armed the kernels read #96's tensor and no second copy of the expert weights
exists (verified bit-exact against the separate layout in the tests and in the
#111 bench).

**T=0 output changes when this is armed** (measured on Qwen3.6-35B-A3B,
first difference at token 73 of 256): #2054 picks the top-k from the fp32
logits where stock runs ``argpartition`` over the bf16-rounded softmax, so a
bf16 tie at the k-th place can pick a different expert; #2053's fused GLU and
weighted sum round differently from ``gather_qmm`` (bf16, ~1 % relative).
Both are deterministic. That is why this is opt-in and not on any route.

Scope (everything else runs the stock mlx-lm code, byte-for-byte):
- Only ``Qwen3NextSparseMoeBlock`` instances (mlx-lm ``qwen3_next``; also the
  MoE block of ``qwen3_5``/``qwen3_5_moe`` text models, which import it), the
  class the #111 bench measured. Text path only (mlx-vlm classes untouched).
- Only a forward of exactly ONE token (``x.size == x.shape[-1]``): batch-1
  decode. Prefill chunks, a co-batched decode (B > 1) and an MTP verify step
  take the stock path. (Upstream #2054 also routes prefill; the fork does
  not.) NOTE: mlx-lm holds the LAST prompt token back and runs it as the
  first one-token step, so when the request is alone that token's KV/SSM
  state, and every generated token's, is kernel-computed; cached snapshots
  that include them (grown sessions, generated spans) differ in rounding
  from an unarmed route's.
- Not training, Metal GPU is the default device, no ``sharding_group``.
- The combine kernel additionally needs #2053's own guards: SwiGLU
  activation, affine 2/3/4/5/6/8-bit, group size 32/64/128, all three
  projections the same bits/group size, scales in the token's dtype
  (bf16/fp16), quantization biases present, no linear bias, no global
  scale, D and N multiples of 16. Failing those, the routing kernel still
  runs and the experts take the stock ``switch_mlp`` path.

Arming is per INSTANCE (the block's ``__class__`` becomes a subclass of the
stock class), never a patch of the mlx-lm class: an unarmed process, or a
second model in the same process, keeps upstream's ``__call__``.

Failure handling (the #125 hook pattern): armed but cannot bind -- the class
is missing, its pinned ``__call__`` body is not the one mirrored here, or the
kernels fail their load-time self-test (routing checked against the stock
top-k, combine against ``gather_qmm``) on the model's own weights, once per
distinct expert quantization config --
logs ONE ERROR naming what is missing; the model then runs with no kernels.
Model load never raises from here.

Memory: no weight copy (separate layout reads the module's own tensors,
fused layout reads #96's). Per armed forward, one K x N activation (top-8 x
512 bf16 = 8 KB on the 35B) and the JIT pipeline state of one compiled
function per (activation, group size, bits, layout).
"""

from __future__ import annotations

import inspect
import logging
import os
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

ENV = "VLLM_MLX_MOE_DECODE_KERNELS"

# Upstream SwiGLU's activation code in the kernel (glu_act<0>: SiLU(g) * u).
_ACT_SWIGLU = 0

# The self-test's bound on the combine kernel's relative error vs the stock
# gather_qmm path. Measured: 1.3 % worst case on the real 35B layers (#111),
# well under 1 % on the tiny test model. A broken kernel is off by O(1).
SELFTEST_REL_TOL = 5e-2

_status = {"armed": False, "blocks": 0, "error": None}


def kernels_enabled() -> bool:
    return os.environ.get(ENV, "").lower() in ("1", "true", "yes")


def status() -> dict:
    return dict(_status, armed=kernels_enabled())


# ------------------------------------------------------------- Metal source
# From mlx-lm #2053 kernels/switch_glu.py @ 8467a54 (MIT, Apple Inc.).

# x[16] as floats and their sum. The weights of a row are a little-endian stream of
# BITS-bit values; x[i] is prescaled by 2^-s, s the offset of value i's lowest bit in
# the byte (uint16 for 4 bits) that qdot16 masks it from, so that the masked weights
# need no shift (as mlx's qmv).
_HEADER = """
template <int BITS>
inline float prescale(int i) {
    if (BITS == 2) {
        return 1.0f / float(1 << (2 * (i % 4)));
    }
    if (BITS == 3) {
        constexpr int s[8] = {0, 3, 6, 1, 4, 7, 2, 5};
        return 1.0f / float(1 << s[i % 8]);
    }
    if (BITS == 4) {
        return 1.0f / float(1 << (4 * (i % 4)));
    }
    if (BITS == 5) {
        constexpr int s[8] = {0, 5, 2, 7, 4, 1, 6, 3};
        return 1.0f / float(1 << s[i % 8]);
    }
    if (BITS == 6) {
        constexpr int s[4] = {0, 6, 4, 2};
        return 1.0f / float(1 << s[i % 4]);
    }
    return 1.0f;
}

template <int BITS, typename U>
inline float load16(const device U* x, thread float* xs) {
    float s = 0.0f;
    for (int i = 0; i < 16; i += 4) {
        float a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
        s += a + b + c + d;
        xs[i] = a * prescale<BITS>(i);
        xs[i + 1] = b * prescale<BITS>(i + 1);
        xs[i + 2] = c * prescale<BITS>(i + 2);
        xs[i + 3] = d * prescale<BITS>(i + 3);
    }
    return s;
}

template <int ACT>
inline float glu_act(float g) {
    if (ACT == 1) {
        return 0.5f * g * (1.0f + metal::precise::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g)));
    }
    return g / (1.0f + metal::exp(-g));
}

template <int BITS>
inline float qdot16(const device uint8_t* w, thread const float* xs) {
    float a = 0.0f;
    if (BITS == 2) {
        for (int i = 0; i < 4; i++) {
            uint8_t q = w[i];
            a += xs[4 * i] * (q & 0x03) + xs[4 * i + 1] * (q & 0x0c) +
                xs[4 * i + 2] * (q & 0x30) + xs[4 * i + 3] * (q & 0xc0);
        }
    } else if (BITS == 3) {
        // Two packs of 8 values in 3 bytes; values 2 and 5 straddle two bytes.
        for (int p = 0; p < 2; p++) {
            auto q = w + 3 * p;
            auto v = xs + 8 * p;
            a += (q[0] & 0x07) * v[0] + (q[0] & 0x38) * v[1] + (q[0] & 0xc0) * v[2] +
                (q[1] & 0x01) * (v[2] * 256.0f) + (q[1] & 0x0e) * v[3] + (q[1] & 0x70) * v[4] +
                (q[1] & 0x80) * v[5] + (q[2] & 0x03) * (v[5] * 256.0f) + (q[2] & 0x1c) * v[6] +
                (q[2] & 0xe0) * v[7];
        }
    } else if (BITS == 5) {
        // Two packs of 8 values in 5 bytes.
        for (int p = 0; p < 2; p++) {
            auto q = w + 5 * p;
            auto v = xs + 8 * p;
            a += (q[0] & 0x1f) * v[0] + (q[0] & 0xe0) * v[1] + (q[1] & 0x03) * (v[1] * 256.0f) +
                (q[1] & 0x7c) * v[2] + (q[1] & 0x80) * v[3] + (q[2] & 0x0f) * (v[3] * 256.0f) +
                (q[2] & 0xf0) * v[4] + (q[3] & 0x01) * (v[4] * 256.0f) + (q[3] & 0x3e) * v[5] +
                (q[3] & 0xc0) * v[6] + (q[4] & 0x07) * (v[6] * 256.0f) + (q[4] & 0xf8) * v[7];
        }
    } else if (BITS == 6) {
        // Four packs of 4 values in 3 bytes.
        for (int p = 0; p < 4; p++) {
            auto q = w + 3 * p;
            auto v = xs + 4 * p;
            a += (q[0] & 0x3f) * v[0] + (q[0] & 0xc0) * v[1] + (q[1] & 0x0f) * (v[1] * 256.0f) +
                (q[1] & 0xf0) * v[2] + (q[2] & 0x03) * (v[2] * 256.0f) + (q[2] & 0xfc) * v[3];
        }
    } else if (BITS == 4) {
        auto ws = (const device uint16_t*)w;
        for (int i = 0; i < 4; i++) {
            uint16_t q = ws[i];
            a += xs[4 * i] * float(q & 0x000f) + xs[4 * i + 1] * float(q & 0x00f0) +
                xs[4 * i + 2] * float(q & 0x0f00) + xs[4 * i + 3] * float(q & 0xf000);
        }
    } else {
        auto ws = (const device uchar4*)w;
        for (int i = 0; i < 4; i++) {
            uchar4 q = ws[i];
            a += xs[4 * i] * q.x + xs[4 * i + 1] * q.y + xs[4 * i + 2] * q.z + xs[4 * i + 3] * q.w;
        }
    }
    return a;
}
"""

# The gate and up rows of the K experts (K * N rows of D values), the GLU, times
# each expert's weight. A threadgroup takes RPS * SPT rows of one expert: RPS rows
# per simdgroup, 16 values per lane.
# Fork edit (#179): an expert's rows start at inds[slot] * ES and its gate rows
# at +GO, so the kernel also reads #96's fused [up; gate] tensor (ES = 2N,
# GO = N, gw == uw). Upstream's layout is ES = N, GO = 0: the same arithmetic.
_GATE_UP_SOURCE = """
    constexpr int ROWS = RPS * SPT;
    constexpr int TPE = (N + ROWS - 1) / ROWS;
    constexpr int RB = D * BITS / 8;
    constexpr int NG = D / GS;
    uint lane = thread_index_in_simdgroup;
    uint slot = threadgroup_position_in_grid.x / TPE;
    int row0 = (threadgroup_position_in_grid.x % TPE) * ROWS + simdgroup_index_in_threadgroup * RPS;
    if (row0 >= N) {
        return;
    }
    size_t base = size_t(inds[slot]) * ES + row0;
    auto gq = (const device uint8_t*)gw;
    auto uq = (const device uint8_t*)uw;
    float ga[RPS] = {0.0f};
    float ua[RPS] = {0.0f};
    float xs[16];
    for (int k0 = lane * 16; k0 < D; k0 += 512) {
        float xsum = load16<BITS>(x + k0, xs);
        for (int r = 0; r < RPS; r++) {
            size_t row = base + r;
            size_t grow = row + GO;
            size_t g = row * NG + k0 / GS;
            size_t gg = grow * NG + k0 / GS;
            ga[r] += float(gs[gg]) * qdot16<BITS>(gq + grow * RB + k0 * BITS / 8, xs) + float(gb[gg]) * xsum;
            ua[r] += float(us[g]) * qdot16<BITS>(uq + row * RB + k0 * BITS / 8, xs) + float(ub[g]) * xsum;
        }
    }
    float we = weights[slot];
    for (int r = 0; r < RPS; r++) {
        float g = simd_sum(ga[r]);
        float u = simd_sum(ua[r]);
        if (lane == 0) {
            h[slot * N + row0 + r] = static_cast<T>(glu_act<ACT>(g) * u * we);
        }
    }
"""

# The down rows of the K experts side by side are one row of K * N values, and h
# (K x N) the matching vector: the weighted sum over the experts is one dot
# product per output row. A lane's 16 values are of one expert (N % 16 == 0).
_DOWN_SOURCE = """
    constexpr int RB = N * BITS / 8;
    constexpr int NG = N / GS;
    uint lane = thread_index_in_simdgroup;
    int row0 = (threadgroup_position_in_grid.x * SPT + simdgroup_index_in_threadgroup) * RPS;
    if (row0 >= D) {
        return;
    }
    auto dq = (const device uint8_t*)dw;
    float acc[RPS] = {0.0f};
    float xs[16];
    for (int j0 = lane * 16; j0 < K * N; j0 += 512) {
        int slot = j0 / N;
        int j = j0 - slot * N;
        float xsum = load16<BITS>(h + j0, xs);
        size_t base = size_t(inds[slot]) * D + row0;
        for (int r = 0; r < RPS; r++) {
            size_t row = base + r;
            size_t g = row * NG + j / GS;
            acc[r] += float(ds[g]) * qdot16<BITS>(dq + row * RB + j * BITS / 8, xs) + float(db[g]) * xsum;
        }
    }
    for (int r = 0; r < RPS; r++) {
        float a = simd_sum(acc[r]);
        if (lane == 0) {
            y[row0 + r] = static_cast<O>(a);
        }
    }
"""

# From mlx-lm #2054 kernels/softmax_topk.py @ 04ecae9 (MIT, Apple Inc.).
# One simdgroup per token. The weights are the softmax over all experts, or, with
# NORM, renormalized over the top K (the softmax of the top K logits).
_SOFTMAX_TOPK_SOURCE = """
    constexpr int R = (E + 31) / 32;
    uint token = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    auto row = logits + size_t(token) * E;
    float v[R];
    float m = -INFINITY;
    for (int i = 0; i < R; i++) {
        int e = lane + 32 * i;
        v[i] = e < E ? static_cast<float>(row[e]) : -INFINITY;
        m = metal::max(m, v[i]);
    }
    m = simd_max(m);
    float z = 0.0f;
    for (int i = 0; i < R; i++) {
        z += metal::precise::exp(v[i] - m);
    }
    z = simd_sum(z);

    float w[K];
    for (int k = 0; k < K; k++) {
        float best = -INFINITY;
        int best_e = E;
        for (int i = 0; i < R; i++) {
            int e = lane + 32 * i;
            if (v[i] > best || (v[i] == best && e < best_e)) {
                best = v[i];
                best_e = e;
            }
        }
        float top = simd_max(best);
        int winner = simd_min(best == top ? best_e : E);
        for (int i = 0; i < R; i++) {
            if (lane + 32 * i == winner) {
                v[i] = -INFINITY;
            }
        }
        w[k] = metal::precise::exp(top - m);
        if (lane == 0) {
            inds[size_t(token) * K + k] = winner;
        }
    }

    float total = z;
    if (NORM) {
        total = 0.0f;
        for (int k = 0; k < K; k++) {
            total += w[k];
        }
    }
    if (lane == 0) {
        for (int k = 0; k < K; k++) {
            weights[size_t(token) * K + k] = static_cast<WT>(w[k] / total);
        }
    }
"""

# Rows per simdgroup and simdgroups per threadgroup (upstream's tuned values).
_GATE_UP_SHAPE = (1, 2)
_DOWN_SHAPE = (1, 4)


@lru_cache(maxsize=None)
def _kernels():
    """Built lazily: importing this module never touches Metal."""
    return {
        "gate_up": mx.fast.metal_kernel(
            name="vllm_mlx_switch_glu_gate_up",
            input_names=["x", "inds", "weights", "gw", "gs", "gb", "uw", "us", "ub"],
            output_names=["h"],
            header=_HEADER,
            source=_GATE_UP_SOURCE,
        ),
        "down": mx.fast.metal_kernel(
            name="vllm_mlx_switch_glu_down_combine",
            input_names=["h", "inds", "dw", "ds", "db"],
            output_names=["y"],
            header=_HEADER,
            source=_DOWN_SOURCE,
        ),
        "softmax_topk": mx.fast.metal_kernel(
            name="vllm_mlx_softmax_topk",
            input_names=["logits"],
            output_names=["inds", "weights"],
            source=_SOFTMAX_TOPK_SOURCE,
        ),
    }


def on_metal() -> bool:
    return mx.metal.is_available() and mx.default_device() == mx.gpu


def softmax_topk(logits: mx.array, *, top_k: int, normalize: bool):
    """#2054: the top ``top_k`` experts of each token and their softmax
    weights (renormalised over the top k if ``normalize``), from the router
    logits, computed in fp32 and returned in the logits' dtype."""
    E = logits.shape[-1]
    return _kernels()["softmax_topk"](
        inputs=[logits],
        template=[("E", E), ("K", top_k), ("NORM", normalize), ("WT", logits.dtype)],
        grid=(logits.size // E * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[logits.shape[:-1] + (top_k,)] * 2,
        output_dtypes=[mx.uint32, logits.dtype],
    )


def _swiglu_class():
    try:
        from mlx_lm.models.switch_layers import SwiGLU

        return SwiGLU
    except Exception:
        return None


def _act_code(activation):
    swiglu = _swiglu_class()
    if swiglu is not None and type(activation) is swiglu:
        return _ACT_SWIGLU
    return None


def _projections(glu):
    """(gate, up, down, fused) for a stock SwitchGLU or a #96 FusedSwitchGLU;
    None for anything else."""
    if hasattr(glu, "gate_up_proj"):
        fp = glu.gate_up_proj
        return fp, fp, glu.down_proj, True
    if hasattr(glu, "gate_proj") and hasattr(glu, "up_proj"):
        return glu.gate_proj, glu.up_proj, glu.down_proj, False
    return None


def can_combine(glu, x: mx.array) -> bool:
    """#2053's ``can_combine`` guards, for either layout."""
    projs = _projections(glu)
    if projs is None:
        return False
    gp, up, dp, fused = projs
    if not (
        on_metal()
        and x.size == x.shape[-1]
        and x.dtype in (mx.bfloat16, mx.float16)
        and not getattr(glu, "training", False)
        and _act_code(getattr(glu, "activation", None)) is not None
    ):
        return False
    bits = getattr(gp, "bits", None)
    group_size = getattr(gp, "group_size", None)
    if bits not in (2, 3, 4, 5, 6, 8) or group_size not in (32, 64, 128):
        return False
    for p in (gp, up, dp):
        if not (
            getattr(p, "mode", None) == "affine"
            and getattr(p, "bits", None) == bits
            and getattr(p, "group_size", None) == group_size
            and getattr(p, "weight", None) is not None
            and p.weight.ndim == 3
            and p.scales.dtype == x.dtype
            and p.get("biases") is not None
            and "bias" not in p
            and getattr(p, "global_scale", None) is None
        ):
            return False
    n_rows = gp.weight.shape[1] // 2 if fused else gp.weight.shape[1]
    if fused and gp.weight.shape[1] % 2:
        return False
    return x.shape[-1] % 16 == 0 and n_rows % 16 == 0


def _combine(act, group_size, bits, fused):
    kern = _kernels()

    def combine(x, inds, weights, gw, gs, gb, uw, us, ub, dw, ds, db):
        K, D = inds.size, x.shape[-1]
        N = gw.shape[1] // 2 if fused else gw.shape[1]
        inds = inds.reshape(-1).astype(mx.uint32)
        weights = weights.reshape(-1)
        out_dtype = mx.result_type(x.dtype, weights.dtype)
        template = [
            ("T", x.dtype),
            ("O", out_dtype),
            ("ACT", act),
            ("D", D),
            ("N", N),
            ("K", K),
            ("GS", group_size),
            ("BITS", bits),
        ]
        rps, spt = _GATE_UP_SHAPE
        tpe = -(-N // (rps * spt))
        h = kern["gate_up"](
            inputs=[x, inds, weights, gw, gs, gb, uw, us, ub],
            template=template
            + [("RPS", rps), ("SPT", spt)]
            + [("ES", 2 * N if fused else N), ("GO", N if fused else 0)],
            grid=(K * tpe * 32 * spt, 1, 1),
            threadgroup=(32 * spt, 1, 1),
            output_shapes=[(K, N)],
            output_dtypes=[x.dtype],
        )[0]
        rps, spt = _DOWN_SHAPE
        return kern["down"](
            inputs=[h, inds, dw, ds, db],
            template=template + [("RPS", rps), ("SPT", spt)],
            grid=(-(-D // (rps * spt)) * 32 * spt, 1, 1),
            threadgroup=(32 * spt, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[out_dtype],
        )[0]

    return combine


@lru_cache(maxsize=None)
def _combine_fn(act, group_size, bits, fused):
    # Compiled as upstream: one function per configuration, shared by layers.
    return mx.compile(_combine(act, group_size, bits, fused))


def switch_glu_combine(glu, x, inds, weights, compiled=True):
    """``(glu(x, inds) * weights[..., None]).sum(-2)`` for ONE token, in two
    kernels. Needs ``can_combine(glu, x)``."""
    gp, up, dp, fused = _projections(glu)
    key = (_act_code(glu.activation), gp.group_size, gp.bits, fused)
    fn = _combine_fn(*key) if compiled else _combine(*key)
    return fn(
        x, inds, weights,
        gp.weight, gp.scales, gp.biases,
        up.weight, up.scales, up.biases,
        dp.weight, dp.scales, dp.biases,
    )  # fmt: skip


# ------------------------------------------------------------ block binding

# The pinned stock ``Qwen3NextSparseMoeBlock.__call__`` lines the armed
# forward reproduces (routing it replaces, the shared expert it keeps). If
# any is gone the stock body changed under us: refuse to arm (ERROR).
_QWEN3_NEXT_NEEDLES = (
    "gates = self.gate(x)",
    "gates = mx.softmax(gates, axis=-1, precise=True)",
    "inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]",
    "scores = mx.take_along_axis(gates, inds, axis=-1)",
    "if self.norm_topk_prob:",
    "scores = scores / scores.sum(axis=-1, keepdims=True)",
    "y = self.switch_mlp(x, inds)",
    "y = (y * scores[..., None]).sum(axis=-2)",
    "shared_y = self.shared_expert(x)",
    "shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y",
)

_TARGETS = (
    ("mlx_lm.models.qwen3_next", "Qwen3NextSparseMoeBlock", _QWEN3_NEXT_NEEDLES),
)


def _eligible(block, x) -> bool:
    return (
        x.size == x.shape[-1]
        and getattr(block, "sharding_group", None) is None
        and not block.training
        and on_metal()
    )


def _make_armed_class(stock_cls):
    stock_call = stock_cls.__call__

    def __call__(self, x):
        if not _eligible(self, x):
            return stock_call(self, x)
        gates = self.gate(x)
        inds, scores = softmax_topk(
            gates, top_k=self.top_k, normalize=self.norm_topk_prob
        )
        if can_combine(self.switch_mlp, x):
            y = switch_glu_combine(self.switch_mlp, x, inds, scores)
        else:
            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y

    return type(
        stock_cls.__name__,
        (stock_cls,),
        {
            "__call__": __call__,
            "__module__": __name__,
            "__qualname__": stock_cls.__qualname__,
            "_vllm_mlx_moe_kernels": True,
            "_vllm_mlx_stock_class": stock_cls,
        },
    )


def _resolve_targets():
    """[(stock_cls, armed_cls)] and a list of errors naming what is missing."""
    import importlib

    found, errors = [], []
    if _swiglu_class() is None:
        errors.append("mlx_lm.models.switch_layers.SwiGLU not found")
    for mod_name, cls_name, needles in _TARGETS:
        where = f"{mod_name}.{cls_name}"
        try:
            cls = getattr(importlib.import_module(mod_name), cls_name)
        except Exception as e:
            errors.append(f"{where} not found ({type(e).__name__})")
            continue
        try:
            src = inspect.getsource(cls.__call__)
        except Exception as e:
            errors.append(f"{where}.__call__ source unreadable ({type(e).__name__})")
            continue
        gone = [n for n in needles if n not in src]
        if gone:
            hint = (
                " (this mlx-lm already routes through upstream kernels: the "
                "flag is redundant here)"
                if ("softmax_topk" in src or ".combine(" in src)
                else ""
            )
            errors.append(
                f"{where}.__call__ is not the body #179 mirrors; missing "
                f"{gone[0]!r}{hint}"
            )
            continue
        found.append((cls, _make_armed_class(cls)))
    return found, errors


def _walk_modules(model):
    stack, seen = [model], set()
    while stack:
        mod = stack.pop()
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        yield mod
        children = getattr(mod, "children", None)
        if not callable(children):
            continue
        for child in children().values():
            # nn.Module is itself a dict: test for it before plain dicts.
            if isinstance(child, nn.Module):
                stack.append(child)
            elif isinstance(child, (list, tuple)):
                stack.extend(child)
            elif isinstance(child, dict):
                stack.extend(child.values())
            else:
                stack.append(child)


def _config_key(block):
    """What selects a kernel specialisation for ``block``'s experts."""

    def q(p):
        scales = getattr(p, "scales", None)
        return (
            getattr(p, "bits", None),
            getattr(p, "group_size", None),
            getattr(p, "mode", None),
            None if scales is None else scales.dtype,
        )

    projs = _projections(block.switch_mlp)
    if projs is None:
        return (type(block.switch_mlp).__name__,)
    gp, up, dp, fused = projs
    gate = getattr(block, "gate", None)
    return (fused, q(gp), q(up), q(dp), getattr(gate, "bits", None))


def _selftest(block) -> str | None:
    """Run both kernels once on ``block``'s real weights against the stock
    ops. Returns an error string, or None when they bind and agree."""
    try:
        down = block.switch_mlp.down_proj
        n_exp, dim = down.weight.shape[0], down.weight.shape[1]
        dtype = getattr(down, "scales", down.weight).dtype
        key = mx.random.key(179)
        x = (mx.random.normal((1, 1, dim), key=key) * 0.5).astype(dtype)
        logits = block.gate(x)
        inds, scores = softmax_topk(
            logits, top_k=block.top_k, normalize=block.norm_topk_prob
        )
        mx.eval(inds, scores)
        if not int(inds.max().item()) < n_exp:
            return "softmax_topk self-test returned an out-of-range expert"
        # Routing against stock, so a wrong-but-in-range routing cannot pass
        # (the combine check below reuses these inds and would not see it).
        # The kernel takes the top-k LOGITS (ties by index): the chosen
        # logits must be exactly the k largest, each expert once.
        k = block.top_k
        flat = [int(i) for i in inds.reshape(-1).tolist()]
        if len(set(flat)) != k:
            return "softmax_topk self-test picked an expert twice"
        g32 = logits.astype(mx.float32)
        chosen = mx.sort(mx.take_along_axis(g32, inds.astype(mx.int32), axis=-1))
        if not mx.array_equal(chosen, mx.sort(g32, axis=-1)[..., -k:]).item():
            return "softmax_topk self-test did not pick the top-k logits"
        p = mx.take_along_axis(
            mx.softmax(g32, axis=-1, precise=True), inds.astype(mx.int32), axis=-1
        )
        if block.norm_topk_prob:
            p = p / p.sum(axis=-1, keepdims=True)
        w_err = mx.abs(scores.astype(mx.float32) - p).max().item()
        if not w_err <= 1e-2:  # bf16 rounding of a probability <= 1
            return f"softmax_topk self-test weights off by {w_err:.3g}"
        if not can_combine(block.switch_mlp, x):
            return None  # routing kernel only; experts stay on gather_qmm
        ref = (block.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)
        got = switch_glu_combine(block.switch_mlp, x, inds, scores, compiled=False)
        ref32, got32 = ref.astype(mx.float32), got.astype(mx.float32)
        scale = mx.abs(ref32).max().item() + 1e-6
        err = mx.abs(got32 - ref32).max().item() / scale
        if not err <= SELFTEST_REL_TOL:
            return f"switch_glu_combine self-test relative error {err:.3g}"
        return None
    except Exception as e:
        return f"kernel self-test raised {type(e).__name__}: {e}"


def install_decode_kernels(model) -> int:
    """Arm every eligible MoE block under ``model``. Returns the number
    armed. Never raises: any failure leaves the model on the stock path."""
    try:
        if not on_metal():
            return _refuse("Metal GPU is not the default device")
        targets, errors = _resolve_targets()
        if errors:
            return _refuse("; ".join(errors))
        blocks = [
            (m, armed)
            for m in _walk_modules(model)
            for stock, armed in targets
            if type(m) is stock
        ]
        if not blocks:
            _status.update(blocks=0, error=None)
            logger.info("[moe-kernels] %s armed; no supported MoE block found", ENV)
            return 0
        # One self-test per distinct expert configuration, not only the first
        # block: a mixed-precision quant can give later layers other bits /
        # group sizes, i.e. another kernel specialisation.
        tested = set()
        for m, _ in blocks:
            key = _config_key(m)
            if key in tested:
                continue
            tested.add(key)
            err = _selftest(m)
            if err:
                return _refuse(err)
        done = []
        try:
            for m, armed in blocks:
                stock = type(m)
                m.__class__ = armed
                done.append((m, stock))
        except Exception:
            for m, stock in done:  # all or nothing
                m.__class__ = stock
            raise
    except Exception as e:
        return _refuse(f"install raised {type(e).__name__}: {e}")
    _status.update(blocks=len(blocks), error=None)
    logger.warning(
        "[moe-kernels] %s armed: %d MoE block(s) run the single-token decode "
        "kernels (mlx-lm #2053/#2054 port). T=0 output differs from the "
        "unarmed route.",
        ENV,
        len(blocks),
    )
    return len(blocks)


def _refuse(error: str) -> int:
    _status.update(blocks=0, error=error)
    logger.error(
        "[moe-kernels] %s is armed but the kernels cannot bind; the model "
        "runs WITHOUT them (stock MoE path). Missing: %s",
        ENV,
        error,
    )
    return 0


def maybe_install(model, model_name: str = "") -> int:
    """Entry point (via ``moe_fusion.maybe_fuse``): a no-op unless armed."""
    if not kernels_enabled():
        return 0
    n = install_decode_kernels(model)
    if n:
        logger.info("[moe-kernels] %s: %d MoE block(s) armed", model_name, n)
    return n
