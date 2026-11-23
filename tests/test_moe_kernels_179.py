# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in MoE single-token decode kernels (fork patch #179).

Everything here runs the real Metal kernels on a real tiny quantized
Qwen3.5-MoE text model (mlx-lm ``qwen3_5``, whose MoE block is
``Qwen3NextSparseMoeBlock`` -- the class the 35B-A3B runs). No mocks of
mlx or mlx-lm.

What is pinned:
- flag OFF: the model is untouched and its output is byte-identical to the
  path without #179 (with and without #96 fusion armed);
- flag ON: close to the stock path within a stated tolerance, NOT equal
  (that is the documented T=0 change), and the fused #96 layout gives
  exactly the separate layout's numbers (no second weight copy needed);
- the scope guards: prefill and co-batched decode stay byte-identical;
- why expert selection can change: a bf16 probability tie at the k-th place;
- armed-but-cannot-bind: one ERROR naming what is missing, model unchanged.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import qwen3_5 as q35
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from vllm_mlx import moe_kernels
from vllm_mlx.moe_fusion import FusedSwitchGLU, fuse_moe_gate_up, maybe_fuse

pytestmark = pytest.mark.skipif(
    not moe_kernels.on_metal(), reason="the kernels are Metal-only"
)

DIM, VOCAB, TOPK = 128, 96, 4
PROMPT = [5, 9, 17, 23, 31, 42, 7]
DECODE_STEPS = 8

# Flag-on tolerances (max abs error / max abs reference). Measured on this
# tiny model (laptop, mlx 0.32.4): combine kernel 1.1 %, whole block 1.5 %,
# first decode step's logits 2.1 %; the real 35B layers measured 1.3 % at
# the combine (#111 self-test). A broken kernel is off by O(1).
BLOCK_REL_TOL = 3e-2
LOGIT_REL_TOL = 5e-2
ROUTER_SPREAD = 8.0


def _build_model(seed=179):
    mx.random.seed(seed)
    args = q35.TextModelArgs.from_dict(
        {
            "model_type": "qwen3_5_moe",
            "hidden_size": DIM,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": VOCAB,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "full_attention_interval": 2,
            "num_experts": 16,
            "num_experts_per_tok": TOPK,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
        }
    )
    m = q35.TextModel(args)
    # Default init gives a near-uniform router (16 experts at ~1/16 each),
    # where bf16 probability ties are everywhere. Spread the router logits
    # so expert selection is mostly unambiguous, as on a trained model.
    for layer in m.layers:
        layer.mlp.gate.weight = layer.mlp.gate.weight * ROUTER_SPREAD
    m.set_dtype(mx.bfloat16)
    nn.quantize(m, group_size=64, bits=4)
    m.train(False)
    mx.eval(m.parameters())
    return m


def _blocks(model):
    return [layer.mlp for layer in model.layers]


def _greedy(model):
    """Prefill PROMPT, then DECODE_STEPS single-token steps; returns the
    tokens and every step's logits (the decode steps are the armed shape)."""
    cache = model.make_cache()
    logits = model(mx.array([PROMPT]), cache=cache)[:, -1, :]
    out, steps = [], [logits]
    for _ in range(DECODE_STEPS):
        tok = int(mx.argmax(logits, axis=-1).item())
        out.append(tok)
        logits = model(mx.array([[tok]]), cache=cache)[:, -1, :]
        steps.append(logits)
    mx.eval(steps)
    return out, steps


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(moe_kernels.ENV, raising=False)
    monkeypatch.delenv("VLLM_MLX_MOE_GATEUP_FUSION", raising=False)


def _x(shape=(1, 1, DIM), seed=0):
    return (mx.random.normal(shape, key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)


def _rel_err(got, ref):
    ref32 = ref.astype(mx.float32)
    diff = mx.abs(got.astype(mx.float32) - ref32).max().item()
    return diff / (mx.abs(ref32).max().item() + 1e-6)


# ----------------------------------------------------------- flag OFF


@pytest.mark.parametrize("fusion", [False, True], ids=["stock", "with-96"])
def test_flag_off_is_byte_identical_to_the_unpatched_path(monkeypatch, fusion):
    """The reference is what the engines ran before #179: nothing, or #96's
    ``fuse_moe_gate_up`` alone. ``maybe_fuse`` with the kernel flag unset
    must give the same tokens and the same logits bit for bit."""
    ref_model = _build_model()
    if fusion:
        assert fuse_moe_gate_up(ref_model) == 2
    ref_tokens, ref_logits = _greedy(ref_model)

    if fusion:
        monkeypatch.setenv("VLLM_MLX_MOE_GATEUP_FUSION", "1")
    model = _build_model()
    maybe_fuse(model, "tiny")
    assert all(type(b) is Qwen3NextSparseMoeBlock for b in _blocks(model))
    tokens, logits = _greedy(model)

    assert tokens == ref_tokens
    for a, b in zip(logits, ref_logits):
        assert mx.array_equal(a, b).item()


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_flag_values_that_do_not_arm(monkeypatch, value):
    monkeypatch.setenv(moe_kernels.ENV, value)
    model = _build_model()
    assert moe_kernels.maybe_install(model) == 0
    assert all(type(b) is Qwen3NextSparseMoeBlock for b in _blocks(model))


# ------------------------------------------------------------ flag ON


def _stock_routing(block, x):
    p = mx.softmax(block.gate(x), axis=-1, precise=True)
    inds = mx.argpartition(p, kth=-block.top_k, axis=-1)[..., -block.top_k :]
    sc = mx.take_along_axis(p, inds, axis=-1)
    return inds, sc / sc.sum(axis=-1, keepdims=True)


def test_combine_kernel_close_to_gather_qmm_on_the_same_routing():
    """#2053 alone: same experts and weights in, stock vs kernel out."""
    model = _build_model()
    worst, differs = 0.0, False
    for seed in range(16):
        x = _x(seed=seed)
        for b in _blocks(model):
            assert moe_kernels.can_combine(b.switch_mlp, x)
            inds, sc = _stock_routing(b, x)
            want = (b.switch_mlp(x, inds) * sc[..., None]).sum(axis=-2)
            got = moe_kernels.switch_glu_combine(b.switch_mlp, x, inds, sc)
            worst = max(worst, _rel_err(got, want))
            differs |= not mx.array_equal(got, want).item()
    assert worst <= BLOCK_REL_TOL, worst
    assert differs, "kernel output bit-equal to gather_qmm: did it run?"


def test_armed_blocks_are_close_to_stock_not_equal(monkeypatch):
    """Whole block (#2054 routing + #2053 experts). Where the two routings
    pick the same expert set the output is within BLOCK_REL_TOL; where they
    do not (a bf16 tie, see the tie test) it is a different mixture and is
    only counted."""
    ref = _build_model()
    model = _build_model()
    monkeypatch.setenv(moe_kernels.ENV, "1")
    assert moe_kernels.maybe_install(model) == 2
    assert moe_kernels.status()["blocks"] == 2
    assert all(getattr(b, "_vllm_mlx_moe_kernels", False) for b in _blocks(model))
    # isinstance still holds for code that checks the stock class
    assert all(isinstance(b, Qwen3NextSparseMoeBlock) for b in _blocks(model))

    worst, compared, differs = 0.0, 0, False
    for seed in range(32):
        x = _x(seed=seed)
        for b_ref, b in zip(_blocks(ref), _blocks(model)):
            s_inds, _ = _stock_routing(b_ref, x)
            k_inds, _ = moe_kernels.softmax_topk(
                b.gate(x), top_k=b.top_k, normalize=b.norm_topk_prob
            )
            want, got = b_ref(x), b(x)
            differs |= not mx.array_equal(got, want).item()
            if sorted(s_inds.reshape(-1).tolist()) != sorted(
                k_inds.reshape(-1).tolist()
            ):
                continue
            compared += 1
            worst = max(worst, _rel_err(got, want))
    assert compared >= 48, compared  # of 64: ties stay the exception
    assert worst <= BLOCK_REL_TOL, worst
    assert differs, "armed output equal to stock: the kernels did not run"


def test_armed_model_logits_stay_close(monkeypatch):
    _, ref_logits = _greedy(_build_model())
    model = _build_model()
    monkeypatch.setenv(moe_kernels.ENV, "1")
    assert moe_kernels.maybe_install(model) == 2
    _, logits = _greedy(model)
    # Step 0 is the prefill (stock path by the scope guard): exact.
    assert mx.array_equal(logits[0], ref_logits[0]).item()
    # A decode step can follow a different greedy token once outputs drift,
    # so compare the first decode step, which shares its whole history.
    assert _rel_err(logits[1], ref_logits[1]) <= LOGIT_REL_TOL


def test_fused_layout_matches_separate_layout_exactly(monkeypatch):
    """#96 + #179: the gate/up kernel reads #96's [up; gate] tensor (ES=2N,
    GO=N). Same arithmetic as reading the separate tensors, so the outputs
    are bit-identical, and no second copy of the expert weights exists."""
    sep = _build_model()
    fused = _build_model()
    assert fuse_moe_gate_up(fused) == 2
    monkeypatch.setenv(moe_kernels.ENV, "1")
    assert moe_kernels.maybe_install(sep) == 2
    assert moe_kernels.maybe_install(fused) == 2
    for b in _blocks(fused):
        assert isinstance(b.switch_mlp, FusedSwitchGLU)
        assert not hasattr(b.switch_mlp, "gate_proj")
    for seed in range(8):
        x = _x(seed=seed)
        for a, b in zip(_blocks(sep), _blocks(fused)):
            assert mx.array_equal(a(x), b(x)).item()


def test_maybe_fuse_arms_kernels_after_fusion(monkeypatch):
    monkeypatch.setenv("VLLM_MLX_MOE_GATEUP_FUSION", "1")
    monkeypatch.setenv(moe_kernels.ENV, "1")
    model = _build_model()
    assert maybe_fuse(model, "tiny") == 2
    for b in _blocks(model):
        assert isinstance(b.switch_mlp, FusedSwitchGLU)
        assert getattr(b, "_vllm_mlx_moe_kernels", False)


# --------------------------------------------------------- scope guards


@pytest.mark.parametrize(
    "shape", [(1, 5, DIM), (2, 1, DIM), (3, 4, DIM)], ids=["prefill", "B2", "B3T4"]
)
def test_more_than_one_token_takes_the_stock_path_exactly(monkeypatch, shape):
    ref = _build_model()
    model = _build_model()
    monkeypatch.setenv(moe_kernels.ENV, "1")
    assert moe_kernels.maybe_install(model) == 2
    x = _x(shape)
    for b_ref, b in zip(_blocks(ref), _blocks(model)):
        assert mx.array_equal(b_ref(x), b(x)).item()


def test_unsupported_experts_fall_back_to_stock_experts(monkeypatch):
    """fp32 activations fail #2053's dtype guard: the experts run the stock
    gather_qmm path, only routing uses #2054."""
    model = _build_model()
    monkeypatch.setenv(moe_kernels.ENV, "1")
    assert moe_kernels.maybe_install(model) == 2
    x = _x().astype(mx.float32)
    b = _blocks(model)[0]
    assert not moe_kernels.can_combine(b.switch_mlp, x)
    mx.eval(b(x))  # runs, no kernel guard tripped


def test_arming_never_patches_the_mlx_lm_class(monkeypatch):
    stock_call = Qwen3NextSparseMoeBlock.__call__
    other = _build_model()
    monkeypatch.setenv(moe_kernels.ENV, "1")
    assert moe_kernels.maybe_install(_build_model()) == 2
    assert Qwen3NextSparseMoeBlock.__call__ is stock_call
    assert all(type(b) is Qwen3NextSparseMoeBlock for b in _blocks(other))


# ------------------------------------------- why expert selection changes


@pytest.mark.parametrize("pair", [(1, 2), (5, 6)])
def test_bf16_probability_tie_can_pick_a_different_expert_set(pair):
    """Stock routes from ``argpartition`` over the bf16-ROUNDED softmax;
    #2054 picks from the fp32 logits. Two logits one bf16 step apart
    (0.25 and 0.25 + 2^-9) round to the same bf16 probability, so stock
    sees a tie at the k-th place and its choice cannot depend on which
    logit is larger; the kernel always takes the larger logit. Mirror the
    pair: stock sees two IDENTICAL probability vectors and returns the same
    set for both, the kernel returns two different sets -- so at least one
    of the two disagrees with stock. (On the real 35B layers this happened
    in 17 of 160 random single-token cases, #111.)"""
    lo, hi = 0.25, 0.25 + 2**-9
    i, j = pair
    sets_stock, sets_kernel = [], []
    for a, b in ((lo, hi), (hi, lo)):
        vals = [2.0] + [-1.0] * 7
        vals[i], vals[j] = a, b
        g = mx.array([[vals]]).astype(mx.bfloat16)
        p = mx.softmax(g, axis=-1, precise=True)
        assert p[0, 0, i].item() == p[0, 0, j].item(), "no bf16 tie: retune"
        inds = mx.argpartition(p, kth=-2, axis=-1)[..., -2:]
        sets_stock.append(sorted(inds.reshape(-1).tolist()))
        ki, _ = moe_kernels.softmax_topk(g, top_k=2, normalize=True)
        sets_kernel.append(sorted(ki.reshape(-1).tolist()))
    larger_first = [0, j], [0, i]
    assert sets_kernel == list(larger_first)
    assert sets_stock[0] == sets_stock[1]
    assert sets_stock != sets_kernel


def test_routing_weights_close_when_the_sets_agree():
    worst = 0.0
    for seed in range(64):
        g = (mx.random.normal((1, 1, 16), key=mx.random.key(seed)) * 2).astype(
            mx.bfloat16
        )
        p = mx.softmax(g, axis=-1, precise=True)
        inds = mx.argpartition(p, kth=-TOPK, axis=-1)[..., -TOPK:]
        sc = mx.take_along_axis(p, inds, axis=-1)
        sc = sc / sc.sum(axis=-1, keepdims=True)
        ki, kw = moe_kernels.softmax_topk(g, top_k=TOPK, normalize=True)
        a = dict(zip(inds.reshape(-1).tolist(), sc.reshape(-1).tolist()))
        b = dict(zip(ki.reshape(-1).tolist(), kw.reshape(-1).tolist()))
        if set(a) != set(b):
            continue
        worst = max(worst, max(abs(a[e] - b[e]) for e in a))
    assert worst <= 1e-2  # bf16 rounding of a probability <= 1


# ---------------------------------------------- armed but cannot bind


def _assert_refused(caplog, model, needle):
    with caplog.at_level(logging.ERROR, logger="vllm_mlx.moe_kernels"):
        assert moe_kernels.maybe_install(model, "tiny") == 0
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1, errors
    assert needle in errors[0].getMessage()
    assert moe_kernels.ENV in errors[0].getMessage()
    assert needle in moe_kernels.status()["error"]
    assert all(type(b) is Qwen3NextSparseMoeBlock for b in _blocks(model))
    # and the model still runs, on the stock path
    mx.eval(model(mx.array([PROMPT])))


def test_missing_class_is_an_error_naming_it(monkeypatch, caplog):
    monkeypatch.setenv(moe_kernels.ENV, "1")
    monkeypatch.setattr(
        moe_kernels,
        "_TARGETS",
        (("mlx_lm.models.qwen3_next_gone", "Qwen3NextSparseMoeBlock", ()),),
    )
    _assert_refused(
        caplog, _build_model(), "mlx_lm.models.qwen3_next_gone.Qwen3NextSparseMoeBlock"
    )


def test_changed_stock_body_is_an_error_naming_the_line(monkeypatch, caplog):
    monkeypatch.setenv(moe_kernels.ENV, "1")
    mod, cls, needles = moe_kernels._TARGETS[0]
    monkeypatch.setattr(
        moe_kernels, "_TARGETS", ((mod, cls, needles + ("y = self.new_router(x)",)),)
    )
    _assert_refused(caplog, _build_model(), "y = self.new_router(x)")


def test_failed_self_test_is_an_error_and_nothing_arms(monkeypatch, caplog):
    monkeypatch.setenv(moe_kernels.ENV, "1")
    monkeypatch.setattr(moe_kernels, "SELFTEST_REL_TOL", -1.0)
    _assert_refused(caplog, _build_model(), "self-test relative error")


def test_kernel_raising_at_load_never_crashes_load(monkeypatch, caplog):
    def boom(*a, **k):
        raise RuntimeError("metal compile failed")

    monkeypatch.setenv(moe_kernels.ENV, "1")
    monkeypatch.setattr(moe_kernels, "softmax_topk", boom)
    _assert_refused(caplog, _build_model(), "metal compile failed")


def test_wrong_but_in_range_routing_fails_the_self_test(monkeypatch, caplog):
    """Review: a routing kernel that returns valid expert ids but the WRONG
    ones (here: the k smallest logits) must not pass the self-test. The
    combine check reuses the kernel's inds, so only a stock comparison of
    the routing itself catches this."""
    real = moe_kernels.softmax_topk

    def bottom_k(logits, *, top_k, normalize):
        _, w = real(logits, top_k=top_k, normalize=normalize)
        inds = mx.argpartition(logits, kth=top_k - 1, axis=-1)[..., :top_k]
        return inds.astype(mx.uint32), w

    monkeypatch.setenv(moe_kernels.ENV, "1")
    monkeypatch.setattr(moe_kernels, "softmax_topk", bottom_k)
    _assert_refused(caplog, _build_model(), "did not pick the top-k logits")


def test_self_test_runs_once_per_expert_config(monkeypatch):
    """Review: a mixed-precision model (layer 1's experts at 8 bits) needs a
    second kernel specialisation, so it gets its own self-test."""
    model = _build_model()
    seen = []
    real = moe_kernels._selftest

    def spy(block):
        seen.append(block.switch_mlp.down_proj.bits)
        return real(block)

    monkeypatch.setenv(moe_kernels.ENV, "1")
    monkeypatch.setattr(moe_kernels, "_selftest", spy)
    assert moe_kernels.maybe_install(model) == 2
    assert seen == [4]  # both layers 4-bit: one test

    mixed = _build_model()
    glu = mixed.layers[1].mlp.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        p = getattr(glu, name)
        w = mx.dequantize(
            p.weight, p.scales, p.biases, group_size=p.group_size, bits=p.bits
        )
        p.weight, p.scales, p.biases = mx.quantize(w, group_size=64, bits=8)
        p.bits = 8
    mx.eval(mixed.parameters())
    seen.clear()
    assert moe_kernels.maybe_install(mixed) == 2
    assert sorted(seen) == [4, 8]
    mx.eval(mixed(mx.array([[5]]), cache=mixed.make_cache()))


def test_no_moe_block_is_not_an_error(monkeypatch, caplog):
    monkeypatch.setenv(moe_kernels.ENV, "1")
    with caplog.at_level(logging.ERROR, logger="vllm_mlx.moe_kernels"):
        assert moe_kernels.maybe_install(nn.Linear(4, 4)) == 0
    assert not [r for r in caplog.records if r.levelno == logging.ERROR]
