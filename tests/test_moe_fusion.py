# SPDX-License-Identifier: Apache-2.0
"""Tests for MoE gate/up fusion (fork patch #96).

The fusion concatenates gate/up quantized weights along the output-row axis
so the expert MLP runs one gather_qmm instead of two. Rows are independent,
so the fused forward must be exactly equal to the unfused one.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

switch_layers = pytest.importorskip("mlx_lm.models.switch_layers")
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

from vllm_mlx.moe_fusion import (  # noqa: E402
    FusedSwitchGLU,
    fuse_moe_gate_up,
    maybe_fuse,
)

DIM, HID, EXPERTS, TOPK = 128, 64, 8, 2


def _quantized_glu(bits=4):
    glu = SwitchGLU(DIM, HID, EXPERTS, bias=False)
    nn.quantize(glu, group_size=64, bits=bits)
    mx.eval(glu.parameters())
    return glu


def test_fused_forward_is_exactly_equal_decode_shape():
    glu = _quantized_glu()
    x = mx.random.normal((1, 1, DIM)).astype(mx.float16)
    idx = mx.array([[[1, 5]]])
    ref = glu(x, idx)
    fused = FusedSwitchGLU(glu)
    out = fused(x, idx)
    assert mx.array_equal(ref, out), "fused expert MLP diverged from stock"


def test_fused_forward_is_exactly_equal_sorted_path():
    """indices.size >= 64 takes the gather-sort path — cover it too."""
    glu = _quantized_glu()
    x = mx.random.normal((1, 40, DIM)).astype(mx.float16)
    idx = mx.random.randint(0, EXPERTS, (1, 40, TOPK))
    ref = glu(x, idx)
    out = FusedSwitchGLU(glu)(x, idx)
    assert mx.array_equal(ref, out)


def test_walk_replaces_nested_and_listed_modules():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.switch_mlp = _quantized_glu()

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Block(), Block()]
            self.lone = _quantized_glu()

    toy = Toy()
    n = fuse_moe_gate_up(toy)
    assert n == 3
    assert isinstance(toy.lone, FusedSwitchGLU)
    assert isinstance(toy.layers[0].switch_mlp, FusedSwitchGLU)
    # fused modules still produce stock output through the toy tree
    x = mx.random.normal((1, 1, DIM)).astype(mx.float16)
    idx = mx.array([[[0, 3]]])
    mx.eval(toy.lone(x, idx))


def test_env_gate_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_MLX_MOE_GATEUP_FUSION", raising=False)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.switch_mlp = _quantized_glu()

    toy = Toy()
    assert maybe_fuse(toy, "toy") == 0
    assert not isinstance(toy.switch_mlp, FusedSwitchGLU)
    monkeypatch.setenv("VLLM_MLX_MOE_GATEUP_FUSION", "1")
    assert maybe_fuse(toy, "toy") == 1


def test_mismatched_quant_is_skipped_not_broken():
    glu = _quantized_glu()
    # sabotage: re-quantize just the gate at different bits
    glu8 = _quantized_glu(bits=8)
    glu.gate_proj = glu8.gate_proj

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.switch_mlp = glu

    toy = Toy()
    assert fuse_moe_gate_up(toy) == 0
    assert not isinstance(toy.switch_mlp, FusedSwitchGLU)
