# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #181 — the parity script's near-tie classifier and its float32
switch (``scripts/fork/vision_unified_parity.py``). Model-free."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fork/vision_unified_parity.py"


@pytest.fixture(scope="module")
def parity():
    spec = importlib.util.spec_from_file_location("vu_parity_181", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(**logits):
    row = np.full(8, -10.0, dtype=np.float32)
    for k, v in logits.items():
        row[int(k[1:])] = v
    return row


def test_equal_and_early_stop_are_equal(parity):
    rows = [_row(t1=5.0)] * 3
    assert parity.classify_greedy([1, 2, 3], rows, [1, 2, 3], rows, 0.4)[0] == "equal"
    # ours stopped earlier (eos) on the same prefix
    assert parity.classify_greedy([1, 2, 3], rows, [1, 2], rows, 0.4)[0] == "equal"
    assert parity.classify_greedy([1, 2], rows, [], rows, 0.4)[0] == "diverged"


def test_flip_inside_the_bound_on_both_stacks_is_a_near_tie(parity):
    # the 2026-11-27 4B case: 'Red' 23.75 vs ' red' 23.5 on mlx-vlm,
    # ' red' -1.0 vs 'Red' -1.25 on the batch (logprobs), bound 0.41
    ref = [_row(t2=23.75, t3=23.5)]
    ours = [_row(t3=-1.0, t2=-1.25)]
    verdict, detail = parity.classify_greedy([2, 7], ref, [3, 7], ours, 0.406)
    assert verdict == "near_tie"
    assert "step 0" in detail and "0.250" in detail


@pytest.mark.parametrize(
    "ref_b, our_a",
    [
        (22.0, -1.25),  # ref was sure (margin 1.75), ours a tie
        (23.5, -3.0),  # ref a tie, ours sure (margin 2.0)
    ],
)
def test_flip_outside_the_bound_on_either_stack_diverges(parity, ref_b, our_a):
    ref = [_row(t2=23.75, t3=ref_b)]
    ours = [_row(t3=-1.0, t2=our_a)]
    assert parity.classify_greedy([2], ref, [3], ours, 0.406)[0] == "diverged"


def test_a_flip_at_a_later_step_uses_that_steps_rows(parity):
    ref = [_row(t1=9.0), _row(t2=5.0, t3=4.9)]
    ours = [_row(t1=9.0), _row(t3=5.0, t2=4.95)]
    verdict, detail = parity.classify_greedy([1, 2], ref, [1, 3], ours, 0.2)
    assert verdict == "near_tie" and "step 1" in detail
    # no rows for the differing step: never a near-tie
    assert parity.classify_greedy([1, 2], ref[:1], [1, 3], ours, 0.2)[0] == ("diverged")


def test_zero_bound_float32_accepts_no_flip(parity):
    ref = [_row(t2=23.75, t3=23.74)]
    ours = [_row(t3=-1.0, t2=-1.01)]
    assert parity.classify_greedy([2], ref, [3], ours, 0.0)[0] == "diverged"


def test_inconsistent_rows_are_not_a_near_tie(parity):
    # the rows say the other token was the argmax: negative margin
    ref = [_row(t2=1.0, t3=2.0)]
    ours = [_row(t3=-1.0, t2=-0.5)]
    assert parity.classify_greedy([2], ref, [3], ours, 5.0)[0] == "diverged"


def test_fp32_activations_casts_embeddings_and_restores(parity):
    emb = nn.Embedding(16, 8)
    emb.weight = emb.weight.astype(mx.bfloat16)
    q = nn.QuantizedEmbedding.from_embedding(nn.Embedding(16, 64))
    ids = mx.array([[1, 2, 3]])
    saved = (nn.Embedding.__dict__["__call__"], nn.QuantizedEmbedding.__call__)
    assert emb(ids).dtype == mx.bfloat16
    with parity.fp32_activations():
        assert emb(ids).dtype == mx.float32
        assert q(ids).dtype == mx.float32
    assert emb(ids).dtype == mx.bfloat16
    assert (nn.Embedding.__dict__["__call__"], nn.QuantizedEmbedding.__call__) == (
        saved
    )
