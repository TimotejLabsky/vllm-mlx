# SPDX-License-Identifier: Apache-2.0
"""(#168) Qwen3-VL-MoE attention: same output, no host sync per layer."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

lang = pytest.importorskip("mlx_vlm.models.qwen3_vl_moe.language")
from mlx_lm.models.cache import BatchKVCache  # noqa: E402

from vllm_mlx.patches import qwen3_vl_moe_mllm as patch  # noqa: E402


def _attention():
    args = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        rope_scaling={"type": "default", "mrope_section": [2, 3, 3]},
    )
    mx.random.seed(0)
    attn = lang.Attention(args)
    mx.eval(attn.parameters())
    return attn


def _decode(attn, steps=6):
    """Prefill 5 tokens on a left-padded 2-row batch cache, then decode."""
    cache = BatchKVCache(left_padding=[2, 0])
    outs, pos = [], 0
    for i, L in enumerate([5] + [1] * steps):
        x = mx.random.normal((2, L, 64), key=mx.random.key(100 + i))
        mask = lang.create_attention_mask(x, cache)
        # The generator always passes MRoPE position ids on this path.
        ids = mx.tile(mx.arange(pos, pos + L).reshape(1, 1, L), (3, 2, 1))
        pos += L
        outs.append(attn(x, mask=mask, cache=cache, position_ids=ids))
    mx.eval(outs)
    return outs


@pytest.fixture
def restore_attention():
    original = lang.Attention.__call__
    yield
    lang.Attention.__call__ = original
    for name in ("_fork_168_patched", "_fork_168_original_call"):
        if name in lang.Attention.__dict__:
            delattr(lang.Attention, name)


def test_168_patched_attention_matches_upstream_and_never_syncs(
    restore_attention, monkeypatch
):
    attn = _attention()
    reference = _decode(attn)

    assert patch.patch_qwen3_vl_moe_attention() is True
    syncs = []
    item = mx.array.item

    def counted(self, *a):
        syncs.append(1)
        return item(self, *a)

    monkeypatch.setattr(mx.array, "item", counted)
    got = _decode(attn)
    monkeypatch.setattr(mx.array, "item", item)

    assert not syncs, f"{len(syncs)} host syncs in the patched attention"
    for a, b in zip(reference, got):
        assert mx.array_equal(a, b).item()


def test_168_steps_aside_when_upstream_changes(restore_attention, monkeypatch):
    monkeypatch.setattr(patch, "_SYNC_LINE", "a line upstream no longer has")
    original = lang.Attention.__call__
    assert patch.patch_qwen3_vl_moe_attention() is False
    assert lang.Attention.__call__ is original
