# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #181 — images through the text scheduler (``--vision-unified``).

Stage 1: the position mechanics on a tiny random Qwen3.5 text model (no
weights downloaded). The real-checkpoint proof is
``scripts/fork/vision_unified_parity.py``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from vllm_mlx import vision_unified as vu


@pytest.fixture
def batch_patch():
    """Install the BatchKVCache delta patch for one test, then restore the
    stock methods so no other test sees it."""
    from mlx_lm.models.cache import BatchKVCache

    saved = (BatchKVCache.__dict__["merge"], BatchKVCache.__dict__["extract"])
    flag = vu._BATCH_PATCHED
    vu._BATCH_PATCHED = False
    vu.install_batch_cache_delta()
    try:
        yield BatchKVCache
    finally:
        BatchKVCache.merge, BatchKVCache.extract = saved
        vu._BATCH_PATCHED = flag


def _tiny(seed=181):
    from mlx_lm.models import qwen3_5

    mx.random.seed(seed)
    args = qwen3_5.ModelArgs.from_dict(
        {
            "model_type": "qwen3_5",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 64,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "full_attention_interval": 2,
            "rope_parameters": {
                "type": "default",
                "mrope_section": [3, 3, 2],
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
            },
        }
    )
    model = qwen3_5.Model(args)
    model.train(False)
    mx.eval(model.parameters())
    return model.language_model


def _greedy_tokens(model, prompt, n=6):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt]), cache=cache)
    first = np.array(logits[0, -1].astype(mx.float32))
    out = [int(mx.argmax(logits[0, -1]).item())]
    for _ in range(n - 1):
        logits = model(mx.array([[out[-1]]]), cache=cache)
        out.append(int(mx.argmax(logits[0, -1]).item()))
    return first, out


def test_181_selector_matches_mlx_vlm():
    from mlx_vlm.models.rope_utils import _interleaved_position_selector

    for section, f in (([11, 11, 10], 32), ([24, 20, 20], 64), ([3, 3, 2], 8)):
        assert vu.interleaved_selector(section, f).tolist() == (
            _interleaved_position_selector(section, f).tolist()
        )


def test_181_dispatch_is_a_pass_through_without_positions():
    model = _tiny()
    prompt = [3, 9, 1, 22, 5, 7, 40, 2]
    before = _greedy_tokens(model, prompt)
    assert vu.install_mrope(model, [3, 3, 2]) == 2
    after = _greedy_tokens(model, prompt)
    assert np.array_equal(before[0], after[0])
    assert before[1] == after[1]
    # idempotent
    assert vu.install_mrope(model, [3, 3, 2]) == 0


def test_181_equal_axes_reproduce_plain_rope():
    model = _tiny()
    vu.install_mrope(model, [3, 3, 2])
    rope = vu._attention_modules(model)[0].rope
    x = mx.random.normal((2, 4, 7, 32))
    for offset in (0, 13):
        pos = mx.broadcast_to((mx.arange(7) + offset)[None, None, :], (3, 2, 7)).astype(
            mx.int32
        )
        with vu.mrope_positions(pos):
            got = rope(x, offset=offset)
        want = rope.inner(x, offset=offset)
        assert mx.allclose(got, want, atol=1e-5).item()


def test_181_distinct_axes_match_mlx_vlm_mrope():
    from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

    model = _tiny()
    vu.install_mrope(model, [3, 3, 2])
    rope = vu._attention_modules(model)[0].rope  # dims 16 of head_dim 32
    ref = Qwen3_5RotaryEmbedding(16, base=10000, mrope_section=[3, 3, 2])
    L = 12
    t = mx.array([0, 1, 2, 3] + [4] * 6 + [7, 8])
    h = mx.array([0, 1, 2, 3] + [4, 4, 4, 5, 5, 5] + [7, 8])
    w = mx.array([0, 1, 2, 3] + [4, 5, 6, 4, 5, 6] + [7, 8])
    pos = mx.stack([t, h, w])[:, None, :].astype(mx.int32)
    x = mx.random.normal((1, 4, L, 32))
    with vu.mrope_positions(pos):
        got = rope(x)
    want, _ = ref.apply_rotary(x, x, pos, unsqueeze_dim=1)
    assert mx.allclose(got, want, atol=1e-5).item()
    # and it is NOT plain 1-D rope on these positions
    assert not mx.allclose(got, rope.inner(x), atol=1e-3).item()


def test_181_positions_context_is_scoped():
    assert vu._active_positions() is None
    pos = mx.zeros((3, 1, 2), dtype=mx.int32)
    with vu.mrope_positions(pos):
        assert vu._active_positions() is pos
    assert vu._active_positions() is None
    with pytest.raises(ValueError):
        with vu.mrope_positions(mx.zeros((1, 2), dtype=mx.int32)):
            pass


def test_181_install_refuses_unknown_rope():
    model = _tiny()
    vu._attention_modules(model)[0].rope = nn.Identity()
    with pytest.raises(ValueError, match="nn.RoPE"):
        vu.install_mrope(model, [3, 3, 2])
    with pytest.raises(ValueError):
        vu.install_mrope(_tiny(), [3, 3])


def _kv(length, delta=0, H=2, D=8):
    from mlx_lm.models.cache import KVCache

    c = KVCache()
    if length:
        k = mx.random.normal((1, H, length, D))
        c.update_and_fetch(k, k)
    if delta:
        c.rope_delta = delta
    return c


def test_181_merge_without_deltas_is_unchanged(batch_patch):
    a, b = _kv(5), _kv(3)
    merged = batch_patch.merge([a, b])
    assert merged.offset.tolist() == [5, 3]
    assert merged.left_padding.tolist() == [0, 2]


def test_181_merge_adds_a_rows_delta_and_extract_returns_it(batch_patch):
    a, b = _kv(5), _kv(7, delta=-4)
    merged = batch_patch.merge([a, b])
    # mask/index state untouched; only the RoPE offset moves
    assert merged.offset.tolist() == [5, 3]
    assert merged._idx == 7 and merged.left_padding.tolist() == [2, 0]
    back = merged.extract(1)
    assert back.offset == 7 and back.rope_delta == -4
    assert not hasattr(merged.extract(0), "rope_delta")
    # decode step, filter and extend carry the delta per row
    k = mx.random.normal((2, 2, 1, 8))
    merged.update_and_fetch(k, k)
    assert merged.offset.tolist() == [6, 4]
    merged.filter([1])
    assert merged.offset.tolist() == [4]
    other = batch_patch.merge([_kv(2)])
    merged.extend(other)
    assert merged.offset.tolist() == [4, 2]
    assert merged.extract(0).rope_delta == -4
    merged.trim(1)
    assert merged.extract(0).rope_delta == -4


def test_181_rope_delta_helpers():
    from mlx_lm.models.cache import ArraysCache

    cache = [ArraysCache(2), _kv(3), _kv(3)]
    assert vu.rope_delta_of(cache) == 0
    vu.set_rope_delta(cache, -9)
    assert vu.rope_delta_of(cache) == -9
    vu.set_rope_delta(cache, 0)
    assert vu.rope_delta_of(cache) == 0
    cache[1].rope_delta = 2
    with pytest.raises(ValueError):
        vu.rope_delta_of(cache)


def test_181_media_row_decodes_in_a_batch_like_single_stream(batch_patch):
    """A media-shaped row (positions that jump, delta < 0) prefilled out of
    band then decoded through BatchGenerator gives the tokens single-stream
    MRoPE decoding gives; the text row beside it equals the text row alone."""
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.models.cache import make_prompt_cache

    model = _tiny()
    vu.install_mrope(model, [3, 3, 2])
    ids = [3, 9, 1, 22, 5, 7, 40, 2, 11, 12]
    L = len(ids)
    t = [0, 1, 2, 3, 3, 3, 3, 5, 6, 7]
    h = [0, 1, 2, 3, 3, 4, 4, 5, 6, 7]
    w = [0, 1, 2, 3, 4, 3, 4, 5, 6, 7]
    pos = mx.array([t, h, w], dtype=mx.int32)[:, None, :]
    delta = 7 + 1 - L
    emb = model.model.embed_tokens(mx.array([ids]))

    # single-stream reference: positions explicit at every step
    cache = make_prompt_cache(model)
    with vu.mrope_positions(pos):
        logits = model(mx.array([ids]), cache=cache, input_embeddings=emb)
    ref = [int(mx.argmax(logits[0, -1]).item())]
    for i in range(5):
        p = mx.full((3, 1, 1), L + i + delta, dtype=mx.int32)
        with vu.mrope_positions(p):
            logits = model(mx.array([[ref[-1]]]), cache=cache)
        ref.append(int(mx.argmax(logits[0, -1]).item()))

    def run(prompts, caches):
        bg = BatchGenerator(model, max_tokens=6)
        uids = bg.insert(prompts, caches=caches)
        res = {u: [] for u in uids}
        while any(len(v) < 6 for v in res.values()):
            _, gen = bg.next()
            for r in gen:
                res[r.uid].append(int(r.token))
        bg.close()
        return [res[u] for u in uids]

    text = [5, 6, 7, 8, 9]
    alone = run([text], None)[0]
    for cut in (L - 1, 7):  # last token only / trailing text in the batch
        mcache = make_prompt_cache(model)
        vu.media_prefill(
            model,
            mx.array([ids[:cut]]),
            emb[:, :cut],
            pos[:, :, :cut],
            mcache,
            step=3,
        )
        vu.set_rope_delta(mcache, delta)
        rows = run([ids[cut:], text], [mcache, make_prompt_cache(model)])
        assert rows[0] == ref, cut
        assert rows[1] == alone
