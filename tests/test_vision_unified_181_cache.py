# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #181 stage 3 — media requests in the batched system-KV.

Content-keyed cache keys (image digest + grid), restores that continue a
media prefix, and the invariants: different images never share a hit past
the image, and a key prefix fixes the positions (so the delta) it was made
with."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_vision_unified_181_scheduler import (
    IMAGE,
    VEND,
    FakeVLM,
    _media_ids,
    _req,
    _run,
    _scheduler,
    _tiny_outer,
    batch_patch,  # noqa: F401 - fixture
)
from vllm_mlx import vision_unified as vu

D1, D2 = b"\x01" * 32, b"\x02" * 32
G = np.array([[1, 2, 3]])


def test_181_key_is_content_and_grid_keyed():
    ids = _media_ids()
    k1 = vu.media_cache_key(ids, IMAGE, (D1,), G)
    assert k1 == vu.media_cache_key(ids, IMAGE, (D1,), G)
    first = ids.index(IMAGE)
    assert k1[:first] == ids[:first] and k1[first + 6 :] == ids[first + 6 :]
    assert all(v < 0 for v in k1[first : first + 6])
    other = vu.media_cache_key(ids, IMAGE, (D2,), G)
    regrid = vu.media_cache_key(ids, IMAGE, (D1,), np.array([[1, 3, 2]]))
    for k in (other, regrid):
        assert k[:first] == k1[:first] and k[first] != k1[first]
    # untrusted digest (remote URL) or a placeholder/image count mismatch
    assert vu.media_cache_key(ids, IMAGE, (None,), G) is None
    assert vu.media_cache_key(ids, IMAGE, (D1, D2), np.array([[1, 2, 3]] * 2)) is None
    assert vu.media_cache_key(ids, IMAGE, (), G) is None


def test_181_two_images_fingerprint_independently():
    ids = [3, 4] + [IMAGE] * 6 + [VEND, 5] + [IMAGE] * 6 + [VEND, 7, 8]
    grids = np.array([[1, 2, 3], [1, 2, 3]])
    a = vu.media_cache_key(ids, IMAGE, (D1, D2), grids)
    b = vu.media_cache_key(ids, IMAGE, (D1, D1), grids)
    assert a[:10] == b[:10] and a[10] != b[10]


def _keyed(pixel, digest):
    ids = _media_ids()
    return vu.MediaInputs(
        input_ids=ids,
        pixel_values=np.full((6, 4), pixel, dtype=np.float32),
        image_grid_thw=G,
        image_digests=(digest,),
        n_images=1,
        image_tokens=6,
        cache_key=vu.media_cache_key(ids, IMAGE, (digest,), G),
    )


@pytest.fixture
def small_bag(monkeypatch):
    monkeypatch.setenv("VLLM_MLX_SYSTEM_KV_PARTIAL_MIN", "2")


def _fresh(monkeypatch, model):
    s = _scheduler(monkeypatch, model)
    vlm = FakeVLM(model)
    s._vision_unified = vu.MediaRows(vlm, [3, 3, 2])
    return s, vlm


def _cold(monkeypatch, model, media, n=6):
    s, _ = _fresh(monkeypatch, model)
    media.keyed = False
    return _run(s, [_req("cold", list(media.input_ids), media, n)], n)["cold"]


def test_181_resend_hits_skips_the_encode_and_decodes_the_same(
    monkeypatch, batch_patch, small_bag  # noqa: F811
):
    model = _tiny_outer()
    vu.install_mrope(model, [3, 3, 2])
    cold = _cold(monkeypatch, model, _keyed(0.5, D1))

    s, vlm = _fresh(monkeypatch, model)
    first = _keyed(0.5, D1)
    out1 = _run(s, [_req("a1", list(first.input_ids), first)])["a1"]
    assert out1 == cold and first.keyed
    again = _keyed(0.5, D1)
    req = _req("a2", list(again.input_ids), again)
    out2 = _run(s, [req])["a2"]
    assert out2 == cold
    assert req.cached_tokens > again.input_ids.index(VEND)  # past the image
    stats = s._vision_unified.stats()
    assert stats["media_cache_hits"] == 1 and stats["media_encode_skipped"] == 1
    assert vlm.calls == 1  # the re-send never ran the vision tower


def test_181_follow_up_turn_hits_and_matches_a_cold_run(
    monkeypatch, batch_patch, small_bag  # noqa: F811
):
    model = _tiny_outer()
    vu.install_mrope(model, [3, 3, 2])
    s, _ = _fresh(monkeypatch, model)
    first = _keyed(0.5, D1)
    out1 = _run(s, [_req("a1", list(first.input_ids), first)])["a1"]
    follow_ids = list(first.input_ids) + out1 + [12, 13, 14]
    grid = G

    def follow():
        return vu.MediaInputs(
            input_ids=follow_ids,
            pixel_values=np.full((6, 4), 0.5, dtype=np.float32),
            image_grid_thw=grid,
            image_digests=(D1,),
            n_images=1,
            image_tokens=6,
            cache_key=vu.media_cache_key(follow_ids, IMAGE, (D1,), grid),
        )

    f = follow()
    req = _req("a2", list(follow_ids), f)
    hot = _run(s, [req])["a2"]
    assert req.cached_tokens >= len(first.input_ids) - 1
    assert hot == _cold(monkeypatch, model, follow())


def test_181_different_image_same_text_never_hits_past_the_image(
    monkeypatch, batch_patch, small_bag  # noqa: F811
):
    model = _tiny_outer()
    vu.install_mrope(model, [3, 3, 2])
    s, vlm = _fresh(monkeypatch, model)
    a = _keyed(0.5, D1)
    _run(s, [_req("a", list(a.input_ids), a)])
    b = _keyed(-2.0, D2)  # other pixels, other digest
    req = _req("b", list(b.input_ids), b)
    out_b = _run(s, [req])["b"]
    assert req.cached_tokens <= b.input_ids.index(IMAGE)
    assert out_b == _cold(monkeypatch, model, _keyed(-2.0, D2))
    assert vlm.calls == 2


def test_181_text_request_never_matches_into_a_media_entry(
    monkeypatch, batch_patch, small_bag  # noqa: F811
):
    model = _tiny_outer()
    vu.install_mrope(model, [3, 3, 2])
    s, _ = _fresh(monkeypatch, model)
    m = _keyed(0.5, D1)
    _run(s, [_req("m", list(m.input_ids), m)])
    # the same ids as plain text (placeholders as tokens) must not hit past
    # the first placeholder: the stored key has fingerprints there
    t = _req("t", list(m.input_ids))
    _run(s, [t])
    assert t.cached_tokens <= m.input_ids.index(IMAGE)


def test_181_unkeyable_media_stays_cold(
    monkeypatch, batch_patch, small_bag  # noqa: F811
):
    model = _tiny_outer()
    vu.install_mrope(model, [3, 3, 2])
    s, vlm = _fresh(monkeypatch, model)
    for rid in ("u1", "u2"):
        m = _keyed(0.5, D1)
        m.cache_key = None  # e.g. an http(s) image URL
        req = _req(rid, list(m.input_ids), m)
        _run(s, [req])
        assert req.cached_tokens == 0 and not m.keyed
    assert vlm.calls == 2
