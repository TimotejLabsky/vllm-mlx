# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #181 — a media request never holds the scheduler step for
more than one prefill chunk: its text before the first image is an
ordinary row, the image span a chunked job interleaved with decode, and
aborts and #48 relief act at chunk boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_vision_unified_181_scheduler import (
    IMAGE,
    VEND,
    FakeVLM,
    Tok,
    _reference,
    _req,
    batch_patch,  # noqa: F401 - fixture
    _tiny_outer,
)
from vllm_mlx import vision_unified as vu

STEP = 4


def _media(prefix=40, grid=(1, 4, 4), digest=b"\x07" * 32):
    n_img = grid[1] * grid[2]
    ids = [((i * 7) % 50) + 1 for i in range(prefix)] + [IMAGE] * n_img
    ids += [VEND, 5, 7, 40, 2]
    g = np.array([list(grid)])
    return vu.MediaInputs(
        input_ids=ids,
        pixel_values=np.full((n_img, 4), 0.5, dtype=np.float32),
        image_grid_thw=g,
        image_digests=(digest,),
        n_images=1,
        image_tokens=n_img,
        cache_key=vu.media_cache_key(ids, IMAGE, (digest,), g),
    )


def _sched(monkeypatch, model):
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    monkeypatch.setenv("VLLM_MLX_BATCHED_SYSTEM_KV", "1")
    monkeypatch.setenv("VLLM_MLX_SYSTEM_KV_PARTIAL_MIN", "2")
    s = Scheduler(
        model,
        Tok(),
        SchedulerConfig(
            max_num_seqs=4, enable_prefix_cache=True, prefill_step_size=STEP
        ),
    )
    vlm = FakeVLM(model)
    s._vision_unified = vu.MediaRows(vlm, [3, 3, 2])
    return s, vlm


@pytest.fixture
def forwards(monkeypatch):
    """Record the sequence length of every text-model forward."""
    from mlx_lm.models import qwen3_5

    seen = []
    orig = qwen3_5.Model.__call__

    def call(self, inputs, cache=None, input_embeddings=None):
        seen.append((int(inputs.shape[1]), input_embeddings is not None))
        return orig(self, inputs, cache=cache, input_embeddings=input_embeddings)

    monkeypatch.setattr(qwen3_5.Model, "__call__", call)
    return seen


def _media_busy(s, rid):
    """True while the media request has produced no token yet."""
    r = s.requests.get(rid)
    return r is not None and r.num_output_tokens == 0


def test_181_long_prefix_media_never_holds_the_step(
    monkeypatch, batch_patch, forwards  # noqa: F811
):
    model = _tiny_outer()
    vu.install_mrope(model, [3, 3, 2])
    ref = _reference(model, FakeVLM(model), _media(), 6)
    s, vlm = _sched(monkeypatch, model)
    text = _req("t", [5, 6, 7, 8], n=200)
    s.add_request(text)
    for _ in range(3):
        s.step()  # the text row is decoding
    media = _media()
    s.add_request(_req("m", list(media.input_ids), media, n=6))
    steps = gaps = 0
    out_m = []
    while _media_busy(s, "m") or len(out_m) < 6:
        forwards.clear()
        out = s.step()
        toks = {o.request_id: o.new_token_ids for o in out.outputs}
        out_m += toks.get("m", [])
        if _media_busy(s, "m"):
            steps += 1
            # one chunk at most per step, and the text row decoded in it
            assert max(n for n, _ in forwards) <= STEP, forwards
            if not toks.get("t"):
                gaps += 1
        if steps > 200:
            raise AssertionError("media request never finished")
    assert gaps == 0
    # prefix 40 / STEP chunks + encode + 16/STEP image chunks + tail
    assert steps >= 40 // STEP + 1 + 16 // STEP
    assert out_m[:6] == ref
    st = s._vision_unified.stats()
    assert st["media_prefill_chunks"] == -(-17 // STEP)  # 16 image + vision_end
    assert st["media_prefill_tokens"] == 16 + 1  # image span incl. vision_end


def test_181_prefix_text_gets_ordinary_checkpoints(
    monkeypatch, batch_patch  # noqa: F811
):
    """The text before the image is prefilled by the batch as a text row
    would be: a checkpoint lands at the first image (and at the uniform
    interval), so another image after the same text restores there."""
    model = _tiny_outer()
    vu.install_mrope(model, [3, 3, 2])
    s, vlm = _sched(monkeypatch, model)
    captured = []
    orig = s.hybrid_kv.capture_segment

    def cap(rid, processed, cache):
        with s.hybrid_kv._lock:
            base = s.hybrid_kv._base_pos.get(rid, 0)
        captured.append((rid, base + processed))
        return orig(rid, processed, cache)

    s.hybrid_kv.capture_segment = cap
    a = _media(digest=b"\x01" * 32)
    s.add_request(_req("a", list(a.input_ids), a))
    while s.has_requests():
        s.step()
    fis = a.input_ids.index(IMAGE)
    assert ("a", fis) in captured  # first image
    b = _media(digest=b"\x02" * 32)
    rb = _req("b", list(b.input_ids), b)
    s.add_request(rb)
    while s.has_requests():
        s.step()
    assert rb.cached_tokens == fis
    assert b.plan["n"] == fis and vlm.calls == 2


def test_181_abort_mid_media_prefill_releases_at_the_next_step(
    monkeypatch, batch_patch, forwards  # noqa: F811
):
    model = _tiny_outer()
    vu.install_mrope(model, [3, 3, 2])
    s, vlm = _sched(monkeypatch, model)
    media = _media(prefix=8)
    s.add_request(_req("m", list(media.input_ids), media))
    rows = s._vision_unified
    for _ in range(100):
        s.step()
        job = rows.jobs.get("m")
        if job is not None and job.span is not None and job.pos > job.cut - 16:
            break
    assert rows.jobs.get("m") is not None, "job never mid-span"
    s.abort_request("m")
    forwards.clear()
    s.step()
    assert "m" not in rows.jobs and not s.running and not s.has_requests()
    assert not any(emb for _, emb in forwards)  # no further media chunk
    assert vu._MEDIA_UIDS == set()


def test_181_relief_runs_between_media_chunks(monkeypatch, batch_patch):  # noqa: F811
    model = _tiny_outer()
    vu.install_mrope(model, [3, 3, 2])
    s, vlm = _sched(monkeypatch, model)
    rows = s._vision_unified
    seen = []

    def relieve():
        seen.append((len(rows.jobs), len(rows.prefix_rows)))
        return 0

    s.hybrid_kv.relieve_pressure = relieve
    media = _media(prefix=20)
    s.add_request(_req("m", list(media.input_ids), media))
    while s.has_requests():
        s.step()
    during_job = sum(1 for j, _ in seen if j)
    during_prefix = sum(1 for _, p in seen if p)
    assert during_job >= 16 // STEP  # once before every image chunk
    assert during_prefix >= 20 // STEP - 1


def test_181_drafter_bypass_covers_a_zero_delta_media_row():
    from types import SimpleNamespace

    vu._MEDIA_UIDS.clear()
    assert not vu.batch_has_media_row(SimpleNamespace(uids=[3]))
    vu._MEDIA_UIDS.add(3)
    try:
        assert vu.batch_has_media_row(SimpleNamespace(uids=[3]))
        assert not vu.batch_has_media_row(SimpleNamespace(uids=[4]))
    finally:
        vu._MEDIA_UIDS.clear()


def test_181_go_cold_resets_the_reported_cache_hit(batch_patch):  # noqa: F811
    from types import SimpleNamespace

    model = _tiny_outer()
    rows = vu.MediaRows(FakeVLM(model), [3, 3, 2])
    media = _media(prefix=4)
    media.keyed = True
    req = SimpleNamespace(
        images=[media],
        prompt_cache=["restored"],
        cached_tokens=len(media.input_ids),  # full coverage
        cache_hit_type="system_kv",
        request_id="r",
    )
    toks, cache = rows.plan(SimpleNamespace(model=model), req)
    assert cache is None and req.cached_tokens == 0
    assert req.cache_hit_type == "miss" and media.plan["kind"] == "prefix"
