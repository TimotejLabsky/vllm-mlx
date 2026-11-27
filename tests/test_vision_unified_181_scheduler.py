# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #181 stage 2 — a media request through the REAL text
``Scheduler`` (tiny random Qwen3.5, batched system-KV on, no weights
downloaded), next to a text request in the same batch."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vllm_mlx import vision_unified as vu
from vllm_mlx.request import Request, SamplingParams

IMAGE = 60
VEND = 61
EOS = 63


@pytest.fixture
def batch_patch():
    from mlx_lm.models.cache import BatchKVCache

    saved = (BatchKVCache.__dict__["merge"], BatchKVCache.__dict__["extract"])
    flag = vu._BATCH_PATCHED
    vu._BATCH_PATCHED = False
    vu.install_batch_cache_delta()
    try:
        yield
    finally:
        BatchKVCache.merge, BatchKVCache.extract = saved
        vu._BATCH_PATCHED = flag


def _tiny_outer(seed=181):
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
    return model


class FakeVLM:
    """get_input_embeddings with the shape of mlx-vlm's: placeholder rows
    get 'image features' from the pixels, positions are a 2-D grid per
    image, delta = max position + 1 - length."""

    def __init__(self, model):
        self.model = model
        self.config = SimpleNamespace(
            image_token_id=IMAGE, video_token_id=None, vision_end_token_id=VEND
        )
        self.calls = 0
        self.language_model = SimpleNamespace(get_rope_index=self.get_rope_index)

    def get_input_embeddings(self, ids, pixel_values, image_grid_thw=None):
        self.calls += 1
        emb = self.model.language_model.model.embed_tokens(ids)
        toks = ids[0].tolist()
        mask = mx.array([t == IMAGE for t in toks])[None, :, None]
        feat = mx.broadcast_to(pixel_values.reshape(1, 1, -1)[..., :1], emb.shape)
        emb = mx.where(mask, emb + feat.astype(emb.dtype), emb)
        positions, deltas = self.get_rope_index(ids, image_grid_thw)
        return SimpleNamespace(
            inputs_embeds=emb, position_ids=positions, rope_deltas=deltas
        )

    def get_rope_index(self, ids, image_grid_thw):
        toks = ids[0].tolist()
        grids = image_grid_thw.tolist()
        t, h, w = [], [], []
        pos, i, k = 0, 0, 0
        while i < len(toks):
            if toks[i] == IMAGE:
                _, gh, gw = grids[k]
                k += 1
                for r in range(gh):
                    for c in range(gw):
                        t.append(pos)
                        h.append(pos + r)
                        w.append(pos + c)
                i += gh * gw
                pos += max(gh, gw)
            else:
                t.append(pos)
                h.append(pos)
                w.append(pos)
                pos += 1
                i += 1
        positions = mx.array([t, h, w], dtype=mx.int32)[:, None, :]
        return positions, mx.array([[pos - len(toks)]])


class Tok:
    eos_token_id = EOS

    def encode(self, text, add_special_tokens=True):
        return [int(x) if x.isdigit() else 1 for x in text.split()]

    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)


def _media_ids():
    return [3, 9, 1, 22] + [IMAGE] * 6 + [VEND, 5, 7, 40, 2]


def _media(pixel=0.5):
    ids = _media_ids()
    return vu.MediaInputs(
        input_ids=ids,
        pixel_values=np.full((6, 4), pixel, dtype=np.float32),
        image_grid_thw=np.array([[1, 2, 3]]),
        image_digests=(b"x" * 32,),
        n_images=1,
        image_tokens=6,
    )


def _reference(model, vlm, media, n):
    from mlx_lm.models.cache import make_prompt_cache

    ids = mx.array([media.input_ids])
    f = vlm.get_input_embeddings(
        ids, mx.array(media.pixel_values), mx.array(media.image_grid_thw)
    )
    delta = int(f.rope_deltas.item())
    L = ids.shape[1]
    cache = make_prompt_cache(model)
    with vu.mrope_positions(f.position_ids):
        logits = model(ids, cache=cache, input_embeddings=f.inputs_embeds)
    out = [int(mx.argmax(logits[0, -1]).item())]
    while len(out) < n:
        p = mx.full((3, 1, 1), L + len(out) - 1 + delta, dtype=mx.int32)
        with vu.mrope_positions(p):
            logits = model(mx.array([[out[-1]]]), cache=cache)
        out.append(int(mx.argmax(logits[0, -1]).item()))
    return out


def _scheduler(monkeypatch, model):
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    monkeypatch.setenv("VLLM_MLX_BATCHED_SYSTEM_KV", "1")
    s = Scheduler(
        model, Tok(), SchedulerConfig(max_num_seqs=4, enable_prefix_cache=True)
    )
    assert s.hybrid_kv is not None
    return s


def _run(scheduler, requests, n=6):
    for r in requests:
        scheduler.add_request(r)
    outs = {r.request_id: [] for r in requests}
    for _ in range(200):
        if not scheduler.has_requests():
            break
        out = scheduler.step()
        for o in out.outputs:
            if o.new_token_ids:
                outs[o.request_id].extend(o.new_token_ids)
    return outs


def _req(rid, prompt, media=None, n=6):
    return Request(
        request_id=rid,
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=n, temperature=0.0),
        images=[media] if media is not None else None,
    )


def test_181_media_request_decodes_through_the_text_scheduler(monkeypatch, batch_patch):
    model = _tiny_outer()
    vu.install_mrope(model, [3, 3, 2])
    vlm = FakeVLM(model)
    media = _media()
    ref = _reference(model, vlm, media, 6)

    text = [5, 6, 7, 8, 9, 10, 11]
    alone_s = _scheduler(monkeypatch, model)
    alone = _run(alone_s, [_req("t0", list(text))])["t0"]

    s = _scheduler(monkeypatch, model)
    s._vision_unified = vu.MediaRows(vlm, [3, 3, 2])
    stored, captured = [], []
    orig_store, orig_cap = s.hybrid_kv.store, s.hybrid_kv.capture_segment
    s.hybrid_kv.store = lambda rid, *a, **k: (
        stored.append(rid),
        orig_store(rid, *a, **k),
    )[1]
    s.hybrid_kv.capture_segment = lambda rid, *a, **k: (
        captured.append(rid),
        orig_cap(rid, *a, **k),
    )[1]
    fetched = []
    orig_fetch = s.hybrid_kv.fetch
    s.hybrid_kv.fetch = lambda toks, **k: (
        fetched.append(list(toks)),
        orig_fetch(toks, **k),
    )[1]

    outs = _run(
        s,
        [_req("m1", list(media.input_ids), media), _req("t1", list(text))],
    )
    assert outs["m1"] == ref
    assert outs["t1"] == alone
    # stage 2: media rows read and write no cache; the text row does both
    assert "m1" not in stored and "t1" in stored
    assert "m1" not in captured
    assert all(f != media.input_ids for f in fetched)
    assert vlm.calls == 2  # reference + the scheduler's one encode
    stats = s._vision_unified.stats()
    assert stats["media_requests"] == 1 and stats["media_image_tokens"] == 6
    assert s.requests.get("m1") is None or s.requests["m1"].prompt_cache is None


def test_181_media_prefill_failure_rejects_only_that_request(monkeypatch, batch_patch):
    model = _tiny_outer()
    vu.install_mrope(model, [3, 3, 2])
    vlm = FakeVLM(model)

    def boom(*a, **k):
        raise RuntimeError("bad pixels")

    vlm.get_input_embeddings = boom
    s = _scheduler(monkeypatch, model)
    s._vision_unified = vu.MediaRows(vlm, [3, 3, 2])
    media = _media()
    s.add_request(_req("m1", list(media.input_ids), media))
    s.add_request(_req("t1", [5, 6, 7, 8]))
    finished = {}
    for _ in range(100):
        if not s.has_requests():
            break
        for o in s.step().outputs:
            if o.finished:
                finished[o.request_id] = (o.finish_reason, o.error_kind)
    assert finished["m1"] == ("error", "media_error")
    assert finished["t1"][0] in ("length", "stop")
    assert s._vision_unified.stats()["media_errors"] == 1


def test_181_text_requests_never_read_media_state(monkeypatch):
    """No media on the request: media_of is None and the scheduler never
    touches a _vision_unified hook (a flag-off route has none)."""
    assert vu.media_of(_req("t", [1, 2])) is None
    assert vu.media_of(SimpleNamespace(images=["http://x"])) is None
    assert vu.media_of(SimpleNamespace()) is None
    with pytest.raises(RuntimeError):
        vu.prefill_media_row(SimpleNamespace(), _req("m", [1], _media()))


def test_181_media_cut_ends_after_the_last_image():
    rows = vu.MediaRows(FakeVLM(_tiny_outer()), [3, 3, 2])
    ids = _media_ids()
    assert rows.media_cut(ids) == 11  # 4 text + 6 image + vision_end
    assert rows.media_cut([1, IMAGE, IMAGE]) == 2  # never the final token
    with pytest.raises(ValueError):
        rows.media_cut([1, 2, 3])
