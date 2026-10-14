# SPDX-License-Identifier: Apache-2.0
"""(#142) Vision phase B: content-keyed prefix caching for image requests.

Phase A (#56, tests/test_mllm_prefix_cache_media.py) kept every media request
out of the token-keyed cache because two different images behind identical
prompts alias. Phase B keys image requests on their content instead: each
placeholder token is replaced, in the key only, by a fingerprint of that
image's SHA-256.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx

from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest
from vllm_mlx.mllm_media_key import image_digest, media_prefix_key

IMG = 7  # placeholder token id in these tests
SYS = [100, 101, 102]
# prompt: system, <img x3>, question
IDS = SYS + [IMG, IMG, IMG] + [200, 201]


def _png(tmp_path, name, payload):
    p = tmp_path / name
    p.write_bytes(payload)
    return str(p)


# ---------------------------------------------------------------- the key


def test_same_image_same_key_different_image_parts_at_its_first_token(tmp_path):
    a = _png(tmp_path, "a.png", b"\x89PNG-cat")
    a2 = _png(tmp_path, "copy-of-a.png", b"\x89PNG-cat")  # same bytes, new name
    b = _png(tmp_path, "b.png", b"\x89PNG-dog")
    ka, ka2, kb = (media_prefix_key(IDS, [IMG], [x]) for x in (a, a2, b))
    assert ka == ka2  # content, not file name
    assert ka != kb
    assert ka[:3] == kb[:3] == SYS  # text before the image is shared
    assert ka[3] != kb[3]  # parts at the image's first placeholder
    assert ka[6:] == kb[6:] == IDS[6:]  # text after the image is real ids
    assert all(v < 0 for v in ka[3:6])  # can never equal a real token id


def test_per_image_runs_keep_the_first_image_shared(tmp_path):
    a = _png(tmp_path, "a.png", b"A")
    b = _png(tmp_path, "b.png", b"B")
    c = _png(tmp_path, "c.png", b"C")
    ids = SYS + [IMG] * 2 + [300] + [IMG] * 2 + [301]
    k_ab = media_prefix_key(ids, [IMG], [a, b])
    k_ac = media_prefix_key(ids, [IMG], [a, c])
    assert k_ab[:6] == k_ac[:6]  # system + image a + separator shared
    assert k_ab[6] != k_ac[6]


def test_split_layouts_fall_back_to_one_combined_digest(tmp_path):
    """Pixtral splits one image into rows ([IMG]x2 [BREAK] [IMG]x2): runs !=
    images, so every placeholder carries the combined digest."""
    a = _png(tmp_path, "a.png", b"A")
    b = _png(tmp_path, "b.png", b"B")
    ids = SYS + [IMG, IMG, 9, IMG, IMG, 10]
    ka, kb = media_prefix_key(ids, [IMG], [a]), media_prefix_key(ids, [IMG], [b])
    assert ka is not None and ka != kb and ka[:3] == kb[:3] == SYS
    assert ka[5] == kb[5] == 9  # the break token stays a real id


def test_untrustworthy_or_missing_content_is_never_keyed(tmp_path):
    a = _png(tmp_path, "a.png", b"A")
    assert media_prefix_key(IDS, [IMG], ["https://example.com/x.png"]) is None
    assert image_digest("http://example.com/x.png") is None
    assert media_prefix_key(IDS, [IMG], []) is None
    assert media_prefix_key(SYS + [200], [IMG], [a]) is None  # no placeholders
    assert media_prefix_key(IDS, [], [a]) is None  # placeholder id unknown
    # data URLs / base64 hash the full string
    assert media_prefix_key(IDS, [IMG], ["data:image/png;base64,AAAA"]) != (
        media_prefix_key(IDS, [IMG], ["data:image/png;base64,AAAB"])
    )


# ---------------------------------------------------- generator integration


class _PrefixDict:
    """Minimal MemoryAwarePrefixCache stand-in: longest stored prefix wins,
    remaining = tokens[len(prefix):] (the real cache's contract)."""

    def __init__(self):
        self.entries = {}
        self.fetch_calls = []

    def fetch(self, tokens):
        self.fetch_calls.append(list(tokens))
        best = None
        for key, kv in self.entries.items():
            if len(key) <= len(tokens) and tuple(tokens[: len(key)]) == key:
                if best is None or len(key) > len(best[0]):
                    best = (key, kv)
        if best is None:
            return None, list(tokens)
        return best[1], list(tokens[len(best[0]) :])

    def store(self, tokens, cache, **kwargs):
        self.entries[tuple(tokens)] = cache
        return True


def _generator(cache, enabled=True):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.prefix_cache = cache
    gen.model = SimpleNamespace(config=SimpleNamespace(image_token_id=IMG))
    gen._think_suffix_len = 0
    gen._prefix_checkpoint_lock = threading.Lock()
    gen._request_prefix_checkpoints = {}
    gen._has_empty_rotating_cache = MagicMock(return_value=False)
    gen._rewind_prefix_cache = lambda c, n: c
    gen._media_prefix_enabled = enabled
    gen._media_placeholder_ids = (IMG,)
    gen._media_prefix_hits = 0
    gen._media_prefix_stores = 0
    return gen


def _req(uid, ids, **media):
    req = MLLMBatchRequest(uid=uid, request_id=f"r{uid}", prompt="p", **media)
    req.input_ids = mx.array([ids])
    req.is_text_only = not req.has_media
    return req


def _finish(gen, req):
    gen._maybe_store_prefix_cache(
        SimpleNamespace(
            requests=[req],
            num_tokens=[0],
            extract_cache=MagicMock(return_value=f"kv{req.uid}"),
        ),
        [0],
    )


def test_same_image_follow_up_hits_and_continues_on_real_ids(tmp_path):
    cache, img = _PrefixDict(), _png(tmp_path, "inv.png", b"invoice-1")
    gen = _generator(cache)
    turn1 = _req(1, IDS, images=[img])
    _finish(gen, turn1)
    assert gen._media_prefix_stores == 1

    turn2 = _req(2, IDS + [400, 401, 402], images=[img])  # same image, new question
    kv, remaining, _ = gen._prefix_cache_lookup(turn2)
    assert kv == "kv1"
    assert remaining == [400, 401, 402]  # REAL ids, never fingerprints
    assert gen._media_prefix_hits == 1


def test_different_image_behind_identical_tokens_never_aliases(tmp_path):
    """The phase-A bug, now with caching ON: identical ids, other image."""
    cache = _PrefixDict()
    gen = _generator(cache)
    _finish(gen, _req(1, IDS, images=[_png(tmp_path, "cat.png", b"cat")]))
    dog = _req(2, IDS, images=[_png(tmp_path, "dog.png", b"dog")])
    assert gen._prefix_cache_lookup(dog)[:2] == (None, None)


def test_partial_hit_ending_before_an_uncached_image_is_refused(tmp_path):
    """A text entry for the system prompt matches the image request's prefix,
    but the remaining tokens contain placeholders the language-model-only
    path cannot run - fall through to the full VLM forward."""
    cache = _PrefixDict()
    gen = _generator(cache)
    _finish(gen, _req(1, SYS))  # text-only entry: the system prompt
    req = _req(2, IDS, images=[_png(tmp_path, "a.png", b"A")])
    assert gen._prefix_cache_lookup(req)[:2] == (None, None)


def test_video_audio_urls_and_disabled_stay_phase_a(tmp_path):
    cache = _PrefixDict()
    img = _png(tmp_path, "a.png", b"A")
    gen = _generator(cache)
    for media in (
        {"videos": ["v.mp4"]},
        {"audio": ["a.wav"]},
        {"images": ["https://x/y.png"]},
        {"images": [img], "videos": ["v.mp4"]},
    ):
        req = _req(1, IDS, **media)
        assert gen._prefix_cache_lookup(req)[:2] == (None, None)
        _finish(gen, req)
    off = _generator(cache, enabled=False)
    _finish(off, _req(2, IDS, images=[img]))
    assert cache.entries == {} and cache.fetch_calls == []


def test_text_requests_keep_their_own_ids_as_the_key():
    cache = _PrefixDict()
    gen = _generator(cache)
    _finish(gen, _req(1, SYS + [200]))
    assert list(cache.entries) == [tuple(SYS + [200])]
    kv, remaining, _ = gen._prefix_cache_lookup(_req(2, SYS + [200, 201]))
    assert kv == "kv1" and remaining == [201]


def test_arch_allowlist_and_partial_generators_fail_closed():
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    from vllm_mlx import mllm_batch_generator as mbg

    assert "glm4v" in mbg._MEDIA_PREFIX_DEFAULT_ARCHS.split(",")
    assert getattr(gen, "_media_prefix_enabled", False) is False  # fail closed
