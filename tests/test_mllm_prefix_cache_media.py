# SPDX-License-Identifier: Apache-2.0
"""Image-safe prefix caching on the batched MLLM path (media guard, phase A).

The MLLM prefix cache (MemoryAwarePrefixCache) is keyed on raw token ids.
A media prompt's KV depends on pixel/audio content the placeholder tokens
don't encode, so two different images that tokenize identically aliased
each other's KV: the store site had no media check at all, and the fetch
guard only inspected the *remaining* (uncached) ids — on an exact match
``remaining_ids == []`` and the guard never ran.

Phase A fix: media-bearing requests neither store nor fetch. Text-only
requests are unaffected. (Composite media-hash keys — phase B — come later.)
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest

pytestmark = pytest.mark.skipif(mx is None, reason="MLX not available")


class _RecordingPrefixCache:
    def __init__(self, fetch_result=(None, None)):
        self.fetch_result = fetch_result
        self.fetch_calls = []
        self.store_calls = []

    def fetch(self, tokens):
        self.fetch_calls.append(list(tokens))
        return self.fetch_result

    def store(self, tokens, cache):
        self.store_calls.append((list(tokens), cache))


def _bare_generator(prefix_cache, model_config=None):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.prefix_cache = prefix_cache
    gen.model = SimpleNamespace(config=model_config or SimpleNamespace())
    gen._think_suffix_len = 0
    return gen


def _request(**kwargs) -> MLLMBatchRequest:
    req = MLLMBatchRequest(uid=1, request_id="req-1", prompt="hi", **kwargs)
    if req.input_ids is None:
        req.input_ids = mx.array([[1, 2, 3, 4]])
    return req


class TestFetchMediaGuard:
    def test_media_request_never_fetches(self):
        sentinel_kv = object()
        cache = _RecordingPrefixCache(fetch_result=(sentinel_kv, []))
        gen = _bare_generator(cache)
        req = _request(images=["a.png"])

        cached_kv, remaining = gen._prefix_cache_lookup(req)

        assert cached_kv is None and remaining is None
        assert cache.fetch_calls == []

    def test_audio_request_never_fetches(self):
        cache = _RecordingPrefixCache(fetch_result=(object(), []))
        gen = _bare_generator(cache)
        req = _request(audio=["a.wav"])

        assert gen._prefix_cache_lookup(req) == (None, None)
        assert cache.fetch_calls == []

    def test_text_request_fetches(self):
        sentinel_kv = object()
        cache = _RecordingPrefixCache(fetch_result=(sentinel_kv, [4]))
        gen = _bare_generator(cache)
        gen._has_empty_rotating_cache = MagicMock(return_value=False)
        req = _request()

        cached_kv, remaining = gen._prefix_cache_lookup(req)

        assert cached_kv is sentinel_kv
        assert remaining == [4]
        assert cache.fetch_calls == [[1, 2, 3, 4]]

    def test_exact_match_hole_closed(self):
        """The old guard only looked at remaining ids — an exact match on a
        media request bypassed it entirely. Media requests now never fetch."""
        cache = _RecordingPrefixCache(fetch_result=(object(), []))  # exact hit
        gen = _bare_generator(cache)
        req = _request(images=["different.png"])

        assert gen._prefix_cache_lookup(req) == (None, None)
        assert cache.fetch_calls == []

    @pytest.mark.parametrize(
        "attr", ["image_token_index", "image_token_id", "video_token_index", "video_token_id"]
    )
    def test_media_placeholder_in_remaining_clears_hit(self, attr):
        media_tok = 99
        cache = _RecordingPrefixCache(fetch_result=(object(), [99, 5]))
        gen = _bare_generator(cache, model_config=SimpleNamespace(**{attr: media_tok}))
        gen._has_empty_rotating_cache = MagicMock(return_value=False)
        req = _request()

        assert gen._prefix_cache_lookup(req) == (None, None)

    def test_think_suffix_stripped_and_restored(self):
        sentinel_kv = object()
        cache = _RecordingPrefixCache(fetch_result=(sentinel_kv, [3]))
        gen = _bare_generator(cache)
        gen._think_suffix_len = 1
        gen._has_empty_rotating_cache = MagicMock(return_value=False)
        req = _request()

        cached_kv, remaining = gen._prefix_cache_lookup(req)

        assert cache.fetch_calls == [[1, 2, 3]]  # suffix stripped from lookup
        assert remaining == [3, 4]  # suffix appended back to remaining


class TestStoreMediaGuard:
    def _batch_with(self, req):
        batch = SimpleNamespace(
            requests=[req],
            num_tokens=[0],
            extract_cache=MagicMock(return_value=[]),
        )
        return batch

    def test_media_request_not_stored(self):
        cache = _RecordingPrefixCache()
        gen = _bare_generator(cache)
        req = _request(images=["a.png"])

        gen._maybe_store_prefix_cache(self._batch_with(req), [0])

        assert cache.store_calls == []

    def test_video_request_not_stored(self):
        cache = _RecordingPrefixCache()
        gen = _bare_generator(cache)
        req = _request(videos=["a.mp4"])

        gen._maybe_store_prefix_cache(self._batch_with(req), [0])

        assert cache.store_calls == []

    def test_audio_request_not_stored(self):
        cache = _RecordingPrefixCache()
        gen = _bare_generator(cache)
        req = _request(audio=["a.wav"])

        gen._maybe_store_prefix_cache(self._batch_with(req), [0])

        assert cache.store_calls == []

    def test_text_request_still_stored(self):
        cache = _RecordingPrefixCache()
        gen = _bare_generator(cache)
        req = _request()

        gen._maybe_store_prefix_cache(self._batch_with(req), [0])

        assert len(cache.store_calls) == 1
        assert cache.store_calls[0][0] == [1, 2, 3, 4]


def test_aliasing_scenario_end_to_end_guard():
    """Two requests, identical token ids, different images: the second must
    neither fetch the first's entry nor store its own."""
    cache = _RecordingPrefixCache(fetch_result=(object(), []))
    gen = _bare_generator(cache)

    req_a = _request(images=["cat.png"])
    req_b = MLLMBatchRequest(
        uid=2, request_id="req-2", prompt="hi", images=["dog.png"]
    )
    req_b.input_ids = mx.array([[1, 2, 3, 4]])  # tokenizes identically

    assert gen._prefix_cache_lookup(req_a) == (None, None)
    assert gen._prefix_cache_lookup(req_b) == (None, None)
    gen._maybe_store_prefix_cache(
        SimpleNamespace(
            requests=[req_a, req_b],
            num_tokens=[0, 0],
            extract_cache=MagicMock(return_value=[]),
        ),
        [0, 1],
    )
    assert cache.fetch_calls == []
    assert cache.store_calls == []
