# SPDX-License-Identifier: Apache-2.0
"""Pixel-cache entries must not alias the caller's mutable inputs.

``set_pixel_cache`` stored the request's ``extra_kwargs`` dict by
reference. ``_run_vision_encoding`` clears that dict once the encode
finishes (to release pixel buffers), which emptied the cached entry too
— so every pixel-cache HIT replayed the model call *without* the
processor's arch-specific kwargs. Invisible on arches whose processors
leave ``extra_kwargs`` empty (glm4v, qwen families), but mistral3 keeps
``image_sizes`` there and each HIT crashed its patch_merger with
``TypeError: 'NoneType' object is not iterable``.

Fix: defensive copy at store time (the read side already copies).
"""

import mlx.core as mx
import pytest

from vllm_mlx.vision_embedding_cache import VisionEmbeddingCache

pytestmark = pytest.mark.skipif(mx is None, reason="MLX not available")


def _store(cache, extra_kwargs):
    cache.set_pixel_cache(
        images=["fake-image-a"],
        prompt="what color?",
        pixel_values=mx.zeros((1, 3, 4, 4)),
        input_ids=mx.array([[1, 2, 3]]),
        extra_kwargs=extra_kwargs,
        processing_time=0.1,
    )


class TestPixelCacheAliasing:
    def test_clearing_callers_dict_leaves_entry_intact(self):
        cache = VisionEmbeddingCache()
        kwargs = {"image_sizes": [(64, 64)]}
        _store(cache, kwargs)
        kwargs.clear()  # what _run_vision_encoding does post-encode
        entry = cache.get_pixel_cache(["fake-image-a"], "what color?")
        assert entry is not None
        assert entry.extra_kwargs == {"image_sizes": [(64, 64)]}

    def test_none_extra_kwargs_stored_as_empty_dict(self):
        cache = VisionEmbeddingCache()
        _store(cache, None)
        entry = cache.get_pixel_cache(["fake-image-a"], "what color?")
        assert entry is not None
        assert entry.extra_kwargs == {}
