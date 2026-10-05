# SPDX-License-Identifier: Apache-2.0
"""Fork #136: the MLLM caches key on image hashes IN ORDER.

Both ``compute_images_hash`` copies (``mllm_cache`` and
``vision_embedding_cache``) joined the per-image hashes after ``sorted()``, so
a request carrying the same two images in swapped order produced the same key
and reused the first request's pixel values / vision embeddings — image 1's
pixels at image 2's placeholder and vice versa. Upstream waybarrios/vllm-mlx
PR #726 (open) fixes only ``mllm_cache``; the production pixel cache on the
batched MLLM path is ``vision_embedding_cache``.
"""

import mlx.core as mx
import pytest

from vllm_mlx.mllm_cache import MLLMPrefixCacheManager
from vllm_mlx.mllm_cache import compute_images_hash as mllm_images_hash
from vllm_mlx.vision_embedding_cache import VisionEmbeddingCache
from vllm_mlx.vision_embedding_cache import (
    compute_images_hash as vision_images_hash,
)

A, B = "data:image/png;base64,AAAA", "data:image/png;base64,BBBB"


@pytest.mark.parametrize("h", [mllm_images_hash, vision_images_hash])
def test_combined_hash_is_order_sensitive(h):
    assert h([A, B]) != h([B, A])
    assert h([A, B]) == h([A, B])


def _store_pixels(cache, images, marker):
    cache.set_pixel_cache(
        images=images,
        prompt="compare the two images",
        pixel_values=mx.full((2, 3, 4, 4), marker),
        input_ids=mx.array([[1, 2, 3]]),
        processing_time=0.1,
    )


class TestPixelCache:
    """``VisionEmbeddingCache`` — the batched MLLM path's pixel cache."""

    def test_swapped_order_misses(self):
        cache = VisionEmbeddingCache()
        _store_pixels(cache, [A, B], 1.0)
        assert cache.get_pixel_cache([B, A], "compare the two images") is None

    def test_same_order_hits(self):
        cache = VisionEmbeddingCache()
        _store_pixels(cache, [A, B], 1.0)
        entry = cache.get_pixel_cache([A, B], "compare the two images")
        assert entry is not None
        assert mx.all(entry.pixel_values == 1.0).item()

    def test_each_order_keeps_its_own_entry(self):
        cache = VisionEmbeddingCache()
        _store_pixels(cache, [A, B], 1.0)
        _store_pixels(cache, [B, A], 2.0)
        ab = cache.get_pixel_cache([A, B], "compare the two images")
        ba = cache.get_pixel_cache([B, A], "compare the two images")
        assert mx.all(ab.pixel_values == 1.0).item()
        assert mx.all(ba.pixel_values == 2.0).item()

    def test_image_only_key_is_order_sensitive(self):
        cache = VisionEmbeddingCache()
        assert cache._make_image_only_key([A, B]) != cache._make_image_only_key([B, A])


class TestMLLMPrefixCache:
    """``MLLMPrefixCacheManager`` — exact hits AND the vision-only partial hit
    (``entry.image_hash == image_key``) both read the same order-sensitive
    hash the store wrote."""

    def _manager(self):
        mgr = MLLMPrefixCacheManager(max_entries=10)
        mgr.store(
            [A, B],
            "compare",
            vision_embeddings="emb-AB",
            kv_cache=["kv-AB"],
            token_ids=[1, 2, 3],
        )
        return mgr

    def test_same_order_exact_hit(self):
        entry, _ = self._manager().fetch([A, B], "compare")
        assert entry is not None and entry.kv_cache == ["kv-AB"]

    def test_swapped_order_no_exact_hit(self):
        entry, _ = self._manager().fetch([B, A], "compare")
        assert entry is None

    def test_swapped_order_no_vision_only_hit(self):
        # Different prompt -> the vision-embedding-reuse branch is consulted.
        entry, _ = self._manager().fetch([B, A], "describe")
        assert entry is None

    def test_same_order_vision_only_hit_still_works(self):
        entry, match_len = self._manager().fetch([A, B], "describe")
        assert entry is not None and entry.vision_embeddings == "emb-AB"
        assert match_len == 0
