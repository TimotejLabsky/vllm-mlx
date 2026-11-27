# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #181 — the per-image token cap of a --vision-unified route
(``VLLM_MLX_VISION_MAX_IMAGE_TOKENS``, default 4096): over-cap images are
downscaled by the processor's own resize, the cache key follows what is
encoded, and nothing changes off the flag."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vllm_mlx import vision_unified as vu
from vllm_mlx.vision_unified_engine import (
    DEFAULT_MAX_IMAGE_TOKENS,
    ENV_MAX_IMAGE_TOKENS,
    VisionUnifiedError,
    apply_image_token_cap,
    max_image_tokens,
    smart_resize_tokens,
)

QWEN35_MAX_PX = 16777216  # Qwen3.5 preprocessor longest_edge (16,384 tokens)
QWEN35_MIN_PX = 65536


def _processor():
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

    ip = Qwen3VLImageProcessor(
        patch_size=16, merge_size=2, max_pixels=QWEN35_MAX_PX, min_pixels=QWEN35_MIN_PX
    )
    return SimpleNamespace(image_processor=ip)


def _grid(processor, image):
    out = processor.image_processor(images=[image])
    grid = np.asarray(out["image_grid_thw"]).reshape(-1, 3)
    return grid


def _tokens(grid):
    t, h, w = grid[0].tolist()
    return t * h * w // 4


def _key(grid, digest=b"\x05" * 32):
    n = _tokens(grid)
    ids = [11, 12] + [9] * n + [13]
    return vu.media_cache_key(ids, 9, (digest,), grid)


def test_181_default_cap_and_env_parsing(monkeypatch):
    monkeypatch.delenv(ENV_MAX_IMAGE_TOKENS, raising=False)
    assert max_image_tokens() == DEFAULT_MAX_IMAGE_TOKENS == 4096
    monkeypatch.setenv(ENV_MAX_IMAGE_TOKENS, "0")
    assert max_image_tokens() == 0
    for bad in ("x", "-1", "1.5"):
        monkeypatch.setenv(ENV_MAX_IMAGE_TOKENS, bad)
        with pytest.raises(VisionUnifiedError):
            max_image_tokens()


def test_181_over_cap_image_is_downscaled_to_fit_keeping_aspect():
    p = _processor()
    info = apply_image_token_cap(p, 1024)
    assert info["max_pixels"] == 1024 * 32 * 32
    assert info["processor_max_image_tokens"] == 16384
    big = Image.new("RGB", (4096, 2048), (10, 200, 30))  # 8192 tokens uncapped
    grid = _grid(p, big)
    assert _tokens(grid) <= 1024
    _, gh, gw = grid[0].tolist()
    assert abs(gw / gh - 2.0) < 0.1  # aspect kept
    assert smart_resize_tokens(2048, 4096, 32, QWEN35_MIN_PX, QWEN35_MAX_PX) == 8192


def test_181_under_cap_image_is_untouched_and_keeps_its_key():
    small = Image.new("RGB", (640, 480), (200, 20, 20))
    uncapped = _grid(_processor(), small)
    capped_p = _processor()
    apply_image_token_cap(capped_p, 4096)
    capped = _grid(capped_p, small)
    assert np.array_equal(uncapped, capped)
    assert _key(uncapped) == _key(capped)  # byte-identical to before the cap


def test_181_same_image_under_two_caps_never_cross_hits():
    big = Image.new("RGB", (3000, 3000), (30, 30, 200))
    a, b = _processor(), _processor()
    apply_image_token_cap(a, 4096)
    apply_image_token_cap(b, 1024)
    ga, gb = _grid(a, big), _grid(b, big)
    assert _tokens(ga) != _tokens(gb)
    ka, kb = _key(ga), _key(gb)
    # same bytes, other encode: keys part at the first placeholder
    assert ka[:2] == kb[:2] and ka[2] != kb[2]


def test_181_cap_zero_is_the_processor_limit():
    p = _processor()
    info = apply_image_token_cap(p, 0)
    assert p.image_processor.max_pixels == QWEN35_MAX_PX
    assert info["max_image_tokens"] == 0


def test_181_cap_refuses_what_it_cannot_honour():
    with pytest.raises(VisionUnifiedError, match="minimum"):
        apply_image_token_cap(_processor(), 10)  # min_pixels is 64 tokens
    with pytest.raises(VisionUnifiedError, match="max_pixels"):
        apply_image_token_cap(SimpleNamespace(image_processor=None), 4096)
    assert apply_image_token_cap(SimpleNamespace(image_processor=None), 0) == {
        "max_image_tokens": 0
    }


def test_181_downscaled_images_are_counted(tmp_path):
    from vllm_mlx.vision_unified_engine import VisionUnifiedEngine

    eng = VisionUnifiedEngine.__new__(VisionUnifiedEngine)
    eng._vlm_processor = _processor()
    eng._image_cap = apply_image_token_cap(eng._vlm_processor, 1024)
    big, small = tmp_path / "big.png", tmp_path / "small.png"
    Image.new("RGB", (3000, 3000)).save(big)
    Image.new("RGB", (640, 480)).save(small)
    grids = np.concatenate(
        [_grid(eng._vlm_processor, Image.open(f)) for f in (big, small)]
    )
    assert eng._count_downscaled([str(big), str(small)], grids) == 1


def test_181_cap_is_read_only_by_the_unified_engine():
    """Flag off, nothing reads the knob: it lives in the --vision-unified
    engine module alone, so MLLM and text routes keep the processor's own
    limit."""
    root = Path(__file__).resolve().parents[1] / "vllm_mlx"
    users = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        if ENV_MAX_IMAGE_TOKENS in p.read_text()
        or "ENV_MAX_IMAGE_TOKENS" in p.read_text()
    ]
    assert users == ["vision_unified_engine.py"]


def test_181_config_check_refuses_a_malformed_cap(monkeypatch):
    from vllm_mlx.vision_unified_engine import check_vision_unified_config

    monkeypatch.delenv("VLLM_MLX_MOE_GATEUP_FUSION", raising=False)
    monkeypatch.setenv(ENV_MAX_IMAGE_TOKENS, "lots")
    with pytest.raises(VisionUnifiedError, match="image cap"):
        check_vision_unified_config(use_batching=True)
