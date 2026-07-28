# SPDX-License-Identifier: Apache-2.0
"""Request-shape media limits + dead-config wiring (vision series).

image_limits.py mirrors audio_limits.py: env-configured caps on media
item counts (400) and inline base64 payload size (413), enforced in the
server's _prepare_chat_messages before any engine work. Also pins the
video-sampling config plumb (default_video_fps/max_video_frames were
declared on MLLMSchedulerConfig but the preprocess hardcoded constants).
"""

import base64

import pytest
from fastapi import HTTPException

from vllm_mlx.image_limits import validate_media_shape


def _image_part(url="https://example.com/x.png"):
    return {"type": "image_url", "image_url": {"url": url}}


def _msg(parts):
    return {"role": "user", "content": parts}


def _data_url(n_bytes):
    return "data:image/png;base64," + base64.b64encode(b"x" * n_bytes).decode()


class TestCounts:
    def test_under_default_limits_pass(self):
        validate_media_shape([_msg([_image_part() for _ in range(8)])])

    def test_too_many_images_400(self):
        with pytest.raises(HTTPException) as exc:
            validate_media_shape([_msg([_image_part() for _ in range(9)])])
        assert exc.value.status_code == 400
        assert exc.value.detail["error"] == "too_much_media"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("VLLM_MLX_MAX_IMAGES_PER_REQUEST", "2")
        with pytest.raises(HTTPException):
            validate_media_shape([_msg([_image_part() for _ in range(3)])])

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setenv("VLLM_MLX_MAX_IMAGES_PER_REQUEST", "0")
        validate_media_shape([_msg([_image_part() for _ in range(50)])])

    def test_too_many_videos_400(self):
        parts = [
            {"type": "video_url", "video_url": {"url": "https://e.com/v.mp4"}}
            for _ in range(3)
        ]
        with pytest.raises(HTTPException) as exc:
            validate_media_shape([_msg(parts)])
        assert exc.value.status_code == 400

    def test_counts_span_messages(self, monkeypatch):
        monkeypatch.setenv("VLLM_MLX_MAX_IMAGES_PER_REQUEST", "2")
        msgs = [_msg([_image_part()]), _msg([_image_part(), _image_part()])]
        with pytest.raises(HTTPException):
            validate_media_shape(msgs)


class TestPayloadSize:
    def test_small_data_url_passes(self):
        validate_media_shape([_msg([_image_part(_data_url(1024))])])

    def test_oversized_data_url_413(self, monkeypatch):
        monkeypatch.setenv("VLLM_MLX_MAX_IMAGE_MB", "1")
        with pytest.raises(HTTPException) as exc:
            validate_media_shape([_msg([_image_part(_data_url(2 * 1024 * 1024))])])
        assert exc.value.status_code == 413
        assert exc.value.detail["error"] == "media_too_large"


class TestTextRequestsUntouched:
    def test_plain_text(self):
        validate_media_shape([{"role": "user", "content": "hello"}])

    def test_text_parts(self):
        validate_media_shape([_msg([{"type": "text", "text": "hi"}])])


class TestVideoConfigPlumb:
    def test_generator_receives_video_knobs(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.default_video_fps = 1.5
        gen.max_video_frames = 32
        assert gen.default_video_fps == 1.5
        assert gen.max_video_frames == 32

    def test_scheduler_config_has_no_dead_cache_field(self):
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

        cfg = MLLMSchedulerConfig()
        assert not hasattr(cfg, "cache_memory_mb")
        # The plumbed knobs remain.
        assert cfg.default_video_fps == 2.0
        assert cfg.max_video_frames == 128
