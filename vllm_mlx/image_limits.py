# SPDX-License-Identifier: Apache-2.0
"""Request-shape limits for media content (vision series, #66).

Bounds what a single chat request may carry BEFORE any engine work:
media item counts and inline base64 payload sizes. The vision encode is
atomic (full VLM forward per row), so item counts multiply directly into
transient memory; remote-URL size is already bounded by the downloader's
``MAX_IMAGE_SIZE``, but inline ``data:`` payloads had no cap at all.

Mirrors ``audio_limits.py``: env-configured, raises ``HTTPException``
(400 for counts, 413 for payload size), wired in the server's
``_prepare_chat_messages`` so both stream and non-stream paths reject
before SSE headers go out.
"""

import os

from fastapi import HTTPException

DEFAULT_MAX_IMAGES_PER_REQUEST = 8
DEFAULT_MAX_VIDEOS_PER_REQUEST = 2
DEFAULT_MAX_AUDIO_PER_REQUEST = 4
# Matches the remote-download cap in models/mllm.py (MAX_IMAGE_SIZE).
DEFAULT_MAX_IMAGE_MB = 20

_IMAGE_TYPES = ("image_url", "image")
_VIDEO_TYPES = ("video_url", "video")
_AUDIO_TYPES = ("audio_url", "audio", "input_audio")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _part_type_and_url(part) -> tuple[str, str]:
    """(type, url) for a content part in dict or pydantic form."""
    if isinstance(part, dict):
        part_type = part.get("type", "") or ""
    else:
        part_type = getattr(part, "type", "") or ""
    url = ""
    for key in ("image_url", "video_url", "audio_url"):
        holder = (
            part.get(key) if isinstance(part, dict) else getattr(part, key, None)
        )
        if holder is None:
            continue
        if isinstance(holder, dict):
            url = holder.get("url", "") or ""
        else:
            url = getattr(holder, "url", None) or (
                holder if isinstance(holder, str) else ""
            )
        if url:
            break
    return part_type, url


def _decoded_data_url_bytes(url: str) -> int:
    """Approximate decoded size of a ``data:`` URL payload (0 for others)."""
    if not url.startswith("data:"):
        return 0
    _, _, payload = url.partition(",")
    if not payload:
        return 0
    # base64 expands 3 bytes to 4 chars; padding makes this an upper-ish
    # estimate that is plenty accurate for a limit check.
    return (len(payload) * 3) // 4


def validate_media_shape(messages) -> None:
    """Enforce per-request media limits; no-op when limits are disabled (0).

    Raises HTTPException 400 (too many items) or 413 (payload too large).
    """
    max_images = _env_int(
        "VLLM_MLX_MAX_IMAGES_PER_REQUEST", DEFAULT_MAX_IMAGES_PER_REQUEST
    )
    max_videos = _env_int(
        "VLLM_MLX_MAX_VIDEOS_PER_REQUEST", DEFAULT_MAX_VIDEOS_PER_REQUEST
    )
    max_audio = _env_int(
        "VLLM_MLX_MAX_AUDIO_PER_REQUEST", DEFAULT_MAX_AUDIO_PER_REQUEST
    )
    max_image_bytes = (
        _env_int("VLLM_MLX_MAX_IMAGE_MB", DEFAULT_MAX_IMAGE_MB) * 1024 * 1024
    )

    n_images = n_videos = n_audio = 0
    for msg in messages:
        content = (
            msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        )
        if not isinstance(content, list):
            continue
        for part in content:
            part_type, url = _part_type_and_url(part)
            if part_type in _IMAGE_TYPES:
                n_images += 1
                decoded = _decoded_data_url_bytes(url)
                if max_image_bytes > 0 and decoded > max_image_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "error": "media_too_large",
                            "message": (
                                f"inline image is ~{decoded} bytes decoded, "
                                f"limit {max_image_bytes} "
                                f"(VLLM_MLX_MAX_IMAGE_MB)"
                            ),
                        },
                    )
            elif part_type in _VIDEO_TYPES:
                n_videos += 1
            elif part_type in _AUDIO_TYPES:
                n_audio += 1

    for count, cap, noun, env in (
        (n_images, max_images, "images", "VLLM_MLX_MAX_IMAGES_PER_REQUEST"),
        (n_videos, max_videos, "videos", "VLLM_MLX_MAX_VIDEOS_PER_REQUEST"),
        (n_audio, max_audio, "audio clips", "VLLM_MLX_MAX_AUDIO_PER_REQUEST"),
    ):
        if cap > 0 and count > cap:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "too_much_media",
                    "message": f"request carries {count} {noun}, limit {cap} ({env})",
                },
            )
