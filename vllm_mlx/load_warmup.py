# SPDX-License-Identifier: Apache-2.0
"""One short generation at load, so the first real request does not pay the
one-time cost of the very first forward in a fresh process (issue #112).

Off by default. ``VLLM_MLX_LOAD_WARMUP=N`` runs one chat request with
``max_tokens=N`` before the server reports ready; ``VLLM_MLX_LOAD_WARMUP_WORDS=W``
sets the prompt length (default 1 word).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


def load_warmup_tokens() -> int:
    try:
        return max(0, int(os.environ.get("VLLM_MLX_LOAD_WARMUP", "0")))
    except ValueError:
        return 0


async def run_load_warmup(engine: Any, max_tokens: int) -> None:
    try:
        words = max(1, int(os.environ.get("VLLM_MLX_LOAD_WARMUP_WORDS", "1")))
    except ValueError:
        words = 1
    content = " ".join(["Hello"] * words)
    t0 = time.perf_counter()
    try:
        async for output in engine.stream_chat(
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0.0,
        ):
            if output.finished:
                break
        logger.info(
            "[load-warmup] %d-token warm-up done in %.2fs",
            max_tokens,
            time.perf_counter() - t0,
        )
    except Exception as e:  # warm-up must never block startup
        logger.warning("[load-warmup] failed (ignored): %s", str(e)[:200])
