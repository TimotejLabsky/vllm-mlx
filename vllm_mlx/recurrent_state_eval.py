# SPDX-License-Identifier: Apache-2.0
"""Per-step evaluation of recurrent cache state (fork patch #127).

mlx-lm's ``GenerationBatch._step`` async-evaluates only the sampled tokens and
logprobs. The recurrent (GatedDeltaNet / conv / Mamba) cache states written by
the step are never evaluated themselves, so each decode step leaves ~one small
live Metal buffer per recurrent layer behind. Bytes stay flat, the resource
COUNT does not: a hybrid model dies with ``[metal::malloc] Resource limit
(499000) exceeded`` after ~10.5K generated tokens on Qwen3.8-27B-4bit (21.6K on
Qwen3.5-4B). ``generate_step`` escapes it only through its per-step ``.item()``.

mlx-lm #1911 adds every cache's ``state`` to the step's ``async_eval``; that
fixes the crash but costs 15-25 % decode at 10-27K tokens, because forcing the
batch KV caches defeats their in-place slice update. Evaluating only the
recurrent states fixes it at no cost (measured on the Studio 2026-10-05:
27B crash at 10,561 -> 14,000 clean, decode +0.1-1.1 %; 4B 21,638 -> 40,000).

Called once per scheduler step, right after ``BatchGenerator.next()``.
``VLLM_MLX_EVAL_RECURRENT_STATES=0`` disables it.
"""

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

try:
    from mlx_lm.models.cache import ArraysCache, CacheList
except ImportError:  # pragma: no cover - very old mlx-lm
    ArraysCache = CacheList = None


def enabled() -> bool:
    return os.environ.get("VLLM_MLX_EVAL_RECURRENT_STATES", "1") != "0"


def recurrent_states(caches) -> list:
    """The ``state`` of every recurrent (``ArraysCache``-family) layer cache,
    recursing into ``CacheList``. Attention KV caches are skipped on purpose:
    evaluating them every step is what makes the upstream fix slow."""
    if ArraysCache is None:
        return []
    out = []
    for cache in caches or ():
        if isinstance(cache, CacheList):
            out.extend(recurrent_states(cache.caches))
        elif isinstance(cache, ArraysCache):
            out.append(cache.state)
    return out


def eval_recurrent_cache_states(batch_generator) -> None:
    """Queue evaluation of the generation batch's recurrent cache states."""
    batch = getattr(batch_generator, "_generation_batch", None)
    if batch is None:
        return
    try:
        states = recurrent_states(getattr(batch, "prompt_cache", None))
    except Exception:
        logger.debug("[recurrent_state_eval] cache walk failed", exc_info=True)
        return
    if states:
        mx.async_eval(states)
