# SPDX-License-Identifier: Apache-2.0
"""Images through the TEXT scheduler (PATCHES.md #181, ``--vision-unified``).

A Qwen3.5-family checkpoint with a vision tower is loaded once through
mlx-vlm; the text model the batched scheduler runs comes from
``build_text_model`` (zero-copy). A media request is vision-encoded with the
mlx-vlm model, its prompt up to the end of the last image is prefilled into a
fresh single-row cache with merged embeddings and multimodal (3-axis) RoPE
positions, and the request then enters the ordinary ``BatchGenerator`` as if
that cache were a prefix-cache hit. Everything after the last image (the
trailing text, the final token, all decode) is ordinary batched text work;
the only difference is the RoPE position, which is ``cache offset + delta``.

Two mechanisms carry the positions, both installed only when the flag is on:

1. **Prefill positions.** Every full-attention layer's ``rope`` is wrapped in
   :class:`MRoPEDispatch`. With no positions set (every text forward, every
   decode step) it calls the wrapped ``nn.RoPE`` with the same arguments: the
   same op on the same inputs. Inside :func:`mrope_positions` it applies the
   checkpoint's interleaved MRoPE with explicit ``(3, B, L)`` positions.
2. **The delta after prefill.** mlx-lm's ``BatchKVCache`` uses ``offset``
   only as the RoPE position (the mask and the buffer index use ``_idx`` and
   ``left_padding``), and ``filter``/``extend``/``trim`` carry it per row.
   A media row's single-row ``KVCache`` layers carry ``rope_delta``; the
   patched ``BatchKVCache.merge`` adds it to that row's offset, and the
   patched ``extract`` writes it back onto the extracted row. A row without
   a delta is untouched (no op is added).
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# Text ``model_type`` prefixes whose mlx-lm class is the qwen3_5 TextModel
# (dense and MoE share the attention module, so the RoPE seam is the same).
SUPPORTED_MODEL_TYPES = ("qwen3_5", "qwen3_5_moe")

_STATE = threading.local()


def _active_positions() -> Optional[mx.array]:
    return getattr(_STATE, "positions", None)


@contextmanager
def mrope_positions(position_ids: mx.array) -> Iterator[None]:
    """Apply explicit ``(3, B, L)`` MRoPE positions to every wrapped layer
    for the forwards run inside the block (this thread only)."""
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError(
            f"MRoPE positions must be (3, B, L), got {tuple(position_ids.shape)}"
        )
    prev = _active_positions()
    _STATE.positions = position_ids
    try:
        yield
    finally:
        _STATE.positions = prev


def interleaved_selector(mrope_section: Sequence[int], freq_dim: int) -> mx.array:
    """Axis (0=T, 1=H, 2=W) feeding each rotary frequency, Qwen3.5's
    interleaved layout (``mrope_interleaved: true``): T everywhere, H on
    indices 1, 4, 7 ... and W on 2, 5, 8 ... up to 3x their section length."""
    selector = [0] * freq_dim
    for axis, start in ((1, 1), (2, 2)):
        for idx in range(start, min(mrope_section[axis] * 3, freq_dim), 3):
            selector[idx] = axis
    return mx.array(selector, dtype=mx.int32)


class MRoPEDispatch(nn.Module):
    """Wraps one attention layer's ``nn.RoPE``. Outside
    :func:`mrope_positions` it is a pass-through; inside, it rotates the
    first ``dims`` features (half-split pairing, as ``nn.RoPE`` with
    ``traditional=False``) by per-token, per-axis positions."""

    def __init__(self, inner: nn.Module, mrope_section: Sequence[int]):
        super().__init__()
        self.inner = inner
        dims = int(inner.dims)
        if getattr(inner, "traditional", False):
            raise ValueError("MRoPE dispatch supports half-split RoPE only")
        if float(getattr(inner, "scale", 1.0)) != 1.0:
            raise ValueError("MRoPE dispatch supports unscaled RoPE only")
        self._dims = dims
        base = float(inner.base)
        self._inv_freq = 1.0 / (
            base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        )
        self._selector = interleaved_selector(list(mrope_section), dims // 2)
        mx.eval(self._inv_freq, self._selector)

    def __call__(self, x: mx.array, offset: Any = 0) -> mx.array:
        positions = _active_positions()
        if positions is None:
            return self.inner(x, offset=offset)
        return self._apply(x, positions)

    def _apply(self, x: mx.array, positions: mx.array) -> mx.array:
        # x: (B, H, L, D); positions: (3, B, L)
        B, _, L, _ = x.shape
        if positions.shape[1] != B or positions.shape[2] != L:
            raise ValueError(
                f"MRoPE positions {tuple(positions.shape)} do not match "
                f"input batch/length ({B}, {L})"
            )
        pos = mx.take(positions, self._selector, axis=0)  # (F, B, L)
        freqs = pos.transpose(1, 2, 0).astype(mx.float32) * self._inv_freq
        cos = mx.cos(freqs)[:, None, :, :]  # (B, 1, L, F)
        sin = mx.sin(freqs)[:, None, :, :]
        half = self._dims // 2
        x1 = x[..., :half].astype(mx.float32)
        x2 = x[..., half : self._dims].astype(mx.float32)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        rotated = mx.concatenate([r1, r2], axis=-1).astype(x.dtype)
        if self._dims == x.shape[-1]:
            return rotated
        return mx.concatenate([rotated, x[..., self._dims :]], axis=-1)


def _attention_modules(text_model: Any) -> list:
    layers = getattr(text_model, "layers", None)
    if layers is None:
        inner = getattr(text_model, "language_model", None)
        layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("text model exposes no .layers")
    return [
        layer.self_attn
        for layer in layers
        if not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn")
    ]


def install_mrope(text_model: Any, mrope_section: Sequence[int]) -> int:
    """Wrap the RoPE of every full-attention layer; idempotent. Returns the
    number of layers wrapped (0 if already installed)."""
    if len(mrope_section) != 3:
        raise ValueError(f"mrope_section must have 3 entries, got {mrope_section}")
    count = 0
    for attn in _attention_modules(text_model):
        rope = attn.rope
        if isinstance(rope, MRoPEDispatch):
            continue
        if not isinstance(rope, nn.RoPE):
            raise ValueError(
                f"unsupported RoPE module {type(rope).__name__} "
                "(vision-unified needs nn.RoPE)"
            )
        attn.rope = MRoPEDispatch(rope, mrope_section)
        count += 1
    if count == 0 and not _attention_modules(text_model):
        raise ValueError("text model has no full-attention layers")
    return count


# ---------------------------------------------------------------------------
# Per-row RoPE delta through mlx-lm's batch caches
# ---------------------------------------------------------------------------

_BATCH_PATCHED = False


def install_batch_cache_delta() -> None:
    """Patch ``BatchKVCache.merge``/``extract`` so a row's ``rope_delta``
    survives into the batch offset and back out. Idempotent; a no-op on
    rows that carry no delta."""
    global _BATCH_PATCHED
    if _BATCH_PATCHED:
        return
    from mlx_lm.models.cache import BatchKVCache

    orig_merge = BatchKVCache.merge.__func__
    orig_extract = BatchKVCache.extract

    def merge(cls, caches):
        batch = orig_merge(cls, caches)
        deltas = [int(getattr(c, "rope_delta", 0) or 0) for c in caches]
        if any(deltas):
            batch.offset = batch.offset + mx.array(deltas, dtype=batch.offset.dtype)
        return batch

    def extract(self, idx):
        cache = orig_extract(self, idx)
        delta = int(self.offset[idx].item()) - int(cache.offset)
        if delta:
            cache.rope_delta = delta
        return cache

    merge.__vllm_mlx_181__ = True
    extract.__vllm_mlx_181__ = True
    BatchKVCache.merge = classmethod(merge)
    BatchKVCache.extract = extract
    _BATCH_PATCHED = True


def set_rope_delta(cache: Sequence[Any], delta: int) -> None:
    """Stamp ``delta`` on every KV layer of a single-row prompt cache."""
    for layer in cache:
        if hasattr(layer, "keys") and hasattr(layer, "offset"):
            if delta:
                layer.rope_delta = int(delta)
            elif hasattr(layer, "rope_delta"):
                del layer.rope_delta


def rope_delta_of(cache: Sequence[Any]) -> int:
    """The delta a single-row cache carries (0 if none). Raises if the KV
    layers disagree: that cache is corrupt for a media row."""
    seen = {
        int(getattr(layer, "rope_delta", 0) or 0)
        for layer in cache
        if hasattr(layer, "keys") and hasattr(layer, "offset")
    }
    if len(seen) > 1:
        raise ValueError(f"cache layers carry different rope deltas: {seen}")
    return seen.pop() if seen else 0


# ---------------------------------------------------------------------------
# Media prefill
# ---------------------------------------------------------------------------


def media_prefill(
    text_model: Any,
    input_ids: mx.array,
    input_embeddings: mx.array,
    position_ids: mx.array,
    cache: list,
    *,
    step: int = 2048,
) -> None:
    """Prefill ``input_ids`` (1, L) with merged embeddings (1, L, D) and
    MRoPE positions (3, 1, L) into a fresh single-row ``cache``."""
    L = int(input_ids.shape[1])
    if input_embeddings.shape[1] != L or position_ids.shape[-1] != L:
        raise ValueError("ids, embeddings and positions must be the same length")
    for start in range(0, L, step):
        end = min(start + step, L)
        with mrope_positions(position_ids[:, :, start:end]):
            text_model(
                input_ids[:, start:end],
                cache=cache,
                input_embeddings=input_embeddings[:, start:end],
            )
        mx.eval([c.state for c in cache])


# ---------------------------------------------------------------------------
# Media inputs
# ---------------------------------------------------------------------------


@dataclass
class MediaInputs:
    """One request's processed images, host-side (numpy) so nothing lazy
    crosses threads. Carried on ``Request.images`` as a one-item list."""

    input_ids: list
    pixel_values: np.ndarray
    image_grid_thw: np.ndarray
    image_digests: tuple = ()
    n_images: int = 0
    image_tokens: int = 0
    preprocess_s: float = 0.0
    stats: dict = field(default_factory=dict)


def media_of(request: Any) -> Optional[MediaInputs]:
    """The :class:`MediaInputs` of a media request on a --vision-unified
    route, else None. A plain attribute read for every text request."""
    images = getattr(request, "images", None)
    if images and isinstance(images[0], MediaInputs):
        return images[0]
    return None


def _np(value: Any) -> np.ndarray:
    import mlx.core as mx

    if isinstance(value, mx.array):
        return np.array(value)
    return np.asarray(value)


# ---------------------------------------------------------------------------
# Per-scheduler media hooks (run on the generation worker)
# ---------------------------------------------------------------------------


class MediaRows:
    """Attached to the text ``Scheduler`` as ``_vision_unified``. Owns the
    mlx-vlm model whose vision tower and ``get_input_embeddings`` encode a
    media request at schedule time."""

    def __init__(self, vlm_model: Any, mrope_section: list):
        self.vlm = vlm_model
        cfg = getattr(vlm_model, "config", None)
        self.image_token_id = getattr(cfg, "image_token_id", None) or getattr(
            cfg, "image_token_index", None
        )
        self.video_token_id = getattr(cfg, "video_token_id", None)
        self.vision_end_id = getattr(cfg, "vision_end_token_id", None)
        self.mrope_section = mrope_section
        self._lock = threading.Lock()
        self.counters = {
            "media_requests": 0,
            "media_images": 0,
            "media_image_tokens": 0,
            "media_prefill_tokens": 0,
            "media_encode_prefill_s": 0.0,
            "media_errors": 0,
        }

    def stats(self) -> dict:
        with self._lock:
            out = dict(self.counters)
        out["media_encode_prefill_s"] = round(out["media_encode_prefill_s"], 3)
        return out

    def count(self, **deltas) -> None:
        with self._lock:
            for k, v in deltas.items():
                self.counters[k] = self.counters.get(k, 0) + v

    def media_cut(self, ids: list) -> int:
        """Tokens prefilled out of band: through the end of the LAST image
        (its vision_end marker when present), never the final token."""
        last = -1
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] == self.image_token_id or ids[i] == self.video_token_id:
                last = i
                break
        if last < 0:
            raise ValueError("media request has no image tokens")
        cut = last + 1
        if (
            self.vision_end_id is not None
            and cut < len(ids)
            and ids[cut] == (self.vision_end_id)
        ):
            cut += 1
        return min(cut, len(ids) - 1)

    def prefill(self, scheduler: Any, request: Any) -> tuple:
        """Encode + prefill one media request. Returns ``(tokens_to_process,
        cache)`` for ``BatchGenerator.insert``."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        media = media_of(request)
        tic = time.perf_counter()
        ids_list = list(media.input_ids)
        ids = mx.array([ids_list])
        feats = self.vlm.get_input_embeddings(
            ids,
            mx.array(media.pixel_values),
            image_grid_thw=mx.array(media.image_grid_thw),
        )
        emb, pos, deltas = feats.inputs_embeds, feats.position_ids, feats.rope_deltas
        if pos is None or pos.ndim != 3:
            raise ValueError("vision model returned no 3-axis positions")
        delta = int(deltas.reshape(-1)[0].item())
        cut = self.media_cut(ids_list)
        cache = make_prompt_cache(scheduler.model)
        step = int(getattr(scheduler.config, "prefill_step_size", 2048) or 2048)
        media_prefill(
            scheduler.model,
            ids[:, :cut],
            emb[:, :cut],
            pos[:, :, :cut],
            cache,
            step=step,
        )
        set_rope_delta(cache, delta)
        del feats, emb, pos
        elapsed = time.perf_counter() - tic
        self.count(
            media_requests=1,
            media_images=media.n_images,
            media_image_tokens=media.image_tokens,
            media_prefill_tokens=cut,
            media_encode_prefill_s=elapsed,
        )
        media.stats.update(delta=delta, cut=cut, encode_prefill_s=round(elapsed, 3))
        logger.info(
            "[vision_unified] request=%s images=%d image_tokens=%d "
            "prefilled=%d/%d delta=%d encode+prefill=%.2fs",
            request.request_id[:12],
            media.n_images,
            media.image_tokens,
            cut,
            len(ids_list),
            delta,
            elapsed,
        )
        return ids_list[cut:], cache


def mark_uncached(request: Any) -> None:
    """(stage 2) A media request reads no cache: a cold miss."""
    request.cache_hit_type = "miss"
    request.prompt_cache = None
    request.cached_tokens = 0
    request.remaining_tokens = request.prompt_token_ids


def prefill_media_row(scheduler: Any, request: Any) -> tuple:
    rows = getattr(scheduler, "_vision_unified", None)
    if rows is None:
        raise RuntimeError("media request on a scheduler without --vision-unified")
    return rows.prefill(scheduler, request)


def count_media_error(scheduler: Any) -> None:
    rows = getattr(scheduler, "_vision_unified", None)
    if rows is not None:
        rows.count(media_errors=1)


def release_media_request(scheduler: Any, request_id: str, request: Any) -> None:
    """_cleanup_finished for a media request: nothing is stored (stage 2);
    drop any pending checkpoint ladder and the extraction reference."""
    hybrid_kv = getattr(scheduler, "hybrid_kv", None)
    if hybrid_kv is not None:
        hybrid_kv.discard_pending(request_id)
    request._extracted_cache = None


def batch_has_media_row(cache: Any) -> bool:
    """True when any row of a batch cache carries a RoPE delta (a media row
    after its prefill). Answers False unless the #181 patch is installed."""
    if not _BATCH_PATCHED or not cache:
        return False
    for layer in cache:
        offset = getattr(layer, "offset", None)
        left = getattr(layer, "left_padding", None)
        idx = getattr(layer, "_idx", None)
        if isinstance(offset, mx.array) and left is not None and idx is not None:
            return bool(mx.any(offset != (idx - left)).item())
    return False
