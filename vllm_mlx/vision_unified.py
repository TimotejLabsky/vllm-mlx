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
   patched ``BatchKVCache.merge`` adds it to that row's offset. A row
   without a delta is untouched (no op is added).
"""

from __future__ import annotations

import hashlib
import logging
import struct
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

import mlx.core as mx
import numpy as np

# ``mlx.nn`` is imported lazily (``_mrope_class``): this module is imported
# by the scheduler, the batched system-KV and the text drafter on EVERY
# route, so with the flag off it must add no import the text path lacks
# (the no-MLX unit tests stub ``mlx`` with ``mlx.core`` only).

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


_MROPE_CLASS = None


def _mrope_class():
    """The :class:`MRoPEDispatch` class, built on first use."""
    global _MROPE_CLASS
    if _MROPE_CLASS is not None:
        return _MROPE_CLASS
    import mlx.nn as nn

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

    _MROPE_CLASS = MRoPEDispatch
    return MRoPEDispatch


def __getattr__(name: str):
    if name == "MRoPEDispatch":
        return _mrope_class()
    raise AttributeError(name)


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
    import mlx.nn as nn

    MRoPEDispatch = _mrope_class()
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
    """Patch ``BatchKVCache.merge`` so a row's ``rope_delta`` (stamped on its
    single-row ``KVCache`` layers) is added to that row's batch offset.
    Idempotent; adds no op when no row carries a delta. ``extract`` is left
    stock: an extracted row (a cache snapshot) carries no delta, because a
    restore recomputes it from the request's own positions (stage 3)."""
    global _BATCH_PATCHED
    if _BATCH_PATCHED:
        return
    from mlx_lm.models.cache import BatchKVCache

    orig_merge = BatchKVCache.merge.__func__

    def merge(cls, caches):
        batch = orig_merge(cls, caches)
        deltas = [int(getattr(c, "rope_delta", 0) or 0) for c in caches]
        if any(deltas):
            batch.offset = batch.offset + mx.array(deltas, dtype=batch.offset.dtype)
        return batch

    merge.__vllm_mlx_181__ = True
    BatchKVCache.merge = classmethod(merge)
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
    crosses threads. Carried on ``Request.images`` as a one-item list.

    ``cache_key`` (stage 3) is the prompt with every image's placeholder run
    replaced by a content fingerprint (see :func:`media_cache_key`); None
    when an image has no trustworthy digest (a remote URL), and then the
    request runs uncached. ``keyed`` is set once the scheduler adopted the
    key as the request's ``prompt_token_ids``."""

    input_ids: list
    pixel_values: np.ndarray
    image_grid_thw: np.ndarray
    image_digests: tuple = ()
    n_images: int = 0
    image_tokens: int = 0
    preprocess_s: float = 0.0
    cache_key: Optional[list] = None
    keyed: bool = False
    prompt_text: Optional[str] = None
    n_downscaled: int = 0
    plan: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def media_of(request: Any) -> Optional[MediaInputs]:
    """The :class:`MediaInputs` of a media request on a --vision-unified
    route, else None. A plain attribute read for every text request."""
    images = getattr(request, "images", None)
    if images and isinstance(images[0], MediaInputs):
        return images[0]
    return None


def media_uncached(request: Any) -> bool:
    """A media request that must not read or write any cache."""
    media = media_of(request)
    return media is not None and not media.keyed


def _fingerprint(seed: bytes, length: int) -> list:
    words = struct.unpack(">8I", hashlib.sha256(seed).digest())
    return [
        -(1 + ((words[j % 8] ^ (j * 0x9E3779B1)) & 0x7FFFFFFE)) for j in range(length)
    ]


def media_cache_key(
    ids: Sequence[int],
    image_token_id: int,
    digests: Sequence[Optional[bytes]],
    grids: Any,
) -> Optional[list]:
    """The cache key of a media prompt: ``ids`` with the k-th contiguous
    image-placeholder run replaced by negative values derived from the k-th
    image's content digest AND its ``(t, h, w)`` grid.

    * same images + same text -> the same key (re-sends and follow-ups hit);
    * a different image -> the keys part at its first placeholder, so only
      the text before it can be shared (where every position is 1-D);
    * the grid is in the fingerprint, so a key prefix fixes every MRoPE
      position (and so the delta) up to its end: an entry made with one
      delta can never be matched by a request whose positions differ.
      Negative values never collide with token ids (vocab ids are >= 0).
    """
    if not digests or any(d is None for d in digests):
        return None
    grid_rows = [list(map(int, g)) for g in np.asarray(grids).reshape(-1, 3)]
    runs, start = [], None
    for i, tok in enumerate(ids):
        if tok == image_token_id:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(ids)))
    if len(runs) != len(digests) or len(grid_rows) != len(digests):
        return None
    key = list(ids)
    for (a, b), digest, grid in zip(runs, digests, grid_rows):
        seed = bytes(digest) + struct.pack(">3q", *grid)
        key[a:b] = _fingerprint(seed, b - a)
    return key


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
        self.jobs: "OrderedDict[str, _MediaJob]" = OrderedDict()
        self.prefix_rows: dict = {}
        self._lock = threading.Lock()
        self.counters = {
            "media_requests": 0,
            "media_images": 0,
            "media_image_tokens": 0,
            "media_prefill_tokens": 0,
            "media_prefill_chunks": 0,
            "media_prefill_s": 0.0,
            "media_encode_s": 0.0,
            "media_errors": 0,
            "media_images_downscaled": 0,
            "media_cache_hits": 0,
            "media_cached_tokens": 0,
            "media_encode_skipped": 0,
        }

    def stats(self) -> dict:
        with self._lock:
            out = dict(self.counters)
        for k in ("media_prefill_s", "media_encode_s"):
            out[k] = round(out[k], 3)
        out["media_jobs_active"] = len(self.jobs)
        out["media_prefix_rows"] = len(self.prefix_rows)
        return out

    def count(self, **deltas) -> None:
        with self._lock:
            for k, v in deltas.items():
                self.counters[k] = self.counters.get(k, 0) + v

    def media_cut(self, ids: list) -> int:
        """End of the image span: through the end of the LAST image
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

    # -- planning (schedule time, cheap) ------------------------------------
    def first_image(self, ids: list) -> int:
        for i, tok in enumerate(ids):
            if tok == self.image_token_id or tok == self.video_token_id:
                return i
        raise ValueError("media request has no image tokens")

    def plan(self, scheduler: Any, request: Any) -> tuple:
        """Decide how one media request enters the batch. Returns
        ``(tokens_to_process, cache)`` for the scheduler's insert; the plan
        kind is kept in ``media.plan``:

        * ``insert`` - a restore covers the last image: no encode; the rest
          is an ordinary row (delta from the grids).
        * ``prefix`` - text before the first image is still to prefill: it
          is inserted as an ordinary row (same chunking, checkpoints, relief
          as text) that stops at the first image; the image span then runs
          as a chunked job (:meth:`step`).
        * ``job`` - a restore ends inside [first image, last image): the
          image span runs as a job right away, nothing is inserted now.
        """
        import mlx.core as mx

        media = media_of(request)
        ids = list(media.input_ids)
        L = len(ids)
        cut = self.media_cut(ids)
        fis = self.first_image(ids)
        cache = request.prompt_cache if media.keyed else None
        n = int(request.cached_tokens or 0) if cache else 0
        if cache is not None and not (0 < n < L):
            # full coverage cannot replay: go cold
            cache, n = None, 0
            request.prompt_cache = None
            request.cached_tokens = 0
            request.cache_hit_type = "miss"
        media.plan = {"n": n, "fis": fis, "cut": cut}
        self.count(
            media_requests=1,
            media_images=media.n_images,
            media_image_tokens=media.image_tokens,
            media_images_downscaled=media.n_downscaled,
            media_cache_hits=1 if n > 0 else 0,
            media_cached_tokens=n,
        )
        if cache is not None and n >= cut:
            grid = mx.array(media.image_grid_thw)
            _, deltas = self.vlm.language_model.get_rope_index(mx.array([ids]), grid)
            set_rope_delta(cache, int(deltas.reshape(-1)[0].item()))
            self.count(media_encode_skipped=1)
            media.plan.update(kind="insert", base=n, end=L)
            return ids[n:], cache
        if n < fis:
            # stop the row at the first image; the extra token is never
            # prefilled (the row leaves the batch at the segment end)
            media.plan.update(kind="prefix", base=n, end=fis + 1)
            return ids[n : fis + 1], cache
        media.plan.update(kind="job", base=n, end=n)
        return [], cache

    def start_job(self, scheduler: Any, request: Any, cache, insert_kwargs) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        media = media_of(request)
        if cache is None:
            cache = make_prompt_cache(scheduler.model)
        self.jobs[request.request_id] = _MediaJob(
            request=request,
            cache=cache,
            pos=media.plan["n"] if media.plan["kind"] == "job" else media.plan["fis"],
            cut=media.plan["cut"],
            insert_kwargs=insert_kwargs,
        )

    def insert(self, scheduler: Any, request: Any, tokens: list, insert_kwargs: dict):
        """_schedule_waiting's insert for a media row (plans insert/prefix)."""
        media = media_of(request)
        plan = media.plan
        if plan["kind"] == "prefix":
            self.prefix_rows[request.request_id] = (
                plan["fis"] - plan["base"],
                insert_kwargs,
            )
            uids = _segmented_insert(
                scheduler,
                request,
                tokens,
                insert_kwargs,
                base=plan["base"],
                stop=len(tokens) - 1,
            )
        else:
            uids = _segmented_insert(
                scheduler, request, tokens, insert_kwargs, base=plan["base"]
            )
            _MEDIA_UIDS.update(uids)
        return uids

    # -- per step ------------------------------------------------------------
    def after_prompt_responses(self, scheduler: Any, prompt_responses) -> None:
        """A prefix row that reached its first image leaves the batch with
        its cache; its image span becomes a job."""
        if not self.prefix_rows or not prompt_responses:
            return
        for pr in prompt_responses:
            rid = scheduler.uid_to_request_id.get(pr.uid)
            entry = self.prefix_rows.get(rid) if rid is not None else None
            if entry is None:
                continue
            progress = getattr(pr, "progress", None)
            done = progress[0] if isinstance(progress, tuple) else int(progress or 0)
            stop, insert_kwargs = entry
            if done < stop:
                continue
            del self.prefix_rows[rid]
            extracted = scheduler.batch_generator.extract_cache([pr.uid])
            cache = list(extracted[pr.uid][0])
            scheduler.batch_generator.remove([pr.uid])
            scheduler.uid_to_request_id.pop(pr.uid, None)
            scheduler.request_id_to_uid.pop(rid, None)
            request = scheduler.requests.get(rid)
            if request is None:
                continue
            request.batch_uid = None
            self.start_job(scheduler, request, cache, insert_kwargs)

    def step(self, scheduler: Any) -> None:
        """One unit of media work per scheduler step: the vision encode of
        the oldest job (atomic), or one ``prefill_step_size`` chunk of its
        image span; a finished span joins the batch."""
        if not self.jobs:
            return
        rid, job = next(iter(self.jobs.items()))
        request = job.request
        try:
            if job.span is None:
                self._encode(job)
                return
            self._chunk(scheduler, job)
            if job.pos >= job.cut:
                del self.jobs[rid]
                self._finish(scheduler, job)
        except Exception as e:
            self.jobs.pop(rid, None)
            if "Insufficient Memory" in str(e):
                raise  # the step's OOM recovery, as for a text prefill
            logger.error(
                "[vision_unified] request=%s media prefill failed: %s",
                rid[:12],
                e,
                exc_info=True,
            )
            self.count(media_errors=1)
            scheduler.running.pop(rid, None)
            scheduler._reject_at_schedule(request, "media_error")

    def _encode(self, job: "_MediaJob") -> None:
        import mlx.core as mx

        media = media_of(job.request)
        tic = time.perf_counter()
        ids = mx.array([list(media.input_ids)])
        feats = self.vlm.get_input_embeddings(
            ids,
            mx.array(media.pixel_values),
            image_grid_thw=mx.array(media.image_grid_thw),
        )
        pos = feats.position_ids
        if pos is None or pos.ndim != 3:
            raise ValueError("vision model returned no 3-axis positions")
        emb = feats.inputs_embeds[:, job.pos : job.cut]
        pos = pos[:, :, job.pos : job.cut]
        mx.eval(emb, pos)
        job.delta = int(feats.rope_deltas.reshape(-1)[0].item())
        job.span = (ids[:, job.pos : job.cut], emb, pos, job.pos)
        del feats
        dt = time.perf_counter() - tic
        self.count(media_encode_s=dt)
        media.stats["encode_s"] = round(dt, 4)
        logger.info(
            "[vision_unified] request=%s encoded %d image(s), %d image tokens "
            "in %.3fs; span [%d, %d) in chunks",
            job.request.request_id[:12],
            media.n_images,
            media.image_tokens,
            dt,
            job.pos,
            job.cut,
        )

    def _chunk(self, scheduler: Any, job: "_MediaJob") -> None:
        import mlx.core as mx

        ids, emb, pos, origin = job.span
        step = int(getattr(scheduler.config, "prefill_step_size", 2048) or 2048)
        a = job.pos - origin
        b = min(a + step, job.cut - origin)
        tic = time.perf_counter()
        with mrope_positions(pos[:, :, a:b]):
            scheduler.model(ids[:, a:b], cache=job.cache, input_embeddings=emb[:, a:b])
        mx.eval([c.state for c in job.cache])
        job.pos = origin + b
        self.count(
            media_prefill_tokens=b - a,
            media_prefill_chunks=1,
            media_prefill_s=time.perf_counter() - tic,
        )

    def _finish(self, scheduler: Any, job: "_MediaJob") -> None:
        request = job.request
        media = media_of(request)
        ids = list(media.input_ids)
        set_rope_delta(job.cache, job.delta)
        job.span = None
        hybrid_kv = getattr(scheduler, "hybrid_kv", None)
        if media.keyed and hybrid_kv is not None:
            # checkpoint the recurrent state at the end of the last image
            try:
                hybrid_kv.note_scheduled(request.request_id, job.cut)
                hybrid_kv.capture_segment(request.request_id, 0, job.cache)
            except Exception:
                logger.debug(
                    "[vision_unified] image-end checkpoint failed", exc_info=True
                )
        kw = dict(job.insert_kwargs)
        kw["caches"] = [job.cache]
        kw["all_tokens"] = [ids[: job.cut]]
        media.plan.update(kind="insert", base=job.cut, end=len(ids))
        uids = _segmented_insert(scheduler, request, ids[job.cut :], kw, base=job.cut)
        uid = uids[0]
        _MEDIA_UIDS.add(uid)
        scheduler.request_id_to_uid[request.request_id] = uid
        scheduler.uid_to_request_id[uid] = request.request_id
        request.batch_uid = uid
        job.cache = None
        logger.info(
            "[vision_unified] request=%s image span done (delta=%d); %d "
            "trailing tokens join the batch as uid=%d",
            request.request_id[:12],
            job.delta,
            len(ids) - job.cut,
            uid,
        )

    # -- release -------------------------------------------------------------
    def drop(self, scheduler: Any, request_id: str) -> bool:
        job = self.jobs.pop(request_id, None)
        prefix = self.prefix_rows.pop(request_id, None)
        uid = scheduler.request_id_to_uid.get(request_id)
        if uid is not None:
            _MEDIA_UIDS.discard(uid)
        if job is not None:
            job.cache = None
            job.span = None
        return job is not None or prefix is not None

    def drop_all(self) -> None:
        self.jobs.clear()
        self.prefix_rows.clear()
        _MEDIA_UIDS.clear()


@dataclass
class _MediaJob:
    request: Any
    cache: Any
    pos: int
    cut: int
    insert_kwargs: dict
    span: Any = None
    delta: int = 0


# uids of media rows in the live batch (the text drafter bypasses them,
# whatever their delta; read only by text_mtp)
_MEDIA_UIDS: set = set()


def _segmented_insert(
    scheduler: Any,
    request: Any,
    tokens: list,
    insert_kwargs: dict,
    *,
    base: int,
    stop: Optional[int] = None,
):
    """Insert a media row's text part like a text row: with the batched
    system-KV and a keyed request, the same message-boundary (#88/#126),
    uniform-interval and divergence (#117) cuts ``insert_segmented`` makes,
    at the row's absolute start ``base``; ``stop`` forces one more cut."""
    media = media_of(request)
    hybrid_kv = getattr(scheduler, "hybrid_kv", None)
    bg = scheduler.batch_generator
    extra = {stop} if stop else set()
    if hybrid_kv is None or not media.keyed:
        cuts = sorted(c for c in extra if 0 < c < len(tokens))
        segs, start = [], 0
        for c in cuts + [len(tokens)]:
            segs.append(list(tokens[start:c]))
            start = c
        return bg.insert_segments([segs], **insert_kwargs)
    from .batched_system_kv import find_message_boundaries

    boundaries, anchor = (), None
    try:
        markers = hybrid_kv.boundary_marker_ids(media.prompt_text, scheduler.tokenizer)
        boundaries, anchor = find_message_boundaries(
            tokens,
            markers,
            hybrid_kv.boundary_min_step,
            first_min=hybrid_kv.partial_min if base == 0 else None,
            with_anchor=True,
        )
        boundaries = hybrid_kv.add_divergence_cut(
            request.request_id, base, len(tokens), boundaries
        )
    except Exception:
        logger.debug("[vision_unified] boundary detection failed", exc_info=True)
    boundaries = tuple(sorted(set(boundaries) | extra))
    hybrid_kv.note_scheduled(
        request.request_id, base, boundaries=boundaries, anchor=anchor
    )
    return bg.insert_segments(
        [hybrid_kv.split_segments(tokens, boundaries=boundaries)], **insert_kwargs
    )


def key_media_request(scheduler: Any, request: Any) -> bool:
    """add_request: adopt the content key as the request's token ids so the
    batched system-KV matches and stores on it (stage 3). False (the request
    stays uncached) without a key or without the batched system-KV."""
    media = media_of(request)
    if media is None or media.cache_key is None:
        return False
    if getattr(scheduler, "hybrid_kv", None) is None:
        return False
    if len(media.cache_key) != len(request.prompt_token_ids or ()):
        return False
    request.prompt_token_ids = list(media.cache_key)
    media.keyed = True
    return True


def mark_uncached(request: Any) -> None:
    """A media request that reads no cache: a cold miss."""
    request.cache_hit_type = "miss"
    request.prompt_cache = None
    request.cached_tokens = 0
    request.remaining_tokens = request.prompt_token_ids


def _rows(scheduler: Any) -> "MediaRows":
    rows = getattr(scheduler, "_vision_unified", None)
    if rows is None:
        raise RuntimeError("media request on a scheduler without --vision-unified")
    return rows


def plan_media_row(scheduler: Any, request: Any) -> tuple:
    return _rows(scheduler).plan(scheduler, request)


def planned_end(media: MediaInputs) -> int:
    """Real-id index where the planned insert's tokens end."""
    return media.plan["end"]


def planned_job(media: MediaInputs) -> bool:
    return media.plan.get("kind") == "job"


def start_media_job(scheduler: Any, request: Any, cache, insert_kwargs) -> None:
    _rows(scheduler).start_job(scheduler, request, cache, insert_kwargs)


def insert_media(scheduler: Any, request: Any, tokens: list, insert_kwargs: dict):
    return _rows(scheduler).insert(scheduler, request, tokens, insert_kwargs)


def on_step(scheduler: Any, prompt_responses) -> None:
    """step() hook: prefix rows that reached their image become jobs, then
    one unit of media work runs. A no-op on every other route."""
    rows = getattr(scheduler, "_vision_unified", None)
    if rows is None:
        return
    rows.after_prompt_responses(scheduler, prompt_responses)
    rows.step(scheduler)


def drop_media(scheduler: Any, request_id: str) -> None:
    """abort hook: release a media request's job / prefix bookkeeping."""
    rows = getattr(scheduler, "_vision_unified", None)
    if rows is not None:
        rows.drop(scheduler, request_id)


def drop_all_media(scheduler: Any) -> None:
    """recovery / reschedule hook: every job and prefix row is gone (the
    requests are aborted or re-planned from scratch)."""
    rows = getattr(scheduler, "_vision_unified", None)
    if rows is not None:
        rows.drop_all()


def count_media_error(scheduler: Any) -> None:
    rows = getattr(scheduler, "_vision_unified", None)
    if rows is not None:
        rows.count(media_errors=1)


def forget_media_row(scheduler: Any, request_id: str) -> None:
    uid = scheduler.request_id_to_uid.get(request_id)
    if uid is not None:
        _MEDIA_UIDS.discard(uid)


def release_media_request(scheduler: Any, request_id: str, request: Any) -> None:
    """_cleanup_finished for an uncached media request: nothing is stored;
    drop any pending checkpoint ladder and the extraction reference."""
    hybrid_kv = getattr(scheduler, "hybrid_kv", None)
    if hybrid_kv is not None:
        hybrid_kv.discard_pending(request_id)
    request._extracted_cache = None


def batch_has_media_row(batch: Any) -> bool:
    """True when a generation batch holds a media row (any delta, 0
    included). False unless a --vision-unified route inserted one."""
    if not _MEDIA_UIDS:
        return False
    return any(uid in _MEDIA_UIDS for uid in getattr(batch, "uids", ()) or ())
