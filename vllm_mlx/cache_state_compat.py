"""Fork compat layer for mlx-lm's two cache-state protocols (PATCHES.md #137).

mlx-lm PR #1778 (``ee19be4``, "Make 'state' of cache return full state")
changed the cache protocol for every class in ``mlx_lm.models.cache``:

- ``meta_state`` is gone; its scalars ride INSIDE ``state``
  (``KVCache.state`` is ``(keys, values, offset)``, ``RotatingKVCache``
  ``(keys, values, offset, keep, max_size, _idx)``, ...);
- ``state`` returns the FULL pre-allocated buffers, not the offset-sliced
  view (the sliced view moved to ``keys_and_values()``);
- ``from_state(state, meta_state)`` became ``from_state(state)``;
- ``ArraysCache.state`` carries ``None`` metadata (was ``mx.array([])``).

The fork's cache stack (system-KV, batched system-KV, SSD tier, prefix /
memory caches, mid-prefill save) stores and restores states in the LEGACY
protocol: an offset-sliced ``state`` plus a separate string-tuple
``meta_state``. Rather than teach every snapshot, ladder and on-disk format
a second shape, this module presents the legacy protocol on BOTH mlx-lm
builds: on the old pin every helper is the exact expression the call sites
used before (byte-identical); on a post-#1778 mlx-lm the helpers convert
per class.

The sliced view is not cosmetic. A post-#1778 ``KVCache.state`` hands out
the live, over-allocated buffer; installing that into two caches makes both
write their next tokens into the SAME array (mx ``__setitem__`` rebinds the
shared object), i.e. a system-KV snapshot restored into two requests would
cross-contaminate them. The legacy view restores an exactly-full buffer, so
the first write after a restore allocates a fresh one — the pre-#1778
behaviour this stack's aliasing discipline (patches #6, #81) relies on.

Conversion is keyed on WHICH class's ``state`` property an object uses, not
on its name: a subclass that inherits ``KVCache.state`` (e.g. the fork's
``BatchMambaCache`` inherits ``ArraysCache.state``) converts like its base;
caches from other libraries (mlx-vlm keeps its own legacy ``_BaseCache``)
and anything unknown pass through untouched.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _detect_new_state_shape(base_cls) -> bool:
    """True when ``base_cls`` (an mlx-lm ``_BaseCache``) speaks the #1778
    protocol: single-argument ``from_state`` and no ``meta_state``."""
    if base_cls is None:
        return False
    try:
        params = list(inspect.signature(base_cls.from_state).parameters)
    except (TypeError, ValueError):
        return not hasattr(base_cls, "meta_state")
    # Bound classmethod: ``cls`` is already consumed.
    return len(params) == 1 and not hasattr(base_cls, "meta_state")


try:  # pragma: no cover - import guard for no-MLX CI lanes
    from mlx_lm.models import cache as _lm_cache
except Exception:  # pragma: no cover
    _lm_cache = None

NEW_STATE_SHAPE: bool = _detect_new_state_shape(getattr(_lm_cache, "_BaseCache", None))


def _state_owner(cls) -> Optional[type]:
    """The class in ``cls``'s MRO that defines the ``state`` property."""
    for k in cls.__mro__:
        if "state" in k.__dict__:
            return k
    return None


def _defines_meta_state(cls) -> bool:
    return any("meta_state" in k.__dict__ for k in cls.__mro__)


def _lm_class(name: str):
    return getattr(_lm_cache, name, None) if _lm_cache is not None else None


def _kind(c) -> Optional[str]:
    """Name of the mlx-lm class whose ``state`` ``c`` uses, when that class
    speaks the #1778 protocol and needs conversion; else None (passthrough).
    """
    if not NEW_STATE_SHAPE:
        return None
    cls = type(c)
    if _defines_meta_state(cls):
        # Some class in the MRO still carries the legacy pair (mlx-vlm's own
        # caches, vendored model caches): speak it natively.
        return None
    owner = _state_owner(cls)
    if owner is None:
        return None
    name = owner.__name__
    if _lm_class(name) is not owner:
        return None
    return name


def _empty_meta_array():
    import mlx.core as mx

    return mx.array([])


def _none_if_empty(a):
    return a if a is not None and getattr(a, "size", 0) > 0 else None


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


def legacy_state(c) -> Any:
    """``c.state`` in the legacy (pre-#1778) shape: offset-sliced arrays,
    scalars excluded (they live in :func:`legacy_meta_state`)."""
    kind = _kind(c)
    if kind is None:
        return c.state
    if kind in ("KVCache", "RotatingKVCache", "QuantizedKVCache"):
        # keys_and_values() is the pre-#1778 state body, verbatim (it raises
        # on an empty cache exactly like the old property did).
        return c.keys_and_values()
    if kind == "ChunkedKVCache":
        if c.offset == c.keys.shape[2]:
            return c.keys, c.values
        return c.keys[..., : c.offset, :], c.values[..., : c.offset, :]
    if kind == "ArraysCache":
        lp = c.left_padding if c.left_padding is not None else _empty_meta_array()
        ln = c.lengths if c.lengths is not None else _empty_meta_array()
        return c.cache, lp, ln
    if kind == "CacheList":
        return [legacy_state(s) for s in c.caches]
    if kind == "BatchKVCache":
        k, v = c.keys, c.values
        if c._idx < k.shape[2]:
            k = k[..., : c._idx, :]
            v = v[..., : c._idx, :]
        return k, v, c.offset, c.left_padding
    if kind == "BatchRotatingKVCache":
        k, v = c.keys, c.values
        if c._offset < k.shape[2]:
            k, v = k[..., : c._offset, :], v[..., : c._offset, :]
        return k, v, c.offset, c.left_padding
    # A class #1778 did not reshape (ConcatenateKVCache, _BaseCache).
    return c.state


_RAISE = object()


def legacy_meta_state(c, default: Any = _RAISE) -> Any:
    """``c.meta_state`` in the legacy shape (string tuple, ``""`` when
    trivial). With ``default`` it is the ``getattr(c, "meta_state",
    default)`` contract; without it a cache with no meta_state raises
    AttributeError, exactly like the bare attribute read."""
    kind = _kind(c)
    if kind is None:
        if default is _RAISE:
            return c.meta_state
        return getattr(c, "meta_state", default)
    if kind == "QuantizedKVCache":
        return tuple(map(str, (c.offset, c.group_size, c.bits)))
    if kind == "RotatingKVCache":
        return tuple(map(str, (c.keep, c.max_size, c.offset, c._idx)))
    if kind == "ChunkedKVCache":
        return tuple(map(str, (c.chunk_size, c.start_position)))
    if kind == "CacheList":
        return (
            [type(s).__name__ for s in c.caches],
            [legacy_meta_state(s) for s in c.caches],
        )
    if kind == "BatchRotatingKVCache":
        return tuple(map(str, (c.max_size, c._offset, c._idx, c.rotated)))
    return ""


def has_meta_state(c) -> bool:
    """``hasattr(c, "meta_state")`` under the legacy protocol: every mlx-lm
    cache had one before #1778 (``""`` on the base class)."""
    if _kind(c) is not None:
        return True
    return hasattr(c, "meta_state")


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------


def _reject_meta(c, meta) -> None:
    # Pre-#1778 _BaseCache.meta_state setter contract.
    if meta is not None and meta:
        raise ValueError("This cache has no meta_state but a meta_state was set.")


def set_legacy_state(c, state, meta: Any = None) -> None:
    """Install a legacy ``(state, meta_state)`` pair into ``c`` — state
    first, then meta (mlx-lm's own ``from_state`` order). ``meta`` falsy
    means "leave meta alone" (the fork's ``if m: c.meta_state = m``)."""
    kind = _kind(c)
    if kind is None:
        c.state = state
        if meta:
            c.meta_state = meta
        return
    if kind in ("KVCache", "ChunkedKVCache"):
        c.keys, c.values = state
        c.offset = c.keys.shape[2]
        if kind == "ChunkedKVCache":
            if meta:
                c.chunk_size, c.start_position = map(int, meta)
        else:
            _reject_meta(c, meta)
        return
    if kind == "QuantizedKVCache":
        c.keys, c.values = state
        if meta:
            c.offset, c.group_size, c.bits = map(int, meta)
        return
    if kind == "RotatingKVCache":
        c.keys, c.values = state
        if meta:
            c.keep, c.max_size, c.offset, c._idx = map(int, meta)
        return
    if kind == "ArraysCache":
        if isinstance(state, list):
            # Pre-1632 bare list (old SSD entries): no row metadata.
            c.cache, c.left_padding, c.lengths = state, None, None
        else:
            cache_items, lp, ln = state
            c.cache = cache_items
            c.left_padding = _none_if_empty(lp)
            c.lengths = _none_if_empty(ln)
        _reject_meta(c, meta)
        return
    if kind == "CacheList":
        metas = meta[1] if meta else [None] * len(state)
        for sub, s, m in zip(c.caches, state, metas):
            set_legacy_state(sub, s, m)
        return
    if kind == "BatchKVCache":
        c.keys, c.values, c.offset, c.left_padding = state
        c._idx = c.keys.shape[2]
        _reject_meta(c, meta)
        return
    if kind == "BatchRotatingKVCache":
        c.keys, c.values, c.offset, c.left_padding = state
        if meta:
            c.max_size, c._offset, c._idx = map(int, meta[:3])
            r = meta[3]
            c.rotated = r if isinstance(r, bool) else str(r) == "True"
        return
    c.state = state
    _reject_meta(c, meta)


def from_legacy_state(cls, state, meta: Any):
    """``cls.from_state(state, meta_state)`` on either protocol."""
    if not NEW_STATE_SHAPE:
        return cls.from_state(state, meta)
    if _defines_meta_state(cls):
        return cls.from_state(state, meta)
    owner = _state_owner(cls)
    name = owner.__name__ if owner is not None else None
    if owner is None or _lm_class(name) is not owner:
        try:
            return cls.from_state(state, meta)
        except TypeError:
            return cls.from_state(state)
    obj = cls.__new__(cls)
    if name == "CacheList":
        names, metas = meta
        obj.caches = [
            from_legacy_state(_lm_class(n), s, m)
            for s, n, m in zip(state, names, metas)
        ]
        return obj
    set_legacy_state(obj, state, meta)
    return obj
