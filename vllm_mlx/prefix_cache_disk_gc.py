# SPDX-License-Identifier: Apache-2.0
"""(#143) Garbage-collect a persisted prefix-cache directory.

``MemoryAwarePrefixCache.save_to_disk`` writes ``<cache_dir>/<provenance>/``
on every shutdown and never deletes anything. Three kinds of file pile up,
none of which any load can read (Studio, 2026-10-14: 1.27 of 1.8 GB):

* **legacy top-level files** — caches from before upstream #747's provenance
  subdirs; ``load_from_disk`` discards them;
* **stale provenance subdirs** — the provenance hash covers the model
  revision, quant signature and mlx / mlx-lm / mlx-vlm versions, so every
  package bump starts a new subdir and orphans the previous one forever;
* **stale entry files** — a save with fewer entries than the last one leaves
  ``entry_N…`` files its index no longer lists.

``gc_prefix_cache_dir`` removes exactly those and nothing else: only regular
files at the top level, only sibling directories whose name looks like a
provenance hash (16 hex chars), and only ``entry_*`` files inside the kept
subdir that its index does not reference (left alone when the index is
unreadable). ``VLLM_MLX_PREFIX_CACHE_DISK_GC=0`` turns it off.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil

logger = logging.getLogger(__name__)

_PROVENANCE_DIR = re.compile(r"^[0-9a-f]{16}$")
_ENTRY_FILE = re.compile(r"^entry_(\d+)(?:\.safetensors|_tokens\.bin)$")
_TOP_LEVEL_FILE = re.compile(r"^(index\.json|entry_\d+(?:\.safetensors|_tokens\.bin))$")


def disk_gc_enabled() -> bool:
    return os.environ.get("VLLM_MLX_PREFIX_CACHE_DISK_GC", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def _size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _top_level_unreachable(cache_dir: str, keep: str) -> bool:
    """True when ``load_from_disk`` can never serve the top-level files: the
    kept subdir has an index (it always wins), or the top-level index is
    missing / unreadable / lacks provenance (discarded as legacy). A
    top-level index WITH provenance is upstream #747's legacy fallback and
    stays while no subdir supersedes it."""
    if os.path.isfile(os.path.join(cache_dir, keep, "index.json")):
        return True
    try:
        with open(os.path.join(cache_dir, "index.json")) as f:
            index = json.load(f)
    except (OSError, ValueError):
        return True
    return index.get("model_revision") is None or "quant_signature" not in index


def gc_prefix_cache_dir(cache_dir: str, keep: str, *, top_level: bool = True) -> int:
    """Delete unreachable files under ``cache_dir``, keeping provenance
    subdir ``keep``. Returns the bytes freed. Never raises.

    ``top_level=False`` (the load-time call) leaves top-level files for
    ``load_from_disk`` to judge and log first; the next save removes them."""
    if not disk_gc_enabled() or not os.path.isdir(cache_dir):
        return 0
    freed = 0
    removed = []
    try:
        drop_top = top_level and _top_level_unreachable(cache_dir, keep)
        for name in os.listdir(cache_dir):
            path = os.path.join(cache_dir, name)
            if drop_top and os.path.isfile(path) and _TOP_LEVEL_FILE.match(name):
                freed += _size(path)
                os.remove(path)
                removed.append(name)
            elif os.path.isdir(path) and name != keep and _PROVENANCE_DIR.match(name):
                freed += _size(path)
                shutil.rmtree(path, ignore_errors=True)
                removed.append(name + "/")
        kept = os.path.join(cache_dir, keep)
        index_path = os.path.join(kept, "index.json")
        if os.path.isfile(index_path):
            try:
                with open(index_path) as f:
                    listed = {int(e["index"]) for e in json.load(f).get("entries", [])}
            except (OSError, ValueError, KeyError, TypeError):
                listed = None
            if listed is not None:
                for name in os.listdir(kept):
                    m = _ENTRY_FILE.match(name)
                    if m and int(m.group(1)) not in listed:
                        path = os.path.join(kept, name)
                        freed += _size(path)
                        os.remove(path)
                        removed.append(f"{keep}/{name}")
    except OSError:
        logger.debug("[cache_persist] disk GC failed for %s", cache_dir, exc_info=True)
    if removed:
        logger.info(
            "[cache_persist] disk GC freed %.1f MB in %s (%d item(s): %s)",
            freed / 1e6,
            cache_dir,
            len(removed),
            ", ".join(removed[:6]) + (" …" if len(removed) > 6 else ""),
        )
    return freed
