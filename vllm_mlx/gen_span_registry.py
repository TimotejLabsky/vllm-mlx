"""Fork #163 — where the model's own output sits inside cached chains.

#139 (``BatchedSystemKV.reconcile_generated``) gives a client's whitespace
re-render of an earlier turn the model's own tokens back, so the prompt keeps
matching the cached chain. It needs to know which stretches of a chain the
model generated, and kept that as ``gen_spans`` on each cache entry, passed on
only when a new entry absorbs an older one whose tokens are a full prefix of
it. That breaks whenever the older entry is gone at the moment of absorption:

* evicted by the RAM budget between a request's restore match and its store
  (live 2026-11-17: two small background jobs stored in between, every second
  opencode turn on the 27B-8bit re-prefilled 7.5-15.5K tokens);
* promoted from the SSD tier, which never stored the spans;
* a route restart, which empties RAM.

This registry keeps the spans apart from entry lifetime. A span is recorded as
``(end, digest of chain[:end]) -> start``: any chain whose first ``end`` tokens
hash the same contains that generated span at the same place, whichever entry
or tier it came from. Bounded (LRU) and, with an SSD tier, persisted next to
it, so it also survives restarts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_FORMAT = 1


def _blob(tokens: Sequence[int]) -> bytes:
    return np.asarray(tokens, dtype=np.int32).tobytes()


def _prefix_digests(tokens: Sequence[int], ends: Iterable[int]) -> dict:
    """``{end: sha1(tokens[:end])}`` for each end, in one pass over tokens."""
    wanted = sorted({int(e) for e in ends if 0 < int(e) <= len(tokens)})
    if not wanted:
        return {}
    blob = _blob(tokens[: wanted[-1]])
    h = hashlib.sha1()
    out, done = {}, 0
    for end in wanted:
        h.update(blob[done * 4 : end * 4])
        done = end
        out[end] = h.copy().hexdigest()
    return out


class GenSpanRegistry:
    def __init__(self, path: Optional[str] = None, capacity: int = 4096):
        self.path = path
        self.capacity = max(16, int(capacity))
        self._spans: "OrderedDict[Tuple[int, str], int]" = OrderedDict()
        self._ends: dict = {}  # end -> how many registered spans end there
        self._lock = threading.Lock()
        self.loaded = 0
        if path:
            self._load()

    def __len__(self) -> int:
        return len(self._spans)

    # -- recording -----------------------------------------------------------
    def add(self, tokens: Sequence[int], spans: Iterable[Tuple[int, int]]) -> int:
        """Record generated spans of ``tokens``. Returns how many were new."""
        spans = [
            (int(s), int(e)) for s, e in spans if 0 <= int(s) < int(e) <= len(tokens)
        ]
        if not spans:
            return 0
        digests = _prefix_digests(tokens, (e for _, e in spans))
        added = 0
        with self._lock:
            for start, end in spans:
                key = (end, digests[end])
                if key in self._spans:
                    self._spans.move_to_end(key)
                    continue
                self._spans[key] = start
                self._ends[end] = self._ends.get(end, 0) + 1
                added += 1
            while len(self._spans) > self.capacity:
                (end, _), _start = self._spans.popitem(last=False)
                left = self._ends.get(end, 1) - 1
                if left:
                    self._ends[end] = left
                else:
                    self._ends.pop(end, None)
        if added:
            self._save()
        return added

    # -- lookup --------------------------------------------------------------
    def spans_for(self, tokens: Sequence[int]) -> List[Tuple[int, int]]:
        """Every registered generated span that ``tokens`` contains."""
        with self._lock:
            ends = [e for e in self._ends if e <= len(tokens)]
        if not ends:
            return []
        digests = _prefix_digests(tokens, ends)
        found = []
        with self._lock:
            for end, digest in digests.items():
                key = (end, digest)
                start = self._spans.get(key)
                if start is not None:
                    self._spans.move_to_end(key)
                    found.append((start, end))
        return sorted(found)

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        try:
            with open(self.path) as f:
                data = json.load(f)
            if data.get("format") != _FORMAT:
                return
            for start, end, digest in data.get("spans", [])[-self.capacity :]:
                key = (int(end), str(digest))
                if key not in self._spans:
                    self._spans[key] = int(start)
                    self._ends[key[0]] = self._ends.get(key[0], 0) + 1
            self.loaded = len(self._spans)
        except FileNotFoundError:
            return
        except Exception:  # noqa: BLE001 - a bad file only loses old spans
            logger.warning(
                "[gen_spans] could not read %s; starting empty",
                self.path,
                exc_info=True,
            )
            self._spans.clear()
            self._ends.clear()

    def _save(self) -> None:
        if not self.path:
            return
        with self._lock:
            spans = [[s, e, d] for (e, d), s in self._spans.items()]
        tmp = f"{self.path}.tmp.{os.getpid()}"
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(tmp, "w") as f:
                json.dump({"format": _FORMAT, "spans": spans}, f)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 - persistence is best effort
            logger.debug("[gen_spans] save failed", exc_info=True)
            try:
                os.unlink(tmp)
            except OSError:
                pass
