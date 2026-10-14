# SPDX-License-Identifier: Apache-2.0
"""(#142) Content-keyed prefix-cache keys for image requests (vision phase B).

The batched MLLM prefix cache (``MemoryAwarePrefixCache``) is keyed on token
ids. An image enters the prompt as placeholder tokens that say nothing about
its pixels, so #56 (phase A) kept every media request out of the cache: two
different images behind an identical prompt would have served each other's
KV. This module makes the key carry the content instead. Every image
placeholder token is replaced, IN THE KEY ONLY, by a negative value derived
from the SHA-256 of that image's bytes:

* same images + same prompt  -> identical key (follow-up turns and re-asks hit);
* a different image          -> the keys part at that image's first placeholder,
                                so the text before it is still shared;
* negative values never collide with real token ids (vocab ids are >= 0), so
  text-only keys - unchanged - can never match an image span.

Per-image fingerprints need the k-th contiguous placeholder run to be the k-th
image (Qwen, GLM, gemma). Layouts that split an image into several runs
(Pixtral/mistral3 rows separated by ``[IMG_BREAK]``) fall back to one
order-sensitive digest of all images on every placeholder: still content-keyed,
it just gives up reuse when a later turn adds an image.

Images are hashed over their full bytes (files) or the full string (data URLs,
base64). ``http(s)`` URLs return ``None``: the same URL can serve new content,
so it is not a content key.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

_URL_PREFIXES = ("http://", "https://")


def image_digest(image: object) -> Optional[bytes]:
    """Full SHA-256 of an image's content, or ``None`` when it has none we
    can trust (remote URLs, unreadable inputs)."""
    if isinstance(image, (bytes, bytearray)):
        return hashlib.sha256(bytes(image)).digest()
    if not isinstance(image, str):
        return None
    if image.startswith(_URL_PREFIXES):
        return None
    try:
        path = Path(image)
        if len(image) < 4096 and path.is_file():
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            return h.digest()
    except (OSError, ValueError):
        return None
    return hashlib.sha256(image.encode()).digest()


def _fingerprint(digest: bytes, length: int) -> List[int]:
    """``length`` negative int32 values spread over the whole digest, so two
    runs differing anywhere in the digest differ at their first token."""
    words = struct.unpack(">8I", digest)
    return [
        -(1 + ((words[j % 8] ^ (j * 0x9E3779B1)) & 0x7FFFFFFE)) for j in range(length)
    ]


def _runs(ids: Sequence[int], placeholders: frozenset) -> List[tuple]:
    out, start = [], None
    for i, tok in enumerate(ids):
        if tok in placeholders:
            if start is None:
                start = i
        elif start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(ids)))
    return out


def media_prefix_key(
    ids: Sequence[int], placeholder_ids: Iterable[int], images: Sequence[object]
) -> Optional[List[int]]:
    """The prefix-cache key for a prompt carrying ``images``; ``None`` when it
    must not be cached (no placeholders, an image without a content digest,
    or nothing to fingerprint)."""
    placeholders = frozenset(int(p) for p in placeholder_ids if p is not None)
    if not images or not placeholders:
        return None
    maybe = [image_digest(img) for img in images]
    digests: List[bytes] = [d for d in maybe if d is not None]
    if len(digests) != len(maybe):
        return None
    runs = _runs(ids, placeholders)
    if not runs:
        return None
    key = list(ids)
    if len(runs) == len(digests):
        for (start, end), digest in zip(runs, digests):
            key[start:end] = _fingerprint(digest, end - start)
    else:
        combined = hashlib.sha256(b"".join(digests)).digest()
        fp = _fingerprint(combined, sum(end - start for start, end in runs))
        k = 0
        for start, end in runs:
            key[start:end] = fp[k : k + end - start]
            k += end - start
    return key
