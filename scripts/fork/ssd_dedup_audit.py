#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only audit: how much of a system-KV SSD store is duplicated prefix?

Walks every ``batched-*`` (or plain) store under ``--root`` — the SQLite index
for each entry's token key, ``meta.json`` for the layer kinds and checkpoint
positions, and the safetensors HEADER only (no tensor data is read) — and
simulates content-addressed dedup:

* attention KV (``kv`` layers) in fixed ``C``-token chunks keyed by the token
  prefix that ends the chunk — two entries share chunk ``c`` iff their tokens
  agree on ``[0, (c+1)*C)``;
* partial-restore checkpoints keyed by the token prefix at their position.

Everything else (the tail of the KV past the last full chunk, the final
recurrent state) stays per-entry. The ``#140`` column is the per-entry
alternative that won (2026-10-13: 31 GB vs ~18 GB cross-entry): checkpoint
bytes ``prune_spill_ladder`` leaves out — at/after the entry's length, or
within ``--min-gap`` tokens of the previous kept one. After #140 is deployed
it reads ~0 on freshly written entries. Stdlib only, so it runs with the system
python on the Studio:

    scp scripts/fork/ssd_dedup_audit.py mac-studio:/tmp/      # laptop
    python3 /tmp/ssd_dedup_audit.py --root ~/.cache/vllm-mlx-ssd   # Studio
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import struct
import sys

_GB = 1024**3


def _header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def _nbytes(t: dict) -> int:
    a, b = t["data_offsets"]
    return b - a


def audit_store(store: str, chunks: list[int], min_gap: int = 16) -> dict | None:
    db = os.path.join(store, "index.db")
    if not os.path.exists(db):
        return None
    # mode=ro cannot open a WAL database whose -shm is absent (a store whose
    # process is not running); immutable=1 can, and such a store is static.
    rows = None
    for flags in ("mode=ro", "mode=ro&immutable=1"):
        try:
            conn = sqlite3.connect(f"file:{db}?{flags}", uri=True)
            rows = conn.execute("SELECT tokens_blob, file_path FROM entries").fetchall()
            conn.close()
            break
        except sqlite3.OperationalError:
            continue
    if rows is None:
        return None
    entries = []
    for blob, fp in rows:
        d = os.path.join(store, "data", fp)
        try:
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            hdr = _header(os.path.join(d, "snapshot.safetensors"))
        except (OSError, ValueError):
            continue
        total = sum(_nbytes(t) for t in hdr.values())
        kv_layers = {
            lm["i"] for lm in meta.get("layers", []) if lm.get("kind") == "kv"
        }
        kinds = meta.get("kinds")
        if kinds:
            kv_layers = {i for i in kv_layers if kinds[i] == "trim"}
        kv_bytes, kv_len = 0, None
        for i in kv_layers:
            for s in ("k", "v"):
                t = hdr.get(f"l{i}_{s}")
                if t is None:
                    continue
                kv_bytes += _nbytes(t)
                kv_len = t["shape"][2]
        ck = {}
        pat = re.compile(r"^c(\d+)_")
        for name, t in hdr.items():
            m = pat.match(name)
            if m:
                n = int(m.group(1))
                ck[n] = ck.get(n, 0) + _nbytes(t)
        ck_pos = [cm["pos"] for cm in meta.get("checkpoints", [])]
        entries.append(
            {
                "blob": blob,
                "total": total,
                "kv_bytes": kv_bytes,
                "kv_len": kv_len or 0,
                "ckpts": [(ck_pos[n], b) for n, b in ck.items() if n < len(ck_pos)],
            }
        )
    if not entries:
        return None
    logical = sum(e["total"] for e in entries)
    out = {"entries": len(entries), "logical": logical, "by_chunk": {}}
    # Checkpoint dedup is independent of the chunk size.
    seen_ck: set[bytes] = set()
    ck_logical = ck_phys = 0
    for e in entries:
        for pos, b in e["ckpts"]:
            ck_logical += b
            key = hashlib.sha256(e["blob"][: 4 * pos]).digest()
            if key not in seen_ck:
                seen_ck.add(key)
                ck_phys += b
    out["ckpt_logical"], out["ckpt_physical"] = ck_logical, ck_phys
    # #140: the same rule as system_kv_ssd.prune_spill_ladder.
    prunable = 0
    for e in entries:
        n_tok = len(e["blob"]) // 4
        last = None
        for pos, b in sorted(e["ckpts"]):
            if pos >= n_tok or (last is not None and pos - last < min_gap):
                prunable += b
                continue
            last = pos
    out["prunable"] = prunable
    out["kv_logical"] = sum(e["kv_bytes"] for e in entries)
    for c in chunks:
        seen: set[bytes] = set()
        kv_phys = 0
        n_chunks = 0
        for e in entries:
            if not e["kv_len"]:
                continue
            bpt = e["kv_bytes"] / e["kv_len"]
            full = e["kv_len"] // c
            for j in range(full):
                key = hashlib.sha256(e["blob"][: 4 * (j + 1) * c]).digest()
                if key not in seen:
                    seen.add(key)
                    kv_phys += bpt * c
                    n_chunks += 1
            kv_phys += bpt * (e["kv_len"] - full * c)  # per-entry tail
        phys = logical - out["kv_logical"] - ck_logical + kv_phys + ck_phys
        out["by_chunk"][c] = {"physical": phys, "files": n_chunks}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.expanduser("~/.cache/vllm-mlx-ssd"))
    ap.add_argument("--chunks", default="512,1024,2048,4096")
    ap.add_argument("--min-gap", type=int, default=16)
    args = ap.parse_args()
    chunks = [int(x) for x in args.chunks.split(",")]
    stores = sorted(
        os.path.join(args.root, d)
        for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d))
    )
    hdr = f"{'store':48} {'n':>3} {'logical':>8} {'kv%':>4} {'ck%':>4} {'#140':>6} {'ck-dedup':>8}"
    hdr += "".join(f" {'C=' + str(c):>8}" for c in chunks)
    print(hdr)
    tot_l = tot_prune = 0.0
    tot_p = {c: 0.0 for c in chunks}
    for s in stores:
        r = audit_store(s, chunks, args.min_gap)
        if r is None:
            continue
        L = r["logical"]
        tot_l += L
        tot_prune += r["prunable"]
        line = (
            f"{os.path.basename(s)[:48]:48} {r['entries']:>3} {L / _GB:>7.1f}G"
            f" {100 * r['kv_logical'] / L:>3.0f}% {100 * r['ckpt_logical'] / L:>3.0f}%"
            f" {r['prunable'] / _GB:>5.1f}G"
            f" {(r['ckpt_logical'] - r['ckpt_physical']) / _GB:>7.1f}G"
        )
        for c in chunks:
            p = r["by_chunk"][c]["physical"]
            tot_p[c] += p
            line += f" {L / p if p else 0:>7.2f}x"
        print(line)
    print(f"\nTOTAL logical {tot_l / _GB:.1f} GB")
    print(f"  #140 prunable (min_gap={args.min_gap}) {tot_prune / _GB:.1f} GB")
    for c in chunks:
        print(
            f"  C={c:<5} physical {tot_p[c] / _GB:6.1f} GB "
            f"(saves {(tot_l - tot_p[c]) / _GB:5.1f} GB, {tot_l / tot_p[c]:.2f}x)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
