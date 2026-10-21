#!/usr/bin/env python
"""End-to-end check of PATCHES.md #148 (upstream #842) on a REAL server and
model, off the live routes.

``_detach_cache_for_storage`` used to leave every array it copied reachable
from a closure cycle, so each MLLM prefix-cache store kept a second copy of
its snapshot alive until a full cyclic GC pass. Evicting or clearing the
entry did not return the memory.

Starts ``vllm_mlx.cli serve --continuous-batching`` on a vision model with
its own HOME (the media prefix cache persists under ~/.cache/vllm-mlx), runs
a few multi-turn image conversations so the prefix cache stores entries,
then ``DELETE /v1/cache/prefix``. Reads ``metal.active_memory_gb`` from
``/v1/status`` after load+warm-up (baseline), with the cache full, and after
the clear. Asserts that the clear brings active memory back to the
baseline (within ``E2E_TOLERANCE_GB``, default 0.25) and that the cache did
store entries. Each conversation carries ``E2E_PAD_WORDS`` of system text so
the entries are large enough for a leaked copy to show.

Run it once against the fix and once against a control tree to see the
difference; the code under test is whatever ``PYTHONPATH``/cwd resolves:

    E2E_MODEL=mlx-community/GLM-4.6V-Flash-4bit \\
    E2E_SERVE_ARGS="--reasoning-parser glm4" python scripts/fork/e2e_detach_release.py

Exit code 0 only if all pass.
"""

import base64
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib

MODEL = os.environ.get("E2E_MODEL", "mlx-community/GLM-4.6V-Flash-4bit")
PORT = int(os.environ.get("E2E_PORT", "8781"))
BASE = f"http://127.0.0.1:{PORT}"
TOLERANCE_GB = float(os.environ.get("E2E_TOLERANCE_GB", "0.25"))
CONVERSATIONS = int(os.environ.get("E2E_CONVERSATIONS", "4"))
# Text padding per conversation, so each stored entry is hundreds of MB and a
# leaked copy stands well clear of the tolerance.
PAD_WORDS = int(os.environ.get("E2E_PAD_WORDS", "5000"))


def _png(fill, size=768):
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    rows = b"".join(
        b"\x00" + bytes(v for x in range(size) for v in fill(x, y)) for y in range(size)
    )
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _pattern(seed):
    return _png(
        lambda x, y: ((x * (seed + 3)) % 256, (y * (seed + 7)) % 256, (x + y * seed) % 256)
    )


def _pad(seed):
    words = ("alpha", "bravo", "cobalt", "delta", "ember", "fjord", "garnet", "harbor")
    return " ".join(f"{words[(i * 7 + seed) % 8]}{i % 97}" for i in range(PAD_WORDS))


def _img_msg(url, text):
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": text},
        ],
    }


def _chat(messages):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 48,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    out = json.load(urllib.request.urlopen(req, timeout=600))
    return out["choices"][0]["message"].get("content") or ""


def _status():
    return json.load(urllib.request.urlopen(f"{BASE}/v1/status", timeout=30))


def _active_gb():
    # Let the step loop go idle so no in-flight request holds arrays.
    time.sleep(3)
    return _status()["metal"]["active_memory_gb"]


def _find(d, k):
    if isinstance(d, dict):
        if k in d:
            return d[k]
        for v in d.values():
            out = _find(v, k)
            if out is not None:
                return out
    return None


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-detach-release-")
    home = os.path.join(workdir, "home")
    os.makedirs(home)
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        HOME=home,
        HF_HOME=os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
    )
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    log_path = os.path.join(workdir, "server.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL, "--host", "127.0.0.1",
         "--port", str(PORT), "--continuous-batching", "--max-num-seqs", "4",
         *os.environ.get("E2E_SERVE_ARGS", "").split()],
        stdout=log, stderr=subprocess.STDOUT, env=env)
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))

    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(open(log_path).read()[-3000:])
            try:
                if urllib.request.urlopen(f"{BASE}/v1/models", timeout=2).status == 200:
                    break
            except Exception:
                time.sleep(0.5)
            if time.time() - t0 > 600:
                raise RuntimeError("startup timeout")

        # Warm-up on an image the conversations never use, then clear, so the
        # baseline already includes the vision tower's one-off allocations.
        _chat([_img_msg(_pattern(99), "One word: what is this?")])
        _chat([{"role": "user", "content": "Say OK."}])
        urllib.request.urlopen(
            urllib.request.Request(f"{BASE}/v1/cache/prefix", method="DELETE"), timeout=60
        )
        base = _active_gb()

        for i in range(CONVERSATIONS):
            turn1 = [
                {"role": "system", "content": f"Reference notes {i}: {_pad(i)}"},
                _img_msg(_pattern(i), "Describe the colours in one short sentence."),
            ]
            a1 = _chat(turn1)
            _chat(turn1 + [{"role": "assistant", "content": a1},
                           {"role": "user", "content": "Any straight lines? One sentence."}])
        full = _active_gb()
        stores = _find(_status(), "media_stores")

        urllib.request.urlopen(
            urllib.request.Request(f"{BASE}/v1/cache/prefix", method="DELETE"), timeout=60
        )
        cleared = _active_gb()
        print(f"INFO  active GB: baseline {base}  cache-full {full}  after-clear {cleared}"
              f"  (media_stores={stores})")

        check("the prefix cache stored entries", (stores or 0) > 0 and full > base,
              f"stores={stores}, full-baseline={full - base:+.2f} GB")
        check(f"clearing the cache returns active memory to baseline (±{TOLERANCE_GB} GB)",
              cleared - base <= TOLERANCE_GB, f"after-clear - baseline = {cleared - base:+.2f} GB")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    tb = "Traceback (most recent call last)" in open(log_path).read()
    check("no traceback in the server log", not tb, log_path if tb else "")
    print(f"{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
