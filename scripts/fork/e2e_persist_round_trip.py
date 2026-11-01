#!/usr/bin/env python
"""End-to-end check of PATCHES.md #151 on a REAL vision server, off the live
routes: prefix-cache entries survive save/load/resave across three server
runs with their own images' answers.

A round trip only. The corrupting order needs an entry moved to the LRU end
without its arrays being read, which a client cannot force on demand, so that
case is pinned by test_151_* in tests/test_fork_invariants.py.

    python scripts/fork/e2e_persist_round_trip.py

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
PORT = int(os.environ.get("E2E_PORT", "8782"))
BASE = f"http://127.0.0.1:{PORT}"
# Long enough to clear min_prefix_tokens (128), or nothing is ever stored.
QUESTION = (
    "You are checking images for a colour survey. " * 40
    + "What single colour fills this image? Answer with one word."
)
COLOURS = {"red": (220, 20, 20), "blue": (20, 20, 220), "green": (20, 180, 20)}


def png(rgb, size=768):
    def chunk(tag, data):
        body = tag + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
        )

    row = b"\x00" + bytes(rgb) * size
    raw = chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    raw += chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b"")
    data = base64.b64encode(b"\x89PNG\r\n\x1a\n" + raw).decode()
    return f"data:image/png;base64,{data}"


def ask(colour):
    body = {
        "model": MODEL,
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": png(COLOURS[colour])}},
                    {"type": "text", "text": QUESTION},
                ],
            }
        ],
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    text, usage = "", {}
    with urllib.request.urlopen(req, timeout=300) as stream:
        for raw in stream:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                text += (choice.get("delta") or {}).get("content") or ""
    details = usage.get("prompt_tokens_details") or {}
    print(f"INFO  prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
    return text, details.get("cached_tokens", 0) or 0


def session(home, log_path, colours):
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        HOME=home,
        VLLM_MLX_MLLM_MEDIA_PREFIX_CACHE="1",
        HF_HOME=os.environ.get("HF_HOME")
        or os.path.expanduser("~/.cache/huggingface"),
    )
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL,
             "--host", "127.0.0.1", "--port", str(PORT),
             "--continuous-batching", "--max-num-seqs", "4"],
            stdout=log, stderr=subprocess.STDOUT, env=env,
        )  # fmt: skip
        try:
            deadline = time.time() + 300
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(open(log_path).read()[-2000:])
                try:
                    urllib.request.urlopen(f"{BASE}/v1/models", timeout=2)
                    break
                except Exception:
                    if time.time() > deadline:
                        raise
                    time.sleep(0.5)
            return {colour: ask(colour) for colour in colours}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-persist-")
    home = os.path.join(workdir, "home")
    os.makedirs(home)
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    logs = [os.path.join(workdir, f"run{i}.log") for i in range(3)]
    first = session(home, logs[0], ["red", "blue"])
    second = session(home, logs[1], ["red", "green"])  # reorder, then grow
    third = session(home, logs[2], ["green", "blue", "red"])

    for run, answers in (("run1", first), ("run2", second), ("run3", third)):
        for colour, (text, cached) in answers.items():
            check(
                f"{run}: {colour} image answers {colour}",
                colour in text.lower(),
                f"{text!r} cached={cached}",
            )
    check("run3: a persisted entry was reused", third["blue"][1] > 0,
          f"cached={third['blue'][1]}")  # fmt: skip
    text = "".join(open(path).read() for path in logs)
    check("entries were saved", "[cache_persist] SAVED" in text)
    check("entries were loaded back", "[cache_persist] LOADED" in text)
    check("no failed entry saves", "failed to save entry" not in text)
    check("no traceback", "Traceback" not in text)
    leftovers = [
        name
        for _, _, files in os.walk(home)
        for name in files
        if ".tmp." in name
    ]
    check("no temp files left", not leftovers, str(leftovers[:3]))
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
