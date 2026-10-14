#!/usr/bin/env python3
"""Smoke test for a vision (MLLM) route, stdlib only.

    python3 vision_route_smoke.py MODEL [BASE_URL]

Checks: a solid-red image is described as red (non-stream) and a solid-blue
one as blue (stream, ends with [DONE]); a text-only request on the same route
answers; /v1/status is reachable. Exit 0 only if all pass.
"""

import base64
import json
import struct
import sys
import time
import urllib.request
import zlib

MODEL = sys.argv[1]
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))


def png(rgb, size=224):
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    row = b"\x00" + bytes(rgb) * size
    raw = zlib.compress(row * size)
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()


def post(body, stream=False):
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps({"model": MODEL, "temperature": 0, "stream": stream, "max_tokens": 40,
                    "chat_template_kwargs": {"enable_thinking": False}, **body}).encode(),
        {"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=600)


def image_msg(rgb):
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": png(rgb)}},
        {"type": "text", "text": "What single colour fills this image? Answer with one word."},
    ]}]


t0 = time.time()
r = json.load(post({"messages": image_msg((255, 0, 0))}))
red = (r["choices"][0]["message"].get("content") or "").lower()
check("image (non-stream): red", "red" in red, f"{red[:50]!r} ({time.time() - t0:.1f}s)")

text, done = "", False
for raw in post({"messages": image_msg((0, 0, 255))}, stream=True):
    line = raw.decode().strip()
    if line == "data: [DONE]":
        done = True
    elif line.startswith("data: "):
        for ch in json.loads(line[6:]).get("choices") or []:
            text += (ch.get("delta") or {}).get("content") or ""
check("image (stream): blue + [DONE]", done and "blue" in text.lower(), repr(text[:50]))

r = json.load(post({"messages": [{"role": "user", "content": "What is the capital of France? One word."}]}))
txt = r["choices"][0]["message"].get("content") or ""
check("text-only request on the vision route", "paris" in txt.lower(), repr(txt[:50]))

try:
    st = json.load(urllib.request.urlopen(f"{BASE}/upstream/{MODEL}/v1/status", timeout=30))
except Exception:
    st = json.load(urllib.request.urlopen(f"{BASE}/v1/status", timeout=30))
check("route /v1/status reachable", isinstance(st, dict))
sys.exit(0 if all(results) else 1)
