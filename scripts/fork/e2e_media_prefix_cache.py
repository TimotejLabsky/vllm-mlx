#!/usr/bin/env python
"""End-to-end check of PATCHES.md #142 (vision phase B) on a REAL server and
model — off the live routes.

Starts ``vllm_mlx.cli serve --continuous-batching`` on a vision model twice:
media prefix caching ON, then OFF (``VLLM_MLX_MLLM_MEDIA_PREFIX_CACHE=0``,
i.e. phase A). Same conversation on both, T=0, thinking off. Asserts

  * every answer is byte-identical ON vs OFF (a restored image prefix
    continues exactly where a full prefill would — incl. mRoPE positions);
  * ON: the same-image follow-up turn and its exact re-send are served from
    the cache (>= 90 % of the prompt) and faster than OFF;
  * ON: a DIFFERENT image behind byte-identical token ids is NOT served from
    the cache (cross-image aliasing — the phase-A bug), and solid red / solid
    blue behind one prompt still answer red / blue;
  * ``prefix_cache.media_hits`` > 0 ON and == 0 OFF; no traceback.

    E2E_MODEL=mlx-community/GLM-4.6V-Flash-4bit \\
    E2E_SERVE_ARGS="--reasoning-parser glm4" python scripts/fork/e2e_media_prefix_cache.py

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
PORT = int(os.environ.get("E2E_PORT", "8779"))
BASE = f"http://127.0.0.1:{PORT}"


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


PATTERN_A = _png(lambda x, y: ((x * 7 + y * 3) % 256, (x * y) % 256, (y * 5) % 256))
PATTERN_B = _png(lambda x, y: ((x * 3) % 256, (y * 11) % 256, (x + y) % 256))
RED = _png(lambda x, y: (230, 20, 20))
BLUE = _png(lambda x, y: (20, 30, 230))


def _img_msg(url, text):
    return {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": text},
    ]}


def _stream(messages, max_tokens=32):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, text, usage = None, "", {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for ch in chunk.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.perf_counter() - t0
                text += piece
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {"text": text, "ttft": ttft or 0.0, "prompt": usage.get("prompt_tokens", 0),
            "cached": cached}


def _session(enabled, workdir):
    # Own HOME: the server persists its prefix cache under ~/.cache/vllm-mlx
    # and reloads it on start, so a shared HOME would let one run's entries
    # (legitimately) serve the next and blind the aliasing check.
    home = os.path.join(workdir, "home-on" if enabled else "home-off")
    os.makedirs(home, exist_ok=True)
    env = dict(os.environ, VLLM_MLX_MLLM_MEDIA_PREFIX_CACHE="1" if enabled else "0",
               HF_HUB_OFFLINE="1", HOME=home,
               HF_HOME=os.environ.get("HF_HOME")
               or os.path.expanduser("~/.cache/huggingface"))
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    log_path = os.path.join(workdir, f"server-{'on' if enabled else 'off'}.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL, "--host", "127.0.0.1",
         "--port", str(PORT), "--continuous-batching", "--max-num-seqs", "4",
         *os.environ.get("E2E_SERVE_ARGS", "").split()],
        stdout=log, stderr=subprocess.STDOUT, env=env)
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
        q1 = "Describe the dominant colours in one short sentence."
        t1 = [_img_msg(PATTERN_A, q1)]
        r = {"t1": _stream(t1)}
        t2 = t1 + [{"role": "assistant", "content": r["t1"]["text"]},
                   {"role": "user", "content": "Are there any straight lines? One short sentence."}]
        r["t2"] = _stream(t2)
        r["t2_resend"] = _stream(t2)
        r["other_image"] = _stream([_img_msg(PATTERN_B, q1)])  # identical token ids
        q_col = "What single colour fills this image? Answer with one word."
        r["red"] = _stream([_img_msg(RED, q_col)])
        r["blue"] = _stream([_img_msg(BLUE, q_col)])
        r["red_again"] = _stream([_img_msg(RED, q_col)])
        st = json.load(urllib.request.urlopen(f"{BASE}/v1/status", timeout=30))
        r["status_prefix"] = {
            k: _find(st, k) for k in ("media_hits", "media_stores", "media_prefix_enabled")
        }
        return r
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        r_log = open(log_path).read()
        globals().setdefault("_LOGS", []).append(r_log)


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
    workdir = tempfile.mkdtemp(prefix="e2e-media-prefix-")
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))

    on = _session(True, workdir)
    off = _session(False, workdir)
    for k in ("t1", "t2", "t2_resend", "other_image", "red", "blue", "red_again"):
        print(f"INFO  {k:12} ON cached {on[k]['cached']:>5}/{on[k]['prompt']:<5} "
              f"ttft {on[k]['ttft']:.2f}s | OFF cached {off[k]['cached']:>5} ttft "
              f"{off[k]['ttft']:.2f}s | {on[k]['text'][:40]!r}")
    same = [k for k in on if k != "status_prefix" and on[k]["text"] == off[k]["text"]]
    for k in on:
        if k != "status_prefix" and on[k]["text"] != off[k]["text"]:
            print(f"DIFF  {k}:\n      ON : {on[k]['text']!r}\n      OFF: {off[k]['text']!r}")
    check("every answer is byte-identical ON vs OFF (T=0)", len(same) == 7,
          f"{len(same)}/7 identical; differ: {[k for k in on if k != 'status_prefix' and k not in same]}")
    for k in ("t2", "t2_resend"):
        check(f"ON: {k} is served from the cache (>= 90 % of the prompt)",
              on[k]["cached"] >= 0.9 * on[k]["prompt"], f"{on[k]['cached']}/{on[k]['prompt']}")
    check("ON: the same-image follow-up is faster than OFF",
          on["t2"]["ttft"] < off["t2"]["ttft"], f"{on['t2']['ttft']:.2f}s vs {off['t2']['ttft']:.2f}s")
    check("ON: a different image behind identical token ids is NOT served from cache",
          on["other_image"]["cached"] < on["other_image"]["prompt"] * 0.5,
          f"cached {on['other_image']['cached']}/{on['other_image']['prompt']}")
    check("ON: red / blue / red-again answer their own colour",
          "red" in on["red"]["text"].lower() and "blue" in on["blue"]["text"].lower()
          and "red" in on["red_again"]["text"].lower(),
          f"{on['red']['text']!r} {on['blue']['text']!r} {on['red_again']['text']!r}")
    check("ON: the red re-ask is served from cache", on["red_again"]["cached"] > 0,
          f"{on['red_again']['cached']}/{on['red_again']['prompt']}")
    hits_on = on["status_prefix"].get("media_hits")
    hits_off = off["status_prefix"].get("media_hits")
    check("status: media_hits > 0 ON, == 0 OFF", (hits_on or 0) > 0 and not hits_off,
          f"on={hits_on} off={hits_off} enabled_on={on['status_prefix'].get('media_prefix_enabled')}")
    bad = [ln for log in globals().get("_LOGS", []) for ln in log.splitlines()
           if "Traceback" in ln or "no Stream(" in ln or "generation_error_recovery" in ln]
    check("server logs have no traceback / stream-thread error", not bad, "; ".join(bad[:2]))
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
