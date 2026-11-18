#!/usr/bin/env python
"""End-to-end check of PATCHES.md #168 on a REAL server and model, off the
live routes: Qwen3-VL-MoE decode without a host sync per attention layer.

Serves the same model twice on a spare port, with the prod route's flags:
- control: the INSTALLED vllm_mlx (no #168);
- armed: this source tree (cwd first on PYTHONPATH).
Each gets one warmup, then the same image+text requests at T=0, streamed,
and reports decode tok/s (first token to last) and the text. PASS when the
texts are identical and the armed decode is faster.

    E2E_MODEL=mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit \
        python scripts/fork/e2e_qwen3vl_decode_sync.py

Exit code 0 only if all pass.
"""

import base64
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit")
PORT = int(os.environ.get("E2E_PORT", "8783"))
BASE = f"http://127.0.0.1:{PORT}"
REPS = int(os.environ.get("E2E_REPS", "3"))
MAX_TOKENS = int(os.environ.get("E2E_MAX_TOKENS", "300"))


def image_url() -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(img)
    for i in range(0, 448, 32):
        d.rectangle([i, i, i + 60, i + 40], outline=(i % 255, 80, 160), width=3)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def stream(client, max_tokens, url):
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "Describe this image in great detail, then "
             "write a long story about it."}]}],
    }  # fmt: skip
    text, first, last, n = [], None, None, 0
    with client.stream("POST", f"{BASE}/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            piece = delta.get("content") or ""
            if piece:
                now = time.perf_counter()
                first = first or now
                last = now
                n += 1
                text.append(piece)
    rate = (n - 1) / (last - first) if n > 1 and last > first else 0.0
    return "".join(text), rate


def run(tag: str, workdir: str, use_tree: bool) -> dict:
    import httpx

    log_path = os.path.join(workdir, f"server-{tag}.log")
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    if use_tree:
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    else:
        env.pop("PYTHONPATH", None)
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL,
            "--host", "127.0.0.1", "--port", str(PORT),
            "--continuous-batching", "--max-num-seqs", "4",
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
            "--timeout", "600",
        ],
        stdout=log, stderr=subprocess.STDOUT, env=env,
        cwd=workdir if not use_tree else None,
    )  # fmt: skip
    out = {"log": log_path, "rates": [], "texts": []}
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(open(log_path).read()[-2000:])
            try:
                if httpx.get(f"{BASE}/v1/models", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        url = image_url()
        with httpx.Client(timeout=900) as c:
            stream(c, 16, url)
            for _ in range(REPS):
                text, rate = stream(c, MAX_TOKENS, url)
                out["rates"].append(round(rate, 2))
                out["texts"].append(hashlib.sha1(text.encode()).hexdigest()[:12])
        out["patched"] = "no host sync per layer (#168)" in open(log_path).read()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    return out


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-qwen3vl-sync-")
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    control = run("control", workdir, use_tree=False)
    armed = run("armed", workdir, use_tree=True)
    check("control: installed build is unpatched", not control["patched"])
    check("armed: patch applied", armed["patched"])
    for tag, r in (("control", control), ("armed", armed)):
        bad = [ln for ln in open(r["log"]).read().splitlines() if "Traceback" in ln]
        check(f"{tag}: server log has no traceback", not bad, "; ".join(bad[:2]))
    check(
        "T=0 text identical, control vs armed",
        len(set(control["texts"] + armed["texts"])) == 1,
        f"control {control['texts']} armed {armed['texts']}",
    )
    mc, ma = statistics.median(control["rates"]), statistics.median(armed["rates"])
    check(
        "armed decodes faster",
        ma > mc,
        f"median {mc:.1f} -> {ma:.1f} tok/s ({(ma / mc - 1) * 100:+.0f}%) "
        f"control {control['rates']} armed {armed['rates']}",
    )
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
