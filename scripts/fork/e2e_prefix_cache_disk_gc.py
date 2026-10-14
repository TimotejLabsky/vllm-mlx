#!/usr/bin/env python
"""End-to-end check of PATCHES.md #143 (prefix-cache disk GC) on a REAL server
— off the live routes. Isolated HOME, so the live ~/.cache is never touched.

Plants the three kinds of unreachable file in the server's persisted
prefix-cache dir (``~/.cache/vllm-mlx/prefix_cache/<repo>``): an old
provenance subdir, legacy top-level files, and stale ``entry_N`` files in the
current subdir. Then: start the server, one request, SIGTERM (save) — asserts
the junk is gone and the saved entries are exactly what the index lists;
restart — asserts the saved entry is loaded and the re-ask is served from it.

    E2E_MODEL=mlx-community/GLM-4.6V-Flash-4bit python scripts/fork/e2e_prefix_cache_disk_gc.py
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

MODEL = os.environ.get("E2E_MODEL", "mlx-community/GLM-4.6V-Flash-4bit")
PORT = int(os.environ.get("E2E_PORT", "8781"))
BASE = f"http://127.0.0.1:{PORT}"
PROMPT = "Context: " + " ".join(f"fact {i} is {i * 7 % 13}." for i in range(200))


def _ask():
    body = json.dumps({"model": MODEL, "max_tokens": 8, "temperature": 0,
                       "messages": [{"role": "user", "content": PROMPT + " What is fact 3?"}],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"{BASE}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600))
    return (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0), r["usage"]["prompt_tokens"]


def _serve(home, log_path):
    env = dict(os.environ, HOME=home, HF_HUB_OFFLINE="1",
               HF_HOME=os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"))
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen([sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL, "--host", "127.0.0.1",
                             "--port", str(PORT), "--continuous-batching", "--max-num-seqs", "4"],
                            env=env, stdout=open(log_path, "a"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(open(log_path).read()[-3000:])
        try:
            if urllib.request.urlopen(f"{BASE}/v1/models", timeout=2).status == 200:
                return proc
        except Exception:
            time.sleep(0.5)
        if time.time() - t0 > 600:
            raise RuntimeError("startup timeout")


def _stop(proc):
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=120)


def main() -> int:
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))

    home = tempfile.mkdtemp(prefix="e2e-diskgc-")
    log_path = os.path.join(home, "server.log")
    root = os.path.join(home, ".cache", "vllm-mlx", "prefix_cache", MODEL.replace("/", "--"))

    # pass 1 just to learn this build's provenance subdir name
    proc = _serve(home, log_path)
    _ask()
    _stop(proc)
    subs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    check("a provenance subdir was written", len(subs) == 1, f"{subs}")
    keep = subs[0]

    # plant the junk
    old = os.path.join(root, "0123456789abcdef")
    os.makedirs(old)
    open(os.path.join(old, "entry_0.safetensors"), "wb").write(b"x" * 4096)
    json.dump({"entries": [{"index": 0}]}, open(os.path.join(old, "index.json"), "w"))
    json.dump({"version": 3, "entries": []}, open(os.path.join(root, "index.json"), "w"))
    open(os.path.join(root, "entry_7.safetensors"), "wb").write(b"x" * 4096)
    for name in ("entry_42.safetensors", "entry_42_tokens.bin"):
        open(os.path.join(root, keep, name), "wb").write(b"x" * 4096)

    proc = _serve(home, log_path)
    _ask()
    _stop(proc)
    check("old provenance subdir removed", not os.path.exists(old))
    check("legacy top-level files removed",
          not any(os.path.isfile(os.path.join(root, f)) for f in os.listdir(root)),
          f"{[f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]}")
    idx = json.load(open(os.path.join(root, keep, "index.json")))
    listed = {e["index"] for e in idx["entries"]}
    on_disk = {int(f.split("_")[1].split(".")[0]) for f in os.listdir(os.path.join(root, keep))
               if f.startswith("entry_")}
    check("current subdir holds exactly the indexed entries", on_disk == listed,
          f"listed {sorted(listed)} on disk {sorted(on_disk)}")

    proc = _serve(home, log_path)
    cached, prompt = _ask()
    _stop(proc)
    check("restart reloads the saved entry and serves the re-ask from it",
          cached >= 0.9 * prompt, f"{cached}/{prompt}")
    text = open(log_path).read()
    check("the GC logged what it freed", "disk GC freed" in text)
    bad = [ln for ln in text.splitlines() if "Traceback" in ln]
    check("server log has no traceback", not bad, "; ".join(bad[:2]))
    print(f"model: {MODEL} | home: {home}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
