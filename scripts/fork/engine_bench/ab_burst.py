#!/usr/bin/env python3
"""Decode-burst A/B on the fork's BatchedEngine (bench checkout, port 5899).

For each VLLM_MLX_DECODE_BURST setting: spawn server (35B MoE, production
batched env incl. fusion), then measure
  - single-stream decode: 3x 400-token natural prompts (usage-counted)
  - 4-stream concurrent: 4 natural prompts x 256 tokens, 2 rounds
    (aggregate = sum(usage tokens)/wall)
  - T=0 first-run text hash for cross-setting identity check
"""
import hashlib
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request

PORT = 5899
BASE = "/Users/ai/bench-2026-09-04"
REPO = "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ"
PY = BASE + "/burst-env/bin/python"

PROMPT = ("Write a detailed explanation of how TCP congestion control works, "
          "covering slow start, congestion avoidance, fast retransmit and "
          "fast recovery, with examples.")
CONC = [
    "Describe how a B-tree index speeds up database lookups, with examples.",
    "Explain the difference between processes and threads in an OS kernel.",
    "How does public key cryptography enable secure key exchange? Detail.",
    "What happens during DNS resolution for a fresh domain name? Explain.",
]

def chat(prompt, max_tokens, timeout=600):
    payload = {"model": REPO,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    t_first = None
    usage = None
    text = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                t = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if t:
                    if t_first is None:
                        t_first = time.monotonic()
                    text.append(t)
    t_end = time.monotonic()
    ct = (usage or {}).get("completion_tokens")
    dtps = None
    if ct and t_first and t_end > t_first:
        dtps = round((ct - 1) / (t_end - t_first), 1)
    return {"ttft": round(t_first - t0, 3) if t_first else None,
            "decode_tps": dtps, "tokens": ct,
            "sha": hashlib.sha256("".join(text).encode()).hexdigest()[:12],
            "wall": round(t_end - t0, 2)}

def run_setting(burst):
    subprocess.run(["pkill", "-f", f"port {PORT}"], capture_output=True)
    time.sleep(3)
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/unload", timeout=60)
    except Exception:
        pass
    env = dict(os.environ)
    env.update({
        "VLLM_MLX_BATCHED_SYSTEM_KV": "1",
        "VLLM_MLX_BATCHED_KV_BUDGET_MB": "8192",
        "VLLM_MLX_BATCHED_PAD_WASTE_MB": "4096",
        "VLLM_MLX_BATCHED_MEM_WATERMARK_PCT": "80",
        "VLLM_MLX_MOE_GATEUP_FUSION": "1",
        "VLLM_MLX_DECODE_BURST": str(burst),
        "PYTHONUNBUFFERED": "1",
    })
    logf = open(f"{BASE}/burst-{burst}.log", "w")
    proc = subprocess.Popen(
        [PY, "-m", "vllm_mlx.cli", "serve", REPO, "--host", "127.0.0.1",
         "--port", str(PORT), "--continuous-batching", "--max-num-seqs", "8"],
        env=env, stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 600:
            try:
                with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                    break
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError("server died")
                time.sleep(1)
        for _ in range(90):
            try:
                chat("Say OK.", 4, timeout=300)
                break
            except Exception:
                time.sleep(2)
        chat(PROMPT, 64)  # warmup
        print(f"== burst={burst}")
        for i in range(3):
            r = chat(PROMPT + f" (run {i})", 400)
            print(f"  single #{i}: decode={r['decode_tps']} ttft={r['ttft']} "
                  f"tok={r['tokens']} sha={r['sha'] if i == 0 else '-'}",
                  flush=True)
        for rnd in range(2):
            results = [None] * 4
            def w(i):
                results[i] = chat(CONC[i] + f" (r{rnd})", 256)
            th = [threading.Thread(target=w, args=(i,)) for i in range(4)]
            tw = time.monotonic()
            [t.start() for t in th]
            [t.join(600) for t in th]
            wall = time.monotonic() - tw
            toks = sum((r or {}).get("tokens") or 0 for r in results)
            ttfts = [round((r or {}).get("ttft") or -1, 2) for r in results]
            print(f"  conc4 #{rnd}: {toks} tok / {wall:.1f}s = "
                  f"{toks / wall:.1f} tok/s agg, ttfts={ttfts}", flush=True)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        logf.close()

if __name__ == "__main__":
    import sys
    for b in (sys.argv[1:] or ["1", "4", "8", "1"]):
        run_setting(int(b))
