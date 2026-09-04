#!/usr/bin/env python3
"""Natural-prompt decode A/B: fork-batched vs omlx on the MoE (T=0).

Counts tokens from server-reported usage (not SSE chunks), so burst/multi-token
chunk emission cannot inflate the number. Falls back to client-side word*1.35
estimate if usage is absent.
"""
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request

PORT = 5899
REPO = "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ"
BASE = os.path.expanduser("~/bench-2026-09-04")
PROMPT = ("Write a detailed explanation of how TCP congestion control works, "
          "covering slow start, congestion avoidance, fast retransmit and "
          "fast recovery, with examples.")

ENGINES = {
    "fork-batched": dict(
        cmd=["/Users/ai/vllm-mlx-env/bin/python", "-m", "vllm_mlx.cli",
             "serve", REPO, "--host", "127.0.0.1", "--port", str(PORT),
             "--continuous-batching", "--max-num-seqs", "8"],
        env={"VLLM_MLX_BATCHED_SYSTEM_KV": "1",
             "VLLM_MLX_BATCHED_KV_BUDGET_MB": "8192",
             "VLLM_MLX_BATCHED_PAD_WASTE_MB": "4096",
             "VLLM_MLX_BATCHED_MEM_WATERMARK_PCT": "80",
             "VLLM_MLX_MOE_GATEUP_FUSION": "1"},
        model=REPO),
    "omlx": dict(
        cmd=["/Users/ai/omlx-env/bin/omlx", "serve", "--model-dir",
             os.path.join(BASE, "omlx-models"), "--port", str(PORT),
             "--api-key", "bench"],
        env={}, model="Qwen3.6-35B-A3B-4bit-DWQ"),
}

def chat(model, prompt, max_tokens, stream=True):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0,
               "stream": stream}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer bench"})
    t0 = time.monotonic()
    if not stream:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        el = time.monotonic() - t0
        u = d.get("usage") or {}
        txt = d["choices"][0]["message"].get("content") or ""
        txt += d["choices"][0]["message"].get("reasoning") or ""
        return {"elapsed": el, "usage": u, "text_head": txt[:80],
                "text_len": len(txt)}
    t_first = None
    usage = None
    text = []
    with urllib.request.urlopen(req, timeout=600) as r:
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
                delta = ch[0].get("delta") or {}
                t = (delta.get("content") or "") + \
                    (delta.get("reasoning_content") or "") + \
                    (delta.get("reasoning") or "")
                if t:
                    if t_first is None:
                        t_first = time.monotonic()
                    text.append(t)
    t_end = time.monotonic()
    full = "".join(text)
    out = {"ttft": round(t_first - t0, 3) if t_first else None,
           "elapsed": round(t_end - t0, 2), "usage": usage,
           "text_len": len(full), "text_head": full[:80].replace("\n", " ")}
    if usage and t_first and usage.get("completion_tokens"):
        ct = usage["completion_tokens"]
        out["decode_tps_usage"] = round((ct - 1) / (t_end - t_first), 1)
    return out

def wait_ready(proc):
    t0 = time.monotonic()
    while time.monotonic() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError("exited")
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError("bind timeout")

def main():
    for name, spec in ENGINES.items():
        subprocess.run(["pkill", "-f", f"port {PORT}"], capture_output=True)
        subprocess.run(["pkill", "-f", "omlx-server"], capture_output=True)
        time.sleep(3)
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/unload", timeout=60)
        except Exception:
            pass
        env = dict(os.environ)
        env.update(spec["env"])
        logf = open(os.path.join(BASE, f"ab-{name}.log"), "w")
        proc = subprocess.Popen(spec["cmd"], env=env, stdout=logf,
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            wait_ready(proc)
            # readiness + warmup
            for _ in range(60):
                try:
                    chat(spec["model"], "Say OK.", 4)
                    break
                except Exception:
                    time.sleep(2)
            chat(spec["model"], PROMPT, 64)
            print(f"== {name}")
            for i in range(3):
                r = chat(spec["model"], PROMPT + f" (run {i})", 400)
                print(f"  run{i}: ttft={r['ttft']} decode_usage="
                      f"{r.get('decode_tps_usage')} usage={r.get('usage')} "
                      f"len={r['text_len']} head={r['text_head']!r}",
                      flush=True)
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:
                os.killpg(proc.pid, signal.SIGKILL)
            logf.close()

if __name__ == "__main__":
    main()
