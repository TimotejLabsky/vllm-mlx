"""Drafter on vs off at long context on a spare port, production text flags.
For each context size: a cold request and a warm follow-up turn, both T=0,
streamed; compares decode tok/s, TTFT and the exact text.

usage: long_ctx_ab.py [ctx_tokens ...]      (default 21000 60000)
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

MODEL = "mlx-community/Qwen3.8-27B-8bit"
DRAFT = "mlx-community/Qwen3.8-27B-MTP-8bit"
PORT = 5997
BASE = f"http://127.0.0.1:{PORT}"
CTXS = [int(x) for x in sys.argv[1:]] or [21000, 60000]


def serve(env_extra, log_path):
    env = dict(os.environ)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "VLLM_MLX_BATCHED_ADMISSION_ORDER": "sjf",
            "VLLM_MLX_BATCHED_RECONCILE_GENERATED": "1",
            "VLLM_MLX_SYSTEM_KV_RAM_MB": "6144",
            "VLLM_MLX_BATCHED_SYSTEM_KV": "1",
            "VLLM_MLX_BATCHED_KV_BUDGET_MB": "4096",
            "VLLM_MLX_BATCHED_PAD_WASTE_MB": "2048",
            "VLLM_MLX_BATCHED_MEM_WATERMARK_PCT": "80",
            "VLLM_MLX_BATCHED_MAX_QUEUE": "8",
        }
    )
    env.pop("VLLM_MLX_TEXT_MTP_DRAFTER", None)
    env.update(env_extra)
    cmd = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        "serve",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--continuous-batching",
        "--text-only",
        "--max-num-seqs",
        "4",
        "--enable-metrics",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "qwen3",
        "--cache-memory-mb",
        "16384",
        "--timeout",
        "1200",
        "--default-chat-template-kwargs",
        '{"reasoning_effort":"medium","preserve_thinking":false}',
        "--default-thinking-token-budget",
        "6144",
    ]
    proc = subprocess.Popen(
        cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT
    )
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early, see {log_path}")
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=2).read()
            return proc
        except Exception:
            time.sleep(2)
    proc.kill()
    raise RuntimeError("server did not become healthy")


def chat(messages, max_tokens=400):
    body = {
        "model": "x",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    first = last = None
    text, usage = [], {}
    with urllib.request.urlopen(req, timeout=1500) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("reasoning_content") or d.get("reasoning") or "") + (
                    d.get("content") or ""
                )
                if piece:
                    text.append(piece)
                    now = time.time()
                    first, last = first or now, now
    ct = usage.get("completion_tokens") or 0
    span = (last - first) if first and last and last > first else None
    full = "".join(text)
    return {
        "ttft": round(first - t0, 1) if first else None,
        "tps": round((ct - 1) / span, 2) if span and ct > 1 else None,
        "prompt": usage.get("prompt_tokens"),
        "completion": ct,
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "sha": hashlib.sha256(full.encode()).hexdigest()[:12],
        "text": full,
    }


def session(name, env_extra):
    proc = serve(env_extra, os.path.join(tempfile.gettempdir(), f"longctx-{name}.log"))
    out = {}
    try:
        chat([{"role": "user", "content": "Say hi."}], max_tokens=16)  # warm-up
        for ctx in CTXS:
            filler = "\n".join(
                f"def handler_{i}(event):\n    # step {i}: validate, transform and store the record\n"
                f"    return {{'id': {i}, 'ok': event.get('k{i}') is not None}}\n"
                for i in range(ctx // 50)
            )
            msgs = [
                {
                    "role": "system",
                    "content": "You are a code assistant. The project source follows.\n\n"
                    + filler,
                },
                {
                    "role": "user",
                    "content": "Describe in detail what handler_17, handler_200 and handler_399 do, "
                    "how they differ, and write a small pytest file covering all three.",
                },
            ]
            cold = chat(msgs)
            msgs += [
                {"role": "assistant", "content": "Understood."},
                {
                    "role": "user",
                    "content": "Now write a short docstring for each of those three handlers.",
                },
            ]
            warm = chat(msgs)
            out[ctx] = (cold, warm)
            for tag, r in (("cold", cold), ("warm", warm)):
                print(
                    f"{name} ctx={r['prompt']} {tag}: ttft={r['ttft']}s tps={r['tps']} "
                    f"tokens={r['completion']} cached={r['cached']} sha={r['sha']}",
                    flush=True,
                )
        st = json.load(urllib.request.urlopen(f"{BASE}/v1/status", timeout=10))
        m = st.get("mtp") or {}
        print(
            f"{name} mtp:",
            {
                k: m.get(k)
                for k in (
                    "attempted",
                    "acceptance_rate",
                    "errors",
                    "ms_per_round",
                    "disabled",
                )
            },
            "peak_gb:",
            (st.get("metal") or {}).get("peak_memory_gb"),
            flush=True,
        )
    finally:
        proc.terminate()
        proc.wait(timeout=180)
    return out


plain = session("plain", {})
mtp = session("drafter", {"VLLM_MLX_TEXT_MTP_DRAFTER": DRAFT})
ok = True
print()
for ctx in CTXS:
    for i, tag in enumerate(("cold", "warm")):
        a, b = plain[ctx][i], mtp[ctx][i]
        same = a["sha"] == b["sha"]
        ok = ok and same
        print(
            f"ctx~{ctx} {tag}: plain {a['tps']} tok/s -> drafter {b['tps']} tok/s "
            f"({(b['tps'] / a['tps'] - 1) * 100:+.0f}%)  ttft {a['ttft']}s -> {b['ttft']}s  "
            f"text identical: {same}"
        )
print("LONG_CTX_AB", "PASS" if ok else "FAIL (text differs)")
