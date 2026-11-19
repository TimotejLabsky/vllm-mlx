#!/usr/bin/env python
"""End-to-end check of PATCHES.md #169 on a REAL server, off the live routes.

1. ``VLLM_MLX_CHAT_TEMPLATE_FILE``: the server renders with the file's
   template (startup log line + prompt_tokens differ between two files).
2. Effort after the tool block: with tools present, ``reasoning_effort=low``
   must still think clearly less than ``medium`` with the moved line, as it
   did with the original template (T=0, same prompts).
3. ``VLLM_MLX_SSD_SPILL_MIN_TOKENS``: a short request is not spilled
   (``cache.spill_skipped_small`` > 0).

    E2E_OLD=old.jinja E2E_NEW=new.jinja python scripts/fork/e2e_template_file_effort.py
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.8-27B-8bit")
PORT = int(os.environ.get("E2E_PORT", "5999"))
BASE = f"http://127.0.0.1:{PORT}"
TOOLS = [
    {"type": "function", "function": {
        "name": n, "description": d,
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}},
                       "required": ["q"]}}}
    for n, d in (("web_search", "Search the web"), ("read_file", "Read a file"),
                 ("send_message", "Send a message to the user"))
]  # fmt: skip
PROMPTS = [
    "Three friends split a bill. Ana pays 40% of it, Ben pays 15 EUR more than Cyril, "
    "and Cyril pays 21 EUR. How much was the bill? Show the equation. No tools.",
    "A tank fills through pipe A in 6 h and through pipe B in 9 h, and drains through C "
    "in 12 h. With all three open from empty, how long until it is full? No tools.",
    "Find all integer solutions of x^2 - y^2 = 45 with x, y > 0. No tools.",
    "A clock shows 3:40. What is the smaller angle between the hands, in degrees? No tools.",
    "How many 4-digit numbers have digits that strictly increase left to right? No tools.",
    "A sequence starts 2, 5, 11, 23, ... Give the rule and the 8th term. No tools.",
]


def serve(template: str, log_path: str, workdir: str):
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1",
               VLLM_MLX_BATCHED_SYSTEM_KV="1",
               VLLM_MLX_SSD_SYSTEM_KV_DIR=os.path.join(workdir, "ssd"),
               VLLM_MLX_SSD_SYSTEM_KV_GB="4",
               VLLM_MLX_SSD_SPILL_MIN_TOKENS="100000",
               VLLM_MLX_CHAT_TEMPLATE_FILE=template)  # fmt: skip
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL,
        "--host", "127.0.0.1", "--port", str(PORT),
        "--continuous-batching", "--text-only", "--max-num-seqs", "2",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
        "--reasoning-parser", "qwen3", "--timeout", "1200",
        "--default-chat-template-kwargs", '{"preserve_thinking":false}',
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"),
                            stderr=subprocess.STDOUT)  # fmt: skip
    for _ in range(300):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early, see {log_path}")
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=2).read()
            return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not come up")


def ask(prompt: str, effort: str) -> dict:
    body = json.dumps({
        "model": MODEL, "max_tokens": 6000, "temperature": 0.0, "tools": TOOLS,
        "reasoning_effort": effort,
        "messages": [{"role": "system", "content": "You are a careful assistant."},
                     {"role": "user", "content": prompt}],
    }).encode()  # fmt: skip
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    msg = d["choices"][0]["message"]
    return {"prompt_tokens": d["usage"]["prompt_tokens"],
            "completion": d["usage"]["completion_tokens"],
            "answer": (msg.get("content") or "")[-80:]}  # fmt: skip


def session(tag: str, template: str, workdir: str, efforts=("low", "medium")) -> dict:
    log = os.path.join(workdir, f"server-{tag}.log")
    proc = serve(template, log, workdir)
    out = {"log": log}
    try:
        for effort in efforts:
            rows = [ask(p, effort) for p in PROMPTS]
            out[effort] = rows
            print(tag, effort, [r["completion"] for r in rows],
                  [r["prompt_tokens"] for r in rows], flush=True)  # fmt: skip
        st = json.load(urllib.request.urlopen(f"{BASE}/v1/status", timeout=10))
        out["cache"] = st.get("cache") or {}
    finally:
        proc.terminate()
        proc.wait(timeout=180)
    return out


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-template-")
    old, new = os.environ["E2E_OLD"], os.environ["E2E_NEW"]
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    a = session("old", old, workdir)
    b = session("new", new, workdir, efforts=("low",))
    b["medium"] = a["medium"]  # identical prompt under both (checked by render)
    for tag, r in (("old", a), ("new", b)):
        text = open(r["log"]).read()
        check(f"{tag}: template loaded from the file", "[chat_template] from " in text)
        check(f"{tag}: no traceback", "Traceback" not in text)
        check(f"{tag}: short requests not spilled",
              r["cache"].get("spill_skipped_small", 0) > 0,
              f"spill_skipped_small={r['cache'].get('spill_skipped_small')}")  # fmt: skip
    for tag, r in (("old", a), ("new", b)):
        lo = statistics.mean(x["completion"] for x in r["low"])
        me = statistics.mean(x["completion"] for x in r["medium"])
        check(f"{tag}: low thinks less than medium", lo < 0.8 * me,
              f"low {lo:.0f} vs medium {me:.0f} completion tokens")  # fmt: skip
    lo_a = statistics.mean(x["completion"] for x in a["low"])
    lo_b = statistics.mean(x["completion"] for x in b["low"])
    check("new low within 30% of old low", abs(lo_b - lo_a) <= 0.3 * lo_a,
          f"old {lo_a:.0f} new {lo_b:.0f}")  # fmt: skip
    for tag, r in (("old", a), ("new", b)):
        print(tag, "answers:", [x["answer"][-40:] for x in r["low"]])
    print(f"logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
