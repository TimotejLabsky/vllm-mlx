#!/usr/bin/env python
"""Fork vs upstream vllm-mlx on the production box, per llama-swap route.

Each route's PRODUCTION command and env are read from the llama-swap config
(``--config``); the server runs on a spare port with the SSD tier pointed at
a fresh temp dir (no live cache entries). ``fork`` = the installed package;
``upstream`` = an upstream checkout put first on PYTHONPATH (``--upstream``),
with CLI flags upstream does not have dropped. Same model, prompts, sampling.

Scenarios (all streamed, timings from the client):
  decode   3 short prompts, 400 tokens: median decode tok/s, TTFT
  session  agent session: ~8K-token system prompt + tools, then tool-loop
           turns (each appends the model's call + a ~1.2K tool result):
           TTFT per turn, total wall
  depth    one long answer at the end of the session: decode tok/s at depth
  restart  stop the server, start it again, next turn: TTFT
  image    (vision routes) image+text request decode, then a follow-up turn

    python scripts/fork/bench_fork_vs_upstream.py --route Qwen3.8-27B-8bit \
        --engine fork --out fork-27b.json
"""

import argparse
import base64
import io
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--route", required=True)
ap.add_argument("--engine", choices=["fork", "upstream"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--config", default="/Users/ai/llm-stack/config.yaml")
ap.add_argument("--upstream", default="/tmp/upstream-vllm-mlx")
ap.add_argument("--port", type=int, default=5996)
ap.add_argument("--scenarios", default="decode,session,depth,restart")
ap.add_argument("--turns", type=int, default=6)
args = ap.parse_args()
BASE = f"http://127.0.0.1:{args.port}"
PY = "/Users/ai/vllm-mlx-env/bin/python"


def route_cmd():
    import yaml

    cfg = yaml.safe_load(open(args.config))
    models = cfg["models"]
    # paused2.sh renames the benchmarked route so llama-swap cannot load it
    route = models.get(args.route) or models[args.route + "-PAUSED"]
    argv = shlex.split(route["cmd"])
    argv = argv[argv.index("serve") + 1 :]  # model + flags
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--port":
            out += ["--port", str(args.port)]
            i += 2
            continue
        out.append(a)
        i += 1
    env = {}
    for item in route.get("env", []):
        k, _, v = item.partition("=")
        env[k] = v
    return out, env


def drop_unsupported(argv):
    flags = set(open("/tmp/mtp-verify/upstream-flags.txt").read().split())
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and a not in flags:
            nxt = argv[i + 1] if i + 1 < len(argv) else ""
            i += 1 if (not nxt or nxt.startswith("--")) else 2
            print(f"[upstream] dropping unsupported {a}", flush=True)
            continue
        out.append(a)
        i += 1
    return out


SSD_DIR = tempfile.mkdtemp(prefix="bench-ssd-")


def start(log_path):
    argv, env_extra = route_cmd()
    model = argv[0]
    if args.engine == "upstream":
        argv = drop_unsupported(argv)
    env = dict(os.environ, **env_extra)
    env.update(HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1", VLLM_MLX_SSD_SYSTEM_KV_DIR=SSD_DIR)
    env.pop("ROUTE", None)
    if args.engine == "upstream":
        env["PYTHONPATH"] = args.upstream
    else:
        env.pop("PYTHONPATH", None)
    proc = subprocess.Popen([PY, "-m", "vllm_mlx.cli", "serve"] + argv, env=env,
                            stdout=open(log_path, "a"), stderr=subprocess.STDOUT,
                            cwd="/tmp")  # fmt: skip
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited, see {log_path}")
        try:
            urllib.request.urlopen(f"{BASE}/v1/models", timeout=2).read()
            return proc, model
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not come up")


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()


def stream(model, messages, max_tokens, tools=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    t0 = time.perf_counter()
    first = last = None
    n, text, calls = 0, [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[6:])["choices"]
            if not ch:
                continue
            d = ch[0].get("delta", {})
            piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "name": "", "args": ""})
                f = tc.get("function") or {}
                c["id"] = c["id"] or tc.get("id")
                c["name"] += f.get("name") or ""
                c["args"] += f.get("arguments") or ""
                piece = piece or "x"
            if piece:
                now = time.perf_counter()
                first = first or now
                last = now
                n += 1
                if d.get("content"):
                    text.append(d["content"])
    ttft = (first - t0) if first else None
    rate = (n - 1) / (last - first) if n > 1 and last > first else None
    return {"ttft": ttft, "rate": rate, "chunks": n, "wall": time.perf_counter() - t0,
            "content": "".join(text), "calls": list(calls.values())}  # fmt: skip


# ---------------------------------------------------------------- prompts
def tools():
    return [
        {"type": "function", "function": {
            "name": f"tool_{name}", "description": desc + " " + ("Returns JSON. " * 25),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "What to look up " * 6},
                "limit": {"type": "integer", "description": "Max results " * 6}},
                "required": ["query"]}}}
        for name, desc in [
            ("web_search", "Search the web"), ("read_file", "Read a file"),
            ("write_file", "Write a file"), ("list_dir", "List a directory"),
            ("shell", "Run a shell command"), ("calendar", "Query the calendar"),
            ("email", "Search email"), ("weather", "Weather forecast"),
            ("home_assistant", "Control the smart home"), ("notes", "Search notes"),
            ("memory", "Recall long-term memory"), ("send_message", "Message the user"),
            ("browser", "Open a web page"), ("finance", "Query bank transactions"),
            ("tasks", "Manage the task list"), ("news", "Fetch news headlines"),
        ]
    ]  # fmt: skip


SYSTEM = ("You are Hermes, a personal assistant agent running scheduled jobs for one "
          "user. Follow the skill instructions exactly, call tools when you need data, "
          "and finish with a concise report. " * 40)  # fmt: skip
RESULT = ("Result row: id={i} title='Seagate IronWolf 8TB NAS drive' price={p} EUR "
          "shop=alza.sk stock=yes rating=4.6 reviews=312 shipping=free\n")  # fmt: skip


def tool_result(turn):
    return "".join(RESULT.format(i=k, p=180 + turn * 3 + k) for k in range(24))


def image_url():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(img)
    for i in range(0, 448, 32):
        d.rectangle([i, i, i + 60, i + 40], outline=(i % 255, 80, 160), width=3)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


SHORT = [
    "Explain how a hash map handles collisions, in a few paragraphs.",
    "Write a short Python function that merges overlapping intervals, with a docstring.",
    "What are the trade-offs between SQLite and PostgreSQL for a home server?",
]


# --------------------------------------------------------------- scenarios
def run():
    log = args.out + ".server.log"
    res = {"route": args.route, "engine": args.engine, "dropped_flags": []}
    proc, model = start(log)
    res["model"] = model
    T = tools()
    try:
        sc = args.scenarios.split(",")
        if "decode" in sc:
            rows = [stream(model, [{"role": "user", "content": q}], 400) for q in SHORT]
            res["decode"] = {"rates": [r["rate"] for r in rows], "ttfts": [r["ttft"] for r in rows],
                             "median_rate": statistics.median(r["rate"] for r in rows if r["rate"])}  # fmt: skip
            print("decode", res["decode"], flush=True)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "[Scheduled job: price-watch] Find the current "
                 "price of the Seagate IronWolf 8TB and compare with last week. Use "
                 "tool_web_search first."}]  # fmt: skip
        if "session" in sc:
            turns = []
            t_start = time.perf_counter()
            for t in range(args.turns):
                r = stream(model, msgs, 200, tools=T)
                turns.append({"ttft": r["ttft"], "rate": r["rate"], "calls": len(r["calls"])})
                print(f"turn {t}: ttft={r['ttft']:.1f}s calls={len(r['calls'])}", flush=True)
                call = (r["calls"] or [{"id": f"c{t}", "name": "tool_web_search",
                                         "args": '{"query": "ironwolf 8tb price"}'}])[0]  # fmt: skip
                cid = call["id"] or f"c{t}"
                msgs.append({"role": "assistant", "content": r["content"],
                             "tool_calls": [{"id": cid, "type": "function", "function": {
                                 "name": call["name"] or "tool_web_search",
                                 "arguments": call["args"] or "{}"}}]})  # fmt: skip
                msgs.append({"role": "tool", "tool_call_id": cid, "content": tool_result(t)})
            res["session"] = {"turns": turns, "wall": time.perf_counter() - t_start,
                              "first_ttft": turns[0]["ttft"],
                              "followup_ttft_median": statistics.median(x["ttft"] for x in turns[1:])}  # fmt: skip
        if "depth" in sc:
            q = msgs + [{"role": "user", "content": "Now write the full report: a detailed "
                         "comparison of every result row above, at least 300 words."}]  # fmt: skip
            r = stream(model, q, 400, tools=T)
            res["depth"] = {"ttft": r["ttft"], "rate": r["rate"]}
            print("depth", res["depth"], flush=True)
        if "toolrestart" in sc:
            # The agent's NEXT tool turn after a reload (a new user message
            # would make this template drop earlier <think> blocks - a split
            # no cache can repair). Wait for the SSD writer's idle window.
            time.sleep(45)
            stop(proc)
            time.sleep(3)
            proc, model = start(log)
            r = stream(model, msgs, 100, tools=T)
            res["toolrestart"] = {"ttft": r["ttft"]}
            print("toolrestart", res["toolrestart"], flush=True)
        if "restart" in sc:
            stop(proc)
            time.sleep(3)
            proc, model = start(log)
            q = msgs + [{"role": "user", "content": "Summarise what you found in two sentences."}]
            r = stream(model, q, 100, tools=T)
            res["restart"] = {"ttft": r["ttft"]}
            print("restart", res["restart"], flush=True)
        if "image" in sc:
            url = image_url()
            m = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                  {"type": "text", "text": "Describe this image in great detail, then write a "
                   "long story about it."}]}]  # fmt: skip
            stream(model, m, 16)  # warm the vision path
            rows = [stream(model, m, 300) for _ in range(3)]
            m2 = m + [{"role": "assistant", "content": rows[0]["content"]},
                      {"role": "user", "content": "Now list the colours you see."}]
            f = stream(model, m2, 100)
            res["image"] = {"rates": [r["rate"] for r in rows],
                            "median_rate": statistics.median(r["rate"] for r in rows if r["rate"]),
                            "followup_ttft": f["ttft"]}  # fmt: skip
            print("image", res["image"], flush=True)
    finally:
        stop(proc)
        shutil.rmtree(SSD_DIR, ignore_errors=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("BENCH_DONE", args.route, args.engine, flush=True)


run()
