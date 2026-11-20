#!/usr/bin/env python
"""A/B of the 27B-8bit chat template's empty-think rendering (#170), on a
REAL server off the live routes.

The route runs ``preserve_thinking: false`` and its clients (opencode,
LiteLLM) strip ``reasoning_content`` from the history they send back. The old
template rendered such a turn as an empty ``<think></think>`` inside the
current tool loop and as nothing once a user message followed, so each new
user message re-rendered the whole history. The new one renders it with no
think block wherever it sits.

Per template, the same agent tasks are run as a client that strips
reasoning: up to ``E2E_TOOL_TURNS`` tool turns with canned tool results,
then a new user message. Reported per template:

* TTFT of tool turns and of the new user turn, and the server's
  ``tokens_to_prefill`` for the user turn (from the server log);
* tool-call quality: calls the parser could not read (``<tool_call>`` left
  in content), duplicate calls within a turn, a call repeating the previous
  turn's exactly, tasks that never gave a final answer.

The server is started with the live route's own command and env (read from
the llama-swap config, ``-PAUSED`` accepted), with only the template file,
the port and the SSD dir replaced.

    E2E_OLD=old.jinja E2E_NEW=new.jinja python scripts/fork/e2e_template_empty_think.py
"""

import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

ROUTE = os.environ.get("E2E_ROUTE", "Qwen3.8-27B-8bit")
CONFIG = os.environ.get("E2E_CONFIG", "/Users/ai/llm-stack/config.yaml")
PORT = int(os.environ.get("E2E_PORT", "5997"))
TOOL_TURNS = int(os.environ.get("E2E_TOOL_TURNS", "6"))
MAX_TOKENS = int(os.environ.get("E2E_MAX_TOKENS", "2500"))
BASE = f"http://127.0.0.1:{PORT}"
PY = os.environ.get("E2E_PY", "/Users/ai/vllm-mlx-env/bin/python")


def _fn(name, desc, props, req):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": req}}}  # fmt: skip


S = {"type": "string"}
TOOLS = [
    _fn("web_search", "Search the web. Returns result rows.", {"query": S}, ["query"]),
    _fn("read_file", "Read a file from the workspace.", {"path": S}, ["path"]),
    _fn("list_dir", "List a directory in the workspace.", {"path": S}, ["path"]),
    _fn("run_shell", "Run a shell command on the server and return its output.",
        {"command": S}, ["command"]),  # fmt: skip
    _fn("write_file", "Write a file in the workspace.", {"path": S, "content": S},
        ["path", "content"]),  # fmt: skip
]

SYSTEM = (
    "You are a careful operations agent with access to tools. Use the tools to "
    "gather facts before answering; never invent tool output. When the task is "
    "done, answer the user directly without calling a tool."
)

TASKS = [
    "Find the current price of the Seagate IronWolf 8TB NAS drive at two shops "
    "and tell me which is cheaper.",
    "The service 'ledger' should listen on port 8081. Find which file in the "
    "workspace sets its port and change it.",
    "Check disk usage on the server and tell me which directory under /srv is "
    "the largest.",
    "Look up the latest release of the project 'tinyproxy' and summarise what "
    "changed.",
]
FOLLOW_UP = "Thanks. Now summarise what you did and found in three bullet points."


def tool_result(name, args, turn):
    """Deterministic, plausible tool output (~300-900 tokens)."""
    rows = []
    if name == "web_search":
        q = args.get("query", "")
        for i in range(12):
            rows.append(f"Result {i + 1}: '{q}' - shop{i % 4}.example price="
                        f"{180 + (i * 7 + turn) % 40} EUR stock={'yes' if i % 3 else 'no'} "
                        f"updated=2026-11-{10 + i % 9:02d}")  # fmt: skip
        if "tinyproxy" in q:
            rows.append("tinyproxy 1.11.3 released 2026-10-02: fixes CVE-2026-1234 "
                        "(header overflow), adds BasicAuthRealm, drops OpenSSL 1.1.")  # fmt: skip
        return "\n".join(rows)
    if name == "list_dir":
        p = args.get("path", ".")
        return "\n".join(f"{p.rstrip('/')}/{n}" for n in (
            "config/", "config/ledger.toml", "config/proxy.toml", "README.md",
            "docker-compose.yml", "src/", "scripts/deploy.sh"))  # fmt: skip
    if name == "read_file":
        p = args.get("path", "")
        if "ledger" in p:
            return '[server]\nhost = "0.0.0.0"\nport = 8080\nworkers = 4\n\n[db]\npath = "/srv/ledger/db.sqlite"\n'
        if "compose" in p:
            return "services:\n  ledger:\n    image: ledger:2.3\n    volumes: ['./config/ledger.toml:/etc/ledger.toml']\n    ports: ['8080:8080']\n"
        return f"(file {p}: 40 lines of unrelated configuration)\n" + "x = 1\n" * 30
    if name == "run_shell":
        c = args.get("command", "")
        if "du" in c or "df" in c:
            return ("41G\t/srv/media\n12G\t/srv/backups\n3.1G\t/srv/ledger\n"
                    "820M\t/srv/logs\n57G\ttotal")  # fmt: skip
        return f"$ {c}\nexit code: 0\n" + "ok\n" * 20
    if name == "write_file":
        return f"wrote {len(args.get('content', ''))} bytes to {args.get('path')}"
    return "unknown tool"


def route():
    import yaml

    models = yaml.safe_load(open(CONFIG))["models"]
    r = models.get(ROUTE) or models[ROUTE + "-PAUSED"]
    argv = shlex.split(r["cmd"])
    argv = argv[argv.index("serve") + 1 :]
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--port":
            out += ["--port", str(PORT)]
            i += 2
            continue
        out.append(argv[i])
        i += 1
    env = dict(item.partition("=")[::2] for item in r.get("env", []))
    return out, env


def serve(template, log_path, workdir):
    argv, env_route = route()
    env = dict(os.environ, **env_route)
    env.update(HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1",
               VLLM_MLX_SSD_SYSTEM_KV_DIR=os.path.join(workdir, "ssd"),
               VLLM_MLX_CHAT_TEMPLATE_FILE=template)  # fmt: skip
    env.pop("ROUTE", None)
    proc = subprocess.Popen([PY, "-m", "vllm_mlx.cli", "serve"] + argv, env=env,
                            stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                            cwd=workdir)  # fmt: skip
    for _ in range(900):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited, see {log_path}")
        try:
            urllib.request.urlopen(f"{BASE}/v1/models", timeout=2).read()
            return proc, argv[0]
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not come up")


def stream(model, messages):
    body = {"model": model, "messages": messages, "tools": TOOLS,
            "max_tokens": MAX_TOKENS, "stream": True}  # fmt: skip
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    t0 = time.perf_counter()
    first = None
    content, calls, finish = [], {}, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[6:]).get("choices") or []
            if not ch:
                continue
            d = ch[0].get("delta", {})
            finish = ch[0].get("finish_reason") or finish
            if (d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                    or d.get("tool_calls")) and first is None:  # fmt: skip
                first = time.perf_counter()
            if d.get("content"):
                content.append(d["content"])
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(
                    tc.get("index", 0), {"id": None, "name": "", "args": ""}
                )
                f = tc.get("function") or {}
                c["id"] = c["id"] or tc.get("id")
                c["name"] += f.get("name") or ""
                c["args"] += f.get("arguments") or ""
    return {"ttft": (first - t0) if first else None, "content": "".join(content),
            "calls": list(calls.values()), "finish": finish}  # fmt: skip


def run_task(model, task, ti):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    out = {"tool_ttft": [], "unparsed": 0, "dup_in_turn": 0, "repeat_prev": 0,
           "bad_args": 0, "answered": False, "turns": 0}  # fmt: skip
    prev = None
    for turn in range(TOOL_TURNS):
        r = stream(model, msgs)
        out["turns"] += 1
        if turn:
            out["tool_ttft"].append(r["ttft"])
        if "<tool_call>" in r["content"] or "<function=" in r["content"]:
            out["unparsed"] += 1
        if not r["calls"]:
            out["answered"] = bool(r["content"].strip())
            msgs.append({"role": "assistant", "content": r["content"]})
            break
        keys = [(c["name"], c["args"]) for c in r["calls"]]
        out["dup_in_turn"] += len(keys) - len(set(keys))
        if prev is not None and keys == prev:
            out["repeat_prev"] += 1
        prev = keys
        tcs = []
        for k, c in enumerate(r["calls"]):
            cid = c["id"] or f"call_{ti}_{turn}_{k}"
            tcs.append({"id": cid, "type": "function",
                        "function": {"name": c["name"], "arguments": c["args"]}})  # fmt: skip
        # Like opencode: reasoning is NOT sent back, only content + calls.
        msgs.append({"role": "assistant", "content": r["content"], "tool_calls": tcs})
        for tc in tcs:
            try:
                a = json.loads(tc["function"]["arguments"] or "{}")
            except ValueError:
                out["bad_args"] += 1
                a = {}
            msgs.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": tool_result(tc["function"]["name"], a, turn)})  # fmt: skip
    msgs.append({"role": "user", "content": FOLLOW_UP})
    r = stream(model, msgs)
    out["user_ttft"] = r["ttft"]
    out["follow_up_ok"] = bool(r["content"].strip()) and not r["calls"]
    return out


def user_turn_prefill(log_path):
    """tokens_to_prefill of each schedule line, in order."""
    pat = re.compile(r"tokens_to_prefill=(\d+)")
    return [int(m.group(1)) for line in open(log_path) if "[schedule]" in line
            for m in [pat.search(line)] if m]  # fmt: skip


def main():
    results = {}
    for label in ("old", "new"):
        template = os.environ[f"E2E_{label.upper()}"]
        workdir = tempfile.mkdtemp(prefix=f"e2e170-{label}-")
        log = os.path.join(workdir, "server.log")
        proc, model = serve(template, log, workdir)
        try:
            runs = [run_task(model, t, i) for i, t in enumerate(TASKS)]
            for i, r in enumerate(runs):
                print(label, i, json.dumps(r), flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                proc.kill()
        results[label] = {
            "tool_ttft_median": statistics.median(x for r in runs for x in r["tool_ttft"] if x),
            "user_ttft": [round(r["user_ttft"], 1) for r in runs],
            "unparsed": sum(r["unparsed"] for r in runs),
            "dup_in_turn": sum(r["dup_in_turn"] for r in runs),
            "repeat_prev": sum(r["repeat_prev"] for r in runs),
            "bad_args": sum(r["bad_args"] for r in runs),
            "answered": sum(r["answered"] for r in runs),
            "follow_up_ok": sum(r["follow_up_ok"] for r in runs),
            "turns": [r["turns"] for r in runs],
            "prefill_log": user_turn_prefill(log),
            "log": log,
        }  # fmt: skip
    print(json.dumps(results, indent=1))
    old, new = results["old"], results["new"]
    ok = (new["unparsed"] <= old["unparsed"] and new["dup_in_turn"] <= old["dup_in_turn"]
          and new["follow_up_ok"] == len(TASKS)
          and statistics.median(new["user_ttft"]) < statistics.median(old["user_ttft"]))  # fmt: skip
    print("PASS" if ok else "CHECK", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
