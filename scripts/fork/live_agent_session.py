"""A scripted agent session against a live route: a growing multi-turn tool
loop, the shape real agent traffic has. Checks every turn, not speed alone.

usage: agent_session.py <base_url> <model> [turns]
"""

import json
import sys
import time
import urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 10

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return stdout",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a text file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

REPO = "\n".join(
    f"src/handlers/handler_{i}.py: def handle_{i}(event): validates field k{i}, "
    f"stores record {i}, returns status code {200 + i % 7}"
    for i in range(180)
)
SYSTEM = (
    "You are a coding agent working in a repository. Use the tools to inspect and "
    "change files. Work step by step: one or two tool calls per turn, then continue. "
    "When the task is done, reply with a short summary and no tool call.\n\n"
    "Repository map:\n" + REPO
)
TASK = (
    "Task: handler_42 returns the wrong status code. Find it, read it, fix it so it "
    "returns 200, add a short test file, run the tests, then summarize."
)


def fake_tool(name, args):
    if name == "read_file":
        return (
            f"# {args.get('path')}\ndef handle_42(event):\n    if 'k42' not in event:\n"
            "        return 400\n    store(event)\n    return 206\n"
        )
    if name == "write_file":
        return f"wrote {len(args.get('content', ''))} bytes to {args.get('path')}"
    cmd = args.get("command", "")
    if "pytest" in cmd or "test" in cmd:
        return "collected 2 items\n\ntests/test_handler_42.py ..  [100%]\n\n2 passed in 0.04s"
    if "grep" in cmd or "rg " in cmd:
        return "src/handlers/handler_42.py:5:    return 206"
    if "ls" in cmd:
        return "src  tests  README.md  pyproject.toml"
    return "ok"


def chat(messages):
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": 1500,
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
    content, calls, finish, usage = [], {}, None, None
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                got = bool(
                    d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                )
                if d.get("content"):
                    content.append(d["content"])
                for tc in d.get("tool_calls") or []:
                    e = calls.setdefault(
                        tc.get("index", 0), {"id": "", "name": "", "arguments": ""}
                    )
                    e["id"] = tc.get("id") or e["id"]
                    f = tc.get("function", {})
                    e["name"] += f.get("name") or ""
                    e["arguments"] += f.get("arguments") or ""
                    got = True
                if got:
                    now = time.time()
                    first, last = first or now, now
                finish = ch.get("finish_reason") or finish
    usage = usage or {}
    ct = usage.get("completion_tokens") or 0
    span = (last - first) if first and last and last > first else None
    return {
        "ttft": round(first - t0, 2) if first else None,
        "tps": round((ct - 1) / span, 1) if span and ct > 1 else None,
        "prompt": usage.get("prompt_tokens"),
        "completion": ct,
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "finish": finish,
        "content": "".join(content),
        "calls": list(calls.values()),
    }


messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
problems, rows = [], []
for turn in range(TURNS):
    r = chat(messages)
    bad = []
    parsed = []
    for c in r["calls"]:
        try:
            parsed.append((c, json.loads(c["arguments"])))
        except Exception:
            bad.append(
                f"unparseable arguments for {c['name']}: {c['arguments'][:80]!r}"
            )
        if c["name"] not in ("run_shell", "read_file", "write_file"):
            bad.append(f"unknown tool {c['name']!r}")
    if r["finish"] not in ("tool_calls", "stop"):
        bad.append(f"finish_reason {r['finish']}")
    if r["finish"] == "tool_calls" and not r["calls"]:
        bad.append("tool_calls finish without calls")
    if turn > 0 and not r["cached"]:
        bad.append("no cached tokens on a follow-up turn")
    if (
        "<think>" in r["content"]
        or "</think>" in r["content"]
        or "<tool_call>" in r["content"]
    ):
        bad.append("markup leaked into content")
    rows.append(r)
    print(
        f"turn {turn}: finish={r['finish']} calls={[c['name'] for c in r['calls']]} "
        f"prompt={r['prompt']} cached={r['cached']} completion={r['completion']} "
        f"ttft={r['ttft']}s tps={r['tps']} {'PROBLEM: ' + '; '.join(bad) if bad else 'ok'}",
        flush=True,
    )
    problems.extend(f"turn {turn}: {b}" for b in bad)
    if r["finish"] == "stop" or not r["calls"]:
        print("final:", r["content"][:300].replace("\n", " "))
        break
    messages.append(
        {
            "role": "assistant",
            "content": r["content"] or None,
            "tool_calls": [
                {
                    "id": c["id"] or f"call_{turn}_{i}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for i, c in enumerate(r["calls"])
            ],
        }
    )
    for i, (c, args) in enumerate(parsed):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": c["id"] or f"call_{turn}_{i}",
                "content": fake_tool(c["name"], args),
            }
        )

tps = [r["tps"] for r in rows if r["tps"]]
print(
    f"\nturns={len(rows)} finished={'yes' if rows[-1]['finish'] == 'stop' else 'no'} "
    f"decode tok/s min/mean/max={min(tps)}/{round(sum(tps) / len(tps), 1)}/{max(tps)} "
    f"follow-up ttft max={max(r['ttft'] for r in rows[1:]) if len(rows) > 1 else None}s"
)
print("AGENT_SESSION", "PASS" if not problems else "FAIL")
for p in problems:
    print("  ", p)
sys.exit(1 if problems else 0)
