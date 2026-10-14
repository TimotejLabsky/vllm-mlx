#!/usr/bin/env python3
"""Post-deploy smoke test for ONE live llama-swap route (stdlib only).

Runs on the Studio against llama-swap (default http://127.0.0.1:8080):

  * non-stream chat answers, T=0, and a re-send is byte-identical and cached;
  * streaming delivers content and a [DONE];
  * a forced tool call comes back as a parsed ``tool_calls`` entry;
  * with thinking on, the reply carries reasoning separate from content;
  * the route's /v1/status is reachable and reports no request errors.

    scp scripts/fork/live_route_smoke.py mac-studio:/tmp/        # laptop
    python3 /tmp/live_route_smoke.py Qwen3.6-35B-A3B-4bit         # Studio

Exit 0 only if every check passes. Real traffic shares the route: keep it to a
handful of short requests and run it when the route is idle.
"""

import json
import sys
import time
import urllib.request

MODEL = sys.argv[1]
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080"
SYSTEM = "You are a terse assistant used by a deploy smoke test. " + " ".join(
    f"Rule {i}: answer in one short sentence and never invent facts."
    for i in range(1, 40)
)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))


def post(body, stream=False):
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(
            {"model": MODEL, "temperature": 0, "stream": stream, **body}
        ).encode(),
        {"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=600)


def chat(messages, **kw):
    return json.load(post({"messages": messages, "max_tokens": 48, **kw}))


def find(d, k):
    if isinstance(d, dict):
        if k in d:
            return d[k]
        for v in d.values():
            r = find(v, k)
            if r is not None:
                return r
    return None


off = {"chat_template_kwargs": {"enable_thinking": False}}
msgs = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "What is the capital of France?"},
]

t0 = time.time()
r1 = chat(msgs, **off)
c1 = r1["choices"][0]["message"].get("content") or ""
check("chat answers", "paris" in c1.lower(), f"{c1[:60]!r} ({time.time() - t0:.1f}s)")
r2 = chat(msgs, **off)
c2 = r2["choices"][0]["message"].get("content") or ""
cached = (r2["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
check("re-send is T=0 identical", c1 == c2)
check(
    "re-send is served from cache",
    cached >= r2["usage"]["prompt_tokens"] - 16,
    f"{cached}/{r2['usage']['prompt_tokens']}",
)

text, done = "", False
for raw in post({"messages": msgs, "max_tokens": 48, **off}, stream=True):
    line = raw.decode().strip()
    if line == "data: [DONE]":
        done = True
    elif line.startswith("data: "):
        delta = json.loads(line[6:])["choices"][0].get("delta", {})
        text += delta.get("content") or ""
check(
    "streaming delivers content + [DONE]",
    done and "paris" in text.lower(),
    repr(text[:60]),
)

tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
rt = chat(
    [{"role": "user", "content": "What's the weather in Oslo?"}],
    tools=[tool],
    tool_choice="required",
    max_tokens=96,
    **off,
)
calls = rt["choices"][0]["message"].get("tool_calls") or []
args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
check(
    "tool call parsed",
    calls
    and calls[0]["function"]["name"] == "get_weather"
    and "oslo" in str(args.get("city", "")).lower(),
    json.dumps(args),
)

rr = chat(
    [{"role": "user", "content": "Is 91 prime? Answer yes or no."}], max_tokens=512
)
m = rr["choices"][0]["message"]
reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
check(
    "thinking: reasoning separate from content",
    bool(reasoning) and "<think>" not in (m.get("content") or ""),
    f"reasoning {len(reasoning)} chars, content {(m.get('content') or '')[:40]!r}",
)

try:  # via llama-swap, else straight at a server on a spare port
    st = json.load(urllib.request.urlopen(f"{BASE}/upstream/{MODEL}/v1/status", timeout=30))
except Exception:
    st = json.load(urllib.request.urlopen(f"{BASE}/v1/status", timeout=30))
check("route /v1/status reachable", isinstance(st, dict))
print(
    "INFO  ssd:",
    {
        k: find(st, k)
        for k in ("ckpt_pruned", "ckpt_pruned_bytes", "spill_count", "read_failures")
    },
)
sys.exit(0 if all(results) else 1)
