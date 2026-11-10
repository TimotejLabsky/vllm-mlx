#!/usr/bin/env python3
"""Real-server check for gpt-oss (harmony) tool calls on every endpoint shape
(PATCHES.md #155, #156). Stdlib only; run through route_under_test.py:

    python3 route_under_test.py gpt-oss-20b-MXFP4-Q4 --src /tmp/src -- \
        python3 /tmp/src/scripts/fork/e2e_harmony_tools.py

Checks, each on a fresh prompt at T=0:
  * chat completions non-stream and stream return a parsed ``get_weather`` call;
  * /v1/responses non-stream returns a ``function_call`` item (#155: it
    parsed the cleaned text and returned the call's JSON as output_text);
  * /v1/responses stream returns a ``function_call`` item;
  * a second tool turn, with the first call + its result in history (the
    prompt then shows harmony's recipient-first rendering, #156), still
    returns a parsed call and no harmony framing in content.
Exit 0 only if every check passes.
"""

import json
import os
import sys
import urllib.request

BASE = os.environ.get("SERVER_URL", sys.argv[1] if len(sys.argv) > 1 else "")
MODEL = "gpt-oss"
TOOL_FN = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
CHAT_TOOLS = [{"type": "function", "function": TOOL_FN}]
RESP_TOOLS = [{"type": "function", **TOOL_FN}]
ASK = "What is the weather in Paris right now? Use the get_weather tool."
FRAMING = ("<|channel|>", "<|message|>", "to=functions", "<|constrain|>")
results = []


def post(path, body, stream=False):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read().decode()
    if not stream:
        return json.loads(raw)
    events = []
    for line in raw.splitlines():
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            events.append(json.loads(line[6:]))
    return events


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")


def clean(text):
    return not any(tok in (text or "") for tok in FRAMING)


def chat_nonstream():
    r = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": ASK}],
            "tools": CHAT_TOOLS,
            "temperature": 0,
            "max_tokens": 512,
        },
    )
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    ok = [c["function"]["name"] for c in calls] == ["get_weather"]
    if ok:
        ok = "city" in json.loads(calls[0]["function"]["arguments"])
    check("chat non-stream tool call", ok and clean(msg.get("content")),
          f"calls={len(calls)} content={msg.get('content')!r:.80}")
    return calls


def chat_stream():
    events = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": ASK}],
            "tools": CHAT_TOOLS,
            "temperature": 0,
            "max_tokens": 512,
            "stream": True,
        },
        stream=True,
    )
    names, content = [], ""
    for e in events:
        for ch in e.get("choices", []):
            d = ch.get("delta", {})
            content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                if tc.get("function", {}).get("name"):
                    names.append(tc["function"]["name"])
    check("chat stream tool call", names == ["get_weather"] and clean(content),
          f"names={names} content={content!r:.80}")


def responses(stream):
    body = {
        "model": MODEL,
        "input": ASK,
        "tools": RESP_TOOLS,
        "temperature": 0,
        "max_output_tokens": 512,
        "store": False,
    }
    if stream:
        body["stream"] = True
        events = post("/v1/responses", body, stream=True)
        done = [e for e in events if e.get("type") == "response.completed"]
        output = done[-1]["response"]["output"] if done else []
        text = "".join(
            e.get("delta", "")
            for e in events
            if e.get("type") == "response.output_text.delta"
        )
    else:
        r = post("/v1/responses", body)
        output, text = r["output"], r.get("output_text") or ""
    calls = [o for o in output if o.get("type") == "function_call"]
    ok = [c["name"] for c in calls] == ["get_weather"]
    if ok:
        ok = "city" in json.loads(calls[0]["arguments"])
    label = "stream" if stream else "non-stream"
    check(f"responses {label} function_call", ok and clean(text) and "Paris" not in text,
          f"calls={len(calls)} output_text={text!r:.80}")


def second_turn(first_calls):
    call = first_calls[0]
    messages = [
        {"role": "user", "content": ASK},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps({"city": "Paris", "temp_c": 14}),
        },
        {
            "role": "user",
            "content": "Now the weather in Berlin, please. Use the tool again.",
        },
    ]
    r = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": messages,
            "tools": CHAT_TOOLS,
            "temperature": 0,
            "max_tokens": 512,
        },
    )
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    names = [c["function"]["name"] for c in calls]
    check("second tool turn (history has a prior call)",
          names == ["get_weather"] and clean(msg.get("content")),
          f"names={names} content={msg.get('content')!r:.80}")


if __name__ == "__main__":
    if not BASE:
        sys.exit("set SERVER_URL or pass the base URL")
    first = chat_nonstream()
    chat_stream()
    responses(stream=False)
    responses(stream=True)
    if first:
        second_turn(first)
    else:
        check("second tool turn (history has a prior call)", False, "no first call")
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
