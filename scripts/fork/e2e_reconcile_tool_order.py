#!/usr/bin/env python
"""End-to-end check of PATCHES.md #166 on a REAL server and model, off the
live routes: a client that sends the model's tool call back with its
arguments in another order (Hermes, 2026-11-18 07:42: the model wrote
web_search's limit before query, Hermes sent query first, 21K re-prefilled).

Turn 1 makes a tool call with two arguments. Turn 2 sends that call back with
the arguments reversed, plus a tool result. Run twice:
- control (#139 reconcile on, VLLM_MLX_BATCHED_RECONCILE_TOOL_ORDER unset):
  the divergence log must show the split inside the call - else the test
  proves nothing;
- armed: turn 2 must restore the whole turn-1 chain.

    E2E_MODEL=mlx-community/Qwen3.5-4B-4bit python scripts/fork/e2e_reconcile_tool_order.py

Run it from the source tree under test (it puts the cwd first on PYTHONPATH).
Exit code 0 only if all pass.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.5-4B-4bit")
PORT = int(os.environ.get("E2E_PORT", "8772"))
BASE = f"http://127.0.0.1:{PORT}"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query", "limit"],
            },
        },
    }
]


def run(workdir: str, armed: bool) -> dict:
    import httpx

    tag = "armed" if armed else "control"
    log_path = os.path.join(workdir, f"server-{tag}.log")
    env = dict(
        os.environ,
        VLLM_MLX_BATCHED_SYSTEM_KV="1",
        VLLM_MLX_DIVERGENCE_LOG="1",
        VLLM_MLX_BATCHED_RECONCILE_GENERATED="1",
    )
    env.pop("VLLM_MLX_BATCHED_RECONCILE_TOOL_ORDER", None)
    if armed:
        env["VLLM_MLX_BATCHED_RECONCILE_TOOL_ORDER"] = "1"
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL,
            "--host", "127.0.0.1", "--port", str(PORT),
            "--continuous-batching", "--text-only", "--max-num-seqs", "2",
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
            "--reasoning-parser", "qwen3",
        ],
        stdout=log, stderr=subprocess.STDOUT, env=env,
    )  # fmt: skip
    out = {"log": log_path}
    try:
        deadline = time.time() + 240
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(open(log_path).read()[-2000:])
            try:
                if httpx.get(f"{BASE}/v1/status", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        system = "You are a research assistant with a web_search tool. " * 40
        msgs = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "Search the web for 'seagate ironwolf 8TB price' with "
                "limit 5. Call the tool now.",
            },
        ]
        with httpx.Client(timeout=300) as c:

            def ask(messages):
                r = c.post(
                    f"{BASE}/v1/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": messages,
                        "tools": TOOLS,
                        "max_tokens": 200,
                        "temperature": 0.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                r.raise_for_status()
                return r.json()

            first = ask(msgs)
            message = first["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            out["calls"] = len(calls)
            out["turn1_tokens"] = (
                first["usage"]["prompt_tokens"] + first["usage"]["completion_tokens"]
            )
            sent = []
            for call in calls:
                args = json.loads(call["function"]["arguments"])
                out["arg_order"] = list(args)
                reordered = dict(reversed(list(args.items())))
                sent.append(
                    {
                        **call,
                        "function": {
                            **call["function"],
                            "arguments": json.dumps(reordered),
                        },
                    }
                )
            msgs.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": sent,
                }
            )
            for call in sent:
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": "1. Alza.sk - 219 EUR\n2. Datart - 225 EUR\n" * 10,
                    }
                )
            second = ask(msgs)
            usage = second["usage"]
            out["turn2_prompt"] = usage["prompt_tokens"]
            out["turn2_cached"] = (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0
            ) or 0
            reply = second["choices"][0]["message"]
            out["turn2_answered"] = bool(reply.get("content") or reply.get("tool_calls"))
            out["cache"] = c.get(f"{BASE}/v1/status").json().get("cache", {})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    return out


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-tool-order-")
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    control = run(workdir, armed=False)
    armed = run(workdir, armed=True)
    for tag, r in (("control", control), ("armed", armed)):
        check(f"{tag}: turn 1 made a two-argument call",
              r["calls"] >= 1 and len(r.get("arg_order", [])) == 2,
              f"calls={r['calls']} order={r.get('arg_order')}")  # fmt: skip
        check(f"{tag}: turn 2 answered", r["turn2_answered"])
        bad = [ln for ln in open(r["log"]).read().splitlines() if "Traceback" in ln]
        check(f"{tag}: server log has no traceback", not bad, "; ".join(bad[:2]))
    lines = [
        ln for ln in open(control["log"]).read().splitlines() if "] divergence " in ln
    ]
    check(
        "control: turn 2 split inside the tool call",
        any("<parameter=" in ln for ln in lines),
        lines[-1][-240:] if lines else "no divergence line",
    )
    cache = armed["cache"]
    check(
        "armed: the reordered call was reconciled",
        cache.get("reconcile_reorders", 0) >= 1,
        f"reorders={cache.get('reconcile_reorders')} "
        f"splices={cache.get('reconcile_splices')} "
        f"rejects={cache.get('reconcile_rejects')}",
    )
    check(
        "armed: turn 2 restored the whole turn-1 chain",
        armed["turn2_cached"] >= armed["turn1_tokens"] - 2
        and armed["turn2_cached"] > control["turn2_cached"],
        f"cached {control['turn2_cached']} -> {armed['turn2_cached']} "
        f"(turn-1 chain {armed['turn1_tokens']}, prompt {armed['turn2_prompt']})",
    )
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
