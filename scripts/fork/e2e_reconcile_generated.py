#!/usr/bin/env python
"""End-to-end check of PATCHES.md #139 on a REAL server and model - off the
live routes.

Fork issue #75: the Qwen3.8-27B-8bit template joins parallel tool calls with
"\\n\\n" where the model writes "\\n", so the turn after a parallel tool call
parts ways with the cached chain inside the model's own output. This serves a
small hybrid model through a copy of its snapshot whose template carries the
same change (weights are symlinks), and runs the same two turns twice:

- control (VLLM_MLX_BATCHED_RECONCILE_GENERATED unset): the divergence log
  must show the split at the separator - else the test proves nothing;
- armed: turn 2 must restore the whole turn-1 chain (prompt + output).

    E2E_MODEL=mlx-community/Qwen3.5-4B-4bit python scripts/fork/e2e_reconcile_generated.py

Exit code 0 only if all pass.
"""

import os
import subprocess
import sys
import tempfile
import time

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.5-4B-4bit")
PORT = int(os.environ.get("E2E_PORT", "8771"))
BASE = f"http://127.0.0.1:{PORT}"

STOCK = "{{- '\\n<tool_call>\\n<function=' + tool_call.name"
WIDENED = "{{- '\\n\\n<tool_call>\\n<function=' + tool_call.name"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}},
                "required": ["filePath"],
            },
        },
    }
]


def widened_model_dir(workdir: str) -> str:
    """The model's snapshot with the 27B-8bit separator in its template."""
    from huggingface_hub import snapshot_download

    src = snapshot_download(MODEL, local_files_only=True)
    dst = os.path.join(workdir, "model")
    os.makedirs(dst)
    for name in os.listdir(src):
        if name != "chat_template.jinja":
            os.symlink(os.path.join(src, name), os.path.join(dst, name))
    template = open(os.path.join(src, "chat_template.jinja")).read()
    if template.count(STOCK) != 1:
        raise RuntimeError(f"{MODEL}: stock non-first tool-call line not found")
    with open(os.path.join(dst, "chat_template.jinja"), "w") as f:
        f.write(template.replace(STOCK, WIDENED))
    return dst


def run(model_dir: str, workdir: str, armed: bool) -> dict:
    import httpx

    tag = "armed" if armed else "control"
    log_path = os.path.join(workdir, f"server-{tag}.log")
    env = dict(os.environ, VLLM_MLX_BATCHED_SYSTEM_KV="1", VLLM_MLX_DIVERGENCE_LOG="1")
    env.pop("VLLM_MLX_BATCHED_RECONCILE_GENERATED", None)
    if armed:
        env["VLLM_MLX_BATCHED_RECONCILE_GENERATED"] = "1"
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm_mlx.cli", "serve", model_dir,
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
        system = "You are a coding agent working in a repository. " * 40
        msgs = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "Read /repo/README.md and /repo/docs/USAGE.md. Call "
                "the read tool for BOTH files in this one reply, in parallel.",
            },
        ]
        with httpx.Client(timeout=300) as c:

            def ask(messages):
                r = c.post(
                    f"{BASE}/v1/chat/completions",
                    json={
                        "model": model_dir,
                        "messages": messages,
                        "tools": TOOLS,
                        "max_tokens": 300,
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
            msgs.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": calls,
                }
            )
            for call in calls:
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": "# Title\n\nSome documentation text. " * 20,
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
    workdir = tempfile.mkdtemp(prefix="e2e-reconcile-")
    model_dir = widened_model_dir(workdir)
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    control = run(model_dir, workdir, armed=False)
    armed = run(model_dir, workdir, armed=True)

    for tag, r in (("control", control), ("armed", armed)):
        check(f"{tag}: turn 1 made parallel tool calls", r["calls"] >= 2,
              f"calls={r['calls']}")  # fmt: skip
        check(f"{tag}: turn 2 answered", r["turn2_answered"])
        bad = [ln for ln in open(r["log"]).read().splitlines() if "Traceback" in ln]
        check(f"{tag}: server log has no traceback", not bad, "; ".join(bad[:2]))

    lines = [
        ln for ln in open(control["log"]).read().splitlines() if "] divergence " in ln
    ]
    check(
        "control: turn 2 split at the widened separator",
        any("</tool_call>\\n\\n<tool_call>" in ln for ln in lines),
        lines[-1][-240:] if lines else "no divergence line",
    )
    cache = armed["cache"]
    check(
        "armed: the generated span was reconciled",
        cache.get("reconcile_splices", 0) >= 1,
        f"splices={cache.get('reconcile_splices')} "
        f"kept={cache.get('reconcile_tokens_kept')} "
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
