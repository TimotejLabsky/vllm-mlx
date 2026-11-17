#!/usr/bin/env python
"""End-to-end check of PATCHES.md #163 on a REAL server and model, off the
live routes: the 2026-11-17 opencode sequence, forced.

Same widened-separator model as ``e2e_reconcile_generated.py`` (#139). One RAM
slot, and a co-batch deferral threshold of 1 MB, so a request that arrives
while another runs waits for it. Then:

1. turn 1: parallel tool calls (stored, with its generated span);
2. a background request starts; 0.5 s later turn 2 arrives, is matched
   against turn 1's chain and reconciled, and waits. The background request
   finishes and stores, which evicts turn 1's entry. Turn 2 then runs and
   stores - with nothing left to absorb;
3. turn 3 re-renders turn 1's calls again. It must restore the whole turn-2
   chain. Before #163 it re-prefilled from the last checkpoint, because the
   only record of turn 1's span went with the evicted entry.

    E2E_MODEL=mlx-community/Qwen3.5-4B-4bit python scripts/fork/e2e_reconcile_evicted.py

Run it from the source tree under test (it puts the cwd first on PYTHONPATH).
Exit code 0 only if all pass.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_reconcile_generated import (
    BASE,
    MODEL,
    PORT,
    TOOLS,
    widened_model_dir,
)  # noqa: E402

BACKGROUND = (
    "Write a long, detailed summary of this text, at least 200 words.\n\n"
    + "The quick brown fox jumps over the lazy dog near the river bank. " * 60
)
RESULT_1 = (
    "# Title\n\nSome documentation text. " * 20
    + "\nNow call read for /repo/CHANGELOG.md and /repo/LICENSE in this one reply, "
    "in parallel."
)


def serve(model_dir, workdir):
    import httpx

    log_path = os.path.join(workdir, "server.log")
    env = dict(
        os.environ,
        VLLM_MLX_BATCHED_SYSTEM_KV="1",
        VLLM_MLX_DIVERGENCE_LOG="1",
        VLLM_MLX_BATCHED_RECONCILE_GENERATED="1",
        VLLM_MLX_SYSTEM_KV_SLOTS="1",
        VLLM_MLX_BATCHED_PAD_WASTE_MB="1",
    )
    env.pop("VLLM_MLX_SSD_SYSTEM_KV_DIR", None)
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
    deadline = time.time() + 240
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(open(log_path).read()[-2000:])
        try:
            if httpx.get(f"{BASE}/v1/status", timeout=2).status_code == 200:
                return proc, log, log_path
        except Exception:
            time.sleep(1)
    proc.kill()
    raise RuntimeError("server did not become healthy")


def ask(model_dir, messages, tools=True, max_tokens=300):
    import httpx

    body = {
        "model": model_dir,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = TOOLS
    r = httpx.post(f"{BASE}/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    return r.json()


def cached(usage):
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0


def turn(model_dir, msgs, result_text):
    r = ask(model_dir, msgs)
    usage = r["usage"]
    message = r["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    msgs.append(
        {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": calls,
        }
    )
    for call in calls:
        msgs.append(
            {"role": "tool", "tool_call_id": call["id"], "content": result_text}
        )
    return {
        "prompt": usage["prompt_tokens"],
        "cached": cached(usage),
        "chain": usage["prompt_tokens"] + usage["completion_tokens"],
        "calls": len(calls),
    }


def main() -> int:
    import httpx

    workdir = tempfile.mkdtemp(prefix="e2e-reconcile-evicted-")
    model_dir = widened_model_dir(workdir)
    proc, log, log_path = serve(model_dir, workdir)
    try:
        system = "You are a coding agent working in a repository. " * 40
        msgs = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "Read /repo/README.md and /repo/docs/USAGE.md. Call the "
                "read tool for BOTH files in this one reply, in parallel.",
            },
        ]
        t1 = turn(model_dir, msgs, RESULT_1)
        evictions_before = httpx.get(f"{BASE}/v1/status").json()["cache"]["evictions"]

        bg = {}
        worker = threading.Thread(
            target=lambda: bg.update(
                ask(model_dir, [{"role": "user", "content": BACKGROUND}], False, 200)
            )
        )
        worker.start()
        time.sleep(0.5)
        t2 = turn(model_dir, msgs, "More text. " * 20)
        worker.join()
        mid = httpx.get(f"{BASE}/v1/status").json()["cache"]

        # A tool continuation, like the field trace: a new USER message makes
        # the template drop the earlier <think> blocks - a split in the prompt
        # before any generated span, which is not what this checks.
        r3 = ask(model_dir, msgs)
        t3 = {"prompt": r3["usage"]["prompt_tokens"], "cached": cached(r3["usage"])}
        cache = httpx.get(f"{BASE}/v1/status").json()["cache"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    text = open(log_path).read()
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    check(
        "turn 1 made parallel tool calls",
        t1["calls"] >= 2,
        f"calls={t1['calls']} (turn 2: {t2['calls']}, either is fine)",
    )
    check(
        "turn 2 was reconciled at arrival",
        "reconciled generated span" in text,
        f"splices after turn 2: {mid.get('reconcile_splices')}",
    )
    check(
        "the background request evicted turn 1's entry before turn 2 stored",
        mid["evictions"] > evictions_before and "deferring co-batch" in text,
        f"evictions {evictions_before} -> {mid['evictions']}",
    )
    check(
        "turn 3 restored the whole turn-2 chain",
        t3["cached"] >= t2["chain"] - 2,
        f"cached {t3['cached']} of prompt {t3['prompt']} (turn-2 chain {t2['chain']}); "
        f"splices={cache.get('reconcile_splices')} "
        f"registry_spans={cache.get('reconcile_registry_spans')}",
    )
    bad = [ln for ln in text.splitlines() if "Traceback" in ln]
    check("server log has no traceback", not bad, "; ".join(bad[:2]))
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
