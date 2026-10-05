#!/usr/bin/env python
"""End-to-end check of PATCHES.md #126 on a REAL server and model — off the
live routes.

Two conversations share a short (< BOUNDARY_MIN_STEP = 2048 tokens) system
prompt and differ from the first user message. On a hybrid (GDN/Mamba) model
a restore must snap down to a recurrent-state checkpoint, so the second
conversation reuses the system prompt only if the first one checkpointed the
end of the system message. Before #126 it never did (``cached_tokens`` 0).

    E2E_MODEL=mlx-community/Qwen3.5-4B-4bit E2E_SERVE_ARGS=--text-only \\
        python scripts/fork/e2e_short_system_share.py

Must be a hybrid model: a pure-attention model slices KV at any position and
passes with or without the patch. Exit code 0 only if all pass.
"""

import os
import subprocess
import sys
import tempfile
import time

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.5-4B-4bit")
PORT = int(os.environ.get("E2E_PORT", "8767"))
BASE = f"http://127.0.0.1:{PORT}"

# ~800 tokens: over PARTIAL_MIN (256), under BOUNDARY_MIN_STEP (2048)
SYSTEM = "You are a meticulous code reviewer. " + " ".join(
    f"Rule {i}: keep every finding specific, cite the line, and never invent "
    f"behaviour that the diff does not show."
    for i in range(1, 41)
)
USERS = [
    "Review this change: `def add(a, b): return a - b`. One sentence.",
    "Explain what a mutex is to a five-year-old. One sentence.",
    "List two prime numbers above 100, nothing else.",
]


def _ask(client, user):
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "max_tokens": 24,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    r.raise_for_status()
    usage = r.json()["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return usage["prompt_tokens"], cached


def main() -> int:
    import httpx

    workdir = tempfile.mkdtemp(prefix="e2e-sysshare-")
    log_path = os.path.join(workdir, "server.log")
    env = dict(os.environ, VLLM_MLX_BATCHED_SYSTEM_KV="1")
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL,
            "--host", "127.0.0.1", "--port", str(PORT),
            "--continuous-batching", "--max-num-seqs", "4",
            *os.environ.get("E2E_SERVE_ARGS", "").split(),
        ],
        stdout=log, stderr=subprocess.STDOUT, env=env,
    )  # fmt: skip
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

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
        with httpx.Client(timeout=300) as client:
            rows = [_ask(client, user) for user in USERS]
            status = client.get(f"{BASE}/v1/status").json()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    (first_prompt, first_cached), *later = rows
    check("first conversation is cold", first_cached == 0, f"cached={first_cached}")
    check(
        "system prompt is under BOUNDARY_MIN_STEP (the case #126 fixes)",
        first_prompt < 2048,
        f"prompt_tokens={first_prompt}",
    )
    for i, (prompt, cached) in enumerate(later, start=2):
        # the shared prefix is the system message: most of the prompt
        check(
            f"conversation {i} reuses the shared system prompt",
            cached >= 0.6 * prompt and cached < prompt,
            f"cached={cached}/{prompt}",
        )
    cache = status.get("cache", {})
    print(f"cache: hits={cache.get('hits')} misses={cache.get('misses')}")
    bad = [ln for ln in open(log_path).read().splitlines() if "Traceback" in ln]
    check("server log has no traceback", not bad, "; ".join(bad[:2]))
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
