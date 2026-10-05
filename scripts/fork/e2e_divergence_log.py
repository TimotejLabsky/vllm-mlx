#!/usr/bin/env python
"""End-to-end check of PATCHES.md #128 on a REAL server and model - off the
live routes.

Turn 1 thinks; turn 2 sends the history back WITHOUT the reasoning (what most
clients do), so the chat template re-renders the assistant turn without its
<think> block and the prompt parts ways with the cached chain right where the
thinking began. With VLLM_MLX_DIVERGENCE_LOG=1 that must show up as a
divergence at a think marker, and the server must log the decoded window.

    E2E_MODEL=mlx-community/Qwen3.5-4B-4bit python scripts/fork/e2e_divergence_log.py

Exit code 0 only if all pass.
"""

import os
import subprocess
import sys
import tempfile
import time

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.5-4B-4bit")
PORT = int(os.environ.get("E2E_PORT", "8770"))
BASE = f"http://127.0.0.1:{PORT}"


def main() -> int:
    import httpx

    workdir = tempfile.mkdtemp(prefix="e2e-divergence-")
    log_path = os.path.join(workdir, "server.log")
    env = dict(os.environ, VLLM_MLX_BATCHED_SYSTEM_KV="1", VLLM_MLX_DIVERGENCE_LOG="1")
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL,
            "--host", "127.0.0.1", "--port", str(PORT),
            "--continuous-batching", "--text-only", "--max-num-seqs", "2",
            "--reasoning-parser", "qwen3",
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
        system = "You are a careful assistant. " * 40  # > PARTIAL_MIN
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Is 391 prime? Answer in one sentence."},
        ]
        with httpx.Client(timeout=300) as c:

            def ask(messages):
                r = c.post(
                    f"{BASE}/v1/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": messages,
                        "max_tokens": 400,
                        "temperature": 0.0,
                    },
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]

            first = ask(msgs)
            msgs += [
                {"role": "assistant", "content": first.get("content") or ""},
                {"role": "user", "content": "And is 397 prime?"},
            ]
            ask(msgs)
            cache = c.get(f"{BASE}/v1/status").json().get("cache", {})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    check(
        "turn 1 really thought (else nothing is stripped)",
        bool(first.get("reasoning_content") or first.get("reasoning")),
    )
    check(
        "turn 2 counted as a divergence at a think marker",
        cache.get("divergence_events", 0) >= 1
        and cache.get("divergence_at_think", 0) >= 1,
        f"events={cache.get('divergence_events')} "
        f"at_think={cache.get('divergence_at_think')} "
        f"depth={cache.get('divergence_depth_hist')}",
    )
    lines = [ln for ln in open(log_path).read().splitlines() if "] divergence " in ln]
    check("server logged the decoded divergence window", bool(lines),
          lines[0][-200:] if lines else "")  # fmt: skip
    bad = [ln for ln in open(log_path).read().splitlines() if "Traceback" in ln]
    check("server log has no traceback", not bad, "; ".join(bad[:2]))
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
