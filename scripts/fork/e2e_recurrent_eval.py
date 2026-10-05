#!/usr/bin/env python
"""End-to-end check of PATCHES.md #127 on a REAL server and model - off the
live routes.

  * identity: T=0 output with VLLM_MLX_EVAL_RECURRENT_STATES on equals off
    (the eval forces no different maths, only earlier materialisation);
  * endurance (E2E_LONG=1): one generation well past the ~10.5K-token
    ``[metal::malloc] Resource limit (499000)`` wall of Qwen3.8-27B-4bit
    finishes as ``length`` with the patch on.

    E2E_MODEL=mlx-community/Qwen3.8-27B-4bit E2E_LONG=1 \\
        python scripts/fork/e2e_recurrent_eval.py

Exit code 0 only if all pass.
"""

import os
import subprocess
import sys
import tempfile
import time

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.5-4B-4bit")
PORT = int(os.environ.get("E2E_PORT", "8769"))
BASE = f"http://127.0.0.1:{PORT}"
LONG_TOKENS = int(os.environ.get("E2E_LONG_TOKENS", "12000"))
COUNT = "".join(f"{i}\n" for i in range(1, 201))
PROMPTS = [
    "Explain what a mutex is in two sentences.",
    "Write a haiku about recurrent neural networks.",
]


def _session(env_extra, workdir, label, long_run):
    import httpx

    env = dict(os.environ, VLLM_MLX_BATCHED_SYSTEM_KV="1", **env_extra)
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    log_path = os.path.join(workdir, f"server-{label}.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL,
            "--host", "127.0.0.1", "--port", str(PORT),
            "--continuous-batching", "--text-only", "--max-num-seqs", "4",
            "--timeout", "3600",
        ],
        stdout=log, stderr=subprocess.STDOUT, env=env,
    )  # fmt: skip
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(open(log_path).read()[-2000:])
            try:
                if httpx.get(f"{BASE}/v1/status", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(1)

        def ask(text, n):
            r = httpx.post(
                f"{BASE}/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": text}],
                    "max_tokens": n,
                    "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=3600,
            )
            r.raise_for_status()
            j = r.json()
            return j["choices"][0], j["usage"]

        outs = [ask(p, 96)[0]["message"]["content"] for p in PROMPTS]
        long = None
        if long_run:
            # Raw completion of a bare number sequence: a chat turn may end
            # itself (the 27B stopped at 3.9K), a counting continuation runs
            # to max_tokens.
            t0 = time.time()
            r = httpx.post(
                f"{BASE}/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": COUNT,
                    "max_tokens": LONG_TOKENS,
                    "temperature": 0.0,
                },
                timeout=3600,
            )
            if r.status_code != 200:
                return outs, (f"http {r.status_code}", 0, time.time() - t0), (
                    open(log_path).read()
                )
            j = r.json()
            long = (
                j["choices"][0]["finish_reason"],
                j["usage"]["completion_tokens"],
                time.time() - t0,
            )
        return outs, long, open(log_path).read()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-recurrent-")
    long_run = os.environ.get("E2E_LONG") == "1"
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    # E2E_LONG_OFF=1 also runs the long generation unpatched: expected to
    # die at the wall (informational - it shows the e2e can see the bug).
    off, off_long, _ = _session(
        {"VLLM_MLX_EVAL_RECURRENT_STATES": "0"},
        workdir,
        "off",
        long_run and os.environ.get("E2E_LONG_OFF") == "1",
    )
    if off_long:
        print(f"INFO  unpatched long run: finish={off_long[0]} tokens={off_long[1]}")
    on, long, log = _session({}, workdir, "on", long_run)
    check("T=0 output identical with the per-step eval on and off", on == off,
          repr(on[0][:60]))  # fmt: skip
    if long_run:
        reason, tokens, secs = long
        check(
            f"a {LONG_TOKENS}-token generation finishes past the 10.5K-step wall",
            reason == "length" and tokens >= LONG_TOKENS - 1,
            f"finish={reason} tokens={tokens} in {secs:.0f}s "
            f"({tokens / secs:.1f} tok/s)",
        )
    bad = [ln for ln in log.splitlines() if "Resource limit" in ln or "Traceback" in ln]
    check("server log has no resource-limit error / traceback", not bad, "; ".join(bad[:2]))
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
