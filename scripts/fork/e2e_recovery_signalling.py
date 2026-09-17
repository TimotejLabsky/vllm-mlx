#!/usr/bin/env python
"""End-to-end check of PATCHES.md #104 over real HTTP — off the production box.

Runs the REAL server (``vllm_mlx.cli serve --continuous-batching``) on a small
model with a one-shot fault injector around mlx-lm's ``BatchGenerator.next``:
while the trigger file exists, the next batch step raises the Metal OOM string,
so ``generation_error_recovery`` runs exactly as it does in production. Never
point this at a live route — manufacturing an OOM there aborts real requests.

    python scripts/fork/e2e_recovery_signalling.py            # run all cases
    python scripts/fork/e2e_recovery_signalling.py --serve    # (internal)

Exit code 0 only if every case passes.
"""

import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3-0.6B-8bit")
PORT = int(os.environ.get("E2E_PORT", "8765"))
TRIGGER = os.environ.get("E2E_TRIGGER", "")
BASE = f"http://127.0.0.1:{PORT}"
OOM = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory). [injected]"
)
# opencode session/retry.ts: status >= 500, or a message matching this.
AGENT_RETRY_RE = re.compile(
    r"429|500|502|503|504|524|overloaded|service unavailable|internal error", re.I
)


def serve() -> None:
    gen = importlib.import_module("mlx_lm.generate")
    original = gen.BatchGenerator.next

    def faulty_next(self, *args, **kwargs):
        if TRIGGER and os.path.exists(TRIGGER):
            os.remove(TRIGGER)  # one shot
            raise RuntimeError(OOM)
        return original(self, *args, **kwargs)

    gen.BatchGenerator.next = faulty_next
    from vllm_mlx.cli import main

    sys.argv = [
        "vllm-mlx", "serve", MODEL, "--host", "127.0.0.1", "--port", str(PORT),
        "--continuous-batching", "--max-num-seqs", "4", "--enable-metrics",
    ]  # fmt: skip
    main()


def _arm(trigger: str) -> None:
    open(trigger, "w").close()


def _status(client) -> dict:
    return client.get(f"{BASE}/v1/status").json()


def _body(stream: bool, max_tokens: int) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count from 1 to 40, one per line."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }


def run_cases(trigger: str) -> list[tuple[str, bool, str]]:
    import httpx

    results = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    with httpx.Client(timeout=120) as client:
        # 0. happy path — the patch must not change a healthy request
        r = client.post(f"{BASE}/v1/chat/completions", json=_body(False, 16))
        choice = r.json()["choices"][0]
        check(
            "healthy non-stream unchanged",
            r.status_code == 200
            and choice["message"]["content"]
            and choice["finish_reason"] in ("stop", "length"),
            f"status {r.status_code}, finish_reason {choice['finish_reason']!r}",
        )
        before = _status(client).get("generation_recoveries", "<missing>")

        # 1. non-stream: abort -> 503 + Retry-After (was 200 + content:null)
        _arm(trigger)
        r = client.post(f"{BASE}/v1/chat/completions", json=_body(False, 64))
        detail = r.json().get("detail", {}) if r.status_code != 200 else {}
        check(
            "non-stream abort is HTTP 503 + Retry-After",
            r.status_code == 503
            and r.headers.get("retry-after") == "15"
            and detail.get("error") == "generation_aborted"
            and detail.get("kind") == "oom",
            f"status {r.status_code}, retry-after {r.headers.get('retry-after')!r}, "
            f"detail {detail}",
        )

        # 2/3. streaming: before first token, and mid-stream
        for name, arm_after in (("before first token", 0), ("mid-stream", 5)):
            if arm_after == 0:
                _arm(trigger)
            frames, content, finishes, done = [], 0, [], False
            with client.stream(
                "POST", f"{BASE}/v1/chat/completions", json=_body(True, 200)
            ) as resp:
                status = resp.status_code
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        done = True
                        continue
                    obj = json.loads(data)
                    if "error" in obj:
                        frames.append(obj["error"])
                        continue
                    for ch in obj.get("choices", []):
                        if ch.get("delta", {}).get("content"):
                            content += 1
                            if arm_after and content == arm_after:
                                _arm(trigger)
                        if ch.get("finish_reason"):
                            finishes.append(ch["finish_reason"])
            ok = (
                status == 200
                and len(frames) == 1
                and frames[0].get("code") == 503
                and frames[0].get("kind") == "oom"
                and AGENT_RETRY_RE.search(frames[0].get("message", ""))
                and done
                and finishes == []
                and (content == 0 if arm_after == 0 else content >= arm_after)
            )
            check(
                f"stream abort {name} is one 503-shaped frame",
                ok,
                f"content chunks {content}, frames {frames}, "
                f"finish_reasons {finishes}, [DONE] {done}",
            )

        # 4. counted, and the engine keeps serving afterwards
        after = _status(client)
        check(
            "recoveries are counted in /v1/status",
            isinstance(before, int)
            and after.get("generation_recoveries") == before + 3
            and after.get("recovery_aborted_requests", 0) >= 3,
            f"generation_recoveries {before} -> {after.get('generation_recoveries')}, "
            f"recovery_aborted_requests {after.get('recovery_aborted_requests')}",
        )
        r = client.post(f"{BASE}/v1/chat/completions", json=_body(False, 16))
        check(
            "engine serves normally after three recoveries",
            r.status_code == 200 and r.json()["choices"][0]["message"]["content"],
            f"status {r.status_code}",
        )
        metrics = client.get(f"{BASE}/metrics")
        if metrics.status_code == 200:
            lines = [
                ln for ln in metrics.text.splitlines()
                if ln.startswith("vllm_mlx_stream_aborts_total{")
                and 'result="error"' in ln
            ]  # fmt: skip
            total = sum(float(ln.rsplit(" ", 1)[1]) for ln in lines)
            check(
                "stream_aborts_total counts both aborted streams",
                total == 2,
                f"{lines}",
            )
    return results


def main() -> int:
    if "--serve" in sys.argv:
        serve()
        return 0
    import httpx

    workdir = tempfile.mkdtemp(prefix="e2e-recovery-")
    trigger = os.path.join(workdir, "inject-oom")
    log_path = os.path.join(workdir, "server.log")
    env = dict(os.environ, E2E_TRIGGER=trigger, E2E_PORT=str(PORT), E2E_MODEL=MODEL)
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--serve"],
            stdout=log, stderr=subprocess.STDOUT, env=env,
        )  # fmt: skip
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            if server.poll() is not None:
                print(open(log_path).read()[-3000:])
                return 2
            try:
                if httpx.get(f"{BASE}/v1/status", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        results = run_cases(trigger)
        log_text = open(log_path).read()
        print(
            "server log: generation_error_recovery x"
            f"{log_text.count('[generation_error_recovery]')}, "
            f"'aborted by the engine' x{log_text.count('aborted by the engine')}"
        )
        return 0 if results and all(ok for _, ok, _ in results) else 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    sys.exit(main())
