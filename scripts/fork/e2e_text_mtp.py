#!/usr/bin/env python
"""End-to-end check of PATCHES.md #160 on a REAL text server, off the live
routes: the MTP drafter on the cached text path changes speed, not answers,
and what it stores (RAM and SSD tier) is what plain decoding stores.

Three server runs on a spare port, each driven by ``t0_warm_cold.py`` (cold,
byte-identical re-send, follow-up turn, all at T=0):

  1. no drafter, its own SSD dir
  2. drafter, a fresh SSD dir
  3. drafter again on run 2's SSD dir: the first request must come back from
     the SSD tier, written by a speculative run, with the same answer

    E2E_MODEL=mlx-community/Qwen3.8-27B-8bit \\
    E2E_DRAFT=mlx-community/Qwen3.8-27B-MTP-8bit python scripts/fork/e2e_text_mtp.py

Exit code 0 only if all pass.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.8-27B-8bit")
DRAFT = os.environ.get("E2E_DRAFT", "mlx-community/Qwen3.8-27B-MTP-8bit")
PORT = int(os.environ.get("E2E_PORT", "5998"))
BASE = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))


def _serve(env_extra, log_path):
    env = dict(os.environ)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "VLLM_MLX_BATCHED_SYSTEM_KV": "1",
            "VLLM_MLX_SYSTEM_KV_RAM_MB": "6144",
            "VLLM_MLX_SSD_SYSTEM_KV_GB": "4",
        }
    )
    env.update(env_extra)
    cmd = [
        sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL,
        "--host", "127.0.0.1", "--port", str(PORT),
        "--continuous-batching", "--text-only", "--max-num-seqs", "4",
        "--reasoning-parser", "qwen3", "--timeout", "900",
        "--default-thinking-token-budget", "6144",
    ]  # fmt: skip
    proc = subprocess.Popen(
        cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT
    )
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early, see {log_path}")
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=2).read()
            return proc
        except Exception:
            time.sleep(2)
    proc.kill()
    raise RuntimeError("server did not become healthy")


def _session(name, env_extra, workdir):
    log_path = os.path.join(workdir, f"{name}.log")
    proc = _serve(env_extra, log_path)
    try:
        run = subprocess.run(
            [sys.executable, os.path.join(HERE, "t0_warm_cold.py")],
            env={**os.environ, "SERVER_URL": BASE},
            capture_output=True,
            text=True,
        )
        status = json.load(urllib.request.urlopen(f"{BASE}/v1/status", timeout=10))
    finally:
        proc.terminate()
        proc.wait(timeout=120)
    out = run.stdout
    print(f"--- {name}\n{out.strip()}")
    shas = dict(re.findall(r"(\w+)_sha=(\w+)", out))
    cached = dict(re.findall(r"(cold|warm|follow)=(\d+)", out.split("cached:")[-1]))
    return {
        "ok": run.returncode == 0,
        "shas": shas,
        "cached": {k: int(v) for k, v in cached.items()},
        "mtp": status.get("mtp") or {},
        "log": log_path,
    }


def main():
    workdir = tempfile.mkdtemp(prefix="e2e-text-mtp-")
    ssd_plain = os.path.join(workdir, "ssd-plain")
    ssd_mtp = os.path.join(workdir, "ssd-mtp")
    mtp_env = {"VLLM_MLX_TEXT_MTP_DRAFTER": DRAFT}

    plain = _session("plain", {"VLLM_MLX_SSD_SYSTEM_KV_DIR": ssd_plain}, workdir)
    first = _session(
        "drafter", {**mtp_env, "VLLM_MLX_SSD_SYSTEM_KV_DIR": ssd_mtp}, workdir
    )
    again = _session(
        "drafter-restart", {**mtp_env, "VLLM_MLX_SSD_SYSTEM_KV_DIR": ssd_mtp}, workdir
    )

    checks = {
        "plain run passes on its own": plain["ok"],
        "drafter run passes on its own": first["ok"],
        "drafter answers equal plain answers (cold, warm, follow-up)": (
            first["shas"] == plain["shas"] and len(plain["shas"]) == 3
        ),
        "drafter ran: drafts accepted, no errors, not disabled": (
            first["mtp"].get("accepted", 0) > 0
            and first["mtp"].get("errors") == 0
            and first["mtp"].get("disabled") is False
        ),
        "plain run has no text-path drafter": (
            plain["mtp"].get("implementation") != "text_external_assistant"
        ),
        "after restart the first request is served from the SSD tier": (
            again["cached"].get("cold", 0) > 0
        ),
        "after restart the answers still equal plain": again["shas"] == plain["shas"],
    }
    print()
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nlogs: {workdir}")
    print(f"drafter stats: {json.dumps(first['mtp'])}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
