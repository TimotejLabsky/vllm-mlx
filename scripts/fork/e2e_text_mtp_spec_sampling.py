#!/usr/bin/env python
"""End-to-end check of PATCHES.md #167 on a REAL server, off the live routes:
the text drafter with speculative sampling vs exact matching, at the 27B-8bit
route's own sampling defaults (T=1.0, top_p 0.95, top_k 20; the client sends
no temperature, as opencode and Hermes do).

Two servers in turn on a spare port, prod flags, drafter on:
- greedy: VLLM_MLX_TEXT_MTP_SPEC_SAMPLING unset (the #160 rule);
- ss: VLLM_MLX_TEXT_MTP_SPEC_SAMPLING=1.
Agent-shaped prompts (this repo's source as context, a question) at ~20K and
~40K tokens; each is prefilled once, then asked REPS more times (cache hits),
streamed, 400 tokens. Reports median decode tok/s per prompt and the
drafter's acceptance. PASS: no errors, ss rounds happened, ss acceptance
above greedy, ss median decode not below greedy.

    python scripts/fork/e2e_text_mtp_spec_sampling.py      (Studio, ~30 min)
"""

import glob
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.8-27B-8bit")
DRAFT = os.environ.get("E2E_DRAFT", "mlx-community/Qwen3.8-27B-MTP-8bit")
PORT = int(os.environ.get("E2E_PORT", "5998"))
BASE = f"http://127.0.0.1:{PORT}"
CTXS = [int(x) for x in os.environ.get("E2E_CTXS", "20000,40000").split(",")]
REPS = int(os.environ.get("E2E_REPS", "3"))
QUESTIONS = {
    "review": "Find where the batched system-KV cache decides whether to store a "
    "finished request and how it picks RAM vs SSD. Explain it step by step citing "
    "the functions, then propose a concrete patch that adds a counter for skipped "
    "stores.",
    "prose": "Write a ~600 word overview of this codebase's architecture for a new "
    "contributor: the request path, the engines, the caches and how they interact.",
}


def context(ctx_tokens: int) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    src = sorted(glob.glob(os.path.join(here, "..", "..", "vllm_mlx", "*.py")))
    parts, used = [], 0
    for path in src:
        text = open(path).read()
        n = len(text) // 3  # ~3 chars per token on Python source
        if used + n > ctx_tokens:
            continue
        parts.append(f"### File: vllm_mlx/{os.path.basename(path)}\n```python\n{text}\n```")
        used += n
    return "You are a coding agent in the vllm-mlx repository.\n\n" + "\n\n".join(parts)


def serve(ss: bool, log_path: str):
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1",
               VLLM_MLX_BATCHED_SYSTEM_KV="1", VLLM_MLX_SYSTEM_KV_RAM_MB="8192",
               VLLM_MLX_BATCHED_KV_BUDGET_MB="4096",
               VLLM_MLX_BATCHED_MEM_WATERMARK_PCT="80",
               VLLM_MLX_TEXT_MTP_DRAFTER=DRAFT)  # fmt: skip
    env.pop("VLLM_MLX_TEXT_MTP_SPEC_SAMPLING", None)
    if ss:
        env["VLLM_MLX_TEXT_MTP_SPEC_SAMPLING"] = "1"
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL,
        "--host", "127.0.0.1", "--port", str(PORT),
        "--continuous-batching", "--text-only", "--max-num-seqs", "4",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
        "--reasoning-parser", "qwen3", "--cache-memory-mb", "16384",
        "--timeout", "1200",
        "--default-chat-template-kwargs",
        '{"reasoning_effort":"medium","preserve_thinking":false}',
        "--default-temperature", "1.0", "--default-top-p", "0.95",
        "--default-top-k", "20", "--default-min-p", "0.0",
        "--default-thinking-token-budget", "6144",
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"),
                            stderr=subprocess.STDOUT)  # fmt: skip
    for _ in range(300):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early, see {log_path}")
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=2).read()
            return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not come up")


def ask(system: str, question: str) -> float:
    body = json.dumps({
        "model": MODEL, "stream": True, "max_tokens": 400,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": question}],
    }).encode()  # fmt: skip
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    first = last = None
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content") or delta.get(
                "reasoning"
            ):
                now = time.perf_counter()
                first = first or now
                last = now
                n += 1
    return (n - 1) / (last - first) if n > 1 and last > first else 0.0


def session(ss: bool, workdir: str) -> dict:
    tag = "ss" if ss else "greedy"
    log = os.path.join(workdir, f"server-{tag}.log")
    proc = serve(ss, log)
    out = {"log": log, "rates": {}}
    try:
        for ctx in CTXS:
            system = context(ctx)
            for name, q in QUESTIONS.items():
                ask(system, q)  # cold: prefill
                rates = [round(ask(system, q), 2) for _ in range(REPS)]
                out["rates"][f"{name}@{ctx}"] = rates
                print(f"{tag} {name}@{ctx}: {rates}", flush=True)
        st = json.load(urllib.request.urlopen(f"{BASE}/v1/status", timeout=10))
        out["mtp"] = st.get("mtp") or {}
    finally:
        proc.terminate()
        proc.wait(timeout=180)
    return out


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-ss-")
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    greedy = session(False, workdir)
    ss = session(True, workdir)
    for tag, r in (("greedy", greedy), ("ss", ss)):
        m = r["mtp"]
        bad = [ln for ln in open(r["log"]).read().splitlines() if "Traceback" in ln]
        check(f"{tag}: no drafter errors, no traceback",
              not m.get("errors") and not m.get("disabled") and not bad,
              f"errors={m.get('errors')} disabled={m.get('disabled_reason')}")  # fmt: skip
    check("ss: speculative-sampling rounds ran", ss["mtp"].get("ss_rounds", 0) > 100,
          f"ss_rounds={ss['mtp'].get('ss_rounds')}")  # fmt: skip
    ga, sa = greedy["mtp"].get("acceptance_rate", 0), ss["mtp"].get("ss_acceptance_rate", 0)
    check("ss acceptance above greedy", sa > ga, f"greedy {ga:.3f} -> ss {sa:.3f}")
    rows = []
    for key in greedy["rates"]:
        g, s = statistics.median(greedy["rates"][key]), statistics.median(ss["rates"][key])
        rows.append(s / g)
        print(f"  {key}: greedy {g:.1f} -> ss {s:.1f} tok/s ({(s / g - 1) * 100:+.0f}%)")
    gm = statistics.geometric_mean(rows)
    check("ss decode not below greedy (geomean over prompts)", gm >= 1.0,
          f"{(gm - 1) * 100:+.1f}%")  # fmt: skip
    print(f"logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
