#!/usr/bin/env python
"""End-to-end check of PATCHES.md #131 on a REAL server and model - off the
live routes.

Small env budgets make the co-batch gates bite at small sizes, so the
queueing shapes of fork issues #44 / #46 reproduce in seconds:

  head-of-line   a short request R is decoding; a deep request D arrives and
                 the KV budget defers it; a short S arrives behind D. FCFS: S
                 waits for D. SJF: S is admitted past D, D still runs.
  re-match       a leader L prefills a shared prefix; followers F1/F2 with the
                 same prefix arrive while it runs and wait behind the budget.
                 They missed at enqueue; at admission they must restore from
                 L's entry instead of prefilling cold.

    E2E_MODEL=mlx-community/Qwen3.5-4B-4bit python scripts/fork/e2e_admission.py

Exit code 0 only if all pass.
"""

import concurrent.futures
import os
import subprocess
import sys
import tempfile
import time

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.5-4B-4bit")
PORT = int(os.environ.get("E2E_PORT", "8772"))
BASE = f"http://127.0.0.1:{PORT}"
GATES = {
    "VLLM_MLX_BATCHED_SYSTEM_KV": "1",
    "VLLM_MLX_BATCHED_BPT_FLOOR_KB": "64",  # price tokens at 64 KB
    "VLLM_MLX_BATCHED_KV_BUDGET_MB": "600",  # ~2 seats at 4.5K tokens
}


def words(seed, n):
    import random

    r = random.Random(seed)
    vocab = "alpha beta gamma delta kappa sigma omega lambda theta zeta".split()
    return " ".join(r.choice(vocab) for _ in range(n))


def serve(order, workdir):
    import httpx

    env = dict(os.environ, **GATES, VLLM_MLX_BATCHED_ADMISSION_ORDER=order)
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    log = open(os.path.join(workdir, f"server-{order}.log"), "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL,
            "--host", "127.0.0.1", "--port", str(PORT),
            "--continuous-batching", "--text-only", "--max-num-seqs", "4",
        ],
        stdout=log, stderr=subprocess.STDOUT, env=env,
    )  # fmt: skip
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            if httpx.get(f"{BASE}/v1/status", timeout=2).status_code == 200:
                return proc, log
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not start")


def ask(text, max_tokens, t0):
    """Stream one request; return (seconds to first token, seconds to done,
    cached_tokens)."""
    import json

    import httpx

    first = None
    cached = None
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with httpx.stream("POST", f"{BASE}/v1/chat/completions", json=body, timeout=600) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            j = json.loads(line[6:])
            if first is None and j.get("choices") and j["choices"][0]["delta"].get("content"):
                first = time.time() - t0
            if j.get("usage"):
                cached = (j["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
    return first, time.time() - t0, cached


def head_of_line():
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        r = pool.submit(ask, "Count from 1 to 400. " + words(1, 300), 500, t0)
        time.sleep(3)
        d = pool.submit(ask, "Summarise: " + words(2, 5200), 16, t0)
        time.sleep(1)
        s = pool.submit(ask, "Say hello. " + words(3, 200), 16, t0)
        return r.result(), d.result(), s.result()


def rematch():
    t0 = time.time()
    shared = "Reference document:\n" + words(4, 3000)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        lead = pool.submit(ask, shared + "\nQuestion 0: first word?", 300, t0)
        time.sleep(1.5)
        f1 = pool.submit(ask, shared + "\nQuestion 1: last word?", 8, t0)
        f2 = pool.submit(ask, shared + "\nQuestion 2: count alpha?", 8, t0)
        return lead.result(), f1.result(), f2.result()


def main() -> int:
    import httpx

    workdir = tempfile.mkdtemp(prefix="e2e-admission-")
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    out = {}
    for order in ("fcfs", "sjf"):
        proc, log = serve(order, workdir)
        try:
            hol = head_of_line()
            rm = rematch()
            cache = httpx.get(f"{BASE}/v1/status").json().get("cache", {})
        finally:
            proc.terminate()
            proc.wait(timeout=60)
            log.close()
        out[order] = (hol, rm, cache)
        (_, _, rr), (d_first, d_done, _), (s_first, s_done, _) = hol
        print(
            f"INFO  {order}: short-behind-deep first token {s_first:.1f}s, "
            f"deep first token {d_first:.1f}s, deep done {d_done:.1f}s; "
            f"rematches={cache.get('admission_rematches')} "
            f"reorders={cache.get('sjf_reorders')} deferrals={cache.get('admission_deferrals')}",
            flush=True,
        )

    (_, f_deep, f_short), f_rm, f_cache = out["fcfs"]
    (_, s_deep, s_short), s_rm, s_cache = out["sjf"]
    check(
        "FCFS: the short request waited behind the deferred deep one",
        f_short[0] > f_deep[0],
        f"short {f_short[0]:.1f}s vs deep {f_deep[0]:.1f}s",
    )
    check(
        "SJF: the short request was admitted past the deferred deep one",
        s_short[0] < s_deep[0] and s_cache.get("sjf_reorders", 0) >= 1,
        f"short {s_short[0]:.1f}s vs deep {s_deep[0]:.1f}s, "
        f"reorders={s_cache.get('sjf_reorders')}",
    )
    check(
        "SJF: the deep request still completed (no starvation)",
        s_deep[1] is not None and s_deep[0] is not None,
        f"deep done at {s_deep[1]:.1f}s (FCFS {f_deep[1]:.1f}s)",
    )
    for order, (lead, f1, f2), cache in (("fcfs", f_rm, f_cache), ("sjf", s_rm, s_cache)):
        check(
            f"{order}: deferred followers restored the leader's prefix at admission",
            (f1[2] or 0) > 1000 and (f2[2] or 0) > 1000
            and cache.get("admission_rematches", 0) >= 1,
            f"cached {f1[2]}, {f2[2]}; rematches={cache.get('admission_rematches')}",
        )
    bad = []
    for order in ("fcfs", "sjf"):
        text = open(os.path.join(workdir, f"server-{order}.log")).read()
        bad += [ln for ln in text.splitlines() if "Traceback" in ln]
    check("server logs have no traceback", not bad, "; ".join(bad[:2]))
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
