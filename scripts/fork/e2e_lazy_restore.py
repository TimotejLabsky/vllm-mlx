#!/usr/bin/env python
"""End-to-end check of PATCHES.md #106 on a REAL server and model — off the
live routes.

Lazy restore moves the building (and ``mx.eval``) of a restored cache from the
event-loop thread at ``add_request`` to the executor thread at admission. MLX
stream/thread mismatches have bitten this stack before (#28), so unit tests on
fake caches are not enough: this starts ``vllm_mlx.cli serve
--continuous-batching`` twice on the same model and asserts

  * T=0 output with LAZY_RESTORE=1 is byte-identical to the eager server's,
    on the first pass and on the all-restores second pass;
  * the warm turns really were lazy restores (``lazy_restores`` > 0, hits > 0);
  * a concurrent burst sharing one prefix (co-batching + queued restores +
    the projected-admission gate armed) completes, identically, with no
    engine recovery and no traceback in the server log.

    python scripts/fork/e2e_lazy_restore.py
    E2E_MODEL=mlx-community/Qwen3.5-4B-4bit E2E_SERVE_ARGS=--text-only \\
        python scripts/fork/e2e_lazy_restore.py

Use a hybrid (GDN/Mamba) model when one is cached: that is the production
cache topology (ArraysCache + checkpoints). Exit code 0 only if all pass.
"""

import concurrent.futures
import os
import subprocess
import sys
import tempfile
import time

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3-0.6B-8bit")
PORT = int(os.environ.get("E2E_PORT", "8766"))
BASE = f"http://127.0.0.1:{PORT}"

# > the 256-token partial-restore floor, so follow-ups can restore from it
SYSTEM = "You are a meticulous code reviewer. " + " ".join(
    f"Rule {i}: keep every finding specific, cite the line, and never invent "
    f"behaviour that the diff does not show."
    for i in range(1, 41)
)
TURNS = [
    "Review this change: `def add(a, b): return a - b`. One sentence.",
    "Now review: `for i in range(len(xs)): print(xs[i+1])`. One sentence.",
    "And this: `open(path).read()` with no close. One sentence.",
]


def _body(messages, max_tokens=48):
    return {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _conversation(client):
    """A growing multi-turn chat: each turn's prompt extends the previous one,
    which is exactly what restores from the cache."""
    messages = [{"role": "system", "content": SYSTEM}]
    outputs = []
    for turn in TURNS:
        messages.append({"role": "user", "content": turn})
        r = client.post(f"{BASE}/v1/chat/completions", json=_body(messages))
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        outputs.append(text)
        messages.append({"role": "assistant", "content": text})
    return outputs


def _interleaved(client):
    """Two conversations taking turns on a route with ONE cache slot: each
    turn evicts the other chain from RAM, so every second turn has to come
    back from the SSD tier (#36 promote, and #107's lazy fallback path)."""
    chats = {
        # diverge from the FIRST token: a shared system prompt would give a
        # shallow RAM hit on the other chain and nothing would touch the disk
        name: [{"role": "system", "content": f"Project {name} ({name * 3}). " + SYSTEM}]
        for name in ("alpha", "beta")
    }
    outputs = []
    for turn in TURNS[:2]:
        for name in chats:
            chats[name].append({"role": "user", "content": turn})
            r = client.post(f"{BASE}/v1/chat/completions", json=_body(chats[name]))
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            outputs.append(text)
            chats[name].append({"role": "assistant", "content": text})
            time.sleep(2.0)  # let the write-through spill land
    return outputs


def _burst(client, n=4):
    def one(i):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TURNS[i % len(TURNS)]},
        ]
        r = client.post(f"{BASE}/v1/chat/completions", json=_body(messages))
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(one, range(n)))


def _run_server(env_extra, log_path):
    env = dict(os.environ, VLLM_MLX_BATCHED_SYSTEM_KV="1", **env_extra)
    env.setdefault("VLLM_MLX_SSD_SYSTEM_KV_GB", "4")
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
    return proc, log


def _session(env_extra, label, workdir):
    import httpx

    log_path = os.path.join(workdir, f"server-{label}.log")
    proc, log = _run_server(env_extra, log_path)
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
            cold = _conversation(client)
            warm = _conversation(client)  # same prompts again: pure restores
            burst = _burst(client)
            burst_again = _burst(client)
            interleaved = _interleaved(client) if env_extra.get("_SSD") else []
            status = client.get(f"{BASE}/v1/status").json()
        return {
            "cold": cold,
            "warm": warm,
            "burst": burst,
            "burst_again": burst_again,
            "interleaved": interleaved,
            "status": status,
            "log": open(log_path).read(),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-lazy-")
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))

    eager = _session({}, "eager", workdir)
    lazy = _session(
        {
            "VLLM_MLX_BATCHED_LAZY_RESTORE": "1",
            "VLLM_MLX_BATCHED_PROJECTED_ADMISSION": "1",
            "VLLM_MLX_BATCHED_SOLO_TRANSIENT_MB": "512",
            "VLLM_MLX_BATCHED_BPT_FLOOR_KB": "64",
        },
        "lazy",
        workdir,
    )

    for label, session in (("eager", eager), ("lazy", lazy)):
        for i, (c, w) in enumerate(zip(session["cold"], session["warm"])):
            if c != w:
                k = next(
                    (j for j, (a, b) in enumerate(zip(c, w)) if a != b),
                    min(len(c), len(w)),
                )
                print(
                    f"DIFF  {label} turn {i + 1}: first differing char {k} of {len(c)}/{len(w)}"
                )
                print(f"      cold: ...{c[max(0, k - 40):k + 60]!r}")
                print(f"      warm: ...{w[max(0, k - 40):k + 60]!r}")
                break
        restores = [
            ln.split("] ", 1)[-1]
            for ln in session["log"].splitlines()
            if "restore at" in ln
        ]
        print(f"INFO  {label} restores: {restores[:4]}")

    ssd_env = {"_SSD": "1", "VLLM_MLX_SYSTEM_KV_SLOTS": "1"}
    eager_ssd = _session(
        {**ssd_env, "VLLM_MLX_SSD_SYSTEM_KV_DIR": os.path.join(workdir, "ssd-eager")},
        "eager-ssd", workdir,
    )  # fmt: skip
    lazy_ssd = _session(
        {
            **ssd_env,
            "VLLM_MLX_SSD_SYSTEM_KV_DIR": os.path.join(workdir, "ssd-lazy"),
            "VLLM_MLX_BATCHED_LAZY_RESTORE": "1",
            "VLLM_MLX_BATCHED_PROJECTED_ADMISSION": "1",
            "VLLM_MLX_BATCHED_SOLO_TRANSIENT_MB": "512",
            "VLLM_MLX_BATCHED_BPT_FLOOR_KB": "64",
        },
        "lazy-ssd", workdir,
    )  # fmt: skip
    ssd_cache = lazy_ssd["status"].get("cache", {})

    cache = lazy["status"].get("cache", {})
    # Informational, NOT a gate for #106: a warm turn restores at a different
    # position than the first pass did, so the tail is prefilled in different
    # chunks - same maths, ~1e-6 logit drift, and on a small 4-bit hybrid an
    # argmax tie can flip a few dozen tokens in (seen 2026-09-21 on
    # Qwen3.5-4B-4bit turn 3, IDENTICALLY on the unpatched eager server).
    # The property this patch must keep is lazy == eager, checked next.
    same = eager["warm"] == eager["cold"] and lazy["warm"] == lazy["cold"]
    print(f"{'INFO' if same else 'NOTE'}  warm == cold on both servers: {same}")
    check(
        "lazy output is byte-identical to the eager server's",
        lazy["cold"] == eager["cold"] and lazy["warm"] == eager["warm"],
        repr(lazy["cold"][0][:60]),
    )
    check(
        "concurrent burst identical across servers and across repeats",
        lazy["burst"] == eager["burst"] and lazy["burst_again"] == lazy["burst"],
        f"{len(lazy['burst'])} concurrent rows",
    )
    check(
        "the warm turns really were lazy restores",
        cache.get("lazy_restores", 0) > 0 and cache.get("hits", 0) > 0,
        f"lazy_restores={cache.get('lazy_restores')} hits={cache.get('hits')} "
        f"misses={cache.get('misses')} lazy_restore_misses="
        f"{cache.get('lazy_restore_misses')}",
    )
    check(
        "no engine recovery, no rejection",
        lazy["status"].get("generation_recoveries", 0) == 0
        and cache.get("solo_rejections", 0) == 0,
        f"projected_defers={cache.get('projected_defers')} "
        f"projected_relief_passes={cache.get('projected_relief_passes')}",
    )
    check(
        "one-slot route: turns served from the SSD tier, lazy == eager",
        lazy_ssd["interleaved"] == eager_ssd["interleaved"]
        and ssd_cache.get("ssd_promotes", 0) > 0,
        f"ssd_promotes={ssd_cache.get('ssd_promotes')} "
        f"lazy_restores={ssd_cache.get('lazy_restores')} "
        f"lazy_ssd_fallbacks={ssd_cache.get('lazy_ssd_fallbacks')} "
        f"pinned_entries={ssd_cache.get('pinned_entries')}",
    )
    check(
        "no pin is left behind once the route is idle",
        cache.get("pinned_entries", 0) == 0 and ssd_cache.get("pinned_entries", 0) == 0,
    )
    lazy["log"] += lazy_ssd["log"]
    bad = [
        line
        for line in lazy["log"].splitlines()
        if "Traceback" in line
        or "no Stream(" in line
        or "generation_error_recovery" in line
        or "stream/thread mismatch" in line
    ]
    check(
        "server log has no traceback / stream-thread error", not bad, "; ".join(bad[:2])
    )
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
