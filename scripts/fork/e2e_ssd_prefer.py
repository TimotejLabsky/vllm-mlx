#!/usr/bin/env python
"""End-to-end check of PATCHES.md #117 on a REAL server and model — off the
live routes.

Two conversations share a long prefix (tool schemas / Assist context in the HA
voice prompts) and then diverge INSIDE the system message. On a ONE-slot route
storing chain A evicts chain B from RAM (its write-through spill is on SSD), so
B's follow-up turn finds only A in RAM: a divergent partial that snaps down to a
checkpoint far below the shared length. Before #117 that weak RAM match won and
the exact SSD snapshot was never consulted. Starts ``vllm_mlx.cli serve
--continuous-batching`` twice (arbitration on, then ``SSD_PREFER_GAIN=0``) and
asserts

  * B's follow-up is served from the SSD snapshot with far more cached tokens
    than the old path (``usage.prompt_tokens_details.cached_tokens``), the log
    says ``ssd preferred`` and ``ssd_preferred`` > 0 in /v1/status;
  * with GAIN=0 the old behaviour is intact (no SSD pick, fewer cached tokens);
  * chain A's cold prefill cut a checkpoint at the divergence
    (``divergence_cuts`` > 0), so a third chain C sharing only the prefix
    restores at the shared boundary, not at the uniform interval below it;
  * no traceback / stream-thread error in either server log.

    python scripts/fork/e2e_ssd_prefer.py
    E2E_MODEL=mlx-community/Qwen3.5-4B-4bit E2E_SERVE_ARGS=--text-only \\
        python scripts/fork/e2e_ssd_prefer.py

Use a hybrid (GDN/Mamba) model when one is cached: that is the production cache
topology. Exit code 0 only if all pass.
"""

import os
import subprocess
import sys
import tempfile
import time

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3-0.6B-8bit")
PORT = int(os.environ.get("E2E_PORT", "8767"))
BASE = f"http://127.0.0.1:{PORT}"

SHARED = "You are a home assistant. " + " ".join(
    f"Tool {i}: call it only when the request clearly needs it, pass every "
    f"required argument, and never invent entity names."
    for i in range(1, 90)
)


def _tail(word):
    return f"\n\n{word.upper()} PROFILE. " + " ".join(
        f"{word} item {i}: answer in {word} and keep it brief."
        for i in range(1, 60)
    )


SYSTEM = {name: SHARED + _tail(name) for name in ("alpha", "beta", "gamma")}
Q1 = "Turn on the kitchen light. One sentence."
Q2 = "Now turn it off again. One sentence."


def _body(messages, max_tokens=32):
    return {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _chat(client, messages):
    r = client.post(f"{BASE}/v1/chat/completions", json=_body(messages))
    r.raise_for_status()
    body = r.json()
    details = body.get("usage", {}).get("prompt_tokens_details") or {}
    return (
        body["choices"][0]["message"]["content"] or "",
        int(details.get("cached_tokens") or 0),
        int(body["usage"]["prompt_tokens"]),
    )


def _session(env_extra, label, workdir):
    import httpx

    env = dict(
        os.environ,
        VLLM_MLX_BATCHED_SYSTEM_KV="1",
        VLLM_MLX_SYSTEM_KV_SLOTS="1",
        VLLM_MLX_BATCHED_KV_CKPT_INTERVAL="256",
        VLLM_MLX_SSD_SYSTEM_KV_GB="4",
        VLLM_MLX_SSD_SYSTEM_KV_DIR=os.path.join(workdir, f"ssd-{label}"),
        **env_extra,
    )
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    log_path = os.path.join(workdir, f"server-{label}.log")
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
            sys_b = [{"role": "system", "content": SYSTEM["beta"]}]
            b1, _, _ = _chat(client, sys_b + [{"role": "user", "content": Q1}])
            time.sleep(4.0)  # let B's write-through spill land
            sys_a = [{"role": "system", "content": SYSTEM["alpha"]}]
            _chat(client, sys_a + [{"role": "user", "content": Q1}])  # evicts B
            time.sleep(4.0)
            _, b2_cached, b2_prompt = _chat(
                client,
                sys_b
                + [
                    {"role": "user", "content": Q1},
                    {"role": "assistant", "content": b1},
                    {"role": "user", "content": Q2},
                ],
            )
            time.sleep(4.0)
            sys_c = [{"role": "system", "content": SYSTEM["gamma"]}]
            _, c_cached, c_prompt = _chat(
                client, sys_c + [{"role": "user", "content": Q1}]
            )
            status = client.get(f"{BASE}/v1/status").json()
        return {
            "b2_cached": b2_cached,
            "b2_prompt": b2_prompt,
            "c_cached": c_cached,
            "c_prompt": c_prompt,
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


def _shared_tokens():
    """Token length of the prefix chains A/B/C really share (template
    included) — where the divergence cut must land."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)

    def ids(name):
        return tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM[name]},
                {"role": "user", "content": Q1},
            ],
            add_generation_prompt=True,
            tokenize=True,
        )

    a, c = ids("alpha"), ids("gamma")
    if hasattr(a, "input_ids"):  # BatchEncoding on newer transformers
        a, c = a["input_ids"], c["input_ids"]
    n = 0
    while n < min(len(a), len(c)) and a[n] == c[n]:
        n += 1
    return n


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-ssd-prefer-")
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))

    shared = _shared_tokens()
    print(f"INFO  shared prefix = {shared} tokens; workdir {workdir}")
    on = _session({}, "on", workdir)
    off = _session({"VLLM_MLX_SYSTEM_KV_SSD_PREFER_GAIN": "0"}, "off", workdir)
    on_cache = on["status"].get("cache", {})
    off_cache = off["status"].get("cache", {})
    for label, s in (("on", on), ("off", off)):
        picks = [ln.split("] ", 1)[-1] for ln in s["log"].splitlines() if "ssd preferred" in ln]
        restores = [ln.split("] ", 1)[-1] for ln in s["log"].splitlines() if "restore at" in ln]
        print(f"INFO  {label}: B2 cached {s['b2_cached']}/{s['b2_prompt']}, "
              f"C cached {s['c_cached']}/{s['c_prompt']}")
        print(f"INFO  {label} picks: {picks[:2]} restores: {restores[:4]}")  # fmt: skip

    check(
        "B's follow-up is served from the SSD snapshot (>= 90% of the prompt cached)",
        on["b2_cached"] >= 0.9 * on["b2_prompt"] and on_cache.get("ssd_promotes", 0) > 0,
        f"cached {on['b2_cached']}/{on['b2_prompt']} "
        f"ssd_promotes={on_cache.get('ssd_promotes')} "
        f"ssd_preferred={on_cache.get('ssd_preferred')}",
    )
    check(
        "the SSD pick was made and logged",
        on_cache.get("ssd_preferred", 0) > 0 and "ssd preferred" in on["log"],
    )
    check(
        "GAIN=0 keeps the old behaviour (no SSD pick, fewer cached tokens)",
        off_cache.get("ssd_preferred", 0) == 0
        and off["b2_cached"] + 512 < on["b2_cached"],
        f"off cached {off['b2_cached']} vs on {on['b2_cached']}",
    )
    check(
        "a divergent restore cut a checkpoint at the divergence",
        on_cache.get("divergence_cuts", 0) > 0,
        f"divergence_cuts={on_cache.get('divergence_cuts')}",
    )
    interval_floor = shared // 256 * 256
    # Asserted on the GAIN=0 run, where chain A (the one whose cold prefill saw
    # the divergence from B) is the RAM donor for C. In the arbitration run C's
    # donor is B's chain, stored cold BEFORE any divergence was seen: it has no
    # checkpoint at the boundary yet (it learns one from C's own restore).
    check(
        "chain C restores at the shared boundary, not the interval below it",
        off["c_cached"] > interval_floor and off["c_cached"] >= shared - 16,
        f"cached {off['c_cached']} (shared {shared}, interval floor {interval_floor})",
    )
    bad = [
        line
        for line in (on["log"] + off["log"]).splitlines()
        if "Traceback" in line
        or "no Stream(" in line
        or "generation_error_recovery" in line
        or "stream/thread mismatch" in line
    ]
    check("server logs have no traceback / stream-thread error", not bad, "; ".join(bad[:2]))
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
