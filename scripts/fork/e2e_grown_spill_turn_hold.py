#!/usr/bin/env python
"""End-to-end check of PATCHES.md #164 and #165 on a REAL server and model,
off the live routes. Each scenario runs with the feature on and off.

#164 (grown chains spill): four turns of one conversation (turns 2-4 are
grown stores), the server restarts over the same SSD directory, turn 5 must
restore from a chain past turn 1. Off, only turn 1's chain is on disk.

#165 (turn hold): a background request is queued while a conversation turn
runs (a 1 MB co-batch threshold keeps them apart, as padding does at depth on
the 27B); the conversation's next turn arrives 1 s after that turn finishes,
like an agent's tool call. On, the next turn is admitted before the
background request; off, after it.

    E2E_MODEL=mlx-community/Qwen3.5-4B-4bit python scripts/fork/e2e_grown_spill_turn_hold.py

Run it from the source tree under test (it puts the cwd first on PYTHONPATH).
Exit code 0 only if all pass.
"""

import os
import re
import subprocess
import sys
import tempfile
import threading
import time

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.5-4B-4bit")
PORT = int(os.environ.get("E2E_PORT", "8772"))
BASE = f"http://127.0.0.1:{PORT}"
SYSTEM = "You are a careful assistant working through a long task. " * 30
FILLER = "Observation: the build log shows lines like 'step ok' and 'step ok'. " * 25
BACKGROUND = "Write a short summary of: " + "The fox jumps over the dog. " * 60


def serve(workdir, tag, env_extra):
    import httpx

    log_path = os.path.join(workdir, f"server-{tag}.log")
    env = dict(os.environ, VLLM_MLX_BATCHED_SYSTEM_KV="1", HF_HUB_OFFLINE="1")
    for key in list(env):
        if key.startswith(("VLLM_MLX_SSD_", "VLLM_MLX_BATCHED_TURN_HOLD")):
            env.pop(key)
    env.update(env_extra)
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL,
            "--host", "127.0.0.1", "--port", str(PORT),
            "--continuous-batching", "--text-only", "--max-num-seqs", "2",
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


def stop(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


def ask(messages, max_tokens=80):
    import httpx

    r = httpx.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=600,
    )
    r.raise_for_status()
    body = r.json()
    usage = body["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return body["choices"][0]["message"].get("content") or "", usage, cached


def grown_spill(workdir, on):
    tag = "spill-on" if on else "spill-off"
    ssd = os.path.join(workdir, f"ssd-{tag}")
    env = {"VLLM_MLX_SSD_SYSTEM_KV_DIR": ssd, "VLLM_MLX_SSD_SYSTEM_KV_GB": "4"}
    if on:
        env["VLLM_MLX_SSD_SPILL_GROWN_DELTA"] = "300"
    msgs = [{"role": "system", "content": SYSTEM}]
    chains = []
    proc, log, log_a = serve(workdir, tag + "-a", env)
    try:
        for n in range(4):
            msgs.append(
                {"role": "user", "content": f"Step {n}. {FILLER} Reply briefly."}
            )
            text, usage, _ = ask(msgs)
            msgs.append({"role": "assistant", "content": text})
            chains.append(usage["prompt_tokens"] + usage["completion_tokens"])
            time.sleep(4)  # idle window: the SSD writer lands its spills
    finally:
        stop(proc, log)
    proc, log, log_b = serve(workdir, tag + "-b", env)
    try:
        msgs.append({"role": "user", "content": "Step 4. Reply briefly."})
        _, usage, cached = ask(msgs)
    finally:
        stop(proc, log)
    grown = len(re.findall(r"spilled \d+-token snapshot", open(log_a).read()))
    return {"chains": chains, "cached": cached, "prompt": usage["prompt_tokens"],
            "spills": grown}  # fmt: skip


def turn_hold(workdir, on):
    tag = "hold-on" if on else "hold-off"
    env = {"VLLM_MLX_BATCHED_PAD_WASTE_MB": "1"}
    if on:
        env.update(
            VLLM_MLX_BATCHED_TURN_HOLD_S="6",
            VLLM_MLX_BATCHED_TURN_HOLD_MIN_TOKENS="500",
        )
    proc, log, log_path = serve(workdir, tag, env)
    try:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": FILLER + " Describe the log in detail."},
        ]
        ask(msgs, max_tokens=10)  # warm the chain
        msgs.append({"role": "assistant", "content": "Noted."})
        msgs.append({"role": "user", "content": FILLER + " Now write 150 words on it."})
        done = {}
        turn = threading.Thread(target=lambda: done.update(t=ask(msgs, 150)))
        turn.start()
        time.sleep(1.0)
        bg = threading.Thread(
            target=lambda: done.update(bg=(ask([{"role": "user", "content": BACKGROUND}], 120), time.time()))
        )  # fmt: skip
        bg.start()
        turn.join()
        time.sleep(1.0)  # the agent's tool call
        msgs.append({"role": "assistant", "content": done["t"][0]})
        msgs.append({"role": "user", "content": "Thanks, one more sentence."})
        t0 = time.time()
        ask(msgs, 30)
        next_turn_done = time.time()
        bg.join()
        next_first = next_turn_done < done["bg"][1]
        wait = round(next_turn_done - t0, 1)
    finally:
        stop(proc, log)
    held = "turn hold: admitting" in open(log_path).read()
    return {"next_turn_first": next_first, "next_turn_s": wait, "held": held}


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-grown-spill-turn-hold-")
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    on, off = grown_spill(workdir, True), grown_spill(workdir, False)
    check(
        "#164 on: after the restart turn 5 restores past turn 1's chain",
        on["cached"] > on["chains"][0] + 300,
        f"cached {on['cached']} of {on['prompt']}; chains {on['chains']}; "
        f"spills written {on['spills']}",
    )
    check(
        "#164 off (control): only turn 1's chain survives the restart",
        off["cached"] <= off["chains"][0],
        f"cached {off['cached']}; chains {off['chains']}; spills {off['spills']}",
    )
    h_on, h_off = turn_hold(workdir, True), turn_hold(workdir, False)
    check(
        "#165 on: the next turn ran before the background request",
        h_on["next_turn_first"] and h_on["held"],
        f"next turn took {h_on['next_turn_s']} s",
    )
    check(
        "#165 off (control): the background request ran first",
        not h_off["next_turn_first"],
        f"next turn took {h_off['next_turn_s']} s",
    )
    bad = []
    for name in os.listdir(workdir):
        if name.endswith(".log"):
            bad += [ln for ln in open(os.path.join(workdir, name)) if "Traceback" in ln]
    check("server logs have no traceback", not bad, "; ".join(bad[:2]))
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
