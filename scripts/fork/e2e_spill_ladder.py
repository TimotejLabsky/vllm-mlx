#!/usr/bin/env python
"""End-to-end check of PATCHES.md #140 on a REAL server and model — off the
live routes.

#140 leaves two kinds of checkpoint out of SSD spills: one at the entry's own
length (never selected by a restore) and every checkpoint within
``VLLM_MLX_SSD_SYSTEM_KV_CKPT_MIN_GAP`` tokens of the previous kept one (the
end-of-prompt cluster). Per arm (gap 16, then gap 0 = whole cluster kept) this
starts ``vllm_mlx.cli serve --continuous-batching``, plays a two-turn chat plus
an exact re-send, stops the server (close drains the spills), reads the
on-disk ladders, then starts a SECOND server on the same SSD dir and replays
the re-send and a third turn from a cold RAM bag. Asserts

  * gap 16: no spilled checkpoint at/after its entry's length, none closer
    than 16 tokens to the previous one, ``ckpt_pruned`` > 0;
  * gap 16 writes fewer checkpoint bytes than gap 0;
  * after the restart the re-send and turn 3 restore within 16 tokens of
    where gap 0 restores them (turn 2 is a grown entry and does not re-spill,
    so both restore from turn 1's chain), and turn 3 restores >= 90 %;
  * no traceback / stream-thread error in any server log.

T=0 output identity across arms is reported, not asserted: a restore point a
few tokens earlier changes the prefill chunking (the known argmax-tie class).

    python scripts/fork/e2e_spill_ladder.py
    E2E_MODEL=mlx-community/Qwen3.5-4B-4bit E2E_SERVE_ARGS=--text-only \\
        python scripts/fork/e2e_spill_ladder.py

Use a hybrid (GDN/Mamba) model: a pure-attention model has no checkpoints and
the check is vacuous. Exit code 0 only if all pass.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.5-4B-4bit")
PORT = int(os.environ.get("E2E_PORT", "8768"))
BASE = f"http://127.0.0.1:{PORT}"

SYSTEM = "You are a careful assistant for a home lab. " + " ".join(
    f"Rule {i}: answer briefly, name the exact command, and never guess a "
    f"hostname that the user did not give."
    for i in range(1, 50)
)
Q = [
    "How do I list listening TCP ports on macOS? One sentence.",
    "And how do I see which process owns one of them? One sentence.",
    "How would I stop that process gracefully? One sentence.",
]


def _body(messages, max_tokens=40):
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


def _find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


class _Server:
    def __init__(self, ssd_dir, gap, log_path):
        env = dict(
            os.environ,
            VLLM_MLX_BATCHED_SYSTEM_KV="1",
            VLLM_MLX_SSD_SYSTEM_KV_GB="8",
            VLLM_MLX_SSD_SYSTEM_KV_DIR=ssd_dir,
            VLLM_MLX_SSD_SYSTEM_KV_CKPT_MIN_GAP=str(gap),
        )
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        self.log_path = log_path
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(
            [
                sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL,
                "--host", "127.0.0.1", "--port", str(PORT),
                "--continuous-batching", "--max-num-seqs", "4",
                *os.environ.get("E2E_SERVE_ARGS", "").split(),
            ],
            stdout=self.log, stderr=subprocess.STDOUT, env=env,
        )  # fmt: skip

    def __enter__(self):
        import httpx

        deadline = time.time() + 240
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(open(self.log_path).read()[-2000:])
            try:
                if httpx.get(f"{BASE}/v1/status", timeout=2).status_code == 200:
                    return self
            except Exception:
                time.sleep(1)
        raise RuntimeError("server did not come up")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def _disk_ladders(ssd_dir):
    """[(num_tokens, [ckpt pos]), ...] and total snapshot bytes on disk."""
    out, total = [], 0
    for root, _dirs, files in os.walk(ssd_dir):
        if "meta.json" in files and not root.endswith(".tmp"):
            with open(os.path.join(root, "meta.json")) as f:
                meta = json.load(f)
            out.append(
                (meta["num_tokens"], [cp["pos"] for cp in meta.get("checkpoints", [])])
            )
            total += os.path.getsize(os.path.join(root, "snapshot.safetensors"))
    return sorted(out), total


def _arm(gap, workdir):
    import httpx

    ssd_dir = os.path.join(workdir, f"ssd-gap{gap}")
    sys_msg = [{"role": "system", "content": SYSTEM}]
    res = {}
    with _Server(ssd_dir, gap, os.path.join(workdir, f"server-gap{gap}-a.log")):
        with httpx.Client(timeout=300) as client:
            t1 = sys_msg + [{"role": "user", "content": Q[0]}]
            a1, _, _ = _chat(client, t1)
            time.sleep(4.0)
            t2 = t1 + [
                {"role": "assistant", "content": a1},
                {"role": "user", "content": Q[1]},
            ]
            a2, _, _ = _chat(client, t2)
            time.sleep(4.0)
            _chat(client, t2)  # exact re-send: the end-of-prompt cluster
            time.sleep(4.0)
            res["pruned"] = _find(client.get(f"{BASE}/v1/status").json(), "ckpt_pruned")
    res["ladders"], res["disk_bytes"] = _disk_ladders(ssd_dir)
    with _Server(ssd_dir, gap, os.path.join(workdir, f"server-gap{gap}-b.log")):
        with httpx.Client(timeout=300) as client:
            res["resend"], res["resend_cached"], res["resend_prompt"] = _chat(client, t2)
            t3 = t2 + [
                {"role": "assistant", "content": a2},
                {"role": "user", "content": Q[2]},
            ]
            res["t3"], res["t3_cached"], res["t3_prompt"] = _chat(client, t3)
    res["a2"] = a2
    res["log"] = "".join(
        open(os.path.join(workdir, f"server-gap{gap}-{s}.log")).read() for s in "ab"
    )
    return res


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-spill-ladder-")
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))

    on = _arm(16, workdir)
    off = _arm(0, workdir)
    for label, r in (("gap16", on), ("gap0", off)):
        print(
            f"INFO  {label}: ladders {r['ladders']} disk {r['disk_bytes'] / 1e6:.0f} MB "
            f"pruned {r['pruned']} | restart re-send cached "
            f"{r['resend_cached']}/{r['resend_prompt']}, turn 3 "
            f"{r['t3_cached']}/{r['t3_prompt']}"
        )
    check("gap16 ladders exist (hybrid model)", any(p for _n, p in on["ladders"]))
    check(
        "gap16: no spilled checkpoint at/after its entry's length",
        all(p < n for n, ps in on["ladders"] for p in ps),
    )
    gaps = [b - a for _n, ps in on["ladders"] for a, b in zip(ps, ps[1:])]
    check("gap16: no two spilled checkpoints < 16 tokens apart", all(g >= 16 for g in gaps), f"{gaps}")
    check("gap16: ckpt_pruned > 0 in /v1/status", (on["pruned"] or 0) > 0, f"{on['pruned']}")
    check(
        "gap16 writes fewer bytes than gap0",
        on["disk_bytes"] < off["disk_bytes"],
        f"{on['disk_bytes'] / 1e6:.0f} vs {off['disk_bytes'] / 1e6:.0f} MB",
    )
    for turn in ("resend", "t3"):
        check(
            f"restarted {turn}: gap16 restores within 16 tokens of gap0",
            on[f"{turn}_cached"] >= off[f"{turn}_cached"] - 15,
            f"{on[f'{turn}_cached']} vs {off[f'{turn}_cached']}",
        )
    check(
        "restarted turn 3 restores >= 90 % (gap16)",
        on["t3_cached"] >= 0.9 * on["t3_prompt"],
        f"{on['t3_cached']}/{on['t3_prompt']}",
    )
    print(
        f"INFO  T=0 identity gap16 vs gap0: re-send {on['resend'] == off['resend']}, "
        f"turn 3 {on['t3'] == off['t3']}; gap16 re-send vs first run "
        f"{on['resend'] == on['a2']}"
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
