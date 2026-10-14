#!/usr/bin/env python
"""A/B of PATCHES.md #140 (spill-ladder-prune) on a REAL server — off the
live routes. Old vs new code via PYTHONPATH on the same venv.

Production SSD stores hold COLD, distinct conversations (a turn grown from a
cached donor never re-spills), so each arm:

  phase A  three distinct deep conversations sent cold (one request each,
           history embedded), then a follow-up turn and an exact re-send of
           it; then a decode-speed probe; then SIGTERM (close drains spills).
  phase B  a new server on the same SSD dir: per conversation, the exact
           re-send and a next turn — both restore through an SSD promote.

Per arm it records spill bytes, shutdown (drain) time, ready time, promote
latency, TTFT / cached tokens of every request, decode tok/s and the T=0
texts (identity is compared across arms).

    AB_OLD=/tmp/ab140/old AB_NEW=/tmp/ab140/new AB_MODEL=mlx-community/Qwen3.5-9B-4bit \\
        AB_DEPTHS=6000,10000,14000 AB_ROUNDS=2 python ab_spill_ladder.py

Extra route env (JSON object) via AB_ENV. Writes AB_OUT (default
./ab140.json) and prints a summary table.
"""

import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

MODEL = os.environ.get("AB_MODEL", "mlx-community/Qwen3.5-9B-4bit")
PORT = int(os.environ.get("AB_PORT", "8769"))
BASE = f"http://127.0.0.1:{PORT}"
DEPTHS = [int(x) for x in os.environ.get("AB_DEPTHS", "6000,10000,14000").split(",")]
ROUNDS = int(os.environ.get("AB_ROUNDS", "2"))
ARMS = {"old": os.environ["AB_OLD"], "new": os.environ["AB_NEW"]}
EXTRA_ENV = json.loads(os.environ.get("AB_ENV", "{}"))
SERVE_ARGS = os.environ.get("AB_SERVE_ARGS", "--text-only").split()


def _filler(tag, n_chars):
    """Distinct, code-review-like history text (~4 chars/token)."""
    lines, i = [], 0
    while sum(len(x) + 1 for x in lines) < n_chars:
        lines.append(
            f"{tag}/src/mod_{i % 97}.py:{i * 7 % 1000}: def handler_{tag}_{i}(req, "
            f"ctx): return ctx.store.get(req.key_{i % 13}) or fallback_{i % 31}(req)"
        )
        i += 1
    return "\n".join(lines)


def conversation(tag, depth):
    sys_msg = f"You are reviewer {tag}. Answer in one short sentence. " + _filler(
        f"sys{tag}", 2400
    )
    msgs = [{"role": "system", "content": sys_msg}]
    turns = 4
    for t in range(turns):
        msgs.append({"role": "user", "content": f"Review chunk {t}:\n"
                     + _filler(f"{tag}t{t}", depth * 4 // turns - 600)})
        if t < turns - 1:
            msgs.append({"role": "assistant", "content": f"Chunk {t} of {tag} looks fine."})
    return msgs


def stream(messages, max_tokens=48):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, text, usage = None, "", {}
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.perf_counter() - t0
                text += piece
    total = time.perf_counter() - t0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {"ttft": ttft or total, "total": total, "text": text,
            "prompt": usage.get("prompt_tokens", 0), "cached": cached,
            "completion": usage.get("completion_tokens", 0)}


def find(d, k):
    if isinstance(d, dict):
        if k in d:
            return d[k]
        for v in d.values():
            r = find(v, k)
            if r is not None:
                return r
    return None


def status():
    return json.load(urllib.request.urlopen(f"{BASE}/v1/status", timeout=30))


class Server:
    def __init__(self, src, ssd_dir, log_path):
        env = dict(os.environ, **EXTRA_ENV)
        env.update(VLLM_MLX_BATCHED_SYSTEM_KV="1", VLLM_MLX_SSD_SYSTEM_KV_DIR=ssd_dir,
                   PYTHONPATH=src, HF_HUB_OFFLINE="1")
        self.log = open(log_path, "w")
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "vllm_mlx.cli", "serve", MODEL, "--host", "127.0.0.1",
             "--port", str(PORT), "--continuous-batching", "--max-num-seqs", "8",
             *SERVE_ARGS],
            stdout=self.log, stderr=subprocess.STDOUT, env=env)
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(open(log_path).read()[-3000:])
            try:
                if urllib.request.urlopen(f"{BASE}/v1/status", timeout=2).status == 200:
                    break
            except Exception:
                time.sleep(0.25)
        self.ready_s = time.perf_counter() - self.t0

    def stop(self):
        t = time.perf_counter()
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
        return time.perf_counter() - t


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs
               if f.endswith(".safetensors"))


def wait_spills(expect, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        st = status()
        if (find(st, "spill_count") or 0) >= expect and not (find(st, "queue_depth") or 0):
            return
        time.sleep(1)


def run_arm(arm, rnd, workdir):
    src = ARMS[arm]
    ssd = os.path.join(workdir, f"ssd-{arm}-{rnd}")
    convs = {f"c{i}{rnd}": conversation(f"c{i}{rnd}", d) for i, d in enumerate(DEPTHS)}
    out = {"arm": arm, "round": rnd, "A": {}, "B": {}}
    s = Server(src, ssd, os.path.join(workdir, f"{arm}-{rnd}-A.log"))
    out["ready_A"] = s.ready_s
    followups = {}
    for tag, msgs in convs.items():
        r0 = stream(msgs)
        follow = msgs + [{"role": "assistant", "content": r0["text"]},
                         {"role": "user", "content": "Any security issue? One sentence."}]
        r1 = stream(follow)
        r2 = stream(follow)
        followups[tag] = (follow, r1["text"])
        out["A"][tag] = {"cold": r0, "follow": r1, "resend": r2}
    dec = stream([{"role": "user", "content": "Count from 1 to 300, comma separated."}], 256)
    out["decode_tps"] = (dec["completion"] - 1) / max(1e-9, dec["total"] - dec["ttft"])
    wait_spills(len(convs))
    st = status()
    out["spill_count"] = find(st, "spill_count")
    out["pruned_bytes"] = find(st, "ckpt_pruned_bytes") or 0
    out["stop_A"] = s.stop()
    out["disk_bytes"] = dir_bytes(ssd)
    s = Server(src, ssd, os.path.join(workdir, f"{arm}-{rnd}-B.log"))
    out["ready_B"] = s.ready_s
    for tag, (follow, ftext) in followups.items():
        rs = stream(follow)
        nxt = follow + [{"role": "assistant", "content": ftext},
                        {"role": "user", "content": "And performance? One sentence."}]
        rn = stream(nxt)
        out["B"][tag] = {"resend": rs, "next": rn}
    st = status()
    out["promotes"] = find(st, "promote_count")
    out["promote_ms"] = find(st, "avg_promote_latency_ms")
    out["stop_B"] = s.stop()
    logs = open(os.path.join(workdir, f"{arm}-{rnd}-A.log")).read() + open(
        os.path.join(workdir, f"{arm}-{rnd}-B.log")).read()
    out["errors"] = sum(logs.count(k) for k in
                        ("Traceback", "no Stream(", "generation_error_recovery"))
    return out


def main():
    workdir = tempfile.mkdtemp(prefix="ab140-")
    results = []
    for rnd in range(ROUNDS):
        order = ["old", "new"] if rnd % 2 == 0 else ["new", "old"]
        for arm in order:
            r = run_arm(arm, rnd, workdir)
            results.append(r)
            print(f"done {arm} round {rnd}: disk {r['disk_bytes'] / 1e6:.0f} MB, "
                  f"promote {r['promote_ms']} ms, errors {r['errors']}", flush=True)
    json.dump(results, open(os.environ.get("AB_OUT", "ab140.json"), "w"), indent=1)

    def agg(arm, fn):
        vals = [fn(r) for r in results if r["arm"] == arm]
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else float("nan")

    rows = [
        ("SSD bytes written (MB)", lambda r: r["disk_bytes"] / 1e6),
        ("ckpt bytes pruned (MB)", lambda r: r["pruned_bytes"] / 1e6),
        ("shutdown drain (s)", lambda r: r["stop_A"]),
        ("restart ready (s)", lambda r: r["ready_B"]),
        ("avg promote latency (ms)", lambda r: r["promote_ms"]),
        ("restart re-send TTFT (s)",
         lambda r: statistics.mean(v["resend"]["ttft"] for v in r["B"].values())),
        ("restart re-send cached (tok)",
         lambda r: statistics.mean(v["resend"]["cached"] for v in r["B"].values())),
        ("restart next-turn TTFT (s)",
         lambda r: statistics.mean(v["next"]["ttft"] for v in r["B"].values())),
        ("restart next-turn cached (tok)",
         lambda r: statistics.mean(v["next"]["cached"] for v in r["B"].values())),
        ("warm re-send TTFT, no restart (s)",
         lambda r: statistics.mean(v["resend"]["ttft"] for v in r["A"].values())),
        ("cold TTFT (s)", lambda r: statistics.mean(v["cold"]["ttft"] for v in r["A"].values())),
        ("decode (tok/s)", lambda r: r["decode_tps"]),
        ("errors in logs", lambda r: r["errors"]),
    ]
    print(f"\nmodel {MODEL}, depths {DEPTHS}, rounds {ROUNDS}")
    print(f"{'metric':36} {'old':>10} {'new':>10} {'delta':>8}")
    for name, fn in rows:
        o, n = agg("old", fn), agg("new", fn)
        d = f"{100 * (n - o) / o:+.1f}%" if o else ""
        print(f"{name:36} {o:>10.3f} {n:>10.3f} {d:>8}")
    texts = {}
    for r in results:
        key = r["round"]
        for tag in r["B"]:
            texts.setdefault((key, tag), {})[r["arm"]] = (
                r["A"][tag]["cold"]["text"], r["A"][tag]["follow"]["text"],
                r["B"][tag]["resend"]["text"], r["B"][tag]["next"]["text"])
    same = sum(1 for v in texts.values() if v.get("old") == v.get("new"))
    print(f"T=0 texts identical old vs new: {same}/{len(texts)} conversations")
    print(f"raw: {os.path.abspath(os.environ.get('AB_OUT', 'ab140.json'))} logs: {workdir}")


if __name__ == "__main__":
    main()
