#!/usr/bin/env python3
"""Real-server check for fork patches #149 + #150 — the 2026-10-23 plateau
OOM (infra issue #847), reproduced on purpose, then shown gone.

The shape: ONE conversation on a batched hybrid route grows turn by turn.
Every turn restores the chain, prefills a few hundred tokens and stores
twice (prompt boundary + finish). Before #149 each grown store kept an mx
VIEW of the finished row as its delta, which pinned the row's whole buffer:
+4-5 GB per turn outside every counter, so a single seat sat at 45-52 GB
between turns with one entry resident. A turn that then prefilled ~1.4K
tokens at ~69K context needed its lazy restore (~4.6 GB) plus the
materialised attention scores (24 heads x chunk x context x 2 B ~ 4.5 GB)
on top — and the #105 solo guard priced neither (#150), so the step ran
into the ~62 GB wall.

With #149 the resting footprint is weights + bag; with #150 a deep chunk
that would not fit sheds cache first. The report prints both: active
memory between turns (should track the bag) and the deep turn's peak.

Phases:
  0  calibrate chars/token (loads the model)
  1  cold turn at --base-tokens (default 45K), then +--turn-tokens turns
     (default 350, max_tokens 40, a few seconds apart like a human/agent)
     until active memory reaches --plateau-gb (default 50) or --max-turns
  2  one deep turn of --deep-tokens (default 1500) prompt tokens
  report: the deep turn's HTTP status / finish, active/peak around it,
     solo_relief_passes, generation recoveries, and the T=0 answer sha

Run it ON the serving host against a spare-port process (route_under_test.py
starts the route's exact cmd+env with PYTHONPATH=<src> and a scratch SSD
dir), never against the live route — a failed run IS a Metal OOM:

  python3 route_under_test.py Qwen3.8-27B-4bit --src /tmp/src --port 8781 -- \\
      python3 e2e_solo_guard_deep_chunk.py --expect oom      # control
  python3 route_under_test.py Qwen3.8-27B-4bit --src /tmp/src-149 --port 8781 -- \\
      python3 e2e_solo_guard_deep_chunk.py --expect held     # fix

Exit 0 when the observed outcome matches --expect (``oom`` = the deep turn
was aborted with the oom kind; ``held`` = it completed with no engine
recovery; the solo-relief count is reported, not required). Stdlib only.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

WORDS = (
    "def class return import self value index buffer request cache token layer "
    "state batch prefill decode memory stream error retry config route model "
    "tensor shape dtype float int list dict none true false async await yield "
    "lock thread queue"
).split()


def doc(seed: int, nchars: int) -> str:
    r = random.Random(seed)
    out, n = [], 0
    while n < nchars:
        line = (
            f"    {r.choice(WORDS)}_{r.randint(0, 9999)} = {r.choice(WORDS)}("
            f"{', '.join(r.choice(WORDS) for _ in range(r.randint(1, 4)))})"
            f"  # {r.choice(WORDS)} {r.randint(0, 99999)}\n"
        )
        out.append(line)
        n += len(line)
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--url", default=os.environ.get("SERVER_URL", "http://127.0.0.1:8781")
    )
    ap.add_argument("--model", default="Qwen3.8-27B-4bit")
    ap.add_argument("--base-tokens", type=int, default=45000)
    ap.add_argument("--turn-tokens", type=int, default=350)
    ap.add_argument("--deep-tokens", type=int, default=1500)
    ap.add_argument("--plateau-gb", type=float, default=50.0)
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--gap-s", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    ap.add_argument("--expect", choices=["oom", "held", "any"], default="any")
    ap.add_argument(
        "--drain-watch-s",
        type=float,
        default=20.0,
        help="trace active memory after the deep turn",
    )
    args = ap.parse_args()

    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

    def get(path, timeout=10):
        with urllib.request.urlopen(args.url + path, timeout=timeout) as r:
            return json.loads(r.read())

    def status():
        try:
            d = get("/v1/status")
            m, c = d.get("metal", {}), d.get("cache", {})
            return {
                "active": m.get("active_memory_gb"),
                "peak": m.get("peak_memory_gb"),
                "solo_relief": c.get("solo_relief_passes"),
                "solo_reject": c.get("solo_rejections"),
                "entries": c.get("entry_count"),
                "recoveries": d.get("generation_recoveries"),
                "spill_pending": c.get("spill_pending_entries"),
                "ladders": c.get("pending_ladders"),
                "running": d.get("num_running"),
            }
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)[:80]}

    def chat(messages, max_tokens):
        body = json.dumps(
            {
                "model": args.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        req = urllib.request.Request(
            args.url + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                j = json.loads(r.read())
            usage = j.get("usage", {})
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            text = j["choices"][0]["message"]["content"] or ""
            return dict(
                ok=True,
                status=200,
                prompt=usage.get("prompt_tokens"),
                cached=cached,
                completion=usage.get("completion_tokens"),
                finish=j["choices"][0].get("finish_reason"),
                text=text,
                secs=round(time.time() - t, 1),
            )
        except urllib.error.HTTPError as e:
            return dict(
                ok=False,
                status=e.code,
                text=e.read().decode(errors="replace")[:300],
                secs=round(time.time() - t, 1),
            )
        except Exception as e:  # noqa: BLE001
            return dict(
                ok=False,
                status=None,
                text=repr(e)[:200],
                secs=round(time.time() - t, 1),
            )

    # monitor: log every change of active / relief / recoveries
    stop = False
    maxima = {"peak": 0.0, "active": 0.0}
    last = {}

    def monitor():
        while not stop:
            s = status()
            if "error" not in s:
                for k in ("peak", "active"):
                    if s[k] is not None:
                        maxima[k] = max(maxima[k], s[k])
                key = (
                    round(s["active"] or 0),
                    s["solo_relief"],
                    s["recoveries"],
                    s["entries"],
                )
                if key != last.get("k"):
                    last["k"] = key
                    log("mon", s)
            time.sleep(2)

    threading.Thread(target=monitor, daemon=True).start()

    # phase 0: calibrate
    probe = doc(args.seed, 20000)
    r = chat([{"role": "user", "content": probe + "\nReply with OK."}], 4)
    if not r["ok"]:
        log("calibration failed", r)
        return 2
    cpt = len(probe) / max(1, r["prompt"] - 20)
    log(f"calibrated {cpt:.2f} chars/token")

    system = "You are a code reviewer. Answer in one short sentence."
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": doc(args.seed + 1, int(args.base_tokens * cpt))
            + "\n\nName one variable from the code above.",
        },
    ]
    r = chat(messages, 40)
    log("turn 0 (cold)", {k: v for k, v in r.items() if k != "text"})
    if not r["ok"]:
        return 2
    messages.append({"role": "assistant", "content": r["text"]})

    # phase 1: grow to the plateau
    turn = 0
    while turn < args.max_turns:
        s = status()
        if (s.get("active") or 0) >= args.plateau_gb:
            log(f"plateau reached: active {s['active']} GB after {turn} turns")
            break
        turn += 1
        time.sleep(args.gap_s)
        messages.append(
            {
                "role": "user",
                "content": doc(args.seed + 100 + turn, int(args.turn_tokens * cpt))
                + f"\n\nTurn {turn}: name one function from the code above.",
            }
        )
        r = chat(messages, 40)
        log(f"turn {turn}", {k: v for k, v in r.items() if k != "text"}, status())
        if not r["ok"]:
            log("growth turn failed; the plateau phase itself should not OOM")
            return 2
        messages.append({"role": "assistant", "content": r["text"]})
    else:
        log(
            f"max turns reached without plateau (active {status().get('active')} GB) — continuing anyway"
        )

    # phase 2: the deep chunk
    time.sleep(args.gap_s)
    before = status()
    messages.append(
        {
            "role": "user",
            "content": doc(args.seed + 999, int(args.deep_tokens * cpt))
            + "\n\nDeep turn: name the first variable in THIS block.",
        }
    )
    r = chat(messages, 40)
    after = status()
    sha = hashlib.sha256((r.get("text") or "").encode()).hexdigest()[:16]
    log("DEEP", {k: v for k, v in r.items() if k != "text"}, "sha", sha)
    log("before", before)
    log("after ", after)
    # how the in-flight state drains once the turn is over (nothing running)
    t_drain = time.time()
    while time.time() - t_drain < args.drain_watch_s:
        s = status()
        log(
            f"drain +{time.time() - t_drain:4.1f}s active={s.get('active')} entries={s.get('entries')} ladders={s.get('ladders')}"
        )
        time.sleep(2)
    stop = True
    final = status()
    log("maxima", maxima, "final", final)

    oom = (not r["ok"]) and (
        "oom" in (r.get("text") or "").lower() or r.get("status") == 503
    )
    held = r["ok"] and (final.get("recoveries") or 0) == (before.get("recoveries") or 0)
    relief = (final.get("solo_relief") or 0) - (before.get("solo_relief") or 0)
    outcome = "oom" if oom else ("held" if held else "other")
    log(f"OUTCOME {outcome}  solo_relief_passes +{relief}  answer_sha {sha}")
    if args.expect == "any":
        return 0
    if args.expect == "oom":
        return 0 if oom else 1
    return 0 if held else 1


if __name__ == "__main__":
    sys.exit(main())
