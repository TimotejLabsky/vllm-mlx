#!/usr/bin/env python
"""Deep-chain stress test for a batched hybrid route — the 2026-09-17 OOM
signature, reproduced on purpose (PATCHES.md #106-#108).

What it does, against a route served through llama-swap:

  phase 0  calibrate chars/token with one request (this also loads the model)
  phase 1  seed N DISTINCT deep chains (default 5 x 60K tokens, two at a time)
           so the snapshot bag is genuinely full - distinct, because chains
           that share a prefix grow each other by reference and add no bytes
  phase 2  fire one deep follow-up per chain: the first starts decoding, the
           second is co-batched 4 s later, the rest queue behind them - the
           exact shape that OOMed three times at 63 GB on 2026-09-17
  report   per follow-up: prompt tokens, cached tokens, seconds; phase-2 wall
           time; memory maxima; the admission / cache / SSD counters

A monitor thread logs every change in run/wait, memory, bag, pins,
spill-pending entries, projected defers/relief, lazy restores, SSD
spills/drops/promotes and engine recoveries.

READ BEFORE RUNNING - this deliberately pushes a production box:
  * Only on an IDLE route and with the owner's go-ahead. Loading a heavy route
    swaps other llama-swap groups out (voice) for the duration.
  * It takes ~50 min on Qwen3.8-27B-4bit (60K cold prefill ~ 12 min per pair).
  * If a guard does NOT hold you get a Metal OOM: generation_error_recovery
    aborts the test's own requests (503 since #104) or the process respawns.
    MacStudioLLMGenerationRecovery will page.
  * Every run leaves N synthetic deep entries in the route's SSD tier. Use a
    fresh --seed per run so runs cannot feed each other; LRU ages them out.
  * Run it ON the serving host (it reads the route's /v1/status directly):
        scp scripts/fork/stress_deep_chains.py host:/tmp/ &&
        ssh host 'nohup python /tmp/stress_deep_chains.py --model M \\
            --status http://127.0.0.1:PORT > /tmp/stress.log 2>&1 &'

Reference results (Qwen3.8-27B-4bit, 5 x 60K, 2026-09-21):
  fork 5a3b1aa  2/5 follow-ups cached, phase 2 1,150 s, peak 53.3 GB, 0 OOM
  + #107, cap 9 4/5                   463 s
  + #108        5/5                   123 s, peak 51.4 GB, first live defer
  (2026-09-17, before #106: three OOMs at 63.0-63.6 GB under this shape)
"""

import argparse
import random
import sys
import threading
import time

WORDS = (
    "def class return import self value index buffer request cache token layer "
    "state batch prefill decode memory stream error retry config route model "
    "tensor shape dtype float int list dict none true false async await yield "
    "lock thread queue"
).split()


def doc(seed: int, nchars: int) -> str:
    """Deterministic pseudo-code; a different seed shares no prefix."""
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
    ap.add_argument("--model", required=True, help="llama-swap route name")
    ap.add_argument("--base", default="http://127.0.0.1:8080", help="llama-swap URL")
    ap.add_argument("--status", required=True, help="the route's own URL (…:PORT)")
    ap.add_argument("--tokens", type=int, default=60000, help="tokens per chain")
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    ap.add_argument("--follow-tokens", type=int, default=450)
    args = ap.parse_args()

    import httpx

    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

    def body(msgs, max_tokens):
        return {
            "model": args.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    results = {}

    def ask(msgs, max_tokens, tag):
        t = time.time()
        try:
            r = httpx.post(
                f"{args.base}/v1/chat/completions",
                timeout=3000,
                json=body(msgs, max_tokens),
            )
            if r.status_code != 200:
                log(tag, "HTTP", r.status_code, r.text[:200])
                results[tag] = ("HTTP", r.status_code)
                return None
            j = r.json()
            usage = j.get("usage", {})
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            results[tag] = (usage.get("prompt_tokens"), cached, round(time.time() - t))
            log(
                tag,
                f"ok prompt={usage.get('prompt_tokens')} cached={cached} "
                f"completion={usage.get('completion_tokens')} in {time.time() - t:.0f}s",
            )
            return j["choices"][0]["message"]["content"] or ""
        except Exception as e:  # the point is to observe failures, not die on them
            log(tag, "EXC", repr(e)[:200])
            results[tag] = ("EXC",)
            return None

    stop = False
    maxima = dict(peak=0, active=0, run=0, wait=0, pinned=0, spill_pending=0)
    last = {}

    def monitor():
        while not stop:
            try:
                d = httpx.get(f"{args.status}/v1/status", timeout=5).json()
                c, m = d["cache"], d["metal"]
                s = c.get("ssd", {})
                for k, v in (
                    ("peak", m["peak_memory_gb"]),
                    ("active", m["active_memory_gb"]),
                    ("run", d["num_running"]),
                    ("wait", d["num_waiting"]),
                    ("pinned", c.get("pinned_entries", 0)),
                    ("spill_pending", c.get("spill_pending_entries", 0)),
                ):
                    maxima[k] = max(maxima[k], v)
                cur = (
                    d["num_running"], d["num_waiting"], c["entry_count"],
                    c.get("pinned_entries"), c.get("spill_pending_entries"),
                    c.get("projected_defers"), c.get("projected_relief_passes"),
                    c.get("lazy_restores"), c.get("lazy_ssd_fallbacks"),
                    c.get("lazy_restore_misses"), d.get("generation_recoveries"),
                    s.get("spill_drops"), s.get("spill_count"),
                    c.get("ssd_promotes"), c.get("evictions"),
                )  # fmt: skip
                if cur != last.get("v"):
                    last["v"] = cur
                    log(
                        f"MON run/wait {d['num_running']}/{d['num_waiting']} "
                        f"active {m['active_memory_gb']:.0f}GB peak {m['peak_memory_gb']:.1f}"
                        f" | bag {c['entry_count']} entries {c['memory_mb'] / 1024:.0f}GB"
                        f" pinned {c.get('pinned_entries')} spill_pending "
                        f"{c.get('spill_pending_entries')} evictions {c.get('evictions')}"
                        f" | proj defers/relief {c.get('projected_defers')}/"
                        f"{c.get('projected_relief_passes')}"
                        f" | lazy ok/ssd/miss {c.get('lazy_restores')}/"
                        f"{c.get('lazy_ssd_fallbacks')}/{c.get('lazy_restore_misses')}"
                        f" | ssd spills {s.get('spill_count')} drops {s.get('spill_drops')}"
                        f" promotes {c.get('ssd_promotes')} queued "
                        f"{s.get('queued_bytes', 0) / 1e6:.0f}MB"
                        f" | recoveries {d.get('generation_recoveries')}"
                    )
            except Exception:
                pass
            time.sleep(3)

    sample = doc(args.seed - 1, 40000)
    r = httpx.post(
        f"{args.base}/v1/chat/completions",
        timeout=1200,
        json=body([{"role": "user", "content": sample + "\nReply: ok"}], 2),
    )
    cpt = len(sample) / r.json()["usage"]["prompt_tokens"]
    log(f"calibration {cpt:.2f} chars/token; seed {args.seed}")
    threading.Thread(target=monitor, daemon=True).start()

    chains = {}

    def seed_chain(i):
        msgs = [
            {
                "role": "system",
                "content": f"You review module #{args.seed + i}. Be terse.",
            },
            {
                "role": "user",
                "content": doc(args.seed + i, int(args.tokens * cpt))
                + "\nSummarise this module in one short sentence.",
            },
        ]
        answer = ask(msgs, 24, f"seed{i}")
        if answer is not None:
            chains[i] = msgs + [{"role": "assistant", "content": answer}]

    log(
        f"PHASE 1: seeding {args.chains} distinct {args.tokens}-token chains, 2 at a time"
    )
    for k in range(0, args.chains, 2):
        threads = [
            threading.Thread(target=seed_chain, args=(i,))
            for i in range(k, min(k + 2, args.chains))
        ]
        [t.start() for t in threads]
        [t.join() for t in threads]
    log(f"seeded {len(chains)}/{args.chains}; maxima so far {maxima}")
    for k in maxima:
        maxima[k] = 0

    log("PHASE 2: deep follow-ups - 1 decoding, 1 co-batched, the rest queued")

    def follow(i):
        ask(
            chains[i]
            + [
                {
                    "role": "user",
                    "content": "Now write a detailed, numbered review of at least 30 "
                    "distinct issues in that module.",
                }
            ],
            args.follow_tokens,
            f"follow{i}",
        )

    threads, started = [], time.time()
    for n, i in enumerate(sorted(chains)):
        t = threading.Thread(target=follow, args=(i,))
        t.start()
        threads.append(t)
        time.sleep(4 if n < 2 else 1)
    [t.join() for t in threads]
    wall = time.time() - started
    stop = True
    time.sleep(1)

    d = httpx.get(f"{args.status}/v1/status", timeout=10).json()
    c = d["cache"]
    s = c.get("ssd", {})
    follow_ups = {k: v for k, v in sorted(results.items()) if k.startswith("follow")}
    log("RESULT follow-ups (prompt, cached, seconds):", follow_ups)
    log(f"RESULT phase2 wall {wall:.0f}s maxima:", maxima)
    log(
        "RESULT counters:",
        {
            k: c.get(k)
            for k in (
                "projected_defers", "projected_relief_passes", "lazy_restores",
                "lazy_ssd_fallbacks", "lazy_restore_misses", "pinned_entries",
                "spill_pending_entries", "pressure_evictions", "evictions",
                "entry_count", "hits", "misses", "ssd_promotes",
            )
        },  # fmt: skip
        "ssd",
        {
            k: s.get(k)
            for k in ("spill_count", "spill_drops", "busy_writes", "backlog_drops")
        },
        "recoveries",
        d.get("generation_recoveries"),
    )
    served = sum(1 for v in follow_ups.values() if len(v) == 3 and v[1])
    log(f"DONE: {served}/{len(chains)} follow-ups served from cache")
    ok = served == len(chains) and d.get("generation_recoveries", 0) == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
