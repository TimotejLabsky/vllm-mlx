"""Concurrent load against a live route: overlapping requests of mixed shapes,
then a lone request, checking every answer and the route's health counters.

usage: concurrent_load.py <base_url> <model>
"""

import json
import sys
import threading
import time
import urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]


def status():
    return json.load(urllib.request.urlopen(f"{BASE}/v1/status", timeout=20))


def ask(tag, prompt, out, **extra):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 350,
        "temperature": 0,
        **extra,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1200))
        m = r["choices"][0]["message"]
        out[tag] = {
            "ok": True,
            "s": round(time.time() - t, 1),
            "finish": r["choices"][0]["finish_reason"],
            "tokens": r["usage"]["completion_tokens"],
            "text": (m.get("content") or ""),
        }
    except Exception as e:  # noqa: BLE001
        out[tag] = {"ok": False, "error": f"{type(e).__name__}: {e}"}


QUESTIONS = [
    ("capital", "Answer with one word only: the capital of France.", "paris", {}),
    ("sum", "Answer with the number only: 17 + 25.", "42", {}),
    (
        "dry",
        "Answer with one word only: the chemical symbol for gold.",
        "au",
        {"dry_multiplier": 0.8},
    ),
    (
        "sorted",
        "Sort these numbers ascending and answer with only the list: 9, 3, 7, 1.",
        "1, 3, 7, 9",
        {},
    ),
]

before = status()
problems = []
for wave in range(3):
    out, threads = {}, []
    for i, (tag, q, want, extra) in enumerate(QUESTIONS):
        th = threading.Thread(target=ask, args=(f"{tag}", q, out), kwargs=extra)
        threads.append(th)
        th.start()
        time.sleep(0.4 * (wave + 1))  # staggered: requests join a running batch
    for th in threads:
        th.join()
    for tag, q, want, extra in QUESTIONS:
        r = out[tag]
        good = r.get("ok") and want in r["text"].lower().replace("**", "")
        print(
            f"wave {wave} {tag}: {'ok' if good else 'PROBLEM'} "
            f"{ {k: v for k, v in r.items() if k != 'text'} } text={r.get('text', '')[:50]!r}",
            flush=True,
        )
        if not good:
            problems.append(f"wave {wave} {tag}: {r}")

# a lone request afterwards must speculate again
mid = status()
out = {}
ask("alone", "Explain in about 150 words how a hash map handles collisions.", out)
after = status()
r = out["alone"]
print("alone:", {k: v for k, v in r.items() if k != "text"})
if not r.get("ok") or r["tokens"] < 50:
    problems.append(f"alone: {r}")


def m(s, k):
    return s["mtp"][k]


rounds_alone = m(after, "attempted") - m(mid, "attempted")
print(
    f"\ndrafter rounds during the waves: {m(mid, 'attempted') - m(before, 'attempted')}, "
    f"concurrent bypasses: {m(mid, 'bypass_counts')['concurrent_batch'] - m(before, 'bypass_counts')['concurrent_batch']}, "
    f"rounds for the lone request: {rounds_alone}"
)
print(
    "errors:",
    m(after, "errors") - m(before, "errors"),
    "disabled:",
    m(after, "disabled"),
    "generation_recoveries:",
    after.get("generation_recoveries"),
    "metal peak GB:",
    (after.get("metal") or {}).get("peak_memory_gb"),
)
if m(after, "errors") != m(before, "errors") or m(after, "disabled"):
    problems.append("drafter errors or disabled")
if (after.get("generation_recoveries") or 0) != (
    before.get("generation_recoveries") or 0
):
    problems.append("generation recoveries during the run")
if rounds_alone < 20:
    problems.append(f"the lone request barely speculated ({rounds_alone} rounds)")
print("CONCURRENT_LOAD", "PASS" if not problems else "FAIL")
for p in problems:
    print("  ", p)
sys.exit(1 if problems else 0)
