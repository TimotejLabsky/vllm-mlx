#!/usr/bin/env python3
"""T=0 warm-vs-cold identity at >=2K prompt tokens against SERVER_URL (stdlib).

Cold request, byte-identical re-send (must be a cache hit with identical text),
then a follow-up turn on top of the cached prefix. Prints a sha of each answer
so two environments (old vs new pins) can be compared run against run.

    SERVER_URL=http://127.0.0.1:8771 python3 t0_warm_cold.py [MODEL]
"""

import hashlib
import json
import os
import sys
import urllib.request

URL = os.environ["SERVER_URL"]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "default"
DOC = " ".join(
    f"Ledger line {i}: item {i * 7 % 101} moved from bin {i % 13} to bin {(i * 3) % 17}."
    for i in range(160)
)
SYSTEM = "You are a careful inventory clerk. Answer briefly and exactly."


def chat(messages):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 64}
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=900))
    msg = r["choices"][0]["message"]
    text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + "|" + (
        msg.get("content") or ""
    )
    return text, r.get("usage", {})


msgs = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": DOC + "\n\nWhich bin did item 14 move to last?"},
]
cold, u1 = chat(msgs)
warm, u2 = chat(msgs)
follow, u3 = chat(
    msgs
    + [{"role": "assistant", "content": cold.split("|", 1)[1]}]
    + [{"role": "user", "content": "And item 21?"}]
)
cached = lambda u: (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
sha = lambda t: hashlib.sha256(t.encode()).hexdigest()[:12]
print(f"prompt_tokens={u1.get('prompt_tokens')} cold_sha={sha(cold)} warm_sha={sha(warm)} follow_sha={sha(follow)}")
print(f"cached: cold={cached(u1)} warm={cached(u2)} follow={cached(u3)}")
print(f"cold: {cold[-160:]!r}")
ok = (
    u1.get("prompt_tokens", 0) >= 2000
    and cold == warm
    and cached(u2) > 0
    and cached(u3) > 0
)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
