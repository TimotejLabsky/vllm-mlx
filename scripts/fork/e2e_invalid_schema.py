#!/usr/bin/env python
"""Real-server check of upstream #856 on the fork: an invalid json_schema is
rejected and does not take a co-batched request down with it. Point it at any
small text route's spare port:

    SERVER_URL=http://127.0.0.1:8771 python scripts/fork/e2e_invalid_schema.py
"""

import concurrent.futures
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ["SERVER_URL"]


def post(extra):
    model = json.load(urllib.request.urlopen(f"{BASE}/v1/models"))["data"][0]["id"]
    body = {
        "model": model,
        "max_tokens": 48,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "Count from 1 to 20."}],
        "chat_template_kwargs": {"enable_thinking": False},
        **extra,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:200]


BAD = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "bad",
            "strict": True,
            "schema": {"$ref": "#/nope"},
        },
    }
}

with concurrent.futures.ThreadPoolExecutor(2) as pool:
    peer = pool.submit(post, {})
    bad = pool.submit(post, BAD)
    peer_status, peer_body = peer.result()
    bad_status, bad_body = bad.result()
ok_bad = 400 <= bad_status < 500
ok_peer = peer_status == 200 and bool(
    peer_body["choices"][0]["message"].get("content")
)
print(f"{'PASS' if ok_bad else 'FAIL'}  invalid schema rejected  {bad_status} {str(bad_body)[:140]}")
print(f"{'PASS' if ok_peer else 'FAIL'}  co-batched peer answered  {peer_status}")
sys.exit(0 if ok_bad and ok_peer else 1)
