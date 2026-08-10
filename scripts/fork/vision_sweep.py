"""Vision e2e smoke for one model on a spare port.

Reconstructed sweep (the original 14-check `fleetsmoke` script is gone from the
Studio — only a stale .pyc remains). Ten checks, run against a live server on
the deployed code, biased toward the vectors the fork actually cares about:

  * colour-identification checks prove the vision tower really sees pixels
  * red -> blue -> red ordering targets cross-image cache aliasing (the known
    open gap: a second, different image must not be answered from the first's
    cached embedding)
  * concurrent divergent-size images target the #57 MRoPE co-batch vectors
  * text-only-on-a-vision-route targets the mixed-route path

Usage: vision_sweep.py <port> <label>
Exit code 0 if all checks pass.
"""

import base64
import io
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else "model"
BASE = f"http://127.0.0.1:{PORT}"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    return ok


def solid_png(rgb, size=(96, 96)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ask(image_url, prompt="What colour is this image? Answer with one word.",
        max_tokens=256, extra_images=()):
    parts = [{"type": "text", "text": prompt}]
    for u in (image_url, *extra_images):
        if u:
            parts.append({"type": "image_url", "image_url": {"url": u}})
    return post("/v1/chat/completions", {
        "model": "x",
        "messages": [{"role": "user", "content": parts}],
        # These are reasoning models: a small budget is spent entirely on
        # <think> and the answer never arrives (the 07-30 sweep logged the same
        # artifact for gemma4). Turn thinking off AND leave real headroom.
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })


def content_of(resp):
    """Answer text only — reasoning stripped, so a colour named while thinking
    can't be mistaken for the answer."""
    m = resp["choices"][0]["message"]
    text = m.get("content") or ""
    # Some routes leak an inline <think> block into content when thinking is
    # disabled at the template level; drop it (and an unclosed trailing one).
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*\Z", " ", text, flags=re.DOTALL)
    if not text.strip():
        text = m.get("reasoning_content") or ""
    return text.lower()


RED = solid_png((220, 20, 20))
BLUE = solid_png((20, 40, 220))
# Deliberately different pixel dimensions -> different MRoPE deltas (#57).
GREEN_TALL = solid_png((20, 190, 60), size=(64, 160))

print(f"\n===== {LABEL} (port {PORT}) =====")

# 1. server up
try:
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        models = json.loads(r.read().decode())
    check("server_up", bool(models.get("data")))
except Exception as e:
    check("server_up", False, repr(e)[:120])
    print(json.dumps({"label": LABEL, "results": results}))
    sys.exit(1)

# 2. single image, colour recognised
try:
    c = content_of(ask(RED))
    check("single_image_red", "red" in c, c.strip()[:70])
except Exception as e:
    check("single_image_red", False, repr(e)[:120])

# 3. different image must not be served from the first image's cache
try:
    c = content_of(ask(BLUE))
    check("second_image_blue_not_aliased", "blue" in c and "red" not in c, c.strip()[:70])
except Exception as e:
    check("second_image_blue_not_aliased", False, repr(e)[:120])

# 4. re-send the first image (pixel-cache HIT path) — must still be red
try:
    c = content_of(ask(RED))
    check("resend_red_after_blue", "red" in c and "blue" not in c, c.strip()[:70])
except Exception as e:
    check("resend_red_after_blue", False, repr(e)[:120])

# 5. differently-shaped image (different MRoPE delta)
try:
    c = content_of(ask(GREEN_TALL))
    check("divergent_size_green", "green" in c, c.strip()[:70])
except Exception as e:
    check("divergent_size_green", False, repr(e)[:120])

# 6. multi-image in one request
try:
    r = ask(RED, prompt="How many images are shown? Answer with a digit.",
            extra_images=(BLUE,))
    check("multi_image", len(content_of(r).strip()) > 0, content_of(r).strip()[:70])
except Exception as e:
    check("multi_image", False, repr(e)[:120])

# 7. concurrent divergent-size images co-batched (#57 vectors)
try:
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(ask, RED)
        f2 = ex.submit(ask, GREEN_TALL)
        c1, c2 = content_of(f1.result()), content_of(f2.result())
    check("concurrent_cobatch", "red" in c1 and "green" in c2,
          f"r={c1.strip()[:28]!r} g={c2.strip()[:28]!r}")
except Exception as e:
    check("concurrent_cobatch", False, repr(e)[:120])

# 8. text-only request on the vision route
try:
    r = post("/v1/chat/completions", {
        "model": "x",
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 256, "temperature": 0.0,
    })
    check("text_only_on_vision_route", len(content_of(r).strip()) > 0,
          content_of(r).strip()[:40])
except Exception as e:
    check("text_only_on_vision_route", False, repr(e)[:120])

# 9. streaming with an image
try:
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What colour is this image?"},
            {"type": "image_url", "image_url": {"url": RED}},
        ]}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 64, "temperature": 0.0, "stream": True,
    }
    out = subprocess.run(
        ["curl", "-s", "-N", "-m", "600", "-X", "POST",
         BASE + "/v1/chat/completions", "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True).stdout
    chunks = [ln for ln in out.splitlines() if ln.startswith("data: ") and "[DONE]" not in ln]
    check("streaming_vision", len(chunks) > 1, f"{len(chunks)} chunks")
except Exception as e:
    check("streaming_vision", False, repr(e)[:120])

# 10. vision cache gauges exposed
try:
    with urllib.request.urlopen(BASE + "/v1/status", timeout=60) as r:
        st = json.loads(r.read().decode())
    blob = json.dumps(st)
    check("status_vision_gauges", "vision_embedding_cache" in blob,
          "vision_embedding_cache present" if "vision_embedding_cache" in blob else "missing")
except Exception as e:
    check("status_vision_gauges", False, repr(e)[:120])

passed = sum(1 for _, ok, _ in results if ok)
print(f"  ==> {LABEL}: {passed}/{len(results)}")
print("SWEEPJSON " + json.dumps({"label": LABEL, "passed": passed,
                                 "total": len(results),
                                 "results": results}))
sys.exit(0 if passed == len(results) else 1)
