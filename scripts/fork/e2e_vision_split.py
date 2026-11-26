#!/usr/bin/env python
"""End-to-end check of PATCHES.md #181 (``--vision-split``) on a REAL server
and model, off the live routes.

Starts ``vllm_mlx.cli serve --continuous-batching`` twice on one checkpoint
that ships a vision tower: first ``--text-only`` (the reference: what the
fleet runs today), then ``--vision-split``. Batched system-KV on for both,
own HOME each (no shared on-disk cache). Asserts

  * text at T=0 is byte-identical on the split route vs --text-only
    (non-streamed and streamed, with and without a system prompt);
  * the re-sent text prompt is served from the TEXT cache on the split
    route (``cached_tokens`` > 0, ``system_kv_cache`` hits > 0);
  * an image request is answered on the split route (solid red -> "red",
    solid blue -> "blue") and gets 400 ``media_not_supported`` on --text-only;
    the same on the Anthropic endpoint (``/v1/messages``), whose text answer
    must also match the reference;
  * an image request and a text request in flight TOGETHER both finish, and
    the text answer is still the reference one (the GPU gate took turns);
  * ``/v1/status`` carries both blocks: the text cache at top level and
    ``vision_split`` with the vision scheduler stats and gate turns on BOTH
    sides;
  * no traceback / stream-thread error in either server log.

Also prints (INFO, not asserted) Metal active memory after load on both
routes: the difference is the vision tower plus the vision side's state.

    E2E_MODEL=mlx-community/Qwen3.5-4B-4bit \\
    E2E_SERVE_ARGS="--reasoning-parser qwen3" python scripts/fork/e2e_vision_split.py

Pick a small hybrid vision checkpoint (Qwen3.5 0.8B/2B/4B/9B 4-bit). Exit
code 0 only if all pass.
"""

import base64
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.5-4B-4bit")
PORT = int(os.environ.get("E2E_PORT", "8781"))
BASE = f"http://127.0.0.1:{PORT}"
MAX_TOKENS = int(os.environ.get("E2E_MAX_TOKENS", "48"))


def _png(rgb, size=256):
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    rows = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


RED = _png((230, 20, 20))
BLUE = _png((20, 30, 230))

SYSTEM = (
    "You are a terse assistant for a homelab. Answer in at most two sentences. "
    + "Background notes: "
    + " ".join(f"item {i} is filed under shelf {i % 7}." for i in range(120))
)
TEXT_CASES = {
    "plain": [
        {
            "role": "user",
            "content": "Name three prime numbers and say why 9 is not prime.",
        }
    ],
    "system": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Which shelf holds item 40? One sentence."},
    ],
}


def _img_msg(url, text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": text},
            ],
        }
    ]


def _body(messages, stream):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return json.dumps(body).encode()


def _post(messages, stream=False):
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        _body(messages, stream),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            if not stream:
                out = json.load(r)
                usage = out.get("usage") or {}
                return {
                    "status": 200,
                    "text": out["choices"][0]["message"].get("content") or "",
                    "cached": (usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens", 0
                    ),
                    "prompt": usage.get("prompt_tokens", 0),
                }
            text, usage = "", {}
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                usage = chunk.get("usage") or usage
                for ch in chunk.get("choices") or []:
                    text += (ch.get("delta") or {}).get("content") or ""
            return {
                "status": 200,
                "text": text,
                "cached": (usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens", 0
                ),
                "prompt": usage.get("prompt_tokens", 0),
            }
    except urllib.error.HTTPError as e:
        return {"status": e.code, "text": "", "error": e.read().decode()[:400]}


def _anthropic(content):
    body = json.dumps(
        {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": "disabled"},
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/messages", body, {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.load(r)
            text = "".join(
                b.get("text", "")
                for b in out.get("content") or []
                if b.get("type") == "text"
            )
            return {"status": 200, "text": text}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "text": "", "error": e.read().decode()[:400]}


ANTHROPIC_IMAGE = [
    {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": BLUE.split(",", 1)[1],
        },
    },
    {"type": "text", "text": "What single colour fills this image? One word."},
]


def _status():
    return json.load(urllib.request.urlopen(f"{BASE}/v1/status", timeout=30))


def _session(mode, workdir):
    home = os.path.join(workdir, f"home-{mode}")
    os.makedirs(home, exist_ok=True)
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        HOME=home,
        VLLM_MLX_BATCHED_SYSTEM_KV="1",
        HF_HOME=os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
    )
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    flag = "--text-only" if mode == "text-only" else "--vision-split"
    log_path = os.path.join(workdir, f"server-{mode}.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm_mlx.cli",
            "serve",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--continuous-batching",
            flag,
            "--max-num-seqs",
            "4",
            *os.environ.get("E2E_SERVE_ARGS", "").split(),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    r = {}
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(open(log_path).read()[-3000:])
            try:
                if urllib.request.urlopen(f"{BASE}/v1/models", timeout=2).status == 200:
                    break
            except Exception:
                time.sleep(0.5)
            if time.time() - t0 > 900:
                raise RuntimeError("startup timeout")
        r["mem_after_load_gb"] = (_status().get("metal") or {}).get("active_memory_gb")
        for name, msgs in TEXT_CASES.items():
            r[f"{name}"] = _post(msgs)
            r[f"{name}_stream"] = _post(msgs, stream=True)
        r["system_resend"] = _post(TEXT_CASES["system"])
        r["red"] = _post(
            _img_msg(RED, "What single colour fills this image? One word.")
        )
        r["blue"] = _post(
            _img_msg(BLUE, "What single colour fills this image? One word."),
            stream=True,
        )
        r["anthropic_text"] = _anthropic(
            "Name three prime numbers and say why 9 is not prime."
        )
        r["anthropic_image"] = _anthropic(ANTHROPIC_IMAGE)
        if mode == "vision-split":
            # Both sides in flight at once: the gate must take turns.
            box = {}

            def run(key, msgs, stream):
                box[key] = _post(msgs, stream=stream)

            threads = [
                threading.Thread(
                    target=run,
                    args=(
                        "conc_image",
                        _img_msg(RED, "Describe this image in one sentence."),
                        True,
                    ),
                ),
                threading.Thread(
                    target=run, args=("conc_text", TEXT_CASES["plain"], True)
                ),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=900)
            r.update(box)
        r["status"] = _status()
        return r
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        globals().setdefault("_LOGS", []).append(open(log_path).read())


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-vision-split-")
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))

    ref = _session("text-only", workdir)
    split = _session("vision-split", workdir)

    for key in ("plain", "plain_stream", "system", "system_stream", "system_resend"):
        same = (
            ref[key]["status"] == split[key]["status"] == 200
            and ref[key]["text"] == split[key]["text"]
        )
        check(
            f"text T=0 byte-identical to --text-only: {key}",
            same,
            (
                ""
                if same
                else f"\n      ref  : {ref[key]['text']!r}\n      split: {split[key]['text']!r}"
            ),
        )
    check(
        "split: re-sent text prompt served from the text cache",
        split["system_resend"]["cached"] > 0,
        f"cached {split['system_resend']['cached']}/{split['system_resend']['prompt']} "
        f"(ref {ref['system_resend']['cached']})",
    )
    st = split["status"]
    cache = st.get("cache") or {}
    check(
        "split: /v1/status text cache block shows hits",
        (cache.get("hits") or 0) > 0,
        json.dumps({k: cache.get(k) for k in ("hits", "misses", "tokens_saved")}),
    )
    check(
        "--text-only: image request is a 400 media_not_supported",
        ref["red"]["status"] == 400
        and "media_not_supported" in ref["red"].get("error", ""),
        f"{ref['red']['status']} {ref['red'].get('error', '')[:120]!r}",
    )
    check(
        "split: red image answered red",
        split["red"]["status"] == 200 and "red" in split["red"]["text"].lower(),
        repr(split["red"]["text"][:80]),
    )
    check(
        "split: blue image answered blue (streamed)",
        split["blue"]["status"] == 200 and "blue" in split["blue"]["text"].lower(),
        repr(split["blue"]["text"][:80]),
    )
    check(
        "Anthropic /v1/messages: text T=0 byte-identical to --text-only",
        ref["anthropic_text"]["status"] == split["anthropic_text"]["status"] == 200
        and ref["anthropic_text"]["text"] == split["anthropic_text"]["text"],
        repr(split["anthropic_text"]["text"][:80]),
    )
    check(
        "Anthropic /v1/messages: image 400 on --text-only, answered blue on split",
        ref["anthropic_image"]["status"] == 400
        and split["anthropic_image"]["status"] == 200
        and "blue" in split["anthropic_image"]["text"].lower(),
        f"ref {ref['anthropic_image']['status']} split {split['anthropic_image']['status']} "
        f"{split['anthropic_image']['text'][:60]!r}",
    )
    ci, ct = split.get("conc_image") or {}, split.get("conc_text") or {}
    check(
        "split: concurrent image + text both finish",
        ci.get("status") == 200
        and ct.get("status") == 200
        and ci.get("text")
        and ct.get("text"),
        f"image {ci.get('status')} text {ct.get('status')}",
    )
    check(
        "split: concurrent text answer still equals the --text-only one",
        ct.get("text") == ref["plain_stream"]["text"],
        repr((ct.get("text") or "")[:80]),
    )
    vs = st.get("vision_split") or {}
    gate = vs.get("gate") or {}
    check(
        "split: /v1/status has the vision_split block with vision scheduler stats",
        bool(vs.get("enabled"))
        and isinstance(vs.get("vision"), dict)
        and "memory_aware_cache" in vs["vision"],
        json.dumps(
            {
                k: vs.get(k)
                for k in (
                    "enabled",
                    "text_model_class",
                    "vision_prefix_cache_budget_mb",
                )
            }
        ),
    )
    check(
        "split: the gate gave turns to both sides",
        (gate.get("text_turns") or 0) > 0 and (gate.get("vision_turns") or 0) > 0,
        json.dumps(
            {
                k: gate.get(k)
                for k in (
                    "text_turns",
                    "vision_turns",
                    "max_text_wait_s",
                    "max_vision_wait_s",
                    "text_yield_timeouts",
                )
            }
        ),
    )
    print(
        f"INFO  metal active after load: text-only {ref['mem_after_load_gb']} GB, "
        f"vision-split {split['mem_after_load_gb']} GB"
    )
    bad = [
        ln
        for log in globals().get("_LOGS", [])
        for ln in log.splitlines()
        if "Traceback" in ln or "no Stream(" in ln or "generation_error_recovery" in ln
    ]
    check(
        "server logs have no traceback / stream-thread error",
        not bad,
        "; ".join(bad[:2]),
    )
    print(f"model: {MODEL} | logs: {workdir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
