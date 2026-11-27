#!/usr/bin/env python
"""End-to-end check of PATCHES.md #181 (``--vision-unified``) on a REAL
server and model, off the live routes.

Starts ``vllm_mlx.cli serve --continuous-batching`` on one checkpoint that
ships a vision tower: first ``--text-only`` (the reference: what the fleet
runs today), then ``--vision-unified``, then ``--vision-unified`` again with
a tiny prompt ceiling. Batched system-KV on, own HOME each (no shared
on-disk cache). Asserts

  * text at T=0 is byte-identical to --text-only (non-streamed, streamed,
    with a long system prompt, its re-send, Anthropic ``/v1/messages``) and
    the cache behaves identically (same ``cached_tokens`` per request);
  * images are answered (red/blue, streamed and not, two images, an image
    in an earlier turn, Anthropic); --text-only still 400s them; video is a
    400 on the unified route;
  * an image request and a text request in flight together both finish and
    the text answer is still the reference one;
  * image tokens count against the prompt ceiling (400 prompt_too_long);
  * a client that disconnects mid-stream from an image request leaves
    nothing running, and the route keeps serving;
  * (stage 3) a re-sent image conversation and its follow-up turn are
    served from the cache; the same text with a different image never hits
    past the image;
  * ``/v1/status`` carries the ``vision_unified`` counters;
  * no traceback / stream-thread error in any server log.

    E2E_MODEL=mlx-community/Qwen3.5-0.8B-4bit python scripts/fork/e2e_vision_unified.py

Exit code 0 only if all pass.
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

MODEL = os.environ.get("E2E_MODEL", "mlx-community/Qwen3.5-0.8B-4bit")
PORT = int(os.environ.get("E2E_PORT", "8791"))
BASE = f"http://127.0.0.1:{PORT}"
MAX_TOKENS = int(os.environ.get("E2E_MAX_TOKENS", "48"))
STAGE = int(os.environ.get("E2E_STAGE", "3"))


def _png(rgb, size=256, other=None):
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    rows = []
    for y in range(size):
        if other is not None and y >= size // 2:
            rows.append(b"\x00" + bytes(other) * size)
        else:
            rows.append(b"\x00" + bytes(rgb) * size)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


RED = _png((230, 20, 20))
BLUE = _png((20, 30, 230))
GREEN = _png((20, 200, 40))

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
COLOUR_Q = "What single colour fills this image? One word."


def _img_msg(urls, text, system=None):
    msgs = [{"role": "system", "content": system}] if system else []
    parts = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    msgs.append({"role": "user", "content": parts + [{"type": "text", "text": text}]})
    return msgs


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


def _disconnect_after_first_chunk(messages):
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(
            {
                "model": MODEL,
                "messages": messages,
                "max_tokens": 400,
                "temperature": 0.0,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode(),
        {"Content-Type": "application/json"},
    )
    r = urllib.request.urlopen(req, timeout=300)
    for raw in r:
        if raw.decode().startswith("data: "):
            break
    r.close()


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
    {"type": "text", "text": COLOUR_Q},
]
VIDEO = [
    {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}},
            {"type": "text", "text": "Describe."},
        ],
    }
]


def _stream_times(messages, max_tokens, started):
    """Stream ``messages``; return the arrival time of every content chunk.
    ``started`` is set after the 10th chunk."""
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(
            {
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode(),
        {"Content-Type": "application/json"},
    )
    times = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            for ch in chunk.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    times.append(time.perf_counter())
                    if len(times) == 10:
                        started.set()
    return times


def _long_system(nonce):
    return (
        f"[{nonce}] You are a terse assistant. Answer in one word. Notes: "
        + " ".join(f"crate {i} sits in bay {i % 11}." for i in range(900))
    )


def _gap_probe(extra_messages):
    """Longest inter-token gap of a decoding text stream while
    ``extra_messages`` (or nothing) is sent once it is decoding."""
    started = threading.Event()
    box = {}
    stream = threading.Thread(
        target=lambda: box.update(
            times=_stream_times(
                [
                    {
                        "role": "user",
                        "content": "Count from one to two hundred in words.",
                    }
                ],
                600,
                started,
            )
        )
    )
    stream.start()
    started.wait(timeout=300)
    t_sent = time.perf_counter()
    if extra_messages is not None:
        box["extra"] = _post(extra_messages)
    stream.join(timeout=900)
    times = box.get("times") or []
    gaps = [b - a for a, b in zip(times, times[1:]) if b > t_sent]
    gaps.sort()
    return {
        "max_gap_ms": round(1000 * max(gaps), 1) if gaps else None,
        "p50_gap_ms": round(1000 * gaps[len(gaps) // 2], 1) if gaps else None,
        "tokens": len(times),
        "extra": box.get("extra"),
    }


def _status():
    return json.load(urllib.request.urlopen(f"{BASE}/v1/status", timeout=30))


def _start(mode, workdir, extra_env=None):
    home = os.path.join(workdir, f"home-{mode}")
    os.makedirs(home, exist_ok=True)
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        HOME=home,
        VLLM_MLX_BATCHED_SYSTEM_KV="1",
        HF_HOME=os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
        **(extra_env or {}),
    )
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    flag = "--text-only" if mode == "text-only" else "--vision-unified"
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
    return proc, log, log_path


def _stop(proc, log, log_path):
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    globals().setdefault("_LOGS", []).append(open(log_path).read())


def _session(mode, workdir):
    proc, log, log_path = _start(mode, workdir)
    r = {}
    try:
        r["mem_after_load_gb"] = (_status().get("metal") or {}).get("active_memory_gb")
        for name, msgs in TEXT_CASES.items():
            r[name] = _post(msgs)
            r[f"{name}_stream"] = _post(msgs, stream=True)
        r["system_resend"] = _post(TEXT_CASES["system"])
        r["red"] = _post(_img_msg([RED], COLOUR_Q))
        r["blue"] = _post(_img_msg([BLUE], COLOUR_Q), stream=True)
        r["anthropic_text"] = _anthropic(
            "Name three prime numbers and say why 9 is not prime."
        )
        r["anthropic_image"] = _anthropic(ANTHROPIC_IMAGE)
        if mode == "text-only":
            r["health"] = json.load(
                urllib.request.urlopen(f"{BASE}/health", timeout=30)
            )
            return r
        r["video"] = _post(VIDEO)
        r["two"] = _post(
            _img_msg(
                [RED, BLUE],
                "There are two images. Name the colour of the first and of "
                "the second, as 'first: X, second: Y'.",
            )
        )
        later = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": GREEN}},
                    {"type": "text", "text": "Remember this image."},
                ],
            },
            {"role": "assistant", "content": "Okay, I have seen the image."},
            {"role": "user", "content": COLOUR_Q},
        ]
        r["later_turn"] = _post(later)
        box = {}

        def run(key, msgs, stream):
            box[key] = _post(msgs, stream=stream)

        threads = [
            threading.Thread(
                target=run,
                args=("conc_image", _img_msg([RED], "Describe this image."), True),
            ),
            threading.Thread(target=run, args=("conc_text", TEXT_CASES["plain"], True)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=900)
        r.update(box)
        # abort: disconnect mid-stream from an image request
        _disconnect_after_first_chunk(_img_msg([BLUE], "Write a long poem about it."))
        time.sleep(2.0)
        r["after_abort_status"] = _status()
        r["after_abort"] = _post(_img_msg([RED], COLOUR_Q))
        if STAGE >= 3:
            conv = _img_msg([BLUE], COLOUR_Q, system=SYSTEM)
            r["c_first"] = _post(conv)
            r["c_resend"] = _post(conv)
            follow = conv + [
                {"role": "assistant", "content": r["c_first"]["text"]},
                {"role": "user", "content": "Is that a warm or a cold colour?"},
            ]
            r["c_follow"] = _post(follow)
            r["c_other"] = _post(_img_msg([RED], COLOUR_Q, system=SYSTEM))
            r["c_other_resend"] = _post(_img_msg([RED], COLOUR_Q, system=SYSTEM))
        if os.environ.get("E2E_GAPS", "1") == "1":
            nonce = str(time.time())
            r["gap_none"] = _gap_probe(None)
            r["gap_text"] = _gap_probe(
                [
                    {"role": "system", "content": _long_system("t" + nonce)},
                    {"role": "user", "content": "Which bay holds crate 40?"},
                ]
            )
            r["gap_image"] = _gap_probe(
                _img_msg([RED], COLOUR_Q, system=_long_system("i" + nonce))
            )
        r["health"] = json.load(urllib.request.urlopen(f"{BASE}/health", timeout=30))
        r["status"] = _status()
        return r
    finally:
        _stop(proc, log, log_path)


def _ceiling_session(workdir):
    proc, log, log_path = _start(
        "ceiling", workdir, {"VLLM_MLX_MAX_PROMPT_TOKENS": "80"}
    )
    try:
        return {
            "text": _post(TEXT_CASES["plain"]),
            "image": _post(_img_msg([RED], COLOUR_Q)),
        }
    finally:
        _stop(proc, log, log_path)


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="e2e-vision-unified-")
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))

    ref = _session("text-only", workdir)
    uni = _session("vision-unified", workdir)
    ceil = _ceiling_session(workdir)

    for key in ("plain", "plain_stream", "system", "system_stream", "system_resend"):
        same = (
            ref[key]["status"] == uni[key]["status"] == 200
            and ref[key]["text"] == uni[key]["text"]
        )
        check(
            f"text T=0 byte-identical to --text-only: {key}",
            same,
            (
                ""
                if same
                else f"\n      ref: {ref[key]['text']!r}\n      uni: {uni[key]['text']!r}"
            ),
        )
        check(
            f"text cache behaviour identical: {key}",
            ref[key]["cached"] == uni[key]["cached"]
            and ref[key]["prompt"] == uni[key]["prompt"],
            f"cached {uni[key]['cached']}/{uni[key]['prompt']} "
            f"(ref {ref[key]['cached']}/{ref[key]['prompt']})",
        )
    check(
        "re-sent text prompt served from the batched system-KV",
        uni["system_resend"]["cached"] > 0,
        f"{uni['system_resend']['cached']}/{uni['system_resend']['prompt']}",
    )
    check(
        "--text-only: image request is a 400 media_not_supported",
        ref["red"]["status"] == 400
        and "media_not_supported" in ref["red"].get("error", ""),
        f"{ref['red']['status']}",
    )
    check(
        "unified: red image answered red",
        uni["red"]["status"] == 200 and "red" in uni["red"]["text"].lower(),
        repr(uni["red"]["text"][:80]),
    )
    check(
        "unified: blue image answered blue (streamed)",
        uni["blue"]["status"] == 200 and "blue" in uni["blue"]["text"].lower(),
        repr(uni["blue"]["text"][:80]),
    )
    check(
        "unified: image tokens are in prompt_tokens",
        uni["red"].get("prompt", 0) > 60,
        f"prompt_tokens={uni['red'].get('prompt')}",
    )
    two = uni["two"]["text"].lower()
    check(
        "unified: two images, both colours named in order",
        uni["two"]["status"] == 200
        and "red" in two
        and "blue" in two
        and two.index("red") < two.index("blue"),
        repr(uni["two"]["text"][:100]),
    )
    check(
        "unified: image in an earlier turn, question in the last",
        uni["later_turn"]["status"] == 200
        and "green" in uni["later_turn"]["text"].lower(),
        repr(uni["later_turn"]["text"][:80]),
    )
    check(
        "unified: video is a 400 media_not_supported",
        uni["video"]["status"] == 400
        and "media_not_supported" in uni["video"].get("error", ""),
        f"{uni['video']['status']} {uni['video'].get('error', '')[:100]!r}",
    )
    check(
        "Anthropic /v1/messages: text T=0 byte-identical to --text-only",
        ref["anthropic_text"]["status"] == uni["anthropic_text"]["status"] == 200
        and ref["anthropic_text"]["text"] == uni["anthropic_text"]["text"],
        repr(uni["anthropic_text"]["text"][:80]),
    )
    check(
        "Anthropic /v1/messages: image 400 on --text-only, answered blue on unified",
        ref["anthropic_image"]["status"] == 400
        and uni["anthropic_image"]["status"] == 200
        and "blue" in uni["anthropic_image"]["text"].lower(),
        f"{uni['anthropic_image']['text'][:60]!r}",
    )
    ci, ct = uni.get("conc_image") or {}, uni.get("conc_text") or {}
    check(
        "unified: concurrent image + text both finish",
        ci.get("status") == 200 and ct.get("status") == 200 and ci.get("text"),
        f"image {ci.get('status')} text {ct.get('status')}",
    )
    check(
        "unified: concurrent text answer equals the --text-only one",
        ct.get("text") == ref["plain_stream"]["text"],
        repr((ct.get("text") or "")[:80]),
    )
    st_abort = uni["after_abort_status"]
    check(
        "unified: client disconnect mid image stream leaves nothing running",
        (st_abort.get("num_running") or 0) == 0
        and (st_abort.get("num_waiting") or 0) == 0
        and uni["after_abort"]["status"] == 200
        and "red" in uni["after_abort"]["text"].lower(),
        f"running={st_abort.get('num_running')} waiting={st_abort.get('num_waiting')}",
    )
    check(
        "prompt ceiling counts image tokens (80): text ok, image 400",
        ceil["text"]["status"] == 200
        and ceil["image"]["status"] == 400
        and "prompt_too_long" in ceil["image"].get("error", ""),
        f"text {ceil['text']['status']} image {ceil['image']['status']}",
    )
    if STAGE >= 3:
        f, rs, fo = uni["c_first"], uni["c_resend"], uni["c_follow"]
        o, ors = uni["c_other"], uni["c_other_resend"]
        check(
            "stage 3: re-sent image conversation is a cache hit past the image",
            rs["status"] == 200
            and rs["cached"] > f["prompt"] - 30
            and rs["text"] == f["text"],
            f"cached {rs['cached']}/{rs['prompt']} (first cached {f['cached']})",
        )
        check(
            "stage 3: follow-up text turn in the image conversation hits",
            fo["status"] == 200 and fo["cached"] >= f["prompt"] - 30,
            f"cached {fo['cached']}/{fo['prompt']}",
        )
        # the 256 px test image is 64 placeholder tokens: a restore may end
        # at the image's first token at most, never inside or after it
        image_start_max = o["prompt"] - 64 - 2
        check(
            "stage 3: same text, different image never hits past the image",
            o["status"] == 200
            and o["cached"] <= image_start_max
            and "red" in o["text"].lower(),
            f"cached {o['cached']}/{o['prompt']} (image starts at or below "
            f"{image_start_max})",
        )
        check(
            "stage 3: the other image's own re-send hits",
            ors["cached"] > o["prompt"] - 30 and ors["text"] == o["text"],
            f"cached {ors['cached']}/{ors['prompt']}",
        )
    if "gap_image" in uni:
        for k in ("gap_none", "gap_text", "gap_image"):
            g = uni[k]
            print(
                f"INFO  {k}: longest inter-token gap {g['max_gap_ms']} ms, "
                f"median {g['p50_gap_ms']} ms over {g['tokens']} tokens "
                f"(extra prompt {((g['extra'] or {}).get('prompt'))} tokens)"
            )
        gi = uni["gap_image"]["extra"] or {}
        check(
            "long system prompt + image while a stream decodes: answered red",
            gi.get("status") == 200 and "red" in gi.get("text", "").lower(),
            repr(gi.get("text", "")[:60]),
        )
    check(
        "/health: --text-only says llm; unified says mllm and serves both",
        ref.get("health", {}).get("model_type") == "llm"
        and uni["health"].get("model_type") == "mllm"
        and uni["health"].get("serves") == ["text", "image"],
        f"ref {ref.get('health', {}).get('model_type')} uni "
        f"{uni['health'].get('model_type')} {uni['health'].get('serves')}",
    )
    vu = (uni["status"] or {}).get("vision_unified") or {}
    check(
        "/v1/status has the vision_unified counters",
        (vu.get("media_requests") or 0) >= 7 and vu.get("media_errors") == 0,
        json.dumps(vu),
    )
    print(
        f"INFO  metal active after load: text-only {ref['mem_after_load_gb']} GB, "
        f"vision-unified {uni['mem_after_load_gb']} GB"
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
