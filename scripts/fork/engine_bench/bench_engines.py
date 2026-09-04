#!/usr/bin/env python3
"""Engine comparison bench for the Mac Studio (2026-09-04).

Engines: mlxlm | upstream-simple | upstream-batched | fork-simple |
         fork-batched | omlx | llamacpp | ollama
Models:  moe   = mlx-community/Qwen3.6-35B-A3B-4bit-DWQ
         dense8= mlx-community/Qwen3.8-27B-8bit
         dense4= mlx-community/Qwen3.8-27B-4bit (quant pair for GGUF Q4_K_XL)

Per cell: standard vLLM benchmark_serving (serial c=1 and c=4) + custom
probes (load time, 8K cold prefill, warm-prefix TTFT, peak RSS by pgid).
Stdlib only. Incremental results to results.jsonl. Run under nohup.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request
import uuid

BASE = os.path.expanduser("~/bench-2026-09-04")
PORT = 5899
RESULTS = os.path.join(BASE, "results.jsonl")
LOGDIR = os.path.join(BASE, "logs")
CLIENT_PY = os.path.join(BASE, "client-env/bin/python")
BENCH_SERVING = os.path.join(BASE, "client/benchmark_serving.py")

FORK_PY = "/Users/ai/vllm-mlx-env/bin/python"
UP_PY = os.path.join(BASE, "upstream-env/bin/python")
OMLX = "/Users/ai/omlx-env/bin/omlx"
LLAMA_SERVER = "/opt/homebrew/bin/llama-server"
OLLAMA = "/opt/homebrew/bin/ollama"
GGUF = "/Users/ai/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"
OLLAMA_MODEL = "qwen38-27b-q4"
OMLX_MODEL_DIR = os.path.join(BASE, "omlx-models")

MODELS = {
    "moe": {
        "repo": "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ",
        "max_num_seqs": 8, "kv_budget": "8192", "pad_waste": "4096",
        "max_prompt": "131072",
    },
    "dense8": {
        "repo": "mlx-community/Qwen3.8-27B-8bit",
        "max_num_seqs": 4, "kv_budget": "4096", "pad_waste": "2048",
        "max_prompt": "196608",
    },
    "dense4": {
        "repo": "mlx-community/Qwen3.8-27B-4bit",
        "max_num_seqs": 4, "kv_budget": "4096", "pad_waste": "2048",
        "max_prompt": "196608",
    },
}

def engine_spec(ekey, mkey):
    m = MODELS[mkey]
    repo = m["repo"]
    nseq = m["max_num_seqs"]
    batched_env = {
        "VLLM_MLX_BATCHED_SYSTEM_KV": "1",
        "VLLM_MLX_BATCHED_KV_BUDGET_MB": m["kv_budget"],
        "VLLM_MLX_BATCHED_PAD_WASTE_MB": m["pad_waste"],
        "VLLM_MLX_BATCHED_MEM_WATERMARK_PCT": "80",
        "VLLM_MLX_BATCHED_MAX_QUEUE": "8",
        "VLLM_MLX_MAX_PROMPT_TOKENS": m["max_prompt"],
        "VLLM_MLX_SYSTEM_KV_RAM_MB": "6144",
    }
    if mkey == "moe":
        batched_env["VLLM_MLX_MOE_GATEUP_FUSION"] = "1"
    specs = {
        "mlxlm": dict(
            cmd=[FORK_PY, "-m", "mlx_lm.server", "--model", repo,
                 "--host", "127.0.0.1", "--port", str(PORT)],
            env={}, port=PORT),
        "upstream-simple": dict(
            cmd=[UP_PY, "-m", "vllm_mlx.cli", "serve", repo,
                 "--host", "127.0.0.1", "--port", str(PORT)],
            env={}, port=PORT),
        "upstream-batched": dict(
            cmd=[UP_PY, "-m", "vllm_mlx.cli", "serve", repo,
                 "--host", "127.0.0.1", "--port", str(PORT),
                 "--continuous-batching",
                 "--max-num-seqs", str(nseq)],
            env={}, port=PORT),
        "fork-simple": dict(
            cmd=[FORK_PY, "-m", "vllm_mlx.cli", "serve", repo,
                 "--host", "127.0.0.1", "--port", str(PORT)],
            env={"VLLM_MLX_SYSTEM_KV_RAM_MB": "6144"}, port=PORT),
        "fork-batched": dict(
            cmd=[FORK_PY, "-m", "vllm_mlx.cli", "serve", repo,
                 "--host", "127.0.0.1", "--port", str(PORT),
                 "--continuous-batching", "--text-only",
                 "--max-num-seqs", str(nseq)],
            env=batched_env, port=PORT),
        "omlx": dict(
            cmd=[OMLX, "serve", "--model-dir", OMLX_MODEL_DIR,
                 "--port", str(PORT), "--api-key", "bench"],
            env={}, port=PORT),
        "llamacpp": dict(
            cmd=[LLAMA_SERVER, "-m", GGUF, "--host", "127.0.0.1",
                 "--port", str(PORT), "-ngl", "99", "-c", "65536",
                 "-np", str(nseq), "-fa", "on", "--jinja",
                 "--cache-reuse", "256"],
            env={}, port=PORT),
        "ollama": dict(
            cmd=[OLLAMA, "serve"],
            env={"OLLAMA_HOST": f"127.0.0.1:{PORT}",
                 "OLLAMA_NUM_PARALLEL": str(nseq),
                 "OLLAMA_FLASH_ATTENTION": "1",
                 "OLLAMA_KEEP_ALIVE": "2h"},
            port=PORT, model_name=OLLAMA_MODEL),
    }
    return specs[ekey]

CELLS = [
    ("mlxlm", "moe"), ("upstream-simple", "moe"), ("upstream-batched", "moe"),
    ("fork-simple", "moe"), ("fork-batched", "moe"), ("omlx", "moe"),
    ("mlxlm", "dense8"), ("upstream-simple", "dense8"),
    ("upstream-batched", "dense8"), ("fork-simple", "dense8"),
    ("fork-batched", "dense8"), ("omlx", "dense8"),
    ("fork-batched", "dense4"), ("llamacpp", "dense4"), ("ollama", "dense4"),
]

_WORDS = ("system memory cache latency throughput kernel scheduler batch "
          "tensor gradient attention token stream buffer socket protocol "
          "packet routing storage index shard replica quorum consensus log "
          "compaction snapshot checkpoint recovery migration allocator page "
          "fault interrupt pipeline register vector matrix quantization "
          "precision bandwidth").split()

def long_document(target_words=6200, seed=7):
    out, x = [], seed
    for i in range(target_words):
        x = (x * 1103515245 + 12345) % (2 ** 31)
        out.append(_WORDS[x % len(_WORDS)])
        if i % 13 == 12:
            out.append(".")
    return " ".join(out)

LONG_DOC = long_document()

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def url_for(port):
    return f"http://127.0.0.1:{port}"

def get_model_name(port, fallback):
    try:
        req = urllib.request.Request(url_for(port) + "/v1/models",
                                     headers={"Authorization": "Bearer bench"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        ids = [d.get("id") for d in data.get("data", []) if d.get("id")]
        base = fallback.split("/")[-1]
        for i in ids:
            if base in i:
                return i
        if ids:
            return ids[0]
    except Exception:
        pass
    return fallback

def stream_chat(port, model, prompt, max_tokens, timeout=900):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
    req = urllib.request.Request(
        url_for(port) + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer bench"})
    t0 = time.monotonic()
    t_first = t_last = None
    chunks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if not ch:
                continue
            delta = ch[0].get("delta") or {}
            text = (delta.get("content") or "") + \
                   (delta.get("reasoning_content") or "") + \
                   (delta.get("reasoning") or "")
            if text:
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                t_last = now
                chunks += 1
    ttft = (t_first - t0) if t_first else None
    decode_tps = None
    if t_first and t_last and t_last > t_first and chunks > 1:
        decode_tps = round((chunks - 1) / (t_last - t_first), 2)
    return {"ttft": round(ttft, 3) if ttft else None,
            "decode_tps": decode_tps, "chunks": chunks,
            "total_s": round(time.monotonic() - t0, 2), "usage": usage}

def llama_swap_unload():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/unload",
                                    timeout=60) as r:
            r.read()
    except Exception as e:
        log(f"  llama-swap unload failed: {e}")

class SwapWatchdog(threading.Thread):
    """Flags any production model llama-swap loads mid-cell."""
    def __init__(self):
        super().__init__(daemon=True)
        self.seen = set()
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.wait(20.0):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8080/running",
                                            timeout=10) as r:
                    data = json.loads(r.read())
                for m in data.get("running", []):
                    self.seen.add(m.get("model"))
            except Exception:
                pass

class RssSampler(threading.Thread):
    """Peak RSS summed over the server's process group (catches ollama's
    runner children)."""
    def __init__(self, pgid):
        super().__init__(daemon=True)
        self.pgid = pgid
        self.peak_kb = 0
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.wait(1.0):
            try:
                out = subprocess.run(["ps", "ax", "-o", "pgid=,rss="],
                                     capture_output=True, text=True,
                                     timeout=10)
                total = 0
                for ln in out.stdout.splitlines():
                    parts = ln.split()
                    if len(parts) == 2 and parts[0] == str(self.pgid):
                        total += int(parts[1])
                self.peak_kb = max(self.peak_kb, total)
            except Exception:
                pass

def wait_ready(proc, port, model_fallback, deadline_s=900):
    t0 = time.monotonic()
    bind_s = None
    while time.monotonic() - t0 < deadline_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                bind_s = time.monotonic() - t0
                break
        except OSError:
            time.sleep(0.5)
    if bind_s is None:
        raise RuntimeError("bind timeout")
    model = get_model_name(port, model_fallback)
    while time.monotonic() - t0 < deadline_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        try:
            stream_chat(port, model, "Say OK.", 4, timeout=600)
            return bind_s, time.monotonic() - t0, model
        except Exception:
            time.sleep(2)
    raise RuntimeError("ready timeout")

def run_benchmark_serving(port, model, tokenizer, num_prompts, concurrency,
                          tag, cellname):
    outfile = os.path.join(LOGDIR, f"bs-{cellname}-{tag}.json")
    cmd = [CLIENT_PY, BENCH_SERVING,
           "--backend", "openai-chat",
           "--base-url", url_for(port),
           "--endpoint", "/v1/chat/completions",
           "--model", model,
           "--tokenizer", tokenizer,
           "--dataset-name", "random",
           "--random-input-len", "1024",
           "--random-output-len", "256",
           "--num-prompts", str(num_prompts),
           "--max-concurrency", str(concurrency),
           "--seed", "42",
           "--save-result", "--result-filename", outfile,
           "--disable-tqdm"]
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["OPENAI_API_KEY"] = "bench"
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                       env=env)
    if r.returncode != 0 or not os.path.exists(outfile):
        return {"error": (r.stderr or r.stdout)[-2000:]}
    with open(outfile) as f:
        d = json.load(f)
    keep = {k: d.get(k) for k in (
        "completed", "duration", "request_throughput", "output_throughput",
        "total_token_throughput", "mean_ttft_ms", "median_ttft_ms",
        "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "p99_itl_ms")}
    return keep

def run_cell(ekey, mkey):
    cellname = f"{ekey}-{mkey}"
    spec = engine_spec(ekey, mkey)
    repo = MODELS[mkey]["repo"]
    # tokenizer for counting: MLX repo (GGUF cells use the 4bit repo's tokenizer)
    tokenizer = repo if not ekey in ("llamacpp", "ollama") else \
        MODELS["dense4"]["repo"]
    log(f"=== CELL {cellname} ===")
    subprocess.run(["pkill", "-f", f"port {PORT}"], capture_output=True)
    subprocess.run(["pkill", "-f", f"--port {PORT}"], capture_output=True)
    subprocess.run(["pkill", "-x", "ollama"], capture_output=True)
    subprocess.run(["pkill", "-f", "omlx-server"], capture_output=True)
    llama_swap_unload()
    watchdog = SwapWatchdog()
    watchdog.start()
    time.sleep(3)
    env = dict(os.environ)
    env.update(spec["env"])
    env["PYTHONUNBUFFERED"] = "1"
    logf = open(os.path.join(LOGDIR, f"{cellname}.server.log"), "w")
    logf.write(" ".join(spec["cmd"]) + "\n" + json.dumps(spec["env"]) + "\n\n")
    logf.flush()
    proc = subprocess.Popen(spec["cmd"], env=env, stdout=logf,
                            stderr=subprocess.STDOUT, start_new_session=True)
    res = {"cell": cellname, "engine": ekey, "model": mkey,
           "model_repo": repo, "ts": time.strftime("%F %T"), "notes": []}
    sampler = None
    port = spec["port"]
    try:
        bind_s, ready_s, model = wait_ready(
            proc, port, spec.get("model_name", repo))
        res["bind_s"] = round(bind_s, 2)
        res["load_s"] = round(ready_s, 2)
        res["served_model"] = model
        log(f"  ready in {ready_s:.1f}s (bind {bind_s:.1f}s) model={model}")
        sampler = RssSampler(proc.pid)
        sampler.start()

        # warmup
        stream_chat(port, model, "Warmup question: what is a mutex?", 64)

        # standard bench: serial then 4-way
        res["serving_c1"] = run_benchmark_serving(
            port, model, tokenizer, 6, 1, "c1", cellname)
        log(f"  c1: {json.dumps(res['serving_c1'])[:220]}")
        res["serving_c4"] = run_benchmark_serving(
            port, model, tokenizer, 12, 4, "c4", cellname)
        log(f"  c4: {json.dumps(res['serving_c4'])[:220]}")

        # cold 8K prefill x2 (cache-busted)
        longs = []
        for i in range(2):
            p = (f"RUN {uuid.uuid4().hex} :: The following is a technical "
                 f"document.\n\n{LONG_DOC}\n\nSummarize the above document "
                 f"in one sentence.")
            r = stream_chat(port, model, p, 32)
            pt = (r.get("usage") or {}).get("prompt_tokens")
            if pt and r["ttft"]:
                r["prefill_tps"] = round(pt / r["ttft"], 1)
            longs.append(r)
            log(f"  cold8k #{i}: ttft={r['ttft']}s prompt_tokens={pt} "
                f"prefill={r.get('prefill_tps')}")
        res["long_cold"] = longs

        # warm-prefix: fixed long prompt, 1 store + 2 repeats
        fixed = (f"WARMKEY :: The following is a technical document.\n\n"
                 f"{LONG_DOC}\n\nSummarize the above document in one sentence.")
        res["warm_store"] = stream_chat(port, model, fixed, 32)
        warms = []
        for i in range(2):
            r = stream_chat(port, model, fixed, 32)
            warms.append(r)
            log(f"  warm #{i}: ttft={r['ttft']}s usage={r.get('usage')}")
        res["warm"] = warms
    except Exception as e:
        res["error"] = str(e)
        log(f"  CELL FAILED: {e}")
    finally:
        watchdog.stop_flag.set()
        if watchdog.seen:
            res["notes"].append(
                f"CONTAMINATED: llama-swap loaded {sorted(watchdog.seen)} "
                f"during this cell")
        if sampler:
            sampler.stop_flag.set()
            res["peak_rss_mb"] = round(sampler.peak_kb / 1024)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
        logf.close()
    with open(RESULTS, "a") as f:
        f.write(json.dumps(res) + "\n")
    log(f"  cell done, peak_rss={res.get('peak_rss_mb')}MB")
    time.sleep(5)

def prewarm_model_files(mkey):
    repo = MODELS[mkey]["repo"].replace("/", "--")
    d = os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo}")
    log(f"prewarming page cache for {repo}")
    subprocess.run(
        f"find {d} -name '*.safetensors' -exec cat {{}} + > /dev/null",
        shell=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="cell like fork-batched/moe")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    cells = CELLS
    if args.list:
        for e, m in cells:
            print(f"{e}/{m}")
        return
    if args.only:
        e, m = args.only.split("/")
        cells = [(e, m)]
    os.makedirs(LOGDIR, exist_ok=True)
    last_model = None
    for e, m in cells:
        if m != last_model and e not in ("llamacpp", "ollama"):
            prewarm_model_files(m)
            last_model = m
        run_cell(e, m)
    log("ALL DONE")

if __name__ == "__main__":
    main()
