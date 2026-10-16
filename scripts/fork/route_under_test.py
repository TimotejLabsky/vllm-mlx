#!/usr/bin/env python3
"""Start ONE llama-swap route's exact command + env on a spare port, against
a source tree of your choice, then tear it down. For checking a rebase or a
patch on the real route config without touching the live route.

    python3 route_under_test.py ROUTE --src /tmp/rb1014 [--port 8771]
        [--drop-flag --continuous-batching] [--drop-flag --text-only]
        [--ssd-dir /tmp/x] [--python /tmp/venv/bin/python] -- CHECK_CMD ...

The route's ``cmd`` and ``env`` come from ~/llm-stack/config.yaml; the
route-log-wrapper is skipped (the log goes to --log), ``${PORT}`` becomes
--port, PYTHONPATH is --src, and VLLM_MLX_SSD_SYSTEM_KV_DIR is redirected to
--ssd-dir (default: a fresh temp dir) so test spills never mix with the live
store. CHECK_CMD runs with SERVER_URL set; its exit code is ours. Stdlib only.
"""

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request


def route_spec(name):
    s = open(os.path.expanduser("~/llm-stack/config.yaml")).read()
    m = re.search(r'\n  "%s":\n(.*?)(?=\n  "|\ngroups:|\Z)' % re.escape(name), s, re.S)
    if not m:
        sys.exit(f"route {name!r} not found")
    blk = m.group(1)
    found = re.search(r"cmd: (.*)", blk)
    if not found:
        sys.exit(f"route {name!r} has no cmd")
    cmd = found.group(1).strip()
    env = dict(re.findall(r'^\s*-\s*"([A-Z0-9_]+)=(.*)"\s*$', blk, re.M))
    return cmd, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("route")
    ap.add_argument("--src", required=True)
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--drop-flag", action="append", default=[])
    ap.add_argument("--ssd-dir")
    ap.add_argument("--log")
    ap.add_argument("--python", help="interpreter to run instead of the route's (e.g. a cloned venv)")
    raw = sys.argv[1:]
    cut = raw.index("--") if "--" in raw else len(raw)
    a = ap.parse_args(raw[:cut])
    a.check = raw[cut + 1 :]
    cmd, env = route_spec(a.route)
    argv = shlex.split(cmd.replace("${PORT}", str(a.port)))
    if argv and argv[0].endswith("route-log-wrapper.sh"):
        argv = argv[1:]
    if a.python and argv and argv[0].endswith("/python"):
        argv[0] = a.python
    for flag in a.drop_flag:
        argv = [x for x in argv if x != flag]
    ssd = a.ssd_dir or tempfile.mkdtemp(prefix="rut-ssd-")
    full_env = dict(os.environ, **env)
    full_env.update(PYTHONPATH=a.src, HF_HUB_OFFLINE="1", VLLM_MLX_SSD_SYSTEM_KV_DIR=ssd)
    log_path = a.log or tempfile.mktemp(prefix=f"rut-{a.route}-", suffix=".log")
    log = open(log_path, "w")
    print(f"[rut] {a.route} on :{a.port} src={a.src} ssd={ssd} log={log_path}", flush=True)
    print(f"[rut] argv: {' '.join(argv[1:])[:300]}", flush=True)
    # Neutral cwd: ``python -m`` puts the cwd AHEAD of PYTHONPATH, so a run
    # from inside a checkout would import that tree instead of --src.
    proc = subprocess.Popen(
        argv, env=full_env, stdout=log, stderr=subprocess.STDOUT, cwd=tempfile.gettempdir()
    )
    url = f"http://127.0.0.1:{a.port}"
    t0 = time.time()
    try:
        while True:
            if proc.poll() is not None:
                print(open(log_path).read()[-3000:])
                sys.exit("[rut] server exited during startup")
            try:
                if urllib.request.urlopen(f"{url}/v1/models", timeout=2).status == 200:
                    break
            except Exception:
                time.sleep(0.5)
            if time.time() - t0 > 600:
                sys.exit("[rut] startup timeout")
        print(f"[rut] ready in {time.time() - t0:.1f}s", flush=True)
        check = a.check
        rc = subprocess.call(check, env=dict(os.environ, SERVER_URL=url, RUT_SSD=ssd)) if check else 0
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    text = open(log_path).read()
    bad = [l for l in text.splitlines() if re.search(r"Traceback|no Stream\(|generation_error_recovery", l)]
    for key in ("Model route:", "MLLM text routing", "SSD persistence enabled", "Keeping model_type"):
        for l in text.splitlines():
            if key in l:
                print(f"[rut] log: {l.strip()[:200]}")
                break
    print(f"[rut] log errors: {len(bad)}" + (f" e.g. {bad[0][:160]}" if bad else ""))
    sys.exit(rc or (1 if bad else 0))


if __name__ == "__main__":
    main()
