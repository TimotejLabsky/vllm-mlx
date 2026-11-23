"""Verify-width probe (survey 2026-11, issue #110): what does ONE target
forward cost at S = 1, 2, 4, 8, 16 positions, batch 1, on the text model?

Gate for model-free (suffix / n-gram) drafting, which verifies wide blocks.
Same method as the 2026-11-17 forward-shape spike: prefill a random-token
context, then time ``body + head + argmax`` at S positions, undoing the
positions after each rep so the context stays fixed; median of ``--reps``
after 4 warm-up reps.

Two forward modes:
  plain     - mlx-lm's own forward (GatedDeltaNet runs all S positions in
              one kernel call); what a verify without partial-accept rollback
              would cost.
  recording - the fork's #160 recording forward (``text_mtp``): the
              recurrence is stepped one position at a time so a partial accept
              can select a state. This is the rollback machinery a wide-verify
              drafter would reuse.

Between configurations the script records ``num_running`` of every loaded
vllm-mlx route through llama-swap (``--swap``) and re-runs a configuration
whose window saw traffic (up to 3 tries). A route is only queried when
``/running`` already lists it, so the probe never loads one.

usage: bench_verify_width.py <model> [--ctx 400 4000 16000]
       [--widths 1 2 4 8 16] [--reps 24] [--swap http://localhost:8080]

Run on the Studio from rsync'd source: PYTHONPATH=<src> HF_HUB_OFFLINE=1.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx import text_mtp
from vllm_mlx.utils.tokenizer import load_model_with_fallback


def _get(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def route_load(swap):
    """{model: (state, num_running, num_waiting)} for loaded routes."""
    if not swap:
        return {}
    out = {}
    try:
        running = _get(f"{swap}/running").get("running", [])
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}
    for r in running:
        name, state = r.get("model"), r.get("state")
        nr = nw = None
        if "vllm_mlx" in (r.get("cmd") or "") and state == "ready":
            try:
                st = _get(f"{swap}/upstream/{name}/v1/status")
                nr, nw = st.get("num_running"), st.get("num_waiting")
            except Exception:  # noqa: BLE001
                pass
        out[name] = (state, nr, nw)
    return out


def busy(load):
    if "_error" in load:
        return True
    return any((v[1] or 0) + (v[2] or 0) > 0 for v in load.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, nargs="+", default=[400, 4000, 16000])
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--modes", nargs="+", default=["plain", "recording"])
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--swap", default="")
    args = ap.parse_args()

    model, _ = load_model_with_fallback(args.model)
    model.train(False)
    lm = getattr(model, "language_model", model)
    assert text_mtp._supports_recording(lm), "not an mlx-lm GatedDeltaNet target"
    from mlx_lm.models import qwen3_5 as q35

    text_mtp._orig["gdn_call"] = q35.GatedDeltaNet.__call__
    q35.GatedDeltaNet.__call__ = text_mtp._recording_gdn_call  # inert unless active

    def head(h):
        return text_mtp._head(lm, h)

    def prefill(ctx):
        ids = mx.random.randint(1000, 50000, (1, ctx))
        cache = make_prompt_cache(model)
        for s in range(0, ctx, 2048):
            h = lm.model(ids[:, s : s + 2048], cache)
            mx.eval(h, [c.state for c in cache])
        return cache

    def time_one(cache, S, mode):
        x = mx.random.randint(1000, 50000, (1, S))
        snap = [(list(c.cache) if not c.is_trimmable() else None) for c in cache]
        times = []
        for i in range(args.reps + 4):
            t = time.perf_counter()
            text_mtp._REC["active"] = mode == "recording" and S > 1
            try:
                h = lm.model(x, cache)
            finally:
                text_mtp._REC["active"] = False
            y = mx.argmax(head(h), axis=-1)
            mx.eval(y)
            dt = time.perf_counter() - t
            if i >= 4:
                times.append(dt)
            for c, s in zip(cache, snap):  # undo, so the context stays fixed
                if s is not None:
                    c.cache = list(s)
                    if hasattr(c, "_text_mtp_spec"):
                        del c._text_mtp_spec
                else:
                    c.trim(S)
        return times

    results = []
    print(
        f"mlx {mx.__version__} peak-before {mx.get_peak_memory() / 1e9:.1f} GB",
        flush=True,
    )
    for ctx in args.ctx:
        cache = prefill(ctx)
        base = {}
        for mode in args.modes:
            for S in args.widths:
                if mode == "recording" and S == 1:
                    continue  # identical to plain S=1
                for attempt in range(3):
                    before = route_load(args.swap)
                    times = time_one(cache, S, mode)
                    after = route_load(args.swap)
                    polluted = (
                        busy(before) or busy(after) or before.keys() != after.keys()
                    )
                    if not polluted:
                        break
                    print(
                        f"  polluted window ctx={ctx} {mode} S={S}: {before} -> {after}",
                        flush=True,
                    )
                    time.sleep(20)
                ms = sorted(1000 * t for t in times)
                med = statistics.median(ms)
                if S == 1:
                    base[ctx] = med
                b = base.get(ctx, med)
                row = {
                    "ctx": ctx,
                    "mode": mode,
                    "S": S,
                    "median_ms": round(med, 2),
                    "min_ms": round(ms[0], 2),
                    "max_ms": round(ms[-1], 2),
                    "n": len(ms),
                    "x_of_S1": round(med / b, 3),
                    "positions_per_s": round(1000 * S / med, 1),
                    "polluted": polluted,
                    "routes_before": before,
                    "routes_after": after,
                }
                results.append(row)
                print(
                    f"ctx={ctx:6d} {mode:9s} S={S:2d} {med:7.1f} ms "
                    f"[{ms[0]:.1f}..{ms[-1]:.1f}] x{med / b:5.2f} "
                    f"{1000 * S / med:6.1f} pos/s n={len(ms)}"
                    f"{' POLLUTED' if polluted else ''} routes={after}",
                    flush=True,
                )
        del cache
        mx.clear_cache()
    print(f"peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    print("JSON " + json.dumps(results), flush=True)
    print("VERIFY_WIDTH_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
