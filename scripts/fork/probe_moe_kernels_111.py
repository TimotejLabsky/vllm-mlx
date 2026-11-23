#!/usr/bin/env python
"""Probe for fork issue #111: upstream mlx-lm MoE single-token kernels vs #96.

A/B of three open mlx-lm PRs against the fork's MoE gate/up fusion
(PATCHES.md #96, ``vllm_mlx/moe_fusion.py``) at batch-1 decode:

- #2053 (pierreablin @ 8467a54): ``SwitchGLU.combine`` - one token's expert
  MLP + routing-weighted sum in two Metal JIT kernels.
- #2054 (pierreablin @ 04ecae9): ``softmax_topk`` - router softmax + top-k
  (+ renorm) in one kernel.
- #2044 (Tigerly1 @ 61accb9): grouped-expert kernels. NOT measured: at that
  head ``switch_glu_by_expert`` returns None for T < 2 ("one token is left to
  the generic path") and for T*k/E > 24, so at batch-1 decode and at
  2048-token prefill chunks (2048*8/256 = 64) it never runs on Qwen3.6-35B-A3B.

The kernels are an overlay: the PR files are loaded from ``--kernels DIR``
by path (nothing in the venv or the pinned mlx-lm is touched), and the
Qwen3-Next MoE block's ``__call__`` is monkey-patched in this process only.
Fetch the two files once (laptop or Studio)::

    mkdir -p /tmp/i111/k
    curl -sL -o /tmp/i111/k/switch_glu.py https://raw.githubusercontent.com/\
pierreablin/mlx-lm/8467a54e579d2120fd90a25c27d1b54e79e1c83a/mlx_lm/kernels/switch_glu.py
    curl -sL -o /tmp/i111/k/softmax_topk.py https://raw.githubusercontent.com/\
pierreablin/mlx-lm/04ecae93f511097900a0c0dd599cedcc5eabb679/mlx_lm/kernels/softmax_topk.py

Variants (all run with the deployed MLX_MAX_OPS_PER_BUFFER=1000 /
MLX_MAX_MB_PER_BUFFER=200 unless the caller overrides them):

    a   stock mlx-lm (no fusion)
    b   fork #96 gate/up fusion (as deployed)
    c   upstream #2053 combine + #2054 softmax_topk, no #96
    c1  #2053 combine only
    c2  #2054 softmax_topk only
    d   #96 + #2053 (fused-layout gate/up kernel) + #2054 - the coherent combo
    e   #96 + #2054 only

For ``d`` the #2053 gate/up kernel is re-pointed at #96's fused ``[up; gate]``
tensor (row offset N for gate) so no second copy of the expert weights is
needed; the arithmetic is the same as ``c``'s.

Modes::

    # kernel numerics on the real model's layers (no generation)
    python probe_moe_kernels_111.py selftest --model M --kernels DIR
    # one timed process (warm-up + REPS timed runs), appends JSONL
    python probe_moe_kernels_111.py run --model M --kernels DIR --variant b --out R.jsonl
    # interleaved rounds, one process per (round, variant), live-route snapshots
    python probe_moe_kernels_111.py drive --model M --kernels DIR --rounds 6 \
        --variants a,b,c,d --out R.jsonl
    # medians / min-max / T=0 identity
    python probe_moe_kernels_111.py report R.jsonl

``drive`` snapshots every running vllm-mlx route behind llama-swap
(``/running`` then ``/upstream/<model>/v1/status``; it never queries a route
that is not already running, so it never loads one) before and after each
process and marks the sample polluted if any route was busy or stepped.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

VARIANTS = {
    #       fusion  combine topk
    "a": (False, False, False),
    "b": (True, False, False),
    "c": (False, True, True),
    "c1": (False, True, False),
    "c2": (False, False, True),
    "d": (True, True, True),
    "e": (True, False, True),
}

DEPLOYED_ENV = {"MLX_MAX_OPS_PER_BUFFER": "1000", "MLX_MAX_MB_PER_BUFFER": "200"}


def _load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- overlay


class Overlay:
    def __init__(self, kernels_dir):
        import mlx.core as mx

        self.mx = mx
        self.sg = _load_file(os.path.join(kernels_dir, "switch_glu.py"), "k2053")
        self.st = _load_file(os.path.join(kernels_dir, "softmax_topk.py"), "k2054")
        self._fused_gate_up = self._make_fused_gate_up_kernel()
        self._fused_fns = {}

    def _make_fused_gate_up_kernel(self):
        """#2053's gate/up kernel over #96's fused [up; gate] rows (2N per
        expert, gate at row offset N). Same per-row arithmetic."""
        src = self.sg._GATE_UP_SOURCE
        repl = [
            (
                "size_t base = size_t(inds[slot]) * N + row0;",
                "size_t base = size_t(inds[slot]) * (2 * N) + row0;",
            ),
            (
                "size_t g = row * NG + k0 / GS;",
                "size_t g = row * NG + k0 / GS;\n"
                "            size_t grow = row + N;\n"
                "            size_t gg = grow * NG + k0 / GS;",
            ),
            (
                "ga[r] += float(gs[g]) * qdot16<BITS>(gq + row * RB + k0 * BITS / 8, xs)"
                " + float(gb[g]) * xsum;",
                "ga[r] += float(gs[gg]) * qdot16<BITS>(gq + grow * RB + k0 * BITS / 8, xs)"
                " + float(gb[gg]) * xsum;",
            ),
        ]
        for old, new in repl:
            if src.count(old) != 1:
                raise RuntimeError(f"#2053 kernel source changed; cannot patch: {old!r}")
            src = src.replace(old, new)
        return self.mx.fast.metal_kernel(
            name="switch_glu_gate_up_fused96",
            input_names=["x", "inds", "weights", "gw", "gs", "gb", "uw", "us", "ub"],
            output_names=["h"],
            header=self.sg._HEADER,
            source=src,
        )

    def _fused_fn(self, act, group_size, bits):
        key = (act, group_size, bits)
        if key in self._fused_fns:
            return self._fused_fns[key]
        mx, sg, kern = self.mx, self.sg, self._fused_gate_up

        def combine(x, inds, weights, fw, fs, fb, dw, ds, db):
            K, N, D = inds.size, fw.shape[1] // 2, x.shape[-1]
            inds = inds.reshape(-1).astype(mx.uint32)
            weights = weights.reshape(-1)
            out_dtype = mx.result_type(x.dtype, weights.dtype)
            template = [
                ("T", x.dtype),
                ("O", out_dtype),
                ("ACT", act),
                ("D", D),
                ("N", N),
                ("K", K),
                ("GS", group_size),
                ("BITS", bits),
            ]
            rps, spt = sg._GATE_UP_SHAPE
            tpe = -(-N // (rps * spt))
            h = kern(
                inputs=[x, inds, weights, fw, fs, fb, fw, fs, fb],
                template=template + [("RPS", rps), ("SPT", spt)],
                grid=(K * tpe * 32 * spt, 1, 1),
                threadgroup=(32 * spt, 1, 1),
                output_shapes=[(K, N)],
                output_dtypes=[x.dtype],
            )[0]
            rps, spt = sg._DOWN_SHAPE
            return sg._down_kernel(
                inputs=[h, inds, dw, ds, db],
                template=template + [("RPS", rps), ("SPT", spt)],
                grid=(-(-D // (rps * spt)) * 32 * spt, 1, 1),
                threadgroup=(32 * spt, 1, 1),
                output_shapes=[x.shape],
                output_dtypes=[out_dtype],
            )[0]

        fn = mx.compile(combine)
        self._fused_fns[key] = fn
        return fn

    def combine(self, glu, x, inds, scores):
        """(glu(x, inds) * scores[..., None]).sum(-2), via #2053 where it applies."""
        if hasattr(glu, "gate_up_proj"):  # fork #96 FusedSwitchGLU
            fp, dp = glu.gate_up_proj, glu.down_proj
            ok = (
                x.size == x.shape[-1]
                and x.dtype == self.mx.bfloat16
                and getattr(fp, "mode", "affine") == "affine"
                and fp.bits in (2, 3, 4, 5, 6, 8)
                and fp.group_size in (32, 64, 128)
                and (fp.bits, fp.group_size) == (dp.bits, dp.group_size)
                and fp.scales.dtype == x.dtype
                and getattr(fp, "biases", None) is not None
                and "bias" not in fp
                and "bias" not in dp
                and x.shape[-1] % 16 == 0
                and (fp.weight.shape[1] // 2) % 16 == 0
            )
            if ok:
                fn = self._fused_fn(0, fp.group_size, fp.bits)
                return fn(
                    x, inds, scores, fp.weight, fp.scales, fp.biases,
                    dp.weight, dp.scales, dp.biases,
                )  # fmt: skip
            return (glu(x, inds) * scores[..., None]).sum(axis=-2)
        if self.sg.can_combine(glu, x):
            return self.sg.switch_glu_combine(glu, x, inds, scores)
        return (glu(x, inds) * scores[..., None]).sum(axis=-2)


def install(model, kernels_dir, fusion, use_combine, use_topk):
    """Apply one variant to a loaded model. Returns a dict of what was applied."""
    import inspect

    import mlx.core as mx
    from mlx_lm.models import qwen3_next
    from mlx_lm.models.switch_layers import SwiGLU

    info = {"fused": 0}
    if fusion:
        from vllm_mlx.moe_fusion import fuse_moe_gate_up

        info["fused"] = fuse_moe_gate_up(model)
    if not (use_combine or use_topk):
        return info

    blk = qwen3_next.Qwen3NextSparseMoeBlock
    src = inspect.getsource(blk.__call__)
    # The patched __call__ below mirrors the pinned stock body; refuse to run
    # if the pin's body is not the one it mirrors.
    for needle in (
        "mx.softmax(gates, axis=-1, precise=True)",
        "mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]",
        "y = self.switch_mlp(x, inds)",
        "mx.sigmoid(self.shared_expert_gate(x)) * shared_y",
    ):
        if needle not in src:
            raise RuntimeError(f"pinned Qwen3NextSparseMoeBlock changed: {needle!r}")
    ov = Overlay(kernels_dir)
    SwiGLU.metal_act = 0  # #2053 tags SwiGLU so can_combine accepts it

    def moe_call(self, x):
        gates = self.gate(x)
        if use_topk:
            inds, scores = ov.st.softmax_topk(
                gates, top_k=self.top_k, normalize=self.norm_topk_prob
            )
        else:
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            inds = mx.stop_gradient(inds)
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if self.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
        if use_combine:
            y = ov.combine(self.switch_mlp, x, inds, scores)
        else:
            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y

    blk.__call__ = moe_call
    info["combine"] = use_combine
    info["topk"] = use_topk
    return info


# ---------------------------------------------------------------- prompt


def build_prompt(tokenizer, n_tokens):
    """Deterministic ~n_tokens chat prompt: the pinned mlx-lm's own source."""
    import mlx_lm

    root = os.path.dirname(mlx_lm.__file__)
    text = ""
    for f in ("generate.py", "utils.py", "server.py"):
        with open(os.path.join(root, f)) as fh:
            text += fh.read()
    ids = tokenizer.encode(text)[: n_tokens - 64]
    body = tokenizer.decode(ids)
    msgs = [
        {
            "role": "user",
            "content": "Explain in detail what this code does, function by "
            "function:\n\n" + body,
        }
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
    return tokenizer.encode(prompt, add_special_tokens=False)


def timed_generate(model, prompt_ids, gen):
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    prompt = mx.array(prompt_ids)
    toks = []
    t0 = time.perf_counter()
    t_first = None
    for tok, _ in generate_step(prompt, model, max_tokens=gen):
        toks.append(int(tok))
        if t_first is None:
            t_first = time.perf_counter()
    t_end = time.perf_counter()
    return {
        "prompt_tokens": len(prompt_ids),
        "prefill_s": t_first - t0,
        "prefill_tps": len(prompt_ids) / (t_first - t0),
        "decode_tps": (len(toks) - 1) / (t_end - t_first),
        "gen_tokens": len(toks),
        "tokens": toks,
        "peak_gb": mx.get_peak_memory() / 1e9,
    }


# ---------------------------------------------------------------- modes


def cmd_run(args):
    import mlx.core as mx
    from mlx_lm import load

    fusion, comb, topk = VARIANTS[args.variant]
    t = time.perf_counter()
    model, tok = load(args.model)
    load_s = time.perf_counter() - t
    info = install(model, args.kernels, fusion, comb, topk)
    prompt_ids = build_prompt(tok, args.prompt_tokens)
    timed_generate(model, prompt_ids, 32)  # warm-up: JIT + kernel caches
    mx.clear_cache()
    mx.reset_peak_memory()
    for rep in range(args.reps):
        r = timed_generate(model, prompt_ids, args.gen)
        mx.clear_cache()
        r.update(
            variant=args.variant,
            rep=rep,
            round=args.round,
            load_s=round(load_s, 1),
            applied=info,
            env={k: os.environ.get(k) for k in DEPLOYED_ENV},
            ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        with open(args.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(
            f"[{args.variant} r{args.round} rep{rep}] prefill {r['prefill_tps']:.1f} "
            f"tok/s  decode {r['decode_tps']:.2f} tok/s  peak {r['peak_gb']:.1f} GB",
            flush=True,
        )


def cmd_selftest(args):
    """Per-layer numerics on the real weights: kernels vs the stock ops."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models import qwen3_next
    from vllm_mlx.moe_fusion import FusedSwitchGLU

    model, _ = load(args.model)
    ov = Overlay(args.kernels)
    from mlx_lm.models.switch_layers import SwiGLU

    SwiGLU.metal_act = 0
    blocks = [
        m
        for _, m in model.named_modules()
        if isinstance(m, qwen3_next.Qwen3NextSparseMoeBlock)
    ]
    print(f"{len(blocks)} MoE blocks")
    mx.random.seed(0)
    D = blocks[0].switch_mlp.down_proj.weight.shape[1]
    worst = {"combine": 0.0, "fused": 0.0, "fused_vs_combine": 0.0}
    topk_set_mismatch = 0
    topk_w_maxdiff = 0.0
    n_cases = 0
    for li, b in enumerate(blocks):
        fused = FusedSwitchGLU(b.switch_mlp)
        for s in range(args.cases):
            x = (mx.random.normal((1, 1, D)) * 0.5).astype(mx.bfloat16)
            gates = b.gate(x)
            g = mx.softmax(gates, axis=-1, precise=True)
            k = b.top_k
            inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
            sc = mx.take_along_axis(g, inds, axis=-1)
            if b.norm_topk_prob:
                sc = sc / sc.sum(axis=-1, keepdims=True)
            ref = (b.switch_mlp(x, inds) * sc[..., None]).sum(axis=-2)
            ref32 = ref.astype(mx.float32)
            y = ov.combine(b.switch_mlp, x, inds, sc)
            yf = ov.combine(fused, x, inds, sc)
            yfs = (fused(x, inds) * sc[..., None]).sum(axis=-2)
            scale = mx.abs(ref32).max().item() + 1e-6
            worst["combine"] = max(
                worst["combine"],
                mx.abs(y.astype(mx.float32) - ref32).max().item() / scale,
            )
            worst["fused"] = max(
                worst["fused"],
                mx.abs(yf.astype(mx.float32) - ref32).max().item() / scale,
            )
            worst["fused_vs_combine"] = max(
                worst["fused_vs_combine"],
                mx.abs(yf.astype(mx.float32) - y.astype(mx.float32)).max().item(),
            )
            if not mx.array_equal(yfs, ref).item():
                print(f"layer {li}: #96 fused != stock (unexpected)")
            ti, tw = ov.st.softmax_topk(gates, top_k=k, normalize=b.norm_topk_prob)
            a_set = sorted(inds.reshape(-1).tolist())
            b_set = sorted(ti.reshape(-1).tolist())
            if a_set != b_set:
                topk_set_mismatch += 1
            else:
                order = mx.argsort(inds.reshape(-1))
                order2 = mx.argsort(ti.reshape(-1).astype(mx.int32))
                d = mx.abs(
                    sc.reshape(-1)[order].astype(mx.float32)
                    - tw.reshape(-1)[order2].astype(mx.float32)
                )
                topk_w_maxdiff = max(topk_w_maxdiff, d.max().item())
            n_cases += 1
        del fused
    print(json.dumps({
        "cases": n_cases,
        "combine_rel_err_max": worst["combine"],
        "fused_combine_rel_err_max": worst["fused"],
        "fused_vs_unfused_combine_absdiff_max": worst["fused_vs_combine"],
        "topk_expert_set_mismatches": topk_set_mismatch,
        "topk_weight_absdiff_max": topk_w_maxdiff,
    }, indent=1))  # fmt: skip


def live_snapshot(base):
    try:
        with urllib.request.urlopen(base + "/running", timeout=5) as r:
            running = json.load(r).get("running", [])
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    snap = {}
    for m in running:
        name = m.get("model")
        entry = {"state": m.get("state")}
        if "vllm_mlx" in (m.get("cmd") or "") and m.get("state") == "ready":
            try:
                url = f"{base}/upstream/{name}/v1/status"
                with urllib.request.urlopen(url, timeout=5) as r:
                    st = json.load(r)
                entry.update(
                    num_running=st.get("num_running"),
                    num_waiting=st.get("num_waiting"),
                    steps_executed=st.get("steps_executed"),
                )
            except Exception as e:  # noqa: BLE001
                entry["error"] = str(e)
        snap[name] = entry
    return snap


def polluted(before, after):
    if "error" in before or "error" in after or set(before) != set(after):
        return True
    for name, b in before.items():
        a = after[name]
        if b.get("num_running") or a.get("num_running"):
            return True
        if b.get("num_waiting") or a.get("num_waiting"):
            return True
        if b.get("steps_executed") != a.get("steps_executed"):
            return True
    return False


def cmd_drive(args):
    variants = args.variants.split(",")
    env = dict(os.environ)
    for k, v in DEPLOYED_ENV.items():
        env.setdefault(k, v)
    env.setdefault("HF_HUB_OFFLINE", "1")
    for rnd in range(args.rounds):
        order = variants[rnd % len(variants) :] + variants[: rnd % len(variants)]
        for v in order:
            for attempt in range(args.retries + 1):
                tmp = args.out + ".tmp"
                if os.path.exists(tmp):
                    os.remove(tmp)
                before = live_snapshot(args.llama_swap)
                cmd = [
                    sys.executable, os.path.abspath(__file__), "run",
                    "--model", args.model, "--kernels", args.kernels,
                    "--variant", v, "--out", tmp, "--round", str(rnd),
                    "--reps", str(args.reps), "--gen", str(args.gen),
                    "--prompt-tokens", str(args.prompt_tokens),
                ]  # fmt: skip
                rc = subprocess.call(cmd, env=env)
                after = live_snapshot(args.llama_swap)
                bad = rc != 0 or polluted(before, after)
                rows = []
                if os.path.exists(tmp):
                    with open(tmp) as fh:
                        rows = [json.loads(line) for line in fh if line.strip()]
                    os.remove(tmp)
                with open(args.out, "a") as fh:
                    for r in rows:
                        r.update(
                            live_before=before,
                            live_after=after,
                            polluted=bad,
                            attempt=attempt,
                        )
                        fh.write(json.dumps(r) + "\n")
                print(f"round {rnd} {v} attempt {attempt}: rc={rc} polluted={bad} "
                      f"before={before} after={after}", flush=True)  # fmt: skip
                if rc != 0:
                    sys.exit(rc)
                if not bad:
                    break


def cmd_report(args):
    rows = []
    with open(args.results) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    clean = [r for r in rows if not r.get("polluted")]
    print(f"{len(rows)} rows, {len(rows) - len(clean)} polluted (discarded)")
    by = {}
    for r in clean:
        by.setdefault(r["variant"], []).append(r)
    ref_v = args.ref if args.ref in by else sorted(by)[0]
    ref_toks = by[ref_v][0]["tokens"]
    base = {}
    print(f"{'var':<4}{'n':>3}  {'decode med':>10} {'min-max':>15}  "
          f"{'prefill med':>11} {'min-max':>15}  {'peak GB':>7}  T=0 vs {ref_v}")  # fmt: skip
    for v in sorted(by):
        rs = by[v]
        dec = [r["decode_tps"] for r in rs]
        pre = [r["prefill_tps"] for r in rs]
        base[v] = (statistics.median(dec), statistics.median(pre))
        idents = set()
        for r in rs:
            t = r["tokens"]
            if t == ref_toks:
                idents.add("identical")
            else:
                pos = next(
                    (i for i, (p, q) in enumerate(zip(t, ref_toks)) if p != q),
                    min(len(t), len(ref_toks)),
                )
                idents.add(f"diverges@{pos}")
        self_consistent = len({tuple(r["tokens"]) for r in rs}) == 1
        print(f"{v:<4}{len(rs):>3}  {statistics.median(dec):>10.2f} "
              f"{min(dec):>7.2f}-{max(dec):<7.2f}  {statistics.median(pre):>11.1f} "
              f"{min(pre):>7.1f}-{max(pre):<7.1f}  "
              f"{max(r['peak_gb'] for r in rs):>7.1f}  {','.join(sorted(idents))}"
              f"{'' if self_consistent else ' (NOT self-consistent)'}")  # fmt: skip
    if args.vs in base:
        bd, bp = base[args.vs]
        for v, (d, p) in sorted(base.items()):
            print(f"  {v} vs {args.vs}: decode {100 * (d / bd - 1):+.1f}%  "
                  f"prefill {100 * (p / bp - 1):+.1f}%")  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("run", "drive", "selftest"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--kernels", required=True)
        if name == "selftest":
            p.add_argument("--cases", type=int, default=4)
            continue
        p.add_argument("--out", required=True)
        p.add_argument("--reps", type=int, default=2)
        p.add_argument("--gen", type=int, default=256)
        p.add_argument("--prompt-tokens", type=int, default=4096)
        if name == "run":
            p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
            p.add_argument("--round", type=int, default=0)
        else:
            p.add_argument("--variants", default="a,b,c,d")
            p.add_argument("--rounds", type=int, default=6)
            p.add_argument("--retries", type=int, default=2)
            p.add_argument("--llama-swap", default="http://127.0.0.1:8080")
    p = sub.add_parser("report")
    p.add_argument("results")
    p.add_argument("--ref", default="a")
    p.add_argument("--vs", default="b")
    args = ap.parse_args()
    {"run": cmd_run, "drive": cmd_drive, "selftest": cmd_selftest}.get(
        args.mode, cmd_report
    )(args)


if __name__ == "__main__":
    main()
