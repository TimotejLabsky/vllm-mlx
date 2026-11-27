#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 numerical proof for PATCHES.md #181 (``--vision-unified``).

No server. Loads a small Qwen3.5 vision checkpoint once through mlx-vlm,
builds the mlx-lm text model from it (``build_text_model``), and checks:

  A. text-only: the text model with the #181 RoPE dispatch + batch-cache
     patches installed gives BIT-IDENTICAL logits and greedy tokens to the
     same model without them, single-stream and through ``BatchGenerator``;
  B. image prompts (one image, two images, image then text turn): the mlx-lm
     text model fed mlx-vlm's merged embeddings + MRoPE positions reproduces
     mlx-vlm's own logits (max |diff| reported, tolerance asserted) and its
     greedy tokens;
  C. the same image prompt through a real ``BatchGenerator`` next to a
     text request: the media row's tokens equal (B), and the text row's
     tokens equal the text row run alone;
  F. the same B and C comparisons with every activation in float32: the
     paths must agree to ``--fp32-tol`` and produce IDENTICAL greedy tokens.

Greedy comparisons in bf16 (B, C1, C3) accept a NEAR-TIE: the first
differing token is one whose logit margin, on BOTH stacks, is within the
text-stack noise floor (max |d| between mlx-vlm and mlx-lm at the last
position of a text prompt of the same length). bf16 rounding differs with
the chunk shape (a prompt prefilled in one forward vs. split at the image
end and at the last token, as the batch does), and that alone flips a greedy
pick between two tokens a quarter of a logit apart (2026-11-27,
Qwen3.5-4B-4bit: 'Red' vs ' red', margin 0.25 on mlx-vlm, 0.25 on the
batch, bound 0.41; in float32 every path gives ' red' and agrees to
< 2e-3). A near-tie passes only because F holds: a real error (positions,
the delta, the recurrent state) survives float32, rounding does not.
``--inject-delta-error 1`` (an off-by-one RoPE delta) passes every bf16
greedy check on the 4B and fails F at ~0.1 vs a 2e-4 float32 floor.

Usage:
  HF_HOME=<cache> PYTHONPATH=<worktree> python scripts/fork/vision_unified_parity.py \
      --model mlx-community/Qwen3.5-0.8B-4bit
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw


def classify_greedy(ref_toks, ref_rows, our_toks, our_rows, bound):
    """Compare two greedy runs: ``ours`` may stop earlier than ``ref``.

    ``*_rows[i]`` are the logits (or logprobs: only differences are used) the
    run picked its token ``i`` from. Returns ``(verdict, detail)`` with
    verdict ``"equal"``, ``"near_tie"`` (the first differing pick is within
    ``bound`` of the other stack's pick on both stacks) or ``"diverged"``.
    """
    if our_toks and our_toks == ref_toks[: len(our_toks)]:
        return "equal", ""
    if not our_toks:
        return "diverged", "no tokens"
    n = min(len(ref_toks), len(our_toks))
    diff = [i for i in range(n) if ref_toks[i] != our_toks[i]]
    if not diff:
        return "diverged", f"length: ref {len(ref_toks)} vs ours {len(our_toks)}"
    i = diff[0]
    if i >= len(ref_rows) or i >= len(our_rows):
        return "diverged", f"step {i}: no logits recorded"
    a, b = ref_toks[i], our_toks[i]
    m_ref = float(ref_rows[i][a] - ref_rows[i][b])
    m_our = float(our_rows[i][b] - our_rows[i][a])
    detail = (
        f"step {i}: ref picks {a}, ours {b}; margin on ref {m_ref:.3f}, "
        f"on ours {m_our:.3f}, noise bound {bound:.3f}"
    )
    if 0 <= m_ref <= bound and 0 <= m_our <= bound:
        return "near_tie", detail
    return "diverged", detail


@contextmanager
def fp32_activations():
    """Run every forward inside the block with float32 activations: the
    token embeddings (plain and quantized) are cast to float32, and every
    later op promotes against the bf16 weights. Restores on exit."""
    import mlx.nn as nn

    saved = {}
    for cls in (nn.QuantizedEmbedding, nn.Embedding):
        orig = cls.__dict__["__call__"]
        saved[cls] = orig

        def call(self, x, _orig=orig):
            return _orig(self, x).astype(mx.float32)

        cls.__call__ = call
    try:
        yield
    finally:
        for cls, orig in saved.items():
            cls.__call__ = orig


def make_image(color, shape="square", size=448):
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    if shape == "square":
        d.rectangle([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=color)
    else:
        d.ellipse([size // 6, size // 6, 5 * size // 6, 5 * size // 6], fill=color)
    return img


def greedy(logits):
    return int(mx.argmax(logits[0, -1]).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-4bit")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument(
        "--fp32-tol",
        type=float,
        default=0.02,
        help="max |d| between paths in float32 (observed < 1e-3 at 0.8B-9B)",
    )
    ap.add_argument(
        "--inject-delta-error",
        type=int,
        default=0,
        help="self-test: add this to the media row's RoPE delta in C/F; "
        "the run must then FAIL (a real error is not passed as a near-tie)",
    )
    args = ap.parse_args()

    from mlx_lm.generate import BatchGenerator
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_vlm import load as vlm_load
    from mlx_vlm.utils import prepare_inputs

    import vllm_mlx
    from vllm_mlx import vision_unified as vu
    from vllm_mlx.text_model_from_vlm import build_text_model

    print("vllm_mlx from", vllm_mlx.__file__)
    vlm, processor = vlm_load(args.model)
    tm = build_text_model(vlm, args.model, enable_mtp=False)
    assert tm is not None, "build_text_model returned None"
    tok = processor.tokenizer
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    failures = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    # ---------------- A. text-only bit identity ----------------
    text_msgs = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Name three primary colours, comma separated."},
    ]
    text_prompt = tok.apply_chat_template(
        text_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    text_ids = tok.encode(text_prompt)

    def text_single(model):
        cache = make_prompt_cache(model)
        logits = model(mx.array([text_ids]), cache=cache)
        mx.eval(logits)
        first = logits[0, -1].astype(mx.float32)
        out = [greedy(logits)]
        for _ in range(args.tokens - 1):
            logits = model(mx.array([[out[-1]]]), cache=cache)
            out.append(greedy(logits))
        return np.array(first), out

    def batch_run(model, prompts, caches=None, rows=None):
        """Greedy tokens per prompt; with ``rows`` (a list), also append one
        list per prompt of the float32 logprob rows each token came from."""
        bg = BatchGenerator(
            model, max_tokens=args.tokens, stop_tokens=[[e] for e in eos]
        )
        uids = bg.insert(prompts, caches=caches)
        res = {u: [] for u in uids}
        lps = {u: [] for u in uids}
        while True:
            _, gen = bg.next()
            if not gen and not any(len(v) < args.tokens for v in res.values()):
                break
            if not gen:
                continue
            for r in gen:
                res[r.uid].append(int(r.token))
                if rows is not None:
                    lps[r.uid].append(np.array(r.logprobs.astype(mx.float32)))
            if all(len(v) >= args.tokens or (v and v[-1] in eos) for v in res.values()):
                break
        bg.close()
        if rows is not None:
            rows.extend(lps[u] for u in uids)
        return [res[u] for u in uids]

    base_logits, base_tokens = text_single(tm)
    base_batch = batch_run(tm, [text_ids, text_ids[:-3]])

    cfg = vlm.config.text_config
    rp = getattr(cfg, "rope_parameters", None) or {}
    section = rp.get("mrope_section") if isinstance(rp, dict) else None
    print("mrope_section", section)
    n = vu.install_mrope(tm, section)
    vu.install_batch_cache_delta()
    print("wrapped layers:", n)

    after_logits, after_tokens = text_single(tm)
    after_batch = batch_run(tm, [text_ids, text_ids[:-3]])
    check(
        "A1 text logits bit-identical (single)",
        np.array_equal(base_logits, after_logits),
        f"max|d|={np.abs(base_logits - after_logits).max()}",
    )
    check("A2 text greedy tokens identical (single)", base_tokens == after_tokens)
    check(
        "A3 text tokens identical (BatchGenerator, 2 rows)", base_batch == after_batch
    )

    # ---------------- B. image prompts vs mlx-vlm ----------------
    def vlm_inputs(messages, images):
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = prepare_inputs(
            processor,
            images=images,
            prompts=prompt,
            image_token_index=vlm.config.image_token_index,
        )
        return inputs

    def vlm_reference(inputs):
        ids = inputs["input_ids"]
        cache = vlm.language_model.make_cache()
        out = vlm(
            ids,
            pixel_values=inputs["pixel_values"],
            cache=cache,
            image_grid_thw=inputs.get("image_grid_thw"),
            mask=inputs.get("attention_mask"),
        )
        logits = out.logits
        mx.eval(logits)
        all_logits = np.array(logits[0].astype(mx.float32))
        toks = [greedy(logits)]
        rows = [all_logits[-1]]
        for _ in range(args.tokens - 1):
            if toks[-1] in eos:
                break
            out = vlm.language_model(mx.array([[toks[-1]]]), cache=cache)
            toks.append(greedy(out.logits))
            rows.append(np.array(out.logits[0, -1].astype(mx.float32)))
        return all_logits, toks, rows

    def ours(inputs, decode=True):
        ids = inputs["input_ids"]
        feats = vlm.get_input_embeddings(
            ids,
            inputs["pixel_values"],
            image_grid_thw=inputs.get("image_grid_thw"),
            mask=inputs.get("attention_mask"),
        )
        emb, pos, deltas = feats.inputs_embeds, feats.position_ids, feats.rope_deltas
        delta = int(deltas.reshape(-1)[0].item())
        cache = make_prompt_cache(tm)
        L = ids.shape[1]
        with vu.mrope_positions(pos):
            logits = tm(ids, cache=cache, input_embeddings=emb)
        mx.eval(logits)
        all_logits = np.array(logits[0].astype(mx.float32))
        toks = [greedy(logits)]
        rows = [all_logits[-1]]
        if decode:
            for i in range(args.tokens - 1):
                if toks[-1] in eos:
                    break
                p = L + i + delta
                step_pos = mx.full((3, 1, 1), p, dtype=mx.int32)
                with vu.mrope_positions(step_pos):
                    logits = tm(mx.array([[toks[-1]]]), cache=cache)
                toks.append(greedy(logits))
                rows.append(np.array(logits[0, -1].astype(mx.float32)))
        return all_logits, toks, emb, pos, delta, rows

    red = make_image("red")
    blue = make_image("blue", "circle")
    cases = {
        "one image": (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What colour is the shape? One word."},
                    ],
                }
            ],
            [red],
        ),
        "two images": (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "Describe both shapes and colours briefly.",
                        },
                    ],
                }
            ],
            [red, blue],
        ),
        "image then later text turn": (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What is in the image?"},
                    ],
                },
                {"role": "assistant", "content": "A blue circle on white."},
                {
                    "role": "user",
                    "content": "And what colour was the circle? One word.",
                },
            ],
            [blue],
        ),
    }

    def text_baseline(length):
        """Stack-noise floor: mlx-vlm vs mlx-lm logits on a TEXT prompt of
        about ``length`` tokens (both 1-D positions). The two stacks use
        different kernels, so even text is not bit-equal between them.
        Returns (mean, max, max at the last position): the last is the
        near-tie bound, the noise on the logits a first token is picked
        from (the all-position max is dominated by a few outlier rows)."""
        filler = "The sea is grey and the boats are many colours. " * 40
        ids = tok.encode(filler)[: max(8, length)]
        ids = mx.array([ids])
        a = vlm.language_model(ids, cache=vlm.language_model.make_cache()).logits
        b = tm(ids, cache=make_prompt_cache(tm))
        d = np.abs(
            np.array(a[0].astype(mx.float32)) - np.array(b[0].astype(mx.float32))
        )
        return float(d.mean()), float(d.max()), float(d[-1].max())

    near_ties = []

    def check_greedy(name, ref_toks, ref_rows, our_toks, our_rows, bound):
        """bf16: equal, or a near-tie inside ``bound`` (recorded; section F
        gates it). ``bound=None`` (float32): only exact equality passes."""
        verdict, detail = classify_greedy(
            ref_toks, ref_rows, our_toks, our_rows, 0.0 if bound is None else bound
        )
        if verdict == "near_tie" and bound is not None:
            near_ties.append(name)
            print(f"[NEAR-TIE] {name} ({detail}; float32 section F must agree)")
            return
        check(name, verdict == "equal", detail)

    def run_media_row(inputs, emb, pos, delta, upto):
        """The media row prefilled out of band to ``upto``; the rest of its
        prompt and all decode run in a real BatchGenerator next to the text
        row. Returns (tokens per row, logprob rows per row)."""
        ids_list = inputs["input_ids"][0].tolist()
        mcache = make_prompt_cache(tm)
        vu.media_prefill(
            tm,
            inputs["input_ids"][:, :upto],
            emb[:, :upto],
            pos[:, :, :upto],
            mcache,
        )
        vu.set_rope_delta(mcache, delta + args.inject_delta_error)
        rows = []
        toks = batch_run(
            tm, [ids_list[upto:], text_ids], caches=[mcache, None], rows=rows
        )
        return toks, rows

    def media_cut(inputs):
        """End of the last image (the trailing-text split), or None."""
        ids_list = inputs["input_ids"][0].tolist()
        vision_end = getattr(vlm.config, "vision_end_token_id", None)
        if vision_end is None or vision_end not in ids_list:
            return None
        return len(ids_list) - 1 - ids_list[::-1].index(vision_end) + 1

    def log_softmax(row):
        m = row.max()
        return row - (m + np.log(np.exp(row - m).sum()))

    # ---------------- B. image prompts vs mlx-vlm ----------------
    media_for_c = None
    for name, (msgs, imgs) in cases.items():
        inputs = vlm_inputs(msgs, imgs)
        ref_logits, ref_toks, ref_rows = vlm_reference(inputs)
        my_logits, my_toks, emb, pos, delta, my_rows = ours(inputs)
        diff = np.abs(ref_logits - my_logits)
        last = np.abs(ref_logits[-1] - my_logits[-1]).max()
        L = inputs["input_ids"].shape[1]
        base_mean, base_max, tie_bound = text_baseline(L)
        # negative control: the same embeddings with plain 1-D positions
        wrong = tm(
            inputs["input_ids"], cache=make_prompt_cache(tm), input_embeddings=emb
        )
        wrong_mean = float(
            np.abs(ref_logits - np.array(wrong[0].astype(mx.float32))).mean()
        )
        print(
            f"   {name}: L={L} delta={delta} mean|d|={diff.mean():.5f} "
            f"max|d|={diff.max():.4f} last-pos max|d|={last:.4f} | text-stack "
            f"floor mean={base_mean:.5f} max={base_max:.4f} "
            f"last-pos max={tie_bound:.4f} | 1-D positions "
            f"mean={wrong_mean:.5f}"
        )
        print(f"   ref : {tok.decode(ref_toks)!r}")
        print(f"   ours: {tok.decode(my_toks)!r}")
        check(
            f"B {name}: mean|d| <= 2x the text-stack floor",
            diff.mean() <= 2 * base_mean,
        )
        check(
            f"B {name}: 1-D positions are >= 4x worse (positions matter)",
            wrong_mean >= 4 * diff.mean(),
        )
        check_greedy(
            f"B {name}: greedy tokens equal mlx-vlm",
            ref_toks,
            ref_rows,
            my_toks,
            my_rows,
            tie_bound,
        )
        if media_for_c is None:
            media_for_c = (inputs, emb, pos, delta, ref_toks, ref_rows, tie_bound)

    # ---------------- C. media row through BatchGenerator ----------------
    inputs, emb, pos, delta, ref_toks, ref_rows, tie_bound = media_for_c
    L = inputs["input_ids"].shape[1]
    text_alone = batch_run(tm, [text_ids])[0]
    # prefill all but the last token out of band; the last token + decode
    # run in the batch with offset + delta.
    toks, rows = run_media_row(inputs, emb, pos, delta, L - 1)
    print(f"   batched media row: {tok.decode(toks[0])!r}")
    check_greedy(
        "C1 batched media row equals mlx-vlm greedy",
        ref_toks,
        ref_rows,
        toks[0],
        rows[0],
        tie_bound,
    )
    check("C2 text row next to a media row equals text alone", toks[1] == text_alone)

    # trailing text through the batch: prefill only up to the end of the image
    cut = media_cut(inputs)
    if cut is not None:
        toks, rows = run_media_row(inputs, emb, pos, delta, cut)
        print(f"   cut at {cut}/{L}: {tok.decode(toks[0])!r}")
        check_greedy(
            "C3 trailing text prefilled in the batch equals mlx-vlm greedy",
            ref_toks,
            ref_rows,
            toks[0],
            rows[0],
            tie_bound,
        )
        check("C4 text row unchanged (trailing-text media row)", toks[1] == text_alone)

    # ---------------- F. the same in float32 ----------------
    # bf16 rounding depends on the chunk shape; a real error (positions, the
    # delta, the recurrent state) does not go away in float32. Every path
    # must agree to --fp32-tol and pick identical tokens.
    with fp32_activations():
        first = None
        for name, (msgs, imgs) in cases.items():
            inputs = vlm_inputs(msgs, imgs)
            ref_logits, ref_toks, ref_rows = vlm_reference(inputs)
            my_logits, my_toks, emb, pos, delta, my_rows = ours(inputs)
            d = float(np.abs(ref_logits - my_logits).max())
            print(f"   fp32 {name}: max|d|={d:.5f} ref {tok.decode(ref_toks)!r}")
            check(
                f"F {name}: float32 activations, logits max|d| <= {args.fp32_tol}",
                emb.dtype == mx.float32 and d <= args.fp32_tol,
            )
            check_greedy(
                f"F {name}: greedy tokens identical (float32)",
                ref_toks,
                ref_rows,
                my_toks,
                my_rows,
                None,
            )
            if first is None:
                first = (inputs, emb, pos, delta, ref_toks, ref_rows)
        inputs, emb, pos, delta, ref_toks, ref_rows = first
        L = inputs["input_ids"].shape[1]
        splits = [("C1", L - 1)]
        cut = media_cut(inputs)
        if cut is not None:
            splits.append(("C3", cut))
        for label, upto in splits:
            toks, rows = run_media_row(inputs, emb, pos, delta, upto)
            d = float(np.abs(rows[0][0] - log_softmax(ref_rows[0])).max())
            print(
                f"   fp32 {label}: {tok.decode(toks[0])!r} first logprobs max|d|={d:.5f}"
            )
            check(
                f"F {label}: first-token logprobs max|d| <= {args.fp32_tol} (float32)",
                d <= args.fp32_tol,
            )
            check_greedy(
                f"F {label}: batched media row tokens identical (float32)",
                ref_toks,
                ref_rows,
                toks[0],
                rows[0],
                None,
            )

    if near_ties:
        print(
            "\nNEAR-TIES (bf16 greedy flips inside the noise bound, gated by "
            f"float32 section F): {near_ties}"
        )
    print("\nRESULT:", "PASS" if not failures else f"FAIL {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
