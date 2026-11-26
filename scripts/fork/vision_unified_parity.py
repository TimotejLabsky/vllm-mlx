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
     tokens equal the text row run alone.

Usage:
  HF_HOME=<cache> PYTHONPATH=<worktree> python scripts/fork/vision_unified_parity.py \
      --model mlx-community/Qwen3.5-0.8B-4bit
"""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw


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

    def batch_run(model, prompts, caches=None):
        bg = BatchGenerator(
            model, max_tokens=args.tokens, stop_tokens=[[e] for e in eos]
        )
        uids = bg.insert(prompts, caches=caches)
        res = {u: [] for u in uids}
        while True:
            _, gen = bg.next()
            if not gen and not any(len(v) < args.tokens for v in res.values()):
                break
            if not gen:
                continue
            for r in gen:
                res[r.uid].append(int(r.token))
            if all(len(v) >= args.tokens or (v and v[-1] in eos) for v in res.values()):
                break
        bg.close()
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
        for _ in range(args.tokens - 1):
            if toks[-1] in eos:
                break
            out = vlm.language_model(mx.array([[toks[-1]]]), cache=cache)
            toks.append(greedy(out.logits))
        return all_logits, toks

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
        if decode:
            for i in range(args.tokens - 1):
                if toks[-1] in eos:
                    break
                p = L + i + delta
                step_pos = mx.full((3, 1, 1), p, dtype=mx.int32)
                with vu.mrope_positions(step_pos):
                    logits = tm(mx.array([[toks[-1]]]), cache=cache)
                toks.append(greedy(logits))
        return all_logits, toks, emb, pos, delta

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
        different kernels, so even text is not bit-equal between them."""
        filler = "The sea is grey and the boats are many colours. " * 40
        ids = tok.encode(filler)[: max(8, length)]
        ids = mx.array([ids])
        a = vlm.language_model(ids, cache=vlm.language_model.make_cache()).logits
        b = tm(ids, cache=make_prompt_cache(tm))
        d = np.abs(
            np.array(a[0].astype(mx.float32)) - np.array(b[0].astype(mx.float32))
        )
        return float(d.mean()), float(d.max())

    media_for_c = None
    for name, (msgs, imgs) in cases.items():
        inputs = vlm_inputs(msgs, imgs)
        ref_logits, ref_toks = vlm_reference(inputs)
        my_logits, my_toks, emb, pos, delta = ours(inputs)
        diff = np.abs(ref_logits - my_logits)
        last = np.abs(ref_logits[-1] - my_logits[-1]).max()
        L = inputs["input_ids"].shape[1]
        base_mean, base_max = text_baseline(L)
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
            f"floor mean={base_mean:.5f} max={base_max:.4f} | 1-D positions "
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
        check(f"B {name}: greedy tokens equal mlx-vlm", ref_toks == my_toks)
        if media_for_c is None:
            media_for_c = (inputs, emb, pos, delta, ref_toks)

    # ---------------- C. media row through BatchGenerator ----------------
    inputs, emb, pos, delta, ref_toks = media_for_c
    ids_list = inputs["input_ids"][0].tolist()
    # prefill all but the last token out of band; the last token + decode
    # run in the batch with offset + delta.
    mcache = make_prompt_cache(tm)
    vu.media_prefill(
        tm,
        inputs["input_ids"][:, :-1],
        emb[:, :-1],
        pos[:, :, :-1],
        mcache,
    )
    vu.set_rope_delta(mcache, delta)
    text_alone = batch_run(tm, [text_ids])[0]
    rows = batch_run(tm, [[ids_list[-1]], text_ids], caches=[mcache, None])
    print(f"   batched media row: {tok.decode(rows[0])!r}")
    check(
        "C1 batched media row equals mlx-vlm greedy",
        rows[0] == ref_toks[: len(rows[0])] and len(rows[0]) > 0,
    )
    check("C2 text row next to a media row equals text alone", rows[1] == text_alone)

    # trailing text through the batch: prefill only up to the end of the image
    vision_end = (
        vlm.config.vision_end_token_id
        if hasattr(vlm.config, "vision_end_token_id")
        else None
    )
    if vision_end is not None and vision_end in ids_list:
        cut = len(ids_list) - 1 - ids_list[::-1].index(vision_end) + 1
        mcache2 = make_prompt_cache(tm)
        vu.media_prefill(
            tm, inputs["input_ids"][:, :cut], emb[:, :cut], pos[:, :, :cut], mcache2
        )
        vu.set_rope_delta(mcache2, delta)
        rows2 = batch_run(tm, [ids_list[cut:], text_ids], caches=[mcache2, None])
        print(f"   cut at {cut}/{len(ids_list)}: {tok.decode(rows2[0])!r}")
        check(
            "C3 trailing text prefilled in the batch equals mlx-vlm greedy",
            rows2[0] == ref_toks[: len(rows2[0])] and len(rows2[0]) > 0,
        )
        check("C4 text row unchanged (trailing-text media row)", rows2[1] == text_alone)
        # extract round trip keeps the delta
        from mlx_lm.models.cache import BatchKVCache, KVCache

        kv = [c for c in mcache2 if isinstance(c, KVCache)][0]
        b = BatchKVCache.merge([kv, KVCache()])
        back = b.extract(0)
        check(
            "C5 merge/extract round trip keeps rope_delta",
            getattr(back, "rope_delta", 0) == delta,
            f"(delta={delta})",
        )

    print("\nRESULT:", "PASS" if not failures else f"FAIL {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
