# SPDX-License-Identifier: Apache-2.0
"""Real-model batched-decode correctness gates for the VLM families
(vision series #58; empirical verification the plan requires before any
route flip).

What these tests pin, on a real rope-delta-family model:

- **Divergent-offset co-batch**: two media requests with different-sized
  images (different MRoPE deltas) inserted together decode batched; each
  row's tokens must be byte-identical to its solo run. Without patch #57
  the later prefill's delta corrupts the earlier row's positions.
- **Mid-decode text join**: a text row extending an active media batch
  must not corrupt either row (stale ``_position_ids`` slicing / delta
  clobber — the other two #57 vectors).
- **Multi-image smoke**: one request with several images generates.

Model: ``VLM_TEST_MODEL`` env override; defaults to the smallest cached
rope-delta-family checkpoint. The Studio deploy gate runs this suite with
``mlx-community/GLM-4.6V-Flash-4bit`` and
``mlx-community/Qwen3-VL-4B-Instruct-3bit`` (see PATCHES.md #58).
Video is exercised in the manual Studio e2e protocol, not here.

Run: ``RUN_SLOW_TESTS=1 .venv/bin/python -m pytest tests/test_vlm_batch_correctness.py --run-slow``
"""

import asyncio
import os

import pytest

try:
    import mlx.core as mx  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_MLX, reason="MLX not available"),
    pytest.mark.skipif(
        not os.environ.get("RUN_SLOW_TESTS"), reason="Slow tests disabled"
    ),
]

MODEL_ID = os.environ.get("VLM_TEST_MODEL", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")

MAX_TOKENS = 24

# Batched and solo forwards run different Metal kernels (GEMM vs GEMV
# shapes), so logits are not bit-identical; in low-entropy continuations a
# near-tie can flip a LATE token. MRoPE position corruption — what this
# suite gates on — manifests within the first couple of tokens instead.
# Rows whose comparison crosses a batch-shape boundary therefore compare a
# strict prefix; same-shape comparisons stay full-sequence. (Observed on
# Qwen2.5-VL-3B: a counting prompt tie-flipped at token 10 while the first
# 10 tokens and the co-batched media rows matched byte-for-byte.)
STRICT_PREFIX_TOKENS = 8


def _make_image(size: int, color) -> str:
    """Return a data:image base64 payload (the only non-URL input the
    fork's image hardening accepts)."""
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


@pytest.fixture(scope="module")
def model_and_processor():
    from mlx_vlm import load

    return load(MODEL_ID)


@pytest.fixture()
def images():
    # Different sizes -> different vision grids -> different MRoPE deltas.
    small = _make_image(64, (255, 0, 0))
    large = _make_image(448, (0, 0, 255))
    return small, large


def _scheduler(model, processor):
    from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

    config = MLLMSchedulerConfig(
        max_num_seqs=4,
        enable_prefix_cache=False,  # isolate decode math from caching
    )
    return MLLMScheduler(model, processor, config)


async def _run_jobs(model, processor, jobs, stagger_steps: int = 0):
    """Run jobs on one scheduler; returns {job_index: output_tokens}.

    ``stagger_steps > 0``: add job 0 first, run that many steps, then add
    the remaining jobs (mid-decode join). Otherwise all jobs co-batch.
    """
    sched = _scheduler(model, processor)
    await sched.start()
    try:
        ids = {}
        reqs = {}
        first_delta = None

        def _add(idx):
            job = jobs[idx]
            rid = sched.add_request(
                prompt=job["prompt"],
                images=job.get("images"),
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            ids[idx] = rid
            # Hold the request object: the scheduler pops finished
            # requests from its map (get_request is destructive).
            reqs[idx] = sched.requests[rid]

        if stagger_steps > 0:
            _add(0)
            for _ in range(stagger_steps):
                sched.step()
            batch = sched.batch_generator.active_batch
            if batch is not None and batch.requests:
                first_delta = batch.requests[0].rope_delta
            for idx in range(1, len(jobs)):
                _add(idx)
        else:
            for idx in range(len(jobs)):
                _add(idx)

        finished = set()
        for _ in range(2000):
            output = sched.step()
            finished.update(output.finished_request_ids)
            if len(finished) >= len(jobs):
                break

        tokens = {}
        for idx in ids:
            tokens[idx] = list(reqs[idx].output_tokens)
        return tokens, first_delta
    finally:
        await sched.stop()


def test_cobatched_divergent_images_match_solo(model_and_processor, images):
    model, processor = model_and_processor
    small, large = images
    job_a = {"prompt": "What color is this image? Answer briefly.", "images": [small]}
    job_b = {
        "prompt": "Describe what you see in one short sentence.",
        "images": [large],
    }

    solo_a, _ = asyncio.run(_run_jobs(model, processor, [job_a]))
    solo_b, _ = asyncio.run(_run_jobs(model, processor, [job_b]))
    both, _ = asyncio.run(_run_jobs(model, processor, [job_a, job_b]))

    assert both[0] == solo_a[0], "co-batched row A diverged from its solo run"
    assert both[1] == solo_b[0], "co-batched row B diverged from its solo run"
    assert len(both[0]) > 0 and len(both[1]) > 0


def test_text_join_mid_decode_matches_solo(model_and_processor, images):
    model, processor = model_and_processor
    _, large = images
    job_a = {"prompt": "Describe this image in detail.", "images": [large]}
    # The text prompt must be LONGER than the media row's pre-vision text
    # so that (without the fix) the stale ``_position_ids`` slice reaches
    # into the vision-position region — the proven discriminating case:
    # the 2026-07-28 negative control (arming disabled) diverges from
    # token 0 on this exact scenario, and matches byte-for-byte with it.
    job_b = {
        "prompt": (
            "Please list the seven days of the week, then the twelve months "
            "of the year, and then explain in two sentences why calendars "
            "have leap years. Be precise and complete in your answer."
        )
    }

    solo_a, _ = asyncio.run(_run_jobs(model, processor, [job_a]))
    solo_b, _ = asyncio.run(_run_jobs(model, processor, [job_b]))
    both, delta_a = asyncio.run(
        _run_jobs(model, processor, [job_a, job_b], stagger_steps=3)
    )

    # Sanity: this family tracks rope deltas, so the media row must have
    # captured one (the mechanism under test is actually active).
    assert delta_a is not None, "media row captured no rope delta"

    # Both rows cross a batch-shape boundary vs their solo runs (1-row vs
    # padded 2-row kernels) — prefix compare (see STRICT_PREFIX_TOKENS).
    # The 2026-07-28 negative control diverges from TOKEN 0 without the
    # fix, so the prefix gate keeps full discrimination.
    assert (
        both[0][:STRICT_PREFIX_TOKENS] == solo_a[0][:STRICT_PREFIX_TOKENS]
    ), "media row corrupted by mid-decode text join"
    assert (
        both[1][:STRICT_PREFIX_TOKENS] == solo_b[0][:STRICT_PREFIX_TOKENS]
    ), "joined text row diverged from its solo run within the strict prefix"


def test_multi_image_smoke(model_and_processor, images):
    model, processor = model_and_processor
    small, large = images
    job = {
        "prompt": "How many images do you see? Answer briefly.",
        "images": [small, large],
    }

    tokens, _ = asyncio.run(_run_jobs(model, processor, [job]))

    assert len(tokens[0]) > 0
