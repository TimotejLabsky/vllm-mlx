# SPDX-License-Identifier: Apache-2.0
"""Tests for the repetition-detection stop (fork patch)."""

from __future__ import annotations

import time

import pytest

from vllm_mlx.repetition_stop import (
    RepetitionStopConfig,
    RepetitionStopTracker,
    find_repetition,
    load_config,
    reset_config_cache,
)

CFG = RepetitionStopConfig(
    enabled=True,
    window=512,
    min_period=1,
    max_period=128,
    min_repeats=3,
    min_span=48,
    interval=16,
    min_tokens=64,
)


# ---------------------------------------------------------------- detection


def test_no_repetition_returns_none():
    assert find_repetition(list(range(500)), CFG) is None


def test_single_token_run_needs_min_span():
    # period 1 requires min_span (48) identical tokens
    assert find_repetition([7] * 47, CFG) is None
    assert find_repetition([7] * 48, CFG) == (1, 48)


def test_single_token_run_after_normal_prefix():
    seq = list(range(100)) + [7] * 48
    assert find_repetition(seq, CFG) == (1, 48)


def test_two_token_alternation():
    # "items items items…" tokenizes as a 1-2 token cycle; period 2 needs
    # 24 copies to cover min_span
    seq = list(range(50)) + [11, 12] * 24
    period, reps = find_repetition(seq, CFG)
    assert period in (1, 2)
    assert period * reps >= CFG.min_span


def test_multi_token_cycle_three_repeats():
    cycle = list(range(1000, 1020))  # 20-token cycle: 3 repeats = 60 >= 48
    seq = list(range(100)) + cycle * 3
    assert find_repetition(seq, CFG) == (20, 3)


def test_two_repeats_not_enough():
    cycle = list(range(1000, 1030))  # 30 tokens, 2 copies = 60 span but reps<3
    seq = list(range(100)) + cycle * 2
    assert find_repetition(seq, CFG) is None


def test_rotation_invariance_mid_cycle_tail():
    # Loop ongoing, last sampled token lands mid-cycle: still caught.
    cycle = list(range(1000, 1016))  # 16 tokens
    seq = list(range(60)) + cycle * 4 + cycle[:7]
    hit = find_repetition(seq, CFG)
    assert hit is not None
    assert hit[0] == 16


def test_multiline_cycle_with_newline_tokens_is_caught():
    # Unlike DRY (newline is a sequence breaker), a repeated multi-line
    # block is exactly what this detector is for. 198 = arbitrary "\n" id.
    line = [5001, 5002, 5003, 5004, 198, 5005, 5006, 198]  # 8-token 2-line block
    seq = list(range(80)) + line * 6  # 48 tokens of cycle
    assert find_repetition(seq, CFG) == (8, 6)


def test_near_variant_breaks_exactness():
    # Documented limitation: a single swapped token ("its" for "items")
    # inside the tail window resets the consecutive-copy count.
    cycle = [11, 12, 13, 14]
    seq = list(range(60)) + cycle * 8
    seq[-6] = 999  # corrupt one token near the end
    hit = find_repetition(seq, CFG)
    # only the post-corruption tail is periodic — too short to trigger
    assert hit is None


def test_prompt_like_prefix_never_scanned_beyond_window():
    cfg = RepetitionStopConfig(
        enabled=True, window=64, min_period=1, max_period=16,
        min_repeats=3, min_span=8, interval=4, min_tokens=8,
    ).sanitized()
    tracker = RepetitionStopTracker(cfg)
    # feed 200 varied tokens, then a run; buffer only holds last 64
    hit = None
    for i, tok in enumerate(list(range(200)) + [7] * 12):
        hit = tracker.observe(1, tok, i + 1) or hit
    assert hit == (1, 12) or hit[0] == 1


def test_perf_smoke_non_repetitive():
    seq = [(i * 2654435761) % 50000 for i in range(512)]
    t0 = time.perf_counter()
    for _ in range(200):
        assert find_repetition(seq, CFG) is None
    per_check = (time.perf_counter() - t0) / 200
    # generous bound: one check well under a millisecond-scale budget
    assert per_check < 0.01


# ---------------------------------------------------------------- tracker


def _tracker(**over):
    base = dict(
        enabled=True, window=512, min_period=1, max_period=128,
        min_repeats=3, min_span=48, interval=16, min_tokens=64,
    )
    base.update(over)
    return RepetitionStopTracker(RepetitionStopConfig(**base))


def test_tracker_interval_and_min_tokens_gate():
    tr = _tracker(min_tokens=64, interval=16)
    # 100 identical tokens but probe off-interval counts: no check fires
    for n in range(1, 64):
        assert tr.observe(1, 7, n) is None  # below min_tokens
    assert tr.observe(1, 7, 65) is None  # 65 % 16 != 0
    assert tr.observe(1, 7, 80) is not None  # on-interval, span satisfied


def test_tracker_per_uid_isolation_and_discard():
    tr = _tracker(min_tokens=0, interval=1, min_span=8)
    for n in range(1, 9):
        tr.observe(1, 7, n)
        assert tr.observe(2, n * 100, n) is None  # uid 2 varied: never triggers
    assert tr.observe(1, 7, 9) == (1, 9)
    tr.discard(1)
    assert tr.observe(1, 7, 10) is None  # buffer restarted
    tr.clear()
    assert tr._buffers == {}


def test_tracker_disabled_is_noop():
    tr = _tracker(enabled=False)
    for n in range(1, 200):
        assert tr.observe(1, 7, n) is None
    assert tr._buffers == {}


def test_tracker_counts_stops():
    tr = _tracker(min_tokens=0, interval=1, min_span=4, min_repeats=2)
    for n in range(1, 10):
        tr.observe(1, 7, n)
    assert tr.stops >= 1


# ---------------------------------------------------------------- config


def test_env_config_parsing(monkeypatch):
    reset_config_cache()
    monkeypatch.setenv("VLLM_MLX_REPDETECT", "1")
    monkeypatch.setenv("VLLM_MLX_REPDETECT_WINDOW", "256")
    monkeypatch.setenv("VLLM_MLX_REPDETECT_MIN_SPAN", "32")
    cfg = load_config()
    assert cfg.enabled and cfg.window == 256 and cfg.min_span == 32
    reset_config_cache()


def test_env_config_default_off(monkeypatch):
    reset_config_cache()
    monkeypatch.delenv("VLLM_MLX_REPDETECT", raising=False)
    assert load_config().enabled is False
    reset_config_cache()


def test_env_config_invalid_values_fall_back(monkeypatch):
    reset_config_cache()
    monkeypatch.setenv("VLLM_MLX_REPDETECT", "1")
    monkeypatch.setenv("VLLM_MLX_REPDETECT_WINDOW", "banana")
    monkeypatch.setenv("VLLM_MLX_REPDETECT_MIN_REPEATS", "-5")
    cfg = load_config()
    assert cfg.window == 512  # default
    assert cfg.min_repeats >= 2  # sanitized floor
    reset_config_cache()


def test_sanitize_clamps():
    cfg = RepetitionStopConfig(
        enabled=True, window=0, min_period=0, max_period=0,
        min_repeats=0, min_span=0, interval=0, min_tokens=-1,
    ).sanitized()
    assert cfg.window >= 16 and cfg.min_period >= 1
    assert cfg.max_period >= cfg.min_period
    assert cfg.min_repeats >= 2 and cfg.interval >= 1 and cfg.min_tokens == 0


# ------------------------------------------------------------- wiring smoke


def test_scheduler_imports_tracker():
    import vllm_mlx.scheduler as sched

    assert hasattr(sched, "RepetitionStopTracker")


def test_metrics_observe_exists():
    from vllm_mlx.metrics import metrics

    # must not raise even when metrics collection is disabled
    metrics.observe_repetition_stop()
