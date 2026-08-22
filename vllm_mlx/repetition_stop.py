"""Repetition-detection stop: end a request when its generated tail is an
exact repeating token cycle (patch: repetition-detection-stop).

Why an engine-level stop and not a sampler penalty: penalties (DRY,
repetition_penalty, presence_penalty) reshape logits and forbid FORMS —
a model stuck in a degenerate attractor simply rewords each cycle or
swaps near-variant tokens and stays in it, burning tokens until
max_tokens. Detection-as-stop ends the request the moment the output is
provably periodic, at zero cost to context, speed, or cache. Prior art:
vLLM ships the same mechanism as ``SamplingParams.repetition_detection``
enforced in the scheduler stop-check (vllm/v1/core/sched/utils.py::
check_sequence_repetition, stop_reason="repetition_detected"), and
vllm-project/vllm#52673 extends it to reasoning sections.

Scope: exact periodic repetition over GENERATED tokens only (prompt
never scanned). Semantic loops that reword every cycle are out of scope
— nothing token-level can catch those. Unlike DRY, newlines are NOT
breakers here, so multi-line cycles are caught.

Env config (read once per process; ``reset_config_cache()`` for tests):

    VLLM_MLX_REPDETECT              default 0 = off; 1 = on
    VLLM_MLX_REPDETECT_WINDOW       default 512  generated-tail tokens scanned
    VLLM_MLX_REPDETECT_MIN_PERIOD   default 1    smallest cycle length checked
    VLLM_MLX_REPDETECT_MAX_PERIOD   default 128  largest cycle length checked
    VLLM_MLX_REPDETECT_MIN_REPEATS  default 3    consecutive exact cycle copies
    VLLM_MLX_REPDETECT_MIN_SPAN     default 48   min tokens covered by the copies
    VLLM_MLX_REPDETECT_INTERVAL     default 16   check every N generated tokens
    VLLM_MLX_REPDETECT_MIN_TOKENS   default 64   no checks before N generated

The MIN_SPAN gate makes short periods conservative: a period-1 run needs
48 identical tokens to trigger, while a period-16 cycle triggers after 3
copies. Legitimate text (enumerations, code boilerplate) is not exactly
periodic for 48 consecutive tokens; verbatim runs ("items items items…"),
duplicated tool-call retries and multi-line loops are.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepetitionStopConfig:
    enabled: bool = False
    window: int = 512
    min_period: int = 1
    max_period: int = 128
    min_repeats: int = 3
    min_span: int = 48
    interval: int = 16
    min_tokens: int = 64

    def sanitized(self) -> "RepetitionStopConfig":
        """Clamp nonsensical values instead of failing the server."""
        return RepetitionStopConfig(
            enabled=self.enabled,
            window=max(16, self.window),
            min_period=max(1, self.min_period),
            max_period=max(max(1, self.min_period), self.max_period),
            min_repeats=max(2, self.min_repeats),
            min_span=max(1, self.min_span),
            interval=max(1, self.interval),
            min_tokens=max(0, self.min_tokens),
        )


_config_cache: Optional[RepetitionStopConfig] = None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        logger.warning("[repetition-stop] invalid %s, using default %d", name, default)
        return default


def load_config() -> RepetitionStopConfig:
    global _config_cache
    if _config_cache is None:
        _config_cache = RepetitionStopConfig(
            enabled=os.environ.get("VLLM_MLX_REPDETECT", "0").strip().lower()
            in ("1", "true", "yes", "on"),
            window=_env_int("VLLM_MLX_REPDETECT_WINDOW", 512),
            min_period=_env_int("VLLM_MLX_REPDETECT_MIN_PERIOD", 1),
            max_period=_env_int("VLLM_MLX_REPDETECT_MAX_PERIOD", 128),
            min_repeats=_env_int("VLLM_MLX_REPDETECT_MIN_REPEATS", 3),
            min_span=_env_int("VLLM_MLX_REPDETECT_MIN_SPAN", 48),
            interval=_env_int("VLLM_MLX_REPDETECT_INTERVAL", 16),
            min_tokens=_env_int("VLLM_MLX_REPDETECT_MIN_TOKENS", 64),
        ).sanitized()
        if _config_cache.enabled:
            logger.info("[repetition-stop] enabled: %s", _config_cache)
    return _config_cache


def reset_config_cache() -> None:
    """Testing hook: force re-read of the env on next load_config()."""
    global _config_cache
    _config_cache = None


def find_repetition(
    tokens: Sequence[int], cfg: RepetitionStopConfig
) -> Optional[Tuple[int, int]]:
    """Return (period, repeats) if the tail of ``tokens`` ends in
    ``repeats`` consecutive exact copies of a ``period``-token cycle
    satisfying the config gates, else None.

    Checking blocks from the end makes the test rotation-invariant: an
    ongoing loop triggers regardless of where in the cycle the last
    token landed.
    """
    n = len(tokens)
    if isinstance(tokens, deque):  # deque slicing is O(n) per slice; copy once
        tokens = list(tokens)
    for period in range(cfg.min_period, cfg.max_period + 1):
        # Repeats needed at this period: the configured floor, and enough
        # copies to cover min_span tokens.
        need = max(cfg.min_repeats, -(-cfg.min_span // period))
        if period * need > n:
            if period * cfg.min_repeats > n:
                break  # longer periods only need more tokens — done
            continue
        cycle = tokens[n - period :]
        reps = 1
        while True:
            start = n - (reps + 1) * period
            if start < 0 or tokens[start : start + period] != cycle:
                break
            reps += 1
        if reps >= need:
            return period, reps
    return None


class RepetitionStopTracker:
    """Per-request generated-token ring buffers + interval-gated checks.

    Designed for the BatchedEngine decode loop: ``observe()`` is called
    once per generated token with the request uid and the running count
    of generated tokens; it returns a (period, repeats) hit or None.
    Buffers must be released with ``discard()`` when a request finishes
    or is removed, and ``clear()`` when the whole batch ends.
    """

    def __init__(self, cfg: Optional[RepetitionStopConfig] = None) -> None:
        self.cfg = (cfg or load_config()).sanitized()
        self._buffers: Dict[int, Deque[int]] = {}
        self.stops = 0  # cumulative, process-lifetime

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def observe(
        self, uid: int, token: int, num_generated: int
    ) -> Optional[Tuple[int, int]]:
        if not self.cfg.enabled:
            return None
        buf = self._buffers.get(uid)
        if buf is None:
            buf = deque(maxlen=self.cfg.window)
            self._buffers[uid] = buf
        buf.append(token)
        if num_generated < self.cfg.min_tokens:
            return None
        if num_generated % self.cfg.interval != 0:
            return None
        hit = find_repetition(buf, self.cfg)
        if hit is not None:
            self.stops += 1
        return hit

    def discard(self, uid: int) -> None:
        self._buffers.pop(uid, None)

    def clear(self) -> None:
        self._buffers.clear()
