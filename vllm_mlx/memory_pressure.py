# SPDX-License-Identifier: Apache-2.0
"""Metal memory-watermark pressure relief (fork patches #48/#53, seam #58).

Extracted from ``batched_system_kv.py`` so schedulers whose cache is NOT a
``BatchedSystemKV`` bag (the batched MLLM path) can reuse the exact relief
discipline: threshold from the device's recommended working set × an env
watermark percentage, PEAK-since-last-check trigger (intra-chunk transients
never show up in instantaneous active readings), buffer-cache drop on any
breach even with nothing to evict (#53), then LRU eviction until active
memory is back under the threshold.

The eviction side is generic: callers hand ``relieve()`` a ``drop_lru``
callable returning True while there is something left to drop. Counters
stay on the owning cache object (stats surfaces are unchanged).
"""

import logging
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)


class PressureManager:
    """Watermark math + relief loop; one instance per owning cache."""

    def __init__(self, watermark_pct: int):
        self.watermark_pct = watermark_pct
        # Cached after the first Metal query — this runs once per scheduler
        # step once the relief hook is wired.
        self._ws_ceiling: Optional[int] = None

    # ------------------------------------------------------------- threshold

    def threshold_bytes(self) -> Optional[float]:
        """Watermark threshold in bytes, or None when disabled/unavailable.

        The ceiling (the device's recommended working set — tracks the
        raised ``iogpu.wired_limit_mb`` on the Studio) is cached after the
        first Metal query."""
        if self.watermark_pct <= 0:
            return None
        try:
            import mlx.core as mx

            if self._ws_ceiling is None:
                self._ws_ceiling = mx.device_info()[
                    "max_recommended_working_set_size"
                ]
            if not self._ws_ceiling or self._ws_ceiling <= 0:
                return None
            return self._ws_ceiling * self.watermark_pct / 100
        except Exception:
            return None

    def watermark_status(self) -> tuple:
        """``(over, active_bytes, ceiling_bytes)`` against the watermark.
        ``(False, 0, 0)`` when the watermark env is unset or Metal is
        unavailable."""
        threshold = self.threshold_bytes()
        if threshold is None:
            return False, 0, 0
        try:
            import mlx.core as mx

            active = mx.get_active_memory()
            return active > threshold, active, self._ws_ceiling
        except Exception:
            return False, 0, 0

    def under_pressure(self) -> bool:
        return self.watermark_status()[0]

    # ----------------------------------------------------------------- relief

    def relieve(
        self,
        drop_lru: Callable[[], bool],
        *,
        log_label: str,
        on_cache_clear: Optional[Callable[[], None]] = None,
    ) -> Tuple[bool, int]:
        """Run one relief pass; returns ``(cache_cleared, evicted_count)``.

        ``on_cache_clear`` fires immediately after the breach-triggered
        buffer-cache drop (counter hook — it must fire even if a later
        eviction raises, matching the pre-extraction semantics).

        Peak, not instantaneous active: relief hooks run BETWEEN prefill
        chunks, exactly where each chunk's attention transients (multi-GB
        at deep context) have just been freed — the live 2026-07-09 deploy
        smoke proved a 94K-token prefill peaks at 59.6 GB intra-chunk while
        every inter-chunk active reading sits just UNDER the threshold, so
        an active-based trigger never fires. ``get_peak_memory`` is read
        and reset each call, making the window "since the previous check"
        (side effect: the peak gauge in scheduler stats becomes
        peak-since-last-step on watermark-armed routes — the recent
        transient max, which is the number that actually kills the
        process).

        The MLX buffer cache is dropped on any breach even when there is
        nothing to evict (#53): the 2026-07-13 Coder-Next crash showed a
        137K prefill surviving on a fresh process and the identical repeat
        prefill SIGABRTing — the no-op eviction path left the allocator
        holding round one's multi-GB transient buffers as wired memory.

        ``drop_lru`` drops the owning cache's least-recently-used entry and
        returns False once the cache is empty; eviction stops early once
        instantaneous active is back under the threshold, with a buffer-
        cache clear after each drop so freed buffers actually leave the
        process before the next chunk.
        """
        threshold = self.threshold_bytes()
        if threshold is None:
            return False, 0
        try:
            import mlx.core as mx

            peak = mx.get_peak_memory()
            mx.reset_peak_memory()
            if peak <= threshold:
                return False, 0

            # over the watermark: return the allocator's cached buffers to
            # Metal first — the only relief available on a bagless process
            mx.clear_cache()
            if on_cache_clear is not None:
                on_cache_clear()

            evicted = 0
            while drop_lru():
                evicted += 1
                mx.clear_cache()
                if mx.get_active_memory() <= threshold:
                    break
            if evicted:
                logger.warning(
                    "[%s] memory pressure: evicted %d "
                    "snapshot entr%s (peak %.1f GB since last step, "
                    "watermark %d%% of %.1f GB)",
                    log_label,
                    evicted,
                    "y" if evicted == 1 else "ies",
                    peak / 1e9,
                    self.watermark_pct,
                    (self._ws_ceiling or 0) / 1e9,
                )
            return True, evicted
        except Exception:
            logger.debug("[%s] pressure relief failed", log_label, exc_info=True)
            return False, 0
