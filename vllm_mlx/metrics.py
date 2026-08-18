# SPDX-License-Identifier: Apache-2.0
"""
Prometheus-first server metrics for vllm-mlx.

The public surface is a small internal abstraction that keeps instrumentation
call sites stable even if we add OpenTelemetry export later.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


# Cache-entry timing spans seconds (thrash) to a working day (healthy
# long-lived entries); log-spaced buckets cover both regimes.
_TIMING_BUCKETS = (1, 5, 15, 60, 300, 900, 1800, 3600, 7200, 14400, 43200)


@dataclass
class InferenceTracker:
    """Request-scoped inference timing and token accounting."""

    collector: "MetricsCollector | None"
    endpoint: str
    stream: bool
    start_time: float = field(default_factory=time.perf_counter)
    _finished: bool = False
    _ttft_observed: bool = False

    def observe_ttft(self) -> None:
        if self.collector is None or self._ttft_observed:
            return
        self.collector.observe_ttft(
            endpoint=self.endpoint,
            stream=self.stream,
            value=time.perf_counter() - self.start_time,
        )
        self._ttft_observed = True

    def finish(
        self,
        *,
        result: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        finish_reason: str | None = None,
    ) -> None:
        if self.collector is None or self._finished:
            return
        self.collector.observe_inference(
            endpoint=self.endpoint,
            stream=self.stream,
            result=result,
            duration=time.perf_counter() - self.start_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
        self._finished = True


class MetricsCollector:
    """Lazy Prometheus-backed metrics collector."""

    def __init__(self) -> None:
        self._enabled = False
        self._lock = threading.Lock()
        self._prom = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configure(self, *, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled
            if not enabled or self._prom is not None:
                return
            self._init_prometheus()

    def _init_prometheus(self) -> None:
        # Scrape-time delta bridge state for the cache counter mirrors:
        # metric name -> last cumulative value seen in the stats snapshot.
        self._cache_counter_seen: dict[str, float] = {}
        from prometheus_client import (
            CONTENT_TYPE_LATEST,
            CollectorRegistry,
            Counter,
            Gauge,
            Histogram,
            generate_latest,
        )

        registry = CollectorRegistry(auto_describe=True)
        self._prom = {
            "registry": registry,
            "generate_latest": generate_latest,
            "content_type": CONTENT_TYPE_LATEST,
            "http_requests_total": Counter(
                "vllm_mlx_http_requests_total",
                "HTTP requests handled by the server.",
                ["method", "path", "status_code"],
                registry=registry,
            ),
            "http_request_duration_seconds": Histogram(
                "vllm_mlx_http_request_duration_seconds",
                "HTTP request latency in seconds.",
                ["method", "path"],
                registry=registry,
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            ),
            "http_requests_in_flight": Gauge(
                "vllm_mlx_http_requests_in_flight",
                "HTTP requests currently in flight.",
                ["method", "path"],
                registry=registry,
            ),
            "inference_requests_total": Counter(
                "vllm_mlx_inference_requests_total",
                "Inference requests completed by endpoint.",
                ["endpoint", "stream", "result"],
                registry=registry,
            ),
            "inference_request_duration_seconds": Histogram(
                "vllm_mlx_inference_request_duration_seconds",
                "End-to-end inference latency in seconds.",
                ["endpoint", "stream"],
                registry=registry,
                buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            ),
            "inference_ttft_seconds": Histogram(
                "vllm_mlx_inference_ttft_seconds",
                "Time to first token for streaming endpoints.",
                ["endpoint", "stream"],
                registry=registry,
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            ),
            "prompt_tokens_total": Counter(
                "vllm_mlx_prompt_tokens_total",
                "Prompt/input tokens processed by endpoint.",
                ["endpoint", "stream"],
                registry=registry,
            ),
            "completion_tokens_total": Counter(
                "vllm_mlx_completion_tokens_total",
                "Generated output tokens produced by endpoint.",
                ["endpoint", "stream"],
                registry=registry,
            ),
            "embedding_truncated_total": Counter(
                "vllm_mlx_embedding_truncated_total",
                "Embedding inputs silently truncated under the truncate overflow policy.",
                ["model"],
                registry=registry,
            ),
            "model_loaded": Gauge(
                "vllm_mlx_model_loaded",
                "Whether a generation model is currently loaded.",
                registry=registry,
            ),
            "engine_type": Gauge(
                "vllm_mlx_engine_type",
                "Current engine mode.",
                ["engine_type"],
                registry=registry,
            ),
            "engine_is_mllm": Gauge(
                "vllm_mlx_engine_is_mllm",
                "Whether the loaded engine is multimodal.",
                registry=registry,
            ),
            "scheduler_waiting_requests": Gauge(
                "vllm_mlx_scheduler_waiting_requests",
                "Requests currently waiting in the scheduler.",
                registry=registry,
            ),
            "scheduler_running_requests": Gauge(
                "vllm_mlx_scheduler_running_requests",
                "Requests currently running in the scheduler.",
                registry=registry,
            ),
            "engine_steps_executed": Gauge(
                "vllm_mlx_engine_steps_executed",
                "Scheduler/engine steps executed since startup.",
                registry=registry,
            ),
            "engine_uptime_seconds": Gauge(
                "vllm_mlx_engine_uptime_seconds",
                "Engine uptime in seconds.",
                registry=registry,
            ),
            "metal_memory_bytes": Gauge(
                "vllm_mlx_metal_memory_bytes",
                "Metal memory usage in bytes.",
                ["kind"],
                registry=registry,
            ),
            "cache_type": Gauge(
                "vllm_mlx_cache_type",
                "Current active cache backend.",
                ["cache_type"],
                registry=registry,
            ),
            "cache_entry_count": Gauge(
                "vllm_mlx_cache_entry_count",
                "Cache entries or allocated blocks, depending on cache type.",
                registry=registry,
            ),
            "cache_hits": Gauge(
                "vllm_mlx_cache_hits",
                "Cache hits since startup/reset.",
                registry=registry,
            ),
            "cache_misses": Gauge(
                "vllm_mlx_cache_misses",
                "Cache misses since startup/reset.",
                registry=registry,
            ),
            "cache_evictions": Gauge(
                "vllm_mlx_cache_evictions",
                "Cache evictions since startup/reset.",
                registry=registry,
            ),
            "cache_hit_rate": Gauge(
                "vllm_mlx_cache_hit_rate",
                "Cache hit rate.",
                registry=registry,
            ),
            "cache_utilization_ratio": Gauge(
                "vllm_mlx_cache_utilization_ratio",
                "Cache utilization ratio.",
                registry=registry,
            ),
            "cache_memory_bytes": Gauge(
                "vllm_mlx_cache_memory_bytes",
                "Cache memory usage in bytes.",
                registry=registry,
            ),
            "cache_memory_limit_bytes": Gauge(
                "vllm_mlx_cache_memory_limit_bytes",
                "Cache memory limit in bytes when available.",
                registry=registry,
            ),
            "cache_tokens_saved": Gauge(
                "vllm_mlx_cache_tokens_saved",
                "Prompt tokens saved by cache reuse since startup/reset.",
                registry=registry,
            ),
            "cache_partial_hits": Gauge(
                "vllm_mlx_cache_partial_hits",
                "Checkpointed partial-prefix restores since startup/reset.",
                registry=registry,
            ),
            "cache_partial_tokens_saved": Gauge(
                "vllm_mlx_cache_partial_tokens_saved",
                "Prompt tokens restored via partial-prefix restore.",
                registry=registry,
            ),
            "cache_grown_stores": Gauge(
                "vllm_mlx_cache_grown_stores",
                "Cache stores that reused donor segments (grow-on-hit).",
                registry=registry,
            ),
            "cache_boundary_stores": Gauge(
                "vllm_mlx_cache_boundary_stores",
                "Prompt-boundary cache stores (abort resilience).",
                registry=registry,
            ),
            "cache_ssd_promotes": Gauge(
                "vllm_mlx_cache_ssd_promotes",
                "Entries promoted from the SSD cold tier since startup.",
                registry=registry,
            ),
            "cache_admission_deferrals": Gauge(
                "vllm_mlx_cache_admission_deferrals",
                "Co-batch admissions deferred by the memory-aware gate.",
                registry=registry,
            ),
            "cache_pressure_evictions": Gauge(
                "vllm_mlx_cache_pressure_evictions",
                "Snapshot entries evicted by memory-pressure relief.",
                registry=registry,
            ),
            "cache_pressure_skipped_stores": Gauge(
                "vllm_mlx_cache_pressure_skipped_stores",
                "Cache stores skipped while over the memory watermark.",
                registry=registry,
            ),
            "cache_pressure_clears": Gauge(
                "vllm_mlx_cache_pressure_clears",
                "Watermark breaches relieved with an MLX buffer-cache drop "
                "(#53 — fires even when the cache had nothing to evict).",
                registry=registry,
            ),
            # ---- counter mirrors of the cumulative cache stats (vLLM v1
            # metric shapes): the gauges above snapshot cumulative values,
            # which PromQL cannot window or reset-align; these are true
            # monotonic Counters fed by a scrape-time delta bridge.
            "cache_events_total": Counter(
                "vllm_mlx_cache_events",
                "Cache lifecycle events (monotonic counter mirrors of the "
                "cumulative stats gauges, fed by a scrape-time delta bridge; "
                "the gauges cannot be windowed or reset-aligned in PromQL).",
                ["event"],
                registry=registry,
            ),
            "cache_saved_tokens_total": Counter(
                "vllm_mlx_cache_saved_tokens",
                "Prompt tokens not recomputed thanks to the cache "
                "(monotonic; kind=partial counts the partial-restore subset).",
                ["kind"],
                registry=registry,
            ),
            # ---- entry-lifecycle timing (the instrumentation the
            # prefix-cache landscape verdict asked for before touching
            # eviction policy; observations drained from
            # system_kv.CacheTimingRecorder at scrape time).
            "cache_entry_lifetime_seconds": Histogram(
                "vllm_mlx_cache_entry_lifetime_seconds",
                "Cache entry lifetime: store to eviction.",
                registry=registry,
                buckets=_TIMING_BUCKETS,
            ),
            "cache_entry_idle_before_evict_seconds": Histogram(
                "vllm_mlx_cache_entry_idle_before_evict_seconds",
                "Idle time between an entry's last use and its eviction.",
                registry=registry,
                buckets=_TIMING_BUCKETS,
            ),
            "cache_entry_reuse_gap_seconds": Histogram(
                "vllm_mlx_cache_entry_reuse_gap_seconds",
                "Gap between consecutive uses of a live cache entry.",
                registry=registry,
                buckets=_TIMING_BUCKETS,
            ),
            "cache_evict_to_reuse_gap_seconds": Histogram(
                "vllm_mlx_cache_evict_to_reuse_gap_seconds",
                "Gap between evicting an entry and having to re-store the "
                "same token chain — nonzero traffic here means the eviction "
                "policy discarded something that was still needed.",
                registry=registry,
                buckets=_TIMING_BUCKETS,
            ),
            "finish_reasons_total": Counter(
                "vllm_mlx_finish_reasons_total",
                "Chat/completion requests by terminal finish_reason.",
                ["endpoint", "finish_reason"],
                registry=registry,
            ),
            "metal_resource_limit": Gauge(
                "vllm_mlx_metal_resource_limit",
                "Metal resource limit: max live GPU buffer COUNT (not bytes) "
                "before '[metal::malloc] Resource limit exceeded' — a crash "
                "class byte-denominated relief cannot see. MLX exposes only "
                "the ceiling, not the live count.",
                registry=registry,
            ),
            "metal_recommended_working_set_bytes": Gauge(
                "vllm_mlx_metal_recommended_working_set_bytes",
                "Metal max recommended working set size in bytes.",
                registry=registry,
            ),
            "model_registry_entries": Gauge(
                "vllm_mlx_model_registry_entries",
                "Tracked model ownership entries.",
                registry=registry,
            ),
            "model_registry_active_owners": Gauge(
                "vllm_mlx_model_registry_active_owners",
                "Active model owners in the registry.",
                registry=registry,
            ),
            "mcp_connected_servers": Gauge(
                "vllm_mlx_mcp_connected_servers",
                "Connected MCP servers.",
                registry=registry,
            ),
            "mcp_total_servers": Gauge(
                "vllm_mlx_mcp_total_servers",
                "Configured MCP servers.",
                registry=registry,
            ),
            "mcp_tools_available": Gauge(
                "vllm_mlx_mcp_tools_available",
                "Available MCP tools.",
                registry=registry,
            ),
        }

    def track_inference(self, endpoint: str, *, stream: bool) -> InferenceTracker:
        if not self._enabled:
            return InferenceTracker(None, endpoint, stream)
        return InferenceTracker(self, endpoint, stream)

    def observe_http_start(self, *, method: str, path: str) -> None:
        if not self._enabled or self._prom is None:
            return
        self._prom["http_requests_in_flight"].labels(method=method, path=path).inc()

    def observe_http_finish(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration: float,
    ) -> None:
        if not self._enabled or self._prom is None:
            return
        self._prom["http_requests_in_flight"].labels(method=method, path=path).dec()
        self._prom["http_requests_total"].labels(
            method=method,
            path=path,
            status_code=str(status_code),
        ).inc()
        self._prom["http_request_duration_seconds"].labels(
            method=method,
            path=path,
        ).observe(duration)

    def observe_inference(
        self,
        *,
        endpoint: str,
        stream: bool,
        result: str,
        duration: float,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str | None = None,
    ) -> None:
        if not self._enabled or self._prom is None:
            return
        stream_label = _bool_str(stream)
        if finish_reason:
            self._prom["finish_reasons_total"].labels(
                endpoint=endpoint,
                finish_reason=str(finish_reason),
            ).inc()
        self._prom["inference_requests_total"].labels(
            endpoint=endpoint,
            stream=stream_label,
            result=result,
        ).inc()
        self._prom["inference_request_duration_seconds"].labels(
            endpoint=endpoint,
            stream=stream_label,
        ).observe(duration)
        if prompt_tokens > 0:
            self._prom["prompt_tokens_total"].labels(
                endpoint=endpoint,
                stream=stream_label,
            ).inc(prompt_tokens)
        if completion_tokens > 0:
            self._prom["completion_tokens_total"].labels(
                endpoint=endpoint,
                stream=stream_label,
            ).inc(completion_tokens)

    def observe_embedding_truncated(self, *, model: str) -> None:
        if not self._enabled or self._prom is None:
            return
        self._prom["embedding_truncated_total"].labels(model=model).inc()

    def observe_ttft(self, *, endpoint: str, stream: bool, value: float) -> None:
        if not self._enabled or self._prom is None:
            return
        self._prom["inference_ttft_seconds"].labels(
            endpoint=endpoint,
            stream=_bool_str(stream),
        ).observe(value)

    def _update_engine_gauges(
        self,
        *,
        engine: Any | None,
        mcp_manager: Any | None,
    ) -> None:
        assert self._prom is not None

        stats = engine.get_stats() if engine is not None else {}

        self._prom["model_loaded"].set(1 if engine is not None else 0)
        current_engine_type = (
            stats.get("engine_type", "unknown") if engine else "unknown"
        )
        for engine_type in ("simple", "batched", "unknown"):
            self._prom["engine_type"].labels(engine_type=engine_type).set(
                1 if current_engine_type == engine_type else 0
            )
        self._prom["engine_is_mllm"].set(1 if stats.get("is_mllm") else 0)

        self._prom["scheduler_waiting_requests"].set(
            _coerce_int(stats.get("num_waiting"))
        )
        self._prom["scheduler_running_requests"].set(
            _coerce_int(stats.get("num_running"))
        )
        self._prom["engine_steps_executed"].set(
            _coerce_int(stats.get("steps_executed"))
        )
        self._prom["engine_uptime_seconds"].set(
            _coerce_float(stats.get("uptime_seconds"))
        )

        self._prom["metal_memory_bytes"].labels(kind="active").set(
            _coerce_float(stats.get("metal_active_memory_gb")) * 1e9
        )
        self._prom["metal_memory_bytes"].labels(kind="peak").set(
            _coerce_float(stats.get("metal_peak_memory_gb")) * 1e9
        )
        self._prom["metal_memory_bytes"].labels(kind="cache").set(
            _coerce_float(stats.get("metal_cache_memory_gb")) * 1e9
        )

        # Prefer the cache block with actual ACTIVITY over declaration order.
        # MLLM-classified models whose text goes through the LLM route (e.g.
        # Qwen3.5-4B/9B, Qwen3.6-27B) expose an idle memory_aware_cache
        # (media path, unused) alongside the active system_kv_cache — the old
        # first-present-wins scan masked the live counters behind zeros.
        # This was the root cause of the "dense 27B cache observability"
        # mystery (2026-06-01 deploy notes).
        cache_type = "none"
        cache_stats = None
        _present = [
            (c, stats[c])
            for c in (
                "memory_aware_cache",
                "paged_cache",
                "prefix_cache",
                "system_kv_cache",
            )
            if isinstance(stats.get(c), dict)
        ]

        def _cache_activity(d: dict) -> float:
            return sum(
                _coerce_float(d.get(k, 0))
                for k in ("hits", "misses", "tokens_saved")
            )

        _active = sorted(
            ((c, d) for c, d in _present if _cache_activity(d) > 0),
            key=lambda cd: cd[0] != "system_kv_cache",
        )
        if _active:
            cache_type, cache_stats = _active[0]
        elif _present:
            cache_type, cache_stats = _present[0]

        for candidate in ("none", "prefix_cache", "memory_aware_cache", "paged_cache", "system_kv_cache"):
            self._prom["cache_type"].labels(cache_type=candidate).set(
                1 if cache_type == candidate else 0
            )

        if isinstance(cache_stats, dict):
            self._prom["cache_entry_count"].set(
                _coerce_float(
                    cache_stats.get(
                        "entry_count", cache_stats.get("allocated_blocks", 0)
                    )
                )
            )
            self._prom["cache_hits"].set(
                _coerce_float(cache_stats.get("hits", cache_stats.get("cache_hits", 0)))
            )
            self._prom["cache_misses"].set(
                _coerce_float(
                    cache_stats.get("misses", cache_stats.get("cache_misses", 0))
                )
            )
            self._prom["cache_evictions"].set(
                _coerce_float(cache_stats.get("evictions", 0))
            )
            self._prom["cache_hit_rate"].set(
                _coerce_float(
                    cache_stats.get("hit_rate", cache_stats.get("cache_hit_rate", 0.0))
                )
            )
            self._prom["cache_utilization_ratio"].set(
                _coerce_float(
                    cache_stats.get(
                        "memory_utilization", cache_stats.get("utilization", 0.0)
                    )
                )
            )
            self._prom["cache_tokens_saved"].set(
                _coerce_float(cache_stats.get("tokens_saved", 0))
            )
            self._prom["cache_partial_hits"].set(
                _coerce_float(cache_stats.get("partial_hits", 0))
            )
            self._prom["cache_partial_tokens_saved"].set(
                _coerce_float(cache_stats.get("partial_tokens_saved", 0))
            )
            self._prom["cache_grown_stores"].set(
                _coerce_float(cache_stats.get("grown_stores", 0))
            )
            self._prom["cache_boundary_stores"].set(
                _coerce_float(cache_stats.get("boundary_stores", 0))
            )
            self._prom["cache_ssd_promotes"].set(
                _coerce_float(cache_stats.get("ssd_promotes", 0))
            )
            self._prom["cache_admission_deferrals"].set(
                _coerce_float(cache_stats.get("admission_deferrals", 0))
            )
            self._prom["cache_pressure_evictions"].set(
                _coerce_float(cache_stats.get("pressure_evictions", 0))
            )
            self._prom["cache_pressure_skipped_stores"].set(
                _coerce_float(cache_stats.get("pressure_skipped_stores", 0))
            )
            self._prom["cache_pressure_clears"].set(
                _coerce_float(cache_stats.get("pressure_cache_clears", 0))
            )
            self._prom["cache_memory_bytes"].set(
                _coerce_float(cache_stats.get("current_memory_mb", 0.0)) * 1024 * 1024
            )
            self._prom["cache_memory_limit_bytes"].set(
                _coerce_float(cache_stats.get("max_memory_mb", 0.0)) * 1024 * 1024
            )
        else:
            self._prom["cache_entry_count"].set(0)
            self._prom["cache_hits"].set(0)
            self._prom["cache_misses"].set(0)
            self._prom["cache_evictions"].set(0)
            self._prom["cache_hit_rate"].set(0)
            self._prom["cache_utilization_ratio"].set(0)
            self._prom["cache_tokens_saved"].set(0)
            self._prom["cache_partial_hits"].set(0)
            self._prom["cache_partial_tokens_saved"].set(0)
            self._prom["cache_memory_bytes"].set(0)
            self._prom["cache_memory_limit_bytes"].set(0)

        # Counter mirrors: bridge the cumulative snapshot values into true
        # monotonic Counters. delta < 0 means the engine (and its stats)
        # was reset in-process — count the new cumulative from zero.
        counter_sources = {
            ("cache_events_total", "hit"): ("hits", "cache_hits"),
            ("cache_events_total", "miss"): ("misses", "cache_misses"),
            ("cache_events_total", "eviction"): ("evictions",),
            ("cache_events_total", "partial_hit"): ("partial_hits",),
            ("cache_events_total", "pressure_eviction"): ("pressure_evictions",),
            ("cache_events_total", "ssd_promote"): ("ssd_promotes",),
            ("cache_saved_tokens_total", "full"): ("tokens_saved",),
            ("cache_saved_tokens_total", "partial"): ("partial_tokens_saved",),
        }
        if isinstance(cache_stats, dict):
            for (metric_name, label), keys in counter_sources.items():
                value = 0.0
                for key in keys:
                    if key in cache_stats:
                        value = _coerce_float(cache_stats.get(key))
                        break
                seen_key = f"{metric_name}:{label}"
                seen = self._cache_counter_seen.get(seen_key)
                delta = value if seen is None or value < seen else value - seen
                self._cache_counter_seen[seen_key] = value
                if delta > 0:
                    if metric_name == "cache_events_total":
                        self._prom[metric_name].labels(event=label).inc(delta)
                    else:
                        self._prom[metric_name].labels(kind=label).inc(delta)

        # Entry-lifecycle timing: drain every live CacheTimingRecorder.
        try:
            from .system_kv import drain_all_timing_observations

            observations = drain_all_timing_observations()
        except Exception:
            observations = {}
        for obs_key, metric_name in (
            ("lifetime", "cache_entry_lifetime_seconds"),
            ("idle_before_evict", "cache_entry_idle_before_evict_seconds"),
            ("reuse_gap", "cache_entry_reuse_gap_seconds"),
            ("evict_to_reuse_gap", "cache_evict_to_reuse_gap_seconds"),
        ):
            for value in observations.get(obs_key, ()):
                self._prom[metric_name].observe(value)

        # Metal ceilings (MLX exposes limits, not live buffer counts).
        try:
            import mlx.core as mx

            device_info = mx.device_info()
            self._prom["metal_resource_limit"].set(
                _coerce_float(device_info.get("resource_limit"))
            )
            self._prom["metal_recommended_working_set_bytes"].set(
                _coerce_float(device_info.get("max_recommended_working_set_size"))
            )
        except Exception:
            pass

        try:
            from .model_registry import get_registry

            registry_stats = get_registry().get_stats()
        except Exception:
            registry_stats = {}
        self._prom["model_registry_entries"].set(
            _coerce_int(registry_stats.get("total_entries"))
        )
        self._prom["model_registry_active_owners"].set(
            _coerce_int(registry_stats.get("active_owners"))
        )

        if mcp_manager is not None:
            try:
                statuses = list(mcp_manager.get_server_status())
                connected = sum(1 for s in statuses if s.state.value == "connected")
                total = len(statuses)
                tools = len(mcp_manager.get_all_tools())
            except Exception:
                connected = total = tools = 0
        else:
            connected = total = tools = 0
        self._prom["mcp_connected_servers"].set(connected)
        self._prom["mcp_total_servers"].set(total)
        self._prom["mcp_tools_available"].set(tools)

    def render_metrics(
        self,
        *,
        engine: Any | None,
        mcp_manager: Any | None,
    ) -> tuple[bytes, str]:
        if not self._enabled:
            raise RuntimeError("metrics_disabled")
        if self._prom is None:
            self._init_prometheus()
        self._update_engine_gauges(engine=engine, mcp_manager=mcp_manager)
        return (
            self._prom["generate_latest"](self._prom["registry"]),
            self._prom["content_type"],
        )


metrics = MetricsCollector()
