# SPDX-License-Identifier: Apache-2.0
"""Fork observability (#74): cache timing recorder, counter mirrors,
/health/ready forward-pass probe.

The timing histograms are the instrumentation the prefix-cache landscape
verdict (docs/fork/prefix-caching-landscape-2026-08.md) required before any
eviction-policy work: idle-before-evict vs reuse-gap distributions — plus the
evict-to-re-store tombstone gap — decide empirically whether recency-only
eviction ever discards something still needed on this box.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_mlx.system_kv import (
    CacheTimingRecorder,
    drain_all_timing_observations,
    timing_key,
)


class TestCacheTimingRecorder:
    def test_store_then_evict_observes_lifetime_and_idle(self):
        rec = CacheTimingRecorder()
        rec.note_store("a")
        rec.note_evict("a")
        obs = rec.drain()
        assert len(obs["lifetime"]) == 1
        assert len(obs["idle_before_evict"]) == 1
        assert obs["lifetime"][0] >= 0.0
        assert obs["reuse_gap"] == []
        assert obs["evict_to_reuse_gap"] == []

    def test_hit_observes_reuse_gap_and_refreshes_idle_base(self):
        rec = CacheTimingRecorder()
        rec.note_store("a")
        rec.note_hit("a")
        rec.note_hit("a")
        obs = rec.drain()
        assert len(obs["reuse_gap"]) == 2

    def test_evict_then_restore_hits_the_tombstone(self):
        """The smoking-gun metric: a re-store of an evicted chain."""
        rec = CacheTimingRecorder()
        rec.note_store("a")
        rec.note_evict("a")
        rec.note_store("a")
        obs = rec.drain()
        assert len(obs["evict_to_reuse_gap"]) == 1
        # ... and the second store starts a fresh live entry.
        rec.note_evict("a")
        obs = rec.drain()
        assert len(obs["lifetime"]) == 1

    def test_forget_is_silent(self):
        """Absorbed/subsumed entries observe nothing and leave no tombstone."""
        rec = CacheTimingRecorder()
        rec.note_store("a")
        rec.forget("a")
        obs = rec.drain()
        assert all(v == [] for v in obs.values())
        rec.note_store("a")
        assert rec.drain()["evict_to_reuse_gap"] == []

    def test_hit_on_untracked_key_starts_tracking(self):
        rec = CacheTimingRecorder()
        rec.note_hit("pre-existing")
        assert rec.drain()["reuse_gap"] == []
        rec.note_hit("pre-existing")
        assert len(rec.drain()["reuse_gap"]) == 1

    def test_drain_clears(self):
        rec = CacheTimingRecorder()
        rec.note_store("a")
        rec.note_evict("a")
        assert rec.drain()["lifetime"]
        assert rec.drain()["lifetime"] == []

    def test_bounds_hold(self):
        rec = CacheTimingRecorder()
        for i in range(rec.MAX_TOMBSTONES + 100):
            rec.note_store(i)
            rec.note_evict(i)
        assert len(rec._tombstones) == rec.MAX_TOMBSTONES
        obs = rec.drain()
        assert len(obs["lifetime"]) == rec.MAX_OBS

    def test_global_drain_merges_and_clears(self):
        rec = CacheTimingRecorder()
        rec.note_store("x")
        rec.note_evict("x")
        merged = drain_all_timing_observations()
        assert len(merged["lifetime"]) >= 1
        assert rec.drain()["lifetime"] == []

    def test_timing_key_is_content_stable(self):
        assert timing_key([1, 2, 3]) == timing_key([1, 2, 3])
        assert timing_key([1, 2, 3]) != timing_key([1, 2, 4])


class TestCounterMirrors:
    """The scrape-time delta bridge from cumulative stats to Counters."""

    def _collector(self):
        pytest.importorskip("prometheus_client")
        from vllm_mlx.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.configure(enabled=True)
        collector._init_prometheus()
        return collector

    def _engine_with(self, hits, misses, evictions):
        engine = MagicMock()
        engine.get_stats.return_value = {
            "engine_type": "simple",
            "system_kv_cache": {
                "hits": hits,
                "misses": misses,
                "evictions": evictions,
                "hit_rate": 0.5,
                "tokens_saved": 0,
                "partial_hits": 0,
            },
        }
        return engine

    def _event_value(self, collector, event):
        body, _ = collector.render_metrics(engine=None, mcp_manager=None)
        needle = f'vllm_mlx_cache_events_total{{event="{event}"}} '
        for line in body.decode().splitlines():
            if line.startswith(needle):
                return float(line.rsplit(" ", 1)[1])
        return None

    def test_counters_track_cumulative_deltas(self):
        collector = self._collector()
        collector._update_engine_gauges(
            engine=self._engine_with(5, 2, 1), mcp_manager=None
        )
        collector._update_engine_gauges(
            engine=self._engine_with(9, 4, 1), mcp_manager=None
        )
        assert self._event_value(collector, "hit") == 9.0
        assert self._event_value(collector, "miss") == 4.0
        assert self._event_value(collector, "eviction") == 1.0

    def test_stats_reset_counts_from_zero_not_negative(self):
        """An in-process engine reset must not decrement or explode."""
        collector = self._collector()
        collector._update_engine_gauges(
            engine=self._engine_with(100, 50, 3), mcp_manager=None
        )
        collector._update_engine_gauges(
            engine=self._engine_with(4, 1, 0), mcp_manager=None
        )
        assert self._event_value(collector, "hit") == 104.0

    def test_timing_observations_land_in_histograms(self):
        collector = self._collector()
        rec = CacheTimingRecorder()
        rec.note_store("k")
        rec.note_evict("k")
        collector._update_engine_gauges(engine=None, mcp_manager=None)
        body, _ = collector.render_metrics(engine=None, mcp_manager=None)
        text = body.decode()
        assert "vllm_mlx_cache_entry_lifetime_seconds_count 1.0" in text

    def test_finish_reason_counter(self):
        collector = self._collector()
        collector.observe_inference(
            endpoint="chat",
            stream=False,
            result="success",
            duration=0.1,
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="tool_calls",
        )
        body, _ = collector.render_metrics(engine=None, mcp_manager=None)
        assert (
            'vllm_mlx_finish_reasons_total{endpoint="chat",finish_reason="tool_calls"} 1.0'
            in body.decode()
        )

    def test_empty_completion_counter(self):
        """#87: a "successful" zero-token completion is the silent empty
        turn the gateway layers erase — count it at ground truth."""
        collector = self._collector()
        collector.observe_inference(
            endpoint="chat",
            stream=True,
            result="success",
            duration=0.1,
            prompt_tokens=100,
            completion_tokens=0,
            finish_reason="stop",
        )
        # non-empty and non-success must NOT count
        collector.observe_inference(
            endpoint="chat", stream=True, result="success",
            duration=0.1, prompt_tokens=1, completion_tokens=5,
        )
        collector.observe_inference(
            endpoint="chat", stream=True, result="error",
            duration=0.1, prompt_tokens=1, completion_tokens=0,
        )
        text = collector.render_metrics(engine=None, mcp_manager=None)[0].decode()
        assert 'vllm_mlx_empty_completions_total{endpoint="chat"} 1.0' in text


class TestHealthReady:
    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient

        from vllm_mlx.server import app

        return TestClient(app)

    def test_no_engine_is_503(self, client, monkeypatch):
        import vllm_mlx.server as srv

        monkeypatch.setattr(srv, "_engine", None)
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["detail"]["reason"] == "no engine loaded"

    def test_idle_engine_probes_one_token(self, client, monkeypatch):
        import vllm_mlx.server as srv

        engine = MagicMock()
        engine.get_stats.return_value = {"num_running": 0, "num_waiting": 0}

        async def fake_generate(**kwargs):
            assert kwargs["max_tokens"] == 1
            return SimpleNamespace(text="x", tokens=[42])

        engine.generate = fake_generate
        monkeypatch.setattr(srv, "_engine", engine)
        response = client.get("/health/ready")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ready"
        assert payload["probe"] == "forward_pass"
        assert payload["probe_latency_s"] >= 0

    def test_busy_engine_is_ready_without_probing(self, client, monkeypatch):
        """Busy means alive; the probe must never queue behind real traffic."""
        import vllm_mlx.server as srv

        engine = MagicMock()
        engine.get_stats.return_value = {"num_running": 2, "num_waiting": 1}

        async def must_not_run(**kwargs):  # pragma: no cover
            raise AssertionError("probe ran against a busy engine")

        engine.generate = must_not_run
        monkeypatch.setattr(srv, "_engine", engine)
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["probe"] == "skipped"

    def test_probe_failure_is_503(self, client, monkeypatch):
        import vllm_mlx.server as srv

        engine = MagicMock()
        engine.get_stats.return_value = {"num_running": 0, "num_waiting": 0}

        async def broken(**kwargs):
            raise RuntimeError("Metal exploded")

        engine.generate = broken
        monkeypatch.setattr(srv, "_engine", engine)
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert "Metal exploded" in response.json()["detail"]["reason"]

    def test_empty_probe_output_is_503(self, client, monkeypatch):
        import vllm_mlx.server as srv

        engine = MagicMock()
        engine.get_stats.return_value = {}

        async def empty(**kwargs):
            return SimpleNamespace(text="", tokens=[])

        engine.generate = empty
        monkeypatch.setattr(srv, "_engine", engine)
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["detail"]["reason"] == "probe produced no output"
