# SPDX-License-Identifier: Apache-2.0
"""(#143) Persisted prefix-cache directories shed files no load can read."""

import json
import sys
import types
from unittest.mock import MagicMock, patch

from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from vllm_mlx.prefix_cache_disk_gc import gc_prefix_cache_dir

KEEP = "f54434177bddf73c"
OLD = "0123456789abcdef"


def _touch(path, data=b"x" * 10):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _subdir(root, name, listed, extra=()):
    d = root / name
    for i in list(listed) + list(extra):
        _touch(d / f"entry_{i}.safetensors")
        _touch(d / f"entry_{i}_tokens.bin")
    (d / "index.json").write_text(
        json.dumps({"entries": [{"index": i} for i in listed]})
    )
    return d


def test_stale_entries_old_provenance_and_legacy_go_everything_else_stays(tmp_path):
    _subdir(tmp_path, KEEP, [0, 1], extra=[2, 3])  # last save had 4, this one 2
    _subdir(tmp_path, OLD, [0, 1, 2])  # previous mlx version
    (tmp_path / "index.json").write_text(json.dumps({"version": 4, "entries": []}))
    _touch(tmp_path / "entry_0.safetensors")  # legacy, no provenance
    _touch(tmp_path / "notes" / "keep.txt")  # not a provenance-shaped dir
    _touch(tmp_path / "README")  # not a cache file name

    freed = gc_prefix_cache_dir(str(tmp_path), KEEP)

    assert freed > 0
    assert sorted(p.name for p in (tmp_path / KEEP).iterdir()) == [
        "entry_0.safetensors", "entry_0_tokens.bin",
        "entry_1.safetensors", "entry_1_tokens.bin", "index.json",
    ]  # fmt: skip
    assert not (tmp_path / OLD).exists()
    assert not (tmp_path / "index.json").exists()
    assert not (tmp_path / "entry_0.safetensors").exists()
    assert (tmp_path / "notes" / "keep.txt").exists() and (tmp_path / "README").exists()


def test_a_loadable_top_level_fallback_is_kept_until_a_subdir_supersedes_it(tmp_path):
    """Upstream #747 still loads a top-level index that carries provenance."""
    idx = {"version": 4, "model_revision": "abc", "quant_signature": "", "entries": []}
    (tmp_path / "index.json").write_text(json.dumps(idx))
    gc_prefix_cache_dir(str(tmp_path), KEEP)
    assert (tmp_path / "index.json").exists()
    _subdir(tmp_path, KEEP, [0])
    gc_prefix_cache_dir(str(tmp_path), KEEP)
    assert not (tmp_path / "index.json").exists()


def test_load_time_call_leaves_top_level_for_the_loader_to_judge(tmp_path):
    (tmp_path / "index.json").write_text(json.dumps({"version": 3, "entries": []}))
    _subdir(tmp_path, OLD, [0])
    gc_prefix_cache_dir(str(tmp_path), KEEP, top_level=False)
    assert (tmp_path / "index.json").exists() and not (tmp_path / OLD).exists()


def test_unreadable_index_prunes_no_entries_and_the_switch_disables_all(
    tmp_path, monkeypatch
):
    d = _subdir(tmp_path, KEEP, [0], extra=[1])
    (d / "index.json").write_text("{not json")
    gc_prefix_cache_dir(str(tmp_path), KEEP)
    assert (d / "entry_1.safetensors").exists()

    _subdir(tmp_path, OLD, [0])
    monkeypatch.setenv("VLLM_MLX_PREFIX_CACHE_DISK_GC", "0")
    assert gc_prefix_cache_dir(str(tmp_path), KEEP) == 0
    assert (tmp_path / OLD).exists()


def test_save_with_fewer_entries_leaves_no_stale_files(tmp_path):
    """Real save_to_disk twice: 3 entries, then 1 — only the 1 remains."""
    from tests.test_memory_cache import MockKVCache

    def _fake_save(path, data, metadata=None):
        with open(path, "wb") as f:
            f.write(b"data")

    fake = types.ModuleType("mlx_lm")
    fake.models = types.ModuleType("mlx_lm.models")
    fake.models.cache = types.ModuleType("mlx_lm.models.cache")
    fake.models.cache.save_prompt_cache = _fake_save
    config = MemoryCacheConfig(max_memory_mb=100, min_prefix_tokens=1)
    cache = MemoryAwarePrefixCache(MagicMock(), config)
    for i in range(3):
        cache.store([i + 1, i + 2, i + 3], [MockKVCache(500, 500)])
    mods = {
        "mlx_lm": fake,
        "mlx_lm.models": fake.models,
        "mlx_lm.models.cache": fake.models.cache,
    }
    # Built OUTSIDE the fake mlx_lm: runtime versions are part of the
    # provenance hash, and a cache built under the fake gets its own subdir
    # (which the GC would then rightly remove as a stale provenance).
    smaller = MemoryAwarePrefixCache(MagicMock(), config)
    smaller.store([9, 9, 9], [MockKVCache(500, 500)])
    assert smaller._provenance_hash == cache._provenance_hash
    sub = tmp_path / cache._provenance_hash
    with patch.dict(sys.modules, mods):
        assert cache.save_to_disk(str(tmp_path))
        assert len(list(sub.glob("entry_*.safetensors"))) == 3
        assert smaller.save_to_disk(str(tmp_path))
    assert sorted(p.name for p in sub.glob("entry_*")) == [
        "entry_0.safetensors",
        "entry_0_tokens.bin",
    ]
