"""Patch #175 — mlx-lm symbols resolved on both the 0.32.2 layout and the
post-#2082 ("packaged") one.

The installed-layout tests use the real mlx-lm. The packaged-layout tests
cannot: mlx-lm with #2082 is unreleased and the shared venv must not change.
They build a FAKE package tree in ``tmp_path`` that mirrors the file layout of
mlx-lm at ``2d4df8d`` (the #2082 merge) and point the resolver's candidate
paths at it by rewriting the ``mlx_lm.`` prefix. This is the ONE place the
fork's #175 tests fake module paths, and only to exercise the resolver's
packaged branch; whether the real hooks bind on the installed mlx-lm is
pinned without fakes in ``tests/test_fork_invariants.py`` (``test_175_*``).
"""

import inspect
import sys
import textwrap

import pytest

from vllm_mlx import mlx_lm_compat as compat

FAKE = "_mlxlm_2082_fake"

# Post-#2082 layout, trimmed to the signatures the resolver checks. The
# legacy modules that still exist after the move (models/base.py with only
# BaseModelArgs, models/cache.py with the differently-shaped
# create_attention_mask, models/qwen3_5 as a package exporting only Model and
# ModelArgs) are included, so a resolver that tried the old path first, or
# the cache module at all, would bind the wrong thing or nothing.
_TREE = {
    "__init__.py": "",
    "layers/__init__.py": "",
    "layers/moe.py": """
        class SwitchGLU:
            pass
        def _gather_sort(x, indices):
            pass
        def _scatter_unsort(x, inv_order, shape=None):
            pass
    """,
    "layers/recurrent.py": """
        def normalize_qk(q, k, *, inv_scale, eps):
            pass
        def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None,
                               mask=None, *, use_kernel=True, lower_bound=None):
            pass
        def create_ssm_mask(h, cache=None):
            pass
    """,
    "layers/attention.py": """
        def scaled_dot_product_attention(queries, keys, values, cache, scale,
                                         mask, sinks=None):
            pass
        def create_attention_mask(h, cache=None, window_size=None,
                                  return_array=False):
            pass
    """,
    "models/__init__.py": "",
    "models/base.py": """
        class BaseModelArgs:
            pass
    """,
    "models/cache.py": """
        def create_attention_mask(N, offset, return_array, window_size):
            pass
    """,
    "models/qwen3_5/__init__.py": """
        from .config import ModelArgs
        from .model import Model
    """,
    "models/qwen3_5/config.py": """
        class TextModelArgs:
            pass
        class ModelArgs:
            pass
    """,
    "models/qwen3_5/model.py": """
        class Model:
            pass
    """,
    "models/qwen3_5/layers.py": """
        class GatedDeltaNet:
            pass
        class DecoderLayer:
            pass
    """,
    "models/qwen3_next/__init__.py": """
        class Model:
            pass
    """,
    "models/qwen3_next/layers.py": """
        class Qwen3NextDecoderLayer:
            pass
    """,
}


@pytest.fixture
def packaged(tmp_path, monkeypatch):
    root = tmp_path / FAKE
    for rel, body in _TREE.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    monkeypatch.syspath_prepend(str(tmp_path))
    rewritten = {
        name: [(lay, FAKE + mod[len("mlx_lm") :], attr) for lay, mod, attr in cands]
        for name, cands in compat.CANDIDATES.items()
    }
    monkeypatch.setattr(compat, "CANDIDATES", rewritten)
    monkeypatch.setattr(compat, "_resolved", {})
    yield
    for mod in [m for m in sys.modules if m == FAKE or m.startswith(FAKE + ".")]:
        del sys.modules[mod]


_PACKAGED_HOME = {
    "SwitchGLU": "layers.moe",
    "_gather_sort": "layers.moe",
    "_scatter_unsort": "layers.moe",
    "gated_delta_update": "layers.recurrent",
    "normalize_qk": "layers.recurrent",
    "create_ssm_mask": "layers.recurrent",
    "create_attention_mask": "layers.attention",
    "scaled_dot_product_attention": "layers.attention",
    "qwen3_5.GatedDeltaNet": "models.qwen3_5.layers",
    "qwen3_5.DecoderLayer": "models.qwen3_5.layers",
    "qwen3_5.TextModelArgs": "models.qwen3_5.config",
    "qwen3_next.Qwen3NextDecoderLayer": "models.qwen3_next.layers",
}


def test_175_every_candidate_has_a_packaged_home():
    assert set(compat.CANDIDATES) == set(_PACKAGED_HOME)


@pytest.mark.parametrize("name", sorted(_PACKAGED_HOME))
def test_175_packaged_layout_resolves_at_the_new_path(packaged, name):
    obj, layout, where = compat.resolve_with_origin(name)
    attr = name.rsplit(".", 1)[-1]
    assert layout == compat.PACKAGED
    assert where == f"{FAKE}.{_PACKAGED_HOME[name]}.{attr}"
    assert obj.__module__ == f"{FAKE}.{_PACKAGED_HOME[name]}"


def test_175_packaged_layout_reports_itself(packaged):
    assert compat.layout() == compat.PACKAGED


def test_175_cache_create_attention_mask_is_rejected_by_signature(
    packaged, monkeypatch
):
    """The #2082 trap: the cache module's create_attention_mask takes
    ``(N, offset, ...)``. Pointed at it, the resolver must refuse, not bind."""
    monkeypatch.setitem(
        compat.CANDIDATES,
        "create_attention_mask",
        [(compat.PACKAGED, f"{FAKE}.models.cache", "create_attention_mask")],
    )
    with pytest.raises(compat.MlxLmSymbolMissing, match="signature"):
        compat.resolve("create_attention_mask")


def test_175_real_cache_create_attention_mask_fails_the_check():
    """Same trap on the installed mlx-lm (present on both layouts)."""
    from mlx_lm.models import cache

    assert compat._signature_problem(
        "create_attention_mask", cache.create_attention_mask
    )


def test_175_nowhere_names_the_symbol_and_every_path(monkeypatch):
    monkeypatch.setattr(compat, "_resolved", {})
    monkeypatch.setitem(
        compat.CANDIDATES,
        "SwitchGLU",
        [
            (compat.PACKAGED, "_no_such_pkg_175.layers.moe", "SwitchGLU"),
            (compat.LEGACY, "mlx_lm.models.base", "SwitchGLU"),
        ],
    )
    with pytest.raises(compat.MlxLmSymbolMissing) as e:
        compat.resolve("SwitchGLU")
    msg = str(e.value)
    assert "SwitchGLU" in msg
    assert "_no_such_pkg_175.layers.moe.SwitchGLU" in msg
    assert "mlx_lm.models.base.SwitchGLU (module has no SwitchGLU)" in msg
    assert compat.missing(["SwitchGLU"]) == [msg]
    assert compat.layout() == "unknown"


def test_175_installed_legacy_layout_binds_what_the_old_imports_bound():
    """On 0.32.2 every lookup is the object the replaced hard-coded import
    gave, so behaviour on the current pin is unchanged."""
    if compat.layout() != compat.LEGACY:
        pytest.skip("installed mlx-lm is not the 0.32.2 layout")
    from mlx_lm.models import base, gated_delta, qwen3_5, qwen3_next, switch_layers

    assert compat.resolve("SwitchGLU") is switch_layers.SwitchGLU
    assert compat.resolve("_gather_sort") is switch_layers._gather_sort
    assert compat.resolve("_scatter_unsort") is switch_layers._scatter_unsort
    assert compat.resolve("gated_delta_update") is qwen3_5.gated_delta_update
    assert compat.resolve("normalize_qk") is qwen3_5.normalize_qk
    assert compat.resolve("gated_delta_update") is gated_delta.gated_delta_update
    assert compat.resolve("create_attention_mask") is base.create_attention_mask
    assert compat.resolve("create_ssm_mask") is base.create_ssm_mask
    assert (
        compat.resolve("scaled_dot_product_attention")
        is base.scaled_dot_product_attention
    )
    assert compat.resolve("qwen3_5.GatedDeltaNet") is qwen3_5.GatedDeltaNet
    assert compat.resolve("qwen3_5.DecoderLayer") is qwen3_5.DecoderLayer
    assert compat.resolve("qwen3_5.TextModelArgs") is qwen3_5.TextModelArgs
    assert (
        compat.resolve("qwen3_next.Qwen3NextDecoderLayer")
        is qwen3_next.Qwen3NextDecoderLayer
    )


def test_175_signature_table_matches_the_installed_functions():
    for name, want in compat.SIGNATURES.items():
        have = tuple(inspect.signature(compat.resolve(name)).parameters)
        assert have[: len(want)] == want, name


def test_175_a_packaged_module_that_breaks_on_import_is_a_miss_not_a_crash(
    packaged, monkeypatch
):
    """A half-installed or mlx-mismatched new layout can raise something other
    than ImportError while importing (e.g. an AttributeError on a missing
    ``mx.fast`` kernel). The hooks' contract is fail-open with an ERROR, so the
    resolver must turn that into MlxLmSymbolMissing naming the cause, and an
    armed fusion must still return 0 instead of raising out of model load."""
    from vllm_mlx import moe_fusion

    broken = f"{FAKE}.layers.broken_175"
    path = sys.path[0] + "/" + broken.replace(".", "/") + ".py"
    with open(path, "w") as f:
        f.write("raise AttributeError('mx.fast has no gather_kernel_175')\n")
    for name in ("SwitchGLU", "_gather_sort", "_scatter_unsort"):
        monkeypatch.setitem(
            compat.CANDIDATES, name, [(compat.PACKAGED, broken, name)]
        )
    with pytest.raises(compat.MlxLmSymbolMissing, match="gather_kernel_175"):
        compat.resolve("SwitchGLU")
    monkeypatch.setattr(moe_fusion, "_status", dict(moe_fusion._status))
    monkeypatch.setenv("VLLM_MLX_MOE_GATEUP_FUSION", "1")
    assert moe_fusion.maybe_fuse(object(), "toy") == 0
    assert "gather_kernel_175" in moe_fusion.status()["error"]
