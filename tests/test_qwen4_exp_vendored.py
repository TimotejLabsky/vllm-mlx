# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendored qwen4_exp implementation (PATCHES.md #97).

Laptop-runnable coverage: mlx-vlm registration/path injection, the
prompt_utils media-layout shim, real-checkpoint config parsing, PLE
runtime-mode resolution, and byte-level correctness of the SSD-backed
(mmap) n-gram embedding gather against a resident reference — dense and
affine-quantized layouts. Full-model load/generation is a Studio gate
(the checkpoint is 68 GB); see PATCHES.md #97 for the live verification.
"""

import json
from pathlib import Path

import mlx.core as mx
import pytest

from vllm_mlx.vendored.qwen4_exp import (
    apply_qwen4_exp_compat_patch,
    configure_qwen4_exp_runtime,
    is_applied,
)

FIXTURE_CONFIG = Path(__file__).parent / "fixtures" / "qwen4_exp_reap288_config.json"


@pytest.fixture(scope="module", autouse=True)
def _registered():
    apply_qwen4_exp_compat_patch()
    assert is_applied()


def test_registration_is_idempotent():
    # Second call is a no-op that must not duplicate __path__ entries.
    import mlx_vlm

    before = list(mlx_vlm.__path__)
    assert apply_qwen4_exp_compat_patch() is False
    assert list(mlx_vlm.__path__) == before


def test_qwen4_exp_resolves_from_vendor_tree():
    import mlx_vlm.models.qwen4_exp as pkg

    assert "vllm_mlx/vendored/qwen4_exp" in pkg.__file__.replace("\\", "/")
    # The public model surface mlx_vlm.load() needs.
    assert hasattr(pkg, "Model")
    assert hasattr(pkg, "ModelConfig")
    assert hasattr(pkg, "LanguageModel")
    # And the PLE surface the fork exists for — a resident-only qwen4_exp
    # shipped by some mlx-vlm builds lacks these.
    assert hasattr(pkg.language, "configure_ple_runtime")
    assert hasattr(pkg.language, "DiskBackedShardedEmbedding")


def test_vendor_tree_shadows_installed_qwen4_exp():
    # Some mlx-vlm builds labelled 0.6.17 ship their own (resident-only)
    # qwen4_exp; the vendor path must sit FIRST on both __path__s so the
    # fork's implementation wins regardless of the installed build.
    import mlx_vlm
    import mlx_vlm.models

    from vllm_mlx.vendored import qwen4_exp as reg

    assert mlx_vlm.__path__[0] == str(reg._VENDOR_MLX_VLM)
    assert mlx_vlm.models.__path__[0] == str(reg._VENDOR_MLX_VLM / "models")


def test_prompt_utils_shim_maps_to_qwen3_5_moe():
    import mlx_vlm.prompt_utils as prompt_utils

    assert getattr(prompt_utils.get_message_json, "_vllm_mlx_qwen4_exp", False)
    a = prompt_utils.get_message_json("qwen4_exp", "hi", role="user", skip_image_token=True)
    b = prompt_utils.get_message_json("qwen3_5_moe", "hi", role="user", skip_image_token=True)
    assert a == b


def test_real_reap288_config_parses():
    from mlx_vlm.models.qwen4_exp import ModelConfig

    raw = json.loads(FIXTURE_CONFIG.read_text())
    config = ModelConfig.from_dict(raw)
    assert config.model_type == "qwen4_exp"
    text = config.text_config
    assert text.num_experts == 288  # REAP-pruned from 512
    assert text.num_experts_per_tok == 10
    assert text.num_hidden_layers == 48
    # Hybrid 3:1 layout — three GDN linear layers per qwen-sparse-attention layer.
    assert text.layer_types.count("qwen_sparse_attention") == 12
    assert text.layer_types.count("linear_attention") == 36
    assert len(text.layer_types) == 48


def test_ple_mode_resolution_thresholds():
    from mlx_vlm.models.qwen4_exp.language import resolve_ple_runtime_mode

    kwargs = dict(checkpoint_bytes=68 << 30, physical_memory=64 << 30)
    assert resolve_ple_runtime_mode("auto", **kwargs) == "mmap"
    assert (
        resolve_ple_runtime_mode(
            "auto", checkpoint_bytes=20 << 30, physical_memory=64 << 30
        )
        == "resident"
    )
    assert resolve_ple_runtime_mode("resident", **kwargs) == "resident"
    assert resolve_ple_runtime_mode("ssd_mmap", **kwargs) == "mmap"
    with pytest.raises(ValueError):
        resolve_ple_runtime_mode("turbo", **kwargs)


def test_ple_mode_env_var(tmp_path, monkeypatch):
    # VLLM_MLX_QWEN4_PLE_MODE wins; the oMLX-era name still works as fallback.
    (tmp_path / "model.safetensors").write_bytes(b"\x08" + b"\x00" * 7 + b"{}")
    monkeypatch.setenv("OMLX_QWEN4_PLE_MODE", "resident")
    monkeypatch.setenv("VLLM_MLX_QWEN4_PLE_MODE", "mmap")
    assert configure_qwen4_exp_runtime(tmp_path) == "mmap"
    monkeypatch.delenv("VLLM_MLX_QWEN4_PLE_MODE")
    assert configure_qwen4_exp_runtime(tmp_path) == "resident"


PREFIX = "language_model.model.layers.0.ple.ple_embedding.ngram_embedding"


def _write_model_dir(tmp_path: Path, tensors: dict[str, mx.array]) -> Path:
    weights_name = "model.safetensors"
    mx.save_safetensors(str(tmp_path / weights_name), tensors)
    index = {"weight_map": {key: weights_name for key in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return tmp_path


def _disk_backed(tmp_path, num_embeddings, dims, num_shards):
    from mlx_vlm.models.qwen4_exp.language import DiskBackedShardedEmbedding

    return DiskBackedShardedEmbedding(
        tmp_path, PREFIX, num_embeddings, dims, num_shards
    )


def test_disk_backed_dense_gather_matches_resident(tmp_path):
    num_embeddings, dims = 20, 16
    mx.random.seed(0)
    table = mx.random.normal((num_embeddings, dims)).astype(mx.bfloat16)
    # Shard split mirrors the module's own divmod layout (10 + 10 for 20/2).
    shards = {
        f"{PREFIX}.shard_0.weight": table[:10],
        f"{PREFIX}.shard_1.weight": table[10:],
    }
    _write_model_dir(tmp_path, shards)
    emb = _disk_backed(tmp_path, num_embeddings, dims, 2)

    indices = mx.array([0, 9, 10, 19, 3, 10])  # both shards, duplicates included
    got = emb(indices)
    want = table[indices].astype(mx.bfloat16)
    assert got.shape == (6, dims)
    assert mx.array_equal(got.astype(mx.float32), want.astype(mx.float32))
    assert emb.last_touched_shards == (0, 1)
    assert emb.rows_read == 6


def test_disk_backed_affine_gather_matches_dequantized(tmp_path):
    num_embeddings, dims, bits, group_size = 12, 64, 4, 32
    mx.random.seed(1)
    table = mx.random.normal((num_embeddings, dims))
    w0, s0, b0 = mx.quantize(table[:6], group_size=group_size, bits=bits)
    w1, s1, b1 = mx.quantize(table[6:], group_size=group_size, bits=bits)
    tensors = {
        f"{PREFIX}.shard_0.weight": w0,
        f"{PREFIX}.shard_0.scales": s0.astype(mx.float32),
        f"{PREFIX}.shard_0.biases": b0.astype(mx.float32),
        f"{PREFIX}.shard_1.weight": w1,
        f"{PREFIX}.shard_1.scales": s1.astype(mx.float32),
        f"{PREFIX}.shard_1.biases": b1.astype(mx.float32),
    }
    _write_model_dir(tmp_path, tensors)
    emb = _disk_backed(tmp_path, num_embeddings, dims, 2)

    indices = mx.array([[1, 5], [6, 11]])
    got = emb(indices)
    ref0 = mx.dequantize(w0, s0.astype(mx.float32), b0.astype(mx.float32),
                         group_size=group_size, bits=bits, mode="affine")
    ref1 = mx.dequantize(w1, s1.astype(mx.float32), b1.astype(mx.float32),
                         group_size=group_size, bits=bits, mode="affine")
    want = mx.concatenate([ref0[mx.array([1, 5])], ref1[mx.array([0, 5])]])
    assert got.shape[-1] == dims
    assert mx.allclose(
        got.reshape(-1, dims).astype(mx.float32),
        want.astype(mx.float32),
        atol=1e-2,
    )


def test_disk_backed_rejects_out_of_range(tmp_path):
    num_embeddings, dims = 8, 16
    table = mx.zeros((num_embeddings, dims), dtype=mx.bfloat16)
    _write_model_dir(
        tmp_path,
        {
            f"{PREFIX}.shard_0.weight": table[:4],
            f"{PREFIX}.shard_1.weight": table[4:],
        },
    )
    emb = _disk_backed(tmp_path, num_embeddings, dims, 2)
    with pytest.raises(IndexError):
        emb(mx.array([8]))


def test_fp8_checkpoint_is_rejected_with_clear_error():
    from mlx_vlm.models.qwen4_exp import Model

    weights = {"language_model.model.layers.0.mlp.weight_scale_inv": mx.zeros((1,))}
    # Unbound call with a stub self: sanitize touches no attributes before the
    # FP8 guard unless PLE mode is "mmap", which module state is not here.
    with pytest.raises(NotImplementedError, match="FP8"):
        Model.sanitize(object(), dict(weights))
