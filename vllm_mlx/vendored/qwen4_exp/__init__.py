# SPDX-License-Identifier: Apache-2.0
"""Register the vendored Qwen4-Exp (``qwen4_exp``) implementation with mlx-vlm.

Vendored from oMLX (jundot/omlx, Apache-2.0) at v0.6.4-35-ge69d707 —
``omlx/patches/mlx_vlm_qwen4_exp_compat/`` — because neither mlx-lm at the
fork's pin nor mlx-vlm 0.6.17 ships the ``qwen4_exp`` architecture
(Qwen3.8-Flash-Next and its REAP-pruned variants). See PATCHES.md for the
local-edit inventory (FP8 checkpoint support stripped, oMLX Lightning-MTP
hooks left dormant, PLE mode env var renamed).

The registration appends this package's ``vendor/mlx_vlm`` tree to the real
``mlx_vlm``/``mlx_vlm.models`` package ``__path__``s, so
``mlx_vlm.models.qwen4_exp`` resolves from the vendor tree while the vendored
module's ``..qwen3_5``/``..qwen3_vl``/``..base`` relative imports keep
resolving against the installed mlx-vlm.

The big win this carries: the arch's ~29 GB hashed n-gram PLE embedding
table can stay on SSD (``DiskBackedShardedEmbedding``: mmap + per-token row
gathers), dropping resident memory for Flash-Next-REAP-288 from ~68 GB to
~40 GB — which is what makes a 180B-class model fit this 64 GB box at all.
Mode selection: ``VLLM_MLX_QWEN4_PLE_MODE`` = ``auto`` (default; mmap when
the checkpoint exceeds 70% of physical RAM) | ``resident`` | ``mmap``.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"
_APPLIED = False


def _prepend_package_path(package: Any, path: Path) -> None:
    # PREPEND, don't append: some mlx-vlm builds labelled 0.6.17 already
    # ship a qwen4_exp subpackage (resident-only — no configure_ple_runtime,
    # no DiskBackedShardedEmbedding, found live on the Studio 2026-09-06),
    # and a real subpackage on an earlier __path__ entry wins over an
    # appended one. The vendored tree must shadow it: only it carries the
    # SSD-backed PLE mode this box needs to fit the model at all.
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_string = str(path)
    if path_string in package_path:
        package_path.remove(path_string)
    package_path.insert(0, path_string)


def apply_qwen4_exp_compat_patch() -> bool:
    """Expose ``mlx_vlm.models.qwen4_exp`` from the fork's vendor tree.

    Idempotent; returns True the first time the registration succeeds.
    Failure is logged and reported as False rather than raised so a broken
    vendor tree cannot take down loads of other model families.
    """
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import sys

        import mlx_vlm
        import mlx_vlm.models

        _prepend_package_path(mlx_vlm, _VENDOR_MLX_VLM)
        _prepend_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")
        # Drop any already-imported qwen4_exp (an installed resident-only
        # build imported before registration) so the import below re-resolves
        # through the prepended vendor path.
        for name in [
            key
            for key in sys.modules
            if key == "mlx_vlm.models.qwen4_exp"
            or key.startswith("mlx_vlm.models.qwen4_exp.")
        ]:
            del sys.modules[name]
        importlib.invalidate_caches()
        module = importlib.import_module("mlx_vlm.models.qwen4_exp")
        if not str(Path(module.__file__).resolve()).startswith(
            str(_VENDOR_MLX_VLM)
        ):
            raise ImportError(
                f"qwen4_exp resolved outside the vendor tree: {module.__file__}"
            )
        _patch_prompt_utils()
    except Exception as exc:  # noqa: BLE001
        logger.error("qwen4_exp mlx-vlm registration failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("qwen4_exp vendored implementation registered with mlx-vlm")
    return True


def _patch_prompt_utils() -> None:
    """Teach the pinned formatter Qwen4's Qwen3.5-compatible media layout."""
    import mlx_vlm.prompt_utils as prompt_utils

    current = prompt_utils.get_message_json
    if getattr(current, "_vllm_mlx_qwen4_exp", False):
        return

    def get_message_json(model_type, *args, **kwargs):
        if model_type == "qwen4_exp":
            model_type = "qwen3_5_moe"
        return current(model_type, *args, **kwargs)

    get_message_json._vllm_mlx_qwen4_exp = True
    prompt_utils.get_message_json = get_message_json


def is_applied() -> bool:
    return _APPLIED


def configure_qwen4_exp_runtime(model_path: str | Path, mode: str | None = None) -> str:
    """Bind PLE storage mode for ``model_path`` before model construction.

    Returns the resolved mode (``resident`` or ``mmap``). Lightning MTP is
    deliberately not configured: speculative decode is refuted on this
    hardware (see docs/fork/speed-lever-ledger-2026-09.md) and the MTP hooks
    in the vendored module stay dormant unless explicitly bound.
    """
    apply_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import configure_ple_runtime

    resolved = configure_ple_runtime(model_path, mode=mode)
    logger.info("qwen4_exp PLE mode for %s: %s", model_path, resolved)
    return resolved
