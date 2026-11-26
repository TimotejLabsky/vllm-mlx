# SPDX-License-Identifier: Apache-2.0
"""mlx-lm symbols that the models-directory move (mlx-lm #2082) relocates,
resolved on both layouts (fork patch #175).

mlx-lm #2082 (merged upstream, unreleased as of 0.32.2) turns every
``models/<arch>.py`` into a package whose ``__init__`` exports only ``Model``
and ``ModelArgs``, and moves the shared helpers out of ``models/``:

==========================  ==============================  ==================================
symbol                      0.32.2 ("legacy")               post-#2082 ("packaged")
==========================  ==============================  ==================================
SwitchGLU, _gather_sort,    mlx_lm.models.switch_layers     mlx_lm.layers.moe
_scatter_unsort
gated_delta_update,         mlx_lm.models.gated_delta       mlx_lm.layers.recurrent
normalize_qk
create_ssm_mask             mlx_lm.models.base              mlx_lm.layers.recurrent
create_attention_mask,      mlx_lm.models.base              mlx_lm.layers.attention
scaled_dot_product_attention
qwen3_5 GatedDeltaNet,      mlx_lm.models.qwen3_5           mlx_lm.models.qwen3_5.layers
DecoderLayer
qwen3_5 TextModelArgs       mlx_lm.models.qwen3_5           mlx_lm.models.qwen3_5.config
Qwen3NextDecoderLayer       mlx_lm.models.qwen3_next        mlx_lm.models.qwen3_next.layers
SwiGLU (#179)               mlx_lm.models.switch_layers     mlx_lm.layers.common
Qwen3NextSparseMoeBlock     mlx_lm.models.qwen3_next        mlx_lm.models.qwen3_next.layers
(#179)
==========================  ==============================  ==================================

There are no back-compat shims upstream, so a hard-coded old path is an
ImportError/AttributeError on the new release, and every fork site that
guards it with ``except`` turns the break into a silent fallback.

Each symbol is looked up at the packaged path first, then the legacy one.
On 0.32.2 the packaged modules do not exist, so every lookup lands on the
exact object the old hard-coded import bound: behaviour on the current pin
is unchanged.

**The trap.** ``mlx_lm.models.cache.create_attention_mask`` exists on both
layouts with a DIFFERENT signature, ``(N, offset, return_array,
window_size)``, against ``(h, cache=None, window_size=None,
return_array=False)``. It is never a candidate here, and every function this
module resolves has its leading parameter names checked, so a wrong binding
is an error, not a silently different mask.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PACKAGED = "packaged"
LEGACY = "legacy"

# name -> [(layout, module, attribute), ...], packaged first.
CANDIDATES: Dict[str, List[Tuple[str, str, str]]] = {}


def _both(name: str, packaged: str, legacy: str, attr: Optional[str] = None):
    attr = attr or name.rsplit(".", 1)[-1]
    CANDIDATES[name] = [(PACKAGED, packaged, attr), (LEGACY, legacy, attr)]


_both("SwitchGLU", "mlx_lm.layers.moe", "mlx_lm.models.switch_layers")
_both("_gather_sort", "mlx_lm.layers.moe", "mlx_lm.models.switch_layers")
_both("_scatter_unsort", "mlx_lm.layers.moe", "mlx_lm.models.switch_layers")
_both("gated_delta_update", "mlx_lm.layers.recurrent", "mlx_lm.models.gated_delta")
_both("normalize_qk", "mlx_lm.layers.recurrent", "mlx_lm.models.gated_delta")
_both("create_ssm_mask", "mlx_lm.layers.recurrent", "mlx_lm.models.base")
# NOT mlx_lm.models.cache.create_attention_mask: see the module docstring.
_both("create_attention_mask", "mlx_lm.layers.attention", "mlx_lm.models.base")
_both("scaled_dot_product_attention", "mlx_lm.layers.attention", "mlx_lm.models.base")
_both("qwen3_5.GatedDeltaNet", "mlx_lm.models.qwen3_5.layers", "mlx_lm.models.qwen3_5")
_both("qwen3_5.DecoderLayer", "mlx_lm.models.qwen3_5.layers", "mlx_lm.models.qwen3_5")
_both("qwen3_5.TextModelArgs", "mlx_lm.models.qwen3_5.config", "mlx_lm.models.qwen3_5")
_both(
    "qwen3_next.Qwen3NextDecoderLayer",
    "mlx_lm.models.qwen3_next.layers",
    "mlx_lm.models.qwen3_next",
)
# (#179) the decode kernels' targets. SwiGLU's post-#2082 home is
# layers.common (layers.moe only re-imports it).
_both("SwiGLU", "mlx_lm.layers.common", "mlx_lm.models.switch_layers")
_both(
    "qwen3_next.Qwen3NextSparseMoeBlock",
    "mlx_lm.models.qwen3_next.layers",
    "mlx_lm.models.qwen3_next",
)

# Leading parameter names each function must declare. Both layouts share
# these (bodies unchanged by #2082); the cache-module ``create_attention_mask``
# starts ``(N, offset, ...)`` and fails the check.
SIGNATURES: Dict[str, Tuple[str, ...]] = {
    "create_attention_mask": ("h", "cache", "window_size", "return_array"),
    "create_ssm_mask": ("h", "cache"),
    "scaled_dot_product_attention": (
        "queries",
        "keys",
        "values",
        "cache",
        "scale",
        "mask",
    ),
    "gated_delta_update": (
        "q",
        "k",
        "v",
        "a",
        "b",
        "A_log",
        "dt_bias",
        "state",
        "mask",
    ),
    "normalize_qk": ("q", "k", "inv_scale", "eps"),
    "_gather_sort": ("x", "indices"),
    "_scatter_unsort": ("x", "inv_order", "shape"),
}


class MlxLmSymbolMissing(ImportError):
    """A symbol the fork hooks is at none of the known mlx-lm paths, or is
    there with the wrong signature. The message names the symbol and every
    path tried, so the log line says what to fix."""


_resolved: Dict[str, Tuple[Any, str, str]] = {}


def _signature_problem(name: str, obj: Any) -> Optional[str]:
    want = SIGNATURES.get(name)
    if want is None:
        return None
    try:
        have = tuple(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return "signature unreadable"
    if have[: len(want)] != want:
        return f"signature {have} does not start with {want}"
    return None


def resolve_with_origin(name: str) -> Tuple[Any, str, str]:
    """``(object, layout, "module.attr")`` for ``name``; raises
    :class:`MlxLmSymbolMissing`. Successful lookups are cached."""
    hit = _resolved.get(name)
    if hit is not None:
        return hit
    if name not in CANDIDATES:
        raise KeyError(f"mlx_lm_compat has no candidates for {name!r}")
    tried = []
    for layout, module, attr in CANDIDATES[name]:
        where = f"{module}.{attr}"
        try:
            mod = importlib.import_module(module)
        except Exception as e:  # noqa: BLE001
            # Not only ImportError: a half-installed or mlx-mismatched layout
            # can raise anything at import, and every caller's contract is
            # fail-open with an ERROR, never a crash out of model load.
            tried.append(f"{where} (import failed: {type(e).__name__}: {e})")
            continue
        obj = getattr(mod, attr, None)
        if obj is None:
            tried.append(f"{where} (module has no {attr})")
            continue
        problem = _signature_problem(name, obj)
        if problem is not None:
            tried.append(f"{where} ({problem})")
            continue
        _resolved[name] = (obj, layout, where)
        return _resolved[name]
    raise MlxLmSymbolMissing(
        f"mlx-lm symbol {name} not found; tried: " + "; ".join(tried)
    )


def resolve(name: str) -> Any:
    return resolve_with_origin(name)[0]


def missing(names) -> List[str]:
    """Error strings for the names in ``names`` that do not resolve (empty
    when all do)."""
    out = []
    for name in names:
        try:
            resolve(name)
        except MlxLmSymbolMissing as e:
            out.append(str(e))
    return out


def layout() -> str:
    """Which layout the installed mlx-lm has: ``packaged`` (post-#2082),
    ``legacy`` (0.32.2), or ``unknown`` (neither moe module imports)."""
    try:
        return resolve_with_origin("SwitchGLU")[1]
    except MlxLmSymbolMissing:
        return "unknown"


def _reset_for_tests() -> None:
    _resolved.clear()


# --------------------------------------------------------------------------
# (#180) the qwen3_5 model TREE, not just its symbols, changed in #2082
# --------------------------------------------------------------------------
#
#   0.32.2 ("legacy")                      post-#2082 ("packaged")
#   Model                                  Model
#     .language_model  TextModel             .args.text      TextModelArgs
#       .args          TextModelArgs         .lm_head        (untied only)
#       .lm_head       (untied only)         .model          Qwen3_5Model (not
#       .model         Qwen3_5TextModel                      callable, no embed)
#         .embed_tokens .layers .norm          .language_model Qwen3_5TextModel
#                                                .embed_tokens .layers .norm
#
# ``Qwen3_5TextModel`` itself (``__call__(inputs, cache)`` -> normed hidden)
# is unchanged. Code that needs "the text model" asks :func:`text_tree` for
# an object with the legacy ``TextModel`` surface - ``.model`` (the inner
# text model), ``.args`` (text args) and ``.lm_head`` - on either layout.


class TextTreeView:
    """The legacy ``TextModel`` surface over a packaged qwen3_5 ``Model``:
    ``.model``, ``.args``, ``.lm_head`` (None when tied). Deliberately has no
    ``language_model`` and no ``embed_tokens`` of its own, so mlx-vlm's
    ``AutoregressiveMTPDraftModel.bind`` finds ``.model.embed_tokens`` and
    this ``lm_head`` - the same objects it binds on the legacy tree."""

    __slots__ = ("model", "args", "lm_head", "outer")
    layout = PACKAGED

    def __init__(self, outer: Any, inner: Any, args: Any, lm_head: Any):
        self.outer = outer
        self.model = inner
        self.args = args
        self.lm_head = lm_head


def text_tree(model: Any) -> Tuple[Optional[Any], str]:
    """``(text_model, how)`` for a loaded mlx-lm model: the object with the
    legacy ``TextModel`` surface (``.model`` = inner text model, ``.args`` =
    text args, ``.lm_head`` or tied), and which tree matched.

    * ``legacy``: ``model.language_model`` - the very object 0.32.2 code
      used (``getattr(model, "language_model", model)``).
    * ``packaged``: a :class:`TextTreeView` over ``model.model.language_model``
      and ``model.args.text``.
    * ``flat``: ``model`` itself, the old fallback (a text model whose
      ``.model`` is the inner model, e.g. qwen3_next).

    Only the shape is matched; callers validate what they need (the drafter
    still checks ``callable(text_model.model)``). ``text_model`` is None,
    with ``how`` naming what is missing, only for a packaged tree without
    ``args.text``. Never raises."""
    try:
        lm = getattr(model, "language_model", None)
        if lm is not None:
            return lm, LEGACY
        inner = getattr(getattr(model, "model", None), "language_model", None)
        if inner is not None:
            args = getattr(getattr(model, "args", None), "text", None)
            if args is None:
                return None, (
                    f"{type(model).__name__} has .model.language_model "
                    "(post-#2082 tree) but no .args.text"
                )
            view = TextTreeView(model, inner, args, getattr(model, "lm_head", None))
            return view, PACKAGED
        return model, "flat"
    except Exception as e:  # noqa: BLE001
        return None, f"walking {type(model).__name__} raised {type(e).__name__}: {e}"
