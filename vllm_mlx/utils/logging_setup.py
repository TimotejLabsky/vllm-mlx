"""Timestamped logging setup for the server processes.

Fork-owned (PATCHES.md #90). Lives here rather than inline in ``server.py`` /
``cli.py`` so the upstream-owned files carry a one-line call each and the
rebase conflict surface stays minimal.

Why this exists
---------------
Backend log lines had no timestamps: ``server.py`` called
``logging.basicConfig(level=logging.INFO)`` with the default format
(``LEVEL:name:message``), and uvicorn's own loggers default to
``%(levelprefix)s %(message)s``. Once llama-swap route output is captured to
disk (infra ``route-log-wrapper.sh``), an untimestamped log cannot be
correlated with anything — which is exactly what a post-incident diagnosis
needs.

``basicConfig`` alone is NOT enough. It configures the root logger, which the
fork's own ``logging.getLogger(__name__)`` loggers propagate to, but uvicorn
installs its own handlers on ``uvicorn``/``uvicorn.error``/``uvicorn.access``
and those do not propagate. The ``uvicorn.error`` logger is the one that emits
"Exception in ASGI application" — i.e. the traceback for a mid-stream failure,
the single most valuable line in the file. So uvicorn needs its own
``log_config`` with the same treatment.

Millisecond precision is deliberate: a mid-stream exception and the request
that caused it can land in the same second.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

# "2026-09-01 11:51:44,954 ERROR vllm_mlx.server: ..." — asctime carries ms by
# default, which is the resolution incidents actually get argued at.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_root_logging(level: int = logging.INFO) -> None:
    """Install a timestamped root handler for the fork's own loggers.

    No-op if the root logger already has handlers, matching ``basicConfig``
    semantics, so an embedding application's logging setup still wins.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT)


def timestamped_log_config() -> dict[str, Any]:
    """Return uvicorn's default LOGGING_CONFIG with timestamps added.

    Uvicorn's formatters are subclasses that add ``levelprefix`` /
    ``client_addr``; we keep them and only prepend ``asctime`` to their format
    strings, so colourisation and access-log fields are preserved.
    """
    from uvicorn.config import LOGGING_CONFIG

    config = copy.deepcopy(LOGGING_CONFIG)
    for formatter in config.get("formatters", {}).values():
        fmt = formatter.get("fmt")
        if fmt and "%(asctime)s" not in fmt:
            formatter["fmt"] = f"%(asctime)s {fmt}"
    return config
