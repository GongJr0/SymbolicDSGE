"""Runtime selection between the native (``_ckernels``) and numba kernels.

Each consumer that has both a compiled native kernel and a numba reference picks
between them at import time. The default policy is *prefer native, fall back to
numba* when the extension is not built. Two debug environment variables override
that policy:

* ``ALWAYS_USE_NUMBA`` -- ignore the native extension entirely and pin the numba
  path, even when the extension is built. Useful for benchmarking the fallback or
  bisecting a native/numba parity discrepancy.
* ``NEVER_USE_NUMBA`` -- require the native extension. Instead of silently
  falling back, let the ``ImportError`` propagate so execution halts loudly.
  Useful for confirming the native path is actually the one running.

A variable counts as *set* when present and not one of ``0/false/no/off`` (case
insensitive); an empty value is treated as unset. Setting both is contradictory
and raises at import time.
"""

from __future__ import annotations

import os

__all__ = ["FORCE_NUMBA", "REQUIRE_NATIVE"]

_FALSEY = {"", "0", "false", "no", "off"}


def _flag(name: str) -> bool:
    """Return whether the debug env var ``name`` is set to a truthy value."""
    val = os.environ.get(name)
    if val is None:
        return False
    return val.strip().lower() not in _FALSEY


#: Pin the numba path and skip the native import (``ALWAYS_USE_NUMBA``).
FORCE_NUMBA = _flag("ALWAYS_USE_NUMBA")

#: Require the native extension; let an ``ImportError`` halt instead of falling
#: back to numba (``NEVER_USE_NUMBA``).
REQUIRE_NATIVE = _flag("NEVER_USE_NUMBA")

if FORCE_NUMBA and REQUIRE_NATIVE:
    raise ValueError(
        "ALWAYS_USE_NUMBA and NEVER_USE_NUMBA are mutually exclusive; set at most "
        "one (ALWAYS_USE_NUMBA pins numba, NEVER_USE_NUMBA requires native)."
    )
