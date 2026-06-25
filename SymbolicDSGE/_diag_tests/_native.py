"""Shared handle to the native diagnostic kernels, with the numba overrides.

Each diagnostic module prefers the compiled ``_ckernels.diag`` kernels and falls
back to its numba kernel when (a) the extension is not built, (b)
``ALWAYS_USE_NUMBA`` is set, or (c) a kernel returns ``DIAG_FALLBACK`` -- the
signal that the design is rank-deficient and the statistic must be recomputed
through the numba path (which has the SVD-based lstsq fallback the C side
deliberately lacks). ``NEVER_USE_NUMBA`` makes a missing extension raise instead
of falling back. See ``SymbolicDSGE._native_dispatch`` for the env flags.
"""

from __future__ import annotations

from types import ModuleType

from .._native_dispatch import FORCE_NUMBA, REQUIRE_NATIVE

#: Mirrors the C ``#define DIAG_FALLBACK 1`` (the native kernels' "retry in
#: numba" sentinel). Overwritten below with the authoritative value exported by
#: the extension when it is available.
DIAG_FALLBACK = 1

native: ModuleType | None = None
if not FORCE_NUMBA:
    try:
        from .._ckernels import diag as _diag_ext
    except ImportError:  # pragma: no cover - exercised only without the extension
        if REQUIRE_NATIVE:
            raise
    else:
        native = _diag_ext
        DIAG_FALLBACK = _diag_ext.FALLBACK
