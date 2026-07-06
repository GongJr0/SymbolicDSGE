from typing import Callable

from numba import njit
import numpy as np
from numpy import ascontiguousarray, float64
from numpy.typing import NDArray

NDF = NDArray[np.float64]

# Prefer the compiled native kernels; fall back to the numba kernels below when
# the _ckernels extension is not built (sdist without a compiler, etc.). The
# numba versions stay in place permanently as the fallback and the parity oracle.
# ALWAYS_USE_NUMBA / NEVER_USE_NUMBA override this default (see _native_dispatch).
from .._native_dispatch import FORCE_NUMBA, REQUIRE_NATIVE

# Declared explicitly so the FORCE_NUMBA branch (which binds None first) doesn't
# pin the inferred type to None; the native handle matches the _core.pyi stub.
_SimulateKernel = Callable[[NDF, NDF, NDF, NDF, NDF], None]
_AffineKernel = Callable[[NDF, NDF, NDF, int, NDF], None]
_SecondOrderKernel = Callable[
    [NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF], tuple[NDF, NDF]
]
_simulate_native: _SimulateKernel | None
_affine_native: _AffineKernel | None
_simulate_second_order_native: _SecondOrderKernel | None

if FORCE_NUMBA:
    _affine_native = None
    _simulate_native = None
    _simulate_second_order_native = None
else:
    try:
        from .._ckernels.core import (
            affine_observations_into as _affine_native,
            simulate_linear_states_into as _simulate_native,
            simulate_second_order_pruned as _simulate_second_order_native,
        )
    except ImportError:  # pragma: no cover - exercised only without the extension
        if REQUIRE_NATIVE:
            raise
        _affine_native = None
        _simulate_native = None
        _simulate_second_order_native = None


@njit(cache=True)
def _simulate_linear_states_into_numba(
    A: NDF,
    B: NDF,
    x0: NDF,
    shock_mat: NDF,
    out: NDF,
) -> None:
    T = shock_mat.shape[0]
    n = A.shape[0]
    k = B.shape[1]

    for i in range(n):
        out[0, i] = x0[i]

    for t in range(T):
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i, j] * out[t, j]
            for j in range(k):
                s += B[i, j] * shock_mat[t, j]
            out[t + 1, i] = s


@njit(cache=True)
def _affine_observations_into_numba(
    states: NDF,
    C: NDF,
    d: NDF,
    state_start: int,
    out: NDF,
) -> None:
    T = out.shape[0]
    m = C.shape[0]
    n = C.shape[1]

    for t in range(T):
        state_row = state_start + t
        for i in range(m):
            s = d[i]
            for j in range(n):
                s += C[i, j] * states[state_row, j]
            out[t, i] = s


@njit(cache=True)
def _simulate_second_order_pruned_numba(
    hx: NDF,
    gx: NDF,
    bx: NDF,
    hxx: NDF,
    gxx: NDF,
    hss: NDF,
    gss: NDF,
    x0: NDF,
    shock_mat: NDF,
) -> tuple[NDF, NDF]:
    T = shock_mat.shape[0]
    nx = hx.shape[0]
    ny = gx.shape[0]
    n_exog = bx.shape[1]

    x_out = np.empty((T + 1, nx), dtype=np.float64)
    y_out = np.empty((T + 1, ny), dtype=np.float64)
    x1_cur = np.empty(nx, dtype=np.float64)
    x1_next = np.empty(nx, dtype=np.float64)
    x2_cur = np.empty(nx, dtype=np.float64)
    x2_next = np.empty(nx, dtype=np.float64)
    x1_outer = np.empty((nx, nx), dtype=np.float64)

    for i in range(nx):
        x1_cur[i] = x0[i]
        x2_cur[i] = 0.0

    for t in range(T + 1):
        for i in range(nx):
            x_out[t, i] = x1_cur[i] + x2_cur[i]

        for j in range(nx):
            for k in range(nx):
                x1_outer[j, k] = x1_cur[j] * x1_cur[k]

        for i in range(ny):
            s = 0.5 * gss[i]
            for j in range(nx):
                s += gx[i, j] * x_out[t, j]
            for j in range(nx):
                for k in range(nx):
                    s += 0.5 * gxx[i, j, k] * x1_outer[j, k]
            y_out[t, i] = s

        if t == T:
            break

        for i in range(nx):
            s1 = 0.0
            s2 = 0.5 * hss[i]
            for j in range(nx):
                s1 += hx[i, j] * x1_cur[j]
                s2 += hx[i, j] * x2_cur[j]
            for j in range(n_exog):
                s1 += bx[i, j] * shock_mat[t, j]
            for j in range(nx):
                for k in range(nx):
                    s2 += 0.5 * hxx[i, j, k] * x1_outer[j, k]
            x1_next[i] = s1
            x2_next[i] = s2

        for i in range(nx):
            x1_cur[i] = x1_next[i]
            x2_cur[i] = x2_next[i]

    return x_out, y_out


_simulate_second_order_numba: _SecondOrderKernel = _simulate_second_order_pruned_numba


def simulate_linear_states_into(
    A: NDF,
    B: NDF,
    x0: NDF,
    shock_mat: NDF,
    out: NDF,
) -> None:
    """Simulate ``out[t+1] = A @ out[t] + B @ shock[t]`` in place (``out[0] = x0``).

    Uses the native C kernel when built, else the numba kernel. ``out`` must be a
    C-contiguous float64 array (the caller's output buffer); it is written in
    place. Read-only inputs are coerced to contiguous float64 for the native
    path only, leaving the numba path untouched.
    """
    if _simulate_native is not None:
        _simulate_native(
            ascontiguousarray(A, dtype=float64),
            ascontiguousarray(B, dtype=float64),
            ascontiguousarray(x0, dtype=float64),
            ascontiguousarray(shock_mat, dtype=float64),
            out,
        )
    else:
        _simulate_linear_states_into_numba(A, B, x0, shock_mat, out)


def affine_observations_into(
    states: NDF,
    C: NDF,
    d: NDF,
    state_start: int,
    out: NDF,
) -> None:
    """Compute ``out[t] = d + C @ states[state_start + t]`` in place.

    Uses the native C kernel when built, else the numba kernel. ``out`` must be a
    C-contiguous float64 output buffer; read-only inputs are coerced to
    contiguous float64 for the native path only.
    """
    if _affine_native is not None:
        _affine_native(
            ascontiguousarray(states, dtype=float64),
            ascontiguousarray(C, dtype=float64),
            ascontiguousarray(d, dtype=float64),
            state_start,
            out,
        )
    else:
        _affine_observations_into_numba(states, C, d, state_start, out)


def simulate_second_order_pruned(
    hx: NDF,
    gx: NDF,
    bx: NDF,
    hxx: NDF,
    gxx: NDF,
    hss: NDF,
    gss: NDF,
    x0: NDF,
    shock_mat: NDF,
) -> tuple[NDF, NDF]:
    """Return split state and jump paths from the pruned second order rule."""
    if _simulate_second_order_native is not None:
        return _simulate_second_order_native(
            ascontiguousarray(hx, dtype=float64),
            ascontiguousarray(gx, dtype=float64),
            ascontiguousarray(bx, dtype=float64),
            ascontiguousarray(hxx, dtype=float64),
            ascontiguousarray(gxx, dtype=float64),
            ascontiguousarray(hss, dtype=float64),
            ascontiguousarray(gss, dtype=float64),
            ascontiguousarray(x0, dtype=float64),
            ascontiguousarray(shock_mat, dtype=float64),
        )
    return _simulate_second_order_numba(hx, gx, bx, hxx, gxx, hss, gss, x0, shock_mat)
