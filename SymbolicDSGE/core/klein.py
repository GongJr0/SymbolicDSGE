"""In-house first-order (Klein) linear rational-expectations solve.

Replaces the vendored ``_linearsolve`` model class with a stateless function.
The pipeline is: complex-step Jacobian of the compiled residual -> ordered
generalized Schur (``scipy.linalg.ordqz``, kept) -> Schur-to-solution post-proc.
The post-proc prefers the native C kernel (``_ckernels.core.klein_postprocess``)
and falls back to the numba twin below; ``ordqz`` stays in scipy either way.

Only the first-order, no-forcing-variable path is implemented -- exactly what the
solver uses. ``p``/``f`` are returned complex (the caller applies ``real_if_close``
when assembling the state-space ``A``/``B``), matching the previous behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy import complex128, float64
from numpy.typing import NDArray
from numba import njit
from scipy.linalg import ordqz

from .._native_dispatch import FORCE_NUMBA, REQUIRE_NATIVE

NDF = NDArray[float64]
NDC = NDArray[complex128]

# Prefer the native post-proc; fall back to the numba twin when the extension is
# not built (ALWAYS_USE_NUMBA / NEVER_USE_NUMBA override -- see _native_dispatch).
_klein_postprocess_native: Callable[..., Any] | None
if FORCE_NUMBA:
    _klein_postprocess_native = None
else:
    try:
        from .._ckernels.core import klein_postprocess as _klein_postprocess_native
    except ImportError:  # pragma: no cover - exercised only without the extension
        if REQUIRE_NATIVE:
            raise
        _klein_postprocess_native = None


@dataclass(frozen=True)
class KleinSolution:
    """Solution of ``a E[y_{t+1}] = b y_t``: ``u_t = f s_t``, ``s_{t+1} = p s_t``.

    ``p``/``f`` are complex (the imaginary parts are ~1e-16 roundoff from the
    complex Schur form; the caller collapses them via ``real_if_close``). Stored
    as ``SolvedModel.policy``; downstream reads only ``.f`` and ``.stab``.
    """

    p: NDC
    f: NDC
    stab: int
    eig: NDC


# ---------------------------------------------------------------------------
# Pre-proc: complex-step linearization of the compiled residual (lifted numba).
# ---------------------------------------------------------------------------
@njit
def _evaluate_equilibrium_numeric(eq_func, fwd, cur, params, log_linear):  # type: ignore[no-untyped-def]
    if log_linear:
        return np.log(eq_func(np.exp(fwd), np.exp(cur), params) + 1.0)
    return eq_func(fwd, cur, params)


@njit
def _complex_step_jacobian(eq_func, base_point, params, log_linear, differentiate_fwd):  # type: ignore[no-untyped-def]
    step = float64(1e-30)
    complex_step = complex128(1j * step)
    base_complex = np.ascontiguousarray(base_point.astype(complex128))
    params_complex = np.ascontiguousarray(params.astype(complex128))
    base_residual = _evaluate_equilibrium_numeric(
        eq_func, base_complex, base_complex, params_complex, log_linear
    )
    jac = np.empty((base_residual.shape[0], base_point.shape[0]), dtype=float64)

    for j in range(base_point.shape[0]):
        fwd = base_complex.copy()
        cur = base_complex.copy()
        if differentiate_fwd:
            fwd[j] = fwd[j] + complex_step
        else:
            cur[j] = cur[j] + complex_step
        residual = _evaluate_equilibrium_numeric(
            eq_func, fwd, cur, params_complex, log_linear
        )
        jac[:, j] = np.imag(residual) / step
    return jac


@njit
def _approximate_system_numeric(eq_func, steady_state, params, log_linear):  # type: ignore[no-untyped-def]
    base_point = np.ascontiguousarray(steady_state.astype(float64))
    parameter_vector = np.ascontiguousarray(params.astype(float64))
    if log_linear:
        base_point = np.ascontiguousarray(np.log(base_point))
    a = _complex_step_jacobian(eq_func, base_point, parameter_vector, log_linear, True)
    b = -_complex_step_jacobian(
        eq_func, base_point, parameter_vector, log_linear, False
    )
    return a, b


# ---------------------------------------------------------------------------
# Post-proc twin (numba fallback / parity oracle) -- mirrors klein_postproc.c.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _klein_postprocess_numba(
    s: NDC, t: NDC, z: NDC, n_states: int
) -> tuple[NDC, NDC, int, NDC]:
    N = s.shape[0]
    n = n_states
    z11 = np.ascontiguousarray(z[:n, :n])
    z21 = np.ascontiguousarray(z[n:, :n])
    s11 = np.ascontiguousarray(s[:n, :n])
    t11 = np.ascontiguousarray(t[:n, :n])

    z11i = np.linalg.inv(z11)

    stab = 0
    if np.abs(t[n - 1, n - 1]) > np.abs(s[n - 1, n - 1]):
        stab = -1
    if n < N:
        if np.abs(t[n, n]) < np.abs(s[n, n]):
            stab = 1

    eig = np.empty(N, dtype=complex128)
    for i in range(N):
        if np.abs(s[i, i]) > 1e-12:
            eig[i] = t[i, i] / s[i, i]
        else:
            eig[i] = complex128(np.inf)

    dyn = np.linalg.solve(s11, t11)
    f = z21 @ z11i
    p = z11 @ dyn @ z11i
    return f, p, stab, eig


def _postprocess(
    s: NDArray[Any], t: NDArray[Any], z: NDArray[Any], n_states: int
) -> tuple[NDC, NDC, int, NDC]:
    s = np.ascontiguousarray(s, dtype=complex128)
    t = np.ascontiguousarray(t, dtype=complex128)
    z = np.ascontiguousarray(z, dtype=complex128)

    if _klein_postprocess_native is not None:
        f, p, stab, eig = _klein_postprocess_native(s, t, z, n_states)
        return f, p, int(stab), eig
    f, p, stab, eig = _klein_postprocess_numba(s, t, z, n_states)
    return f, p, int(stab), eig


def klein_solve(
    equations_numeric: Callable[..., NDC],
    params: NDF,
    steady_state: NDF,
    n_states: int,
    *,
    log_linear: bool = False,
) -> KleinSolution:
    """First-order Klein solve of the compiled model at ``(params, steady_state)``.

    ``equations_numeric`` is the compiled residual (complex-capable) from
    ``CompiledModel.construct_objective_vector_func()``. Returns a
    :class:`KleinSolution` with complex ``p``/``f``.
    """
    ss = np.ascontiguousarray(np.asarray(steady_state, dtype=float64))
    par = np.ascontiguousarray(np.asarray(params, dtype=float64))

    a, b = _approximate_system_numeric(equations_numeric, ss, par, log_linear)
    s, t, _, _, _, z = ordqz(a, b, sort="ouc", output="complex")

    f, p, stab, eig = _postprocess(s, t, z, n_states)

    return KleinSolution(
        p=np.ascontiguousarray(p, dtype=complex128),
        f=np.ascontiguousarray(f, dtype=complex128),
        stab=stab,
        eig=np.asarray(eig, dtype=complex128),
    )
