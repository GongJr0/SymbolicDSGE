"""Numba / numpy reference implementations for the native ``_ckernels.core``
kernels, retained as parity oracles.

These were the runtime fallbacks in ``SymbolicDSGE.core.*`` / ``utils.dhm`` before
the native extension became mandatory. They now live here purely so the parity
tests can compare each C kernel against an independent implementation; the
library no longer imports them.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numba import njit
from numpy import complex128, float64
from numpy.typing import NDArray

from SymbolicDSGE._ckernels.core import residual_eval

NDF = NDArray[float64]
NDC = NDArray[complex128]


# --- core.simulation ---------------------------------------------------------
@njit(cache=True)
def _simulate_linear_states_into_numba(
    A: NDF,
    B: NDF,
    x0: NDF,
    shock_mat: NDF,
    out: NDF,
    ss: NDF,
) -> None:
    """Reference for ``core.simulate_linear_states_into``. ``ss`` denominates the
    written rows; pass zeros for deviations. The recursion runs off its own
    deviation buffer, never off ``out``."""
    T = shock_mat.shape[0]
    n = A.shape[0]
    k = B.shape[1]

    cur = x0.copy()
    nxt = np.empty(n, dtype=np.float64)
    for t in range(T):
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i, j] * cur[j]
            for j in range(k):
                s += B[i, j] * shock_mat[t, j]
            nxt[i] = s
            out[t, i] = s + ss[i]
        for i in range(n):
            cur[i] = nxt[i]


@njit(cache=True)
def _affine_observations_into_numba(
    states: NDF,
    C: NDF,
    d: NDF,
    out: NDF,
) -> None:
    T = out.shape[0]
    m = C.shape[0]
    n = C.shape[1]

    for t in range(T):
        for i in range(m):
            s = d[i]
            for j in range(n):
                s += C[i, j] * states[t, j]
            out[t, i] = s


@njit(cache=True)
def _simulate_second_order_pruned_numba(
    hx: NDF,
    gx: NDF,
    bu: NDF,
    hxx: NDF,
    gxx: NDF,
    hxu: NDF,
    gxu: NDF,
    huu: NDF,
    guu: NDF,
    hss: NDF,
    gss: NDF,
    x0: NDF,
    shock_mat: NDF,
    ss: NDF,
) -> NDF:
    """Reference for ``core.simulate_second_order_pruned``, written the way
    Dynare's simult_.m writes the order-2 pruned branch: every row of a period
    is one expression in the previous state and this period's innovation.

    ``ss`` denominates the rows; pass zeros for deviations."""
    T = shock_mat.shape[0]
    nx = hx.shape[0]
    ny = gx.shape[0]
    n_exog = bu.shape[1]

    out = np.empty((T, nx + ny), dtype=np.float64)
    x1_cur = x0.copy()
    x2_cur = np.zeros(nx, dtype=np.float64)
    x1_next = np.empty(nx, dtype=np.float64)
    x2_next = np.empty(nx, dtype=np.float64)

    for t in range(T):
        u = shock_mat[t]
        for i in range(nx):
            s1 = 0.0
            s2 = 0.5 * hss[i]
            for j in range(nx):
                s1 += hx[i, j] * x1_cur[j]
                s2 += hx[i, j] * x2_cur[j]
            for l in range(n_exog):
                s1 += bu[i, l] * u[l]
            for j in range(nx):
                for k in range(nx):
                    s2 += 0.5 * hxx[i, j, k] * x1_cur[j] * x1_cur[k]
                for l in range(n_exog):
                    s2 += hxu[i, j, l] * x1_cur[j] * u[l]
            for l in range(n_exog):
                for m in range(n_exog):
                    s2 += 0.5 * huu[i, l, m] * u[l] * u[m]
            x1_next[i] = s1
            x2_next[i] = s2

        # Controls read the same previous state and the same innovation; their
        # first-order shock response is the control rows of bu.
        for i in range(ny):
            s = 0.5 * gss[i]
            for j in range(nx):
                s += gx[i, j] * (x1_cur[j] + x2_cur[j])
            for l in range(n_exog):
                s += bu[nx + i, l] * u[l]
            for j in range(nx):
                for k in range(nx):
                    s += 0.5 * gxx[i, j, k] * x1_cur[j] * x1_cur[k]
                for l in range(n_exog):
                    s += gxu[i, j, l] * x1_cur[j] * u[l]
            for l in range(n_exog):
                for m in range(n_exog):
                    s += 0.5 * guu[i, l, m] * u[l] * u[m]
            out[t, nx + i] = s + ss[nx + i]

        for i in range(nx):
            x1_cur[i] = x1_next[i]
            x2_cur[i] = x2_next[i]
            out[t, i] = x1_cur[i] + x2_cur[i] + ss[i]

    return out


# --- core.klein --------------------------------------------------------------
@njit
def _complex_step_jacobian(eq_func, base_point, params, n_exog, target):  # type: ignore[no-untyped-def]
    """``target`` selects the date perturbed: 0 fwd, 1 cur, 2 prev, 3 eps."""
    step = float64(1e-30)
    complex_step = complex128(1j * step)
    base_complex = np.ascontiguousarray(base_point.astype(complex128))
    params_complex = np.ascontiguousarray(params.astype(complex128))
    eps_base = np.zeros(n_exog, dtype=complex128)
    base_residual = eq_func(
        base_complex, base_complex, base_complex, eps_base, params_complex
    )
    n_col = n_exog if target == 3 else base_point.shape[0]
    jac = np.empty((base_residual.shape[0], n_col), dtype=float64)

    for j in range(n_col):
        fwd = base_complex.copy()
        cur = base_complex.copy()
        prev = base_complex.copy()
        eps = eps_base.copy()
        if target == 0:
            fwd[j] = fwd[j] + complex_step
        elif target == 1:
            cur[j] = cur[j] + complex_step
        elif target == 2:
            prev[j] = prev[j] + complex_step
        else:
            eps[j] = complex_step
        residual = eq_func(fwd, cur, prev, eps, params_complex)
        jac[:, j] = np.imag(residual) / step
    return jac


@njit
def _approximate_system_numeric(eq_func, steady_state, params, n_exog):  # type: ignore[no-untyped-def]
    """(a, b, c, d) under the kernel's signs: ``a y' = b y + c y_prev + d eps``."""
    base_point = np.ascontiguousarray(steady_state.astype(float64))
    parameter_vector = np.ascontiguousarray(params.astype(float64))
    a = _complex_step_jacobian(eq_func, base_point, parameter_vector, n_exog, 0)
    b = -_complex_step_jacobian(eq_func, base_point, parameter_vector, n_exog, 1)
    c = -_complex_step_jacobian(eq_func, base_point, parameter_vector, n_exog, 2)
    d = -_complex_step_jacobian(eq_func, base_point, parameter_vector, n_exog, 3)
    return a, b, c, d


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


# --- core.second_order -------------------------------------------------------
def _zx_zu(ghx: NDF, hx: NDF, bu: NDF, nx: int, ne: int) -> tuple[NDF, NDF]:
    """``dz/dx`` and ``dz/du`` over ``z = (lag, cur, lead, eps)``, the chain-rule
    factors the residual Hessian contracts against."""
    n = ghx.shape[0]

    zx = np.zeros((3 * n + ne, nx))
    zx[:nx, :] = np.eye(nx)
    zx[n : 2 * n, :] = ghx
    zx[2 * n : 3 * n, :] = ghx @ hx

    # The lead block walks the shock forward through the states it moved, so it
    # takes the state rows of the impact, not the whole of it.
    zu = np.zeros((3 * n + ne, ne))
    zu[n : 2 * n, :] = bu
    zu[2 * n : 3 * n, :] = ghx @ bu[:nx]
    zu[3 * n :, :] = np.eye(ne)
    return zx, zu


def _solve_second_order_numpy(
    a: NDF,
    b: NDF,
    hessian: NDF,
    gx: NDF,
    hx: NDF,
    bu: NDF,
    Q: NDF,
    n_state: int,
) -> tuple[NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF]:
    """Numpy reference for ``core.second_order.solve_second_order``.

    Returns ``(gxx, hxx, gxu, hxu, guu, huu, gss, hss)``. The lag Jacobian is
    absent from every coefficient: y_{t-1} is the differentiation variable, so
    it enters only through the identity block of ``zx``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    f_xx = np.asarray(hessian, dtype=np.float64)
    gx = np.asarray(gx, dtype=np.float64)
    hx = np.asarray(hx, dtype=np.float64)
    bu = np.asarray(bu, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    n = a.shape[0]
    nx = int(n_state)
    ny = n - nx
    ne = bu.shape[1]

    ghx = np.vstack([hx, gx])
    zx, zu = _zx_zu(ghx, hx, bu, nx, ne)

    # A = dF/dy_t with the lead's own state dependence folded into the state
    # columns; B = dF/dy_{t+1}.
    amat = -b.copy()
    amat[:, :nx] += a @ ghx
    bmat = a

    def contract(zl: NDF, zr: NDF) -> NDF:
        return np.einsum("iuv,up,vq->ipq", f_xx, zl, zr)

    # ghxx: A X + B X (hx (x) hx) = -f_xx (zx (x) zx).
    big = n * nx * nx
    kron_hh = np.kron(hx, hx).T
    sysm = (
        np.einsum("ij,kl->ikjl", amat, np.eye(nx * nx))
        + np.einsum("ij,kl->ikjl", bmat, kron_hh)
    ).reshape(big, big)
    ghxx = np.linalg.solve(sysm, -contract(zx, zx).reshape(big)).reshape((n, nx, nx))

    # ghxu / ghuu: A X = -f_xx (z (x) z) - B ghxx (. (x) bu).
    rhs_xu = -contract(zx, zu).reshape(n, nx * ne) - bmat @ np.einsum(
        "jkl,kp,lq->jpq", ghxx, hx, bu[:nx]
    ).reshape(n, nx * ne)
    ghxu = np.linalg.solve(amat, rhs_xu).reshape((n, nx, ne))

    rhs_uu = -contract(zu, zu).reshape(n, ne * ne) - bmat @ np.einsum(
        "jkl,kp,lq->jpq", ghxx, bu[:nx], bu[:nx]
    ).reshape(n, ne * ne)
    ghuu = np.linalg.solve(amat, rhs_uu).reshape((n, ne, ne))

    # ghs2: (A + B) X = -(B ghuu + f_xx (zlead (x) zlead)) vec(Q). Only the lead
    # block enters: the term is the expectation of next period's innovation.
    zlead = np.zeros((3 * n + ne, ne))
    zlead[2 * n : 3 * n, :] = bu
    rhs_ss = -(
        bmat @ ghuu.reshape(n, ne * ne) + contract(zlead, zlead).reshape(n, ne * ne)
    ) @ Q.reshape(ne * ne)
    ghs2 = np.linalg.solve(amat + bmat, rhs_ss)

    return (
        ghxx[nx:],
        ghxx[:nx],
        ghxu[nx:],
        ghxu[:nx],
        ghuu[nx:],
        ghuu[:nx],
        ghs2[nx:],
        ghs2[:nx],
    )


def first_order_residual(a: NDF, b: NDF, c: NDF, gx: NDF, hx: NDF, n_state: int) -> NDF:
    """Linearized FOC ``a ghx hx - b ghx - c`` over the state columns -- ~0 at
    the solution. Guards the pencil's signs independently of the second order."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    gx = np.asarray(gx, dtype=np.float64)
    hx = np.asarray(hx, dtype=np.float64)
    nx = int(n_state)
    ghx = np.vstack([hx, gx])
    return a @ ghx @ hx - b @ ghx - c[:, :nx]


# --- utils.dhm ---------------------------------------------------------------
@njit
def _forward_residuals_numba(
    cur_states: np.ndarray,
    fwd_states: np.ndarray,
    params: np.ndarray,
    objective_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_eq: int,
) -> np.ndarray:
    # Evaluate the numba vector residual over the path into a real
    # (n_steps x n_eq) matrix; reference for the native ``residual_path``.
    n_steps = cur_states.shape[0]
    n_var = cur_states.shape[1]
    residuals = np.empty((n_steps, n_eq), dtype=np.float64)
    cur = np.empty((n_var,), dtype=np.complex128)
    fwd = np.empty((n_var,), dtype=np.complex128)
    for t in range(n_steps):
        cur[:] = cur_states[t]
        fwd[:] = fwd_states[t]
        residual_vec = objective_fn(fwd, cur, params)
        for k in range(n_eq):
            residuals[t, k] = residual_vec[k].real
    return residuals


# --- compiled residual evaluator ---------------------------------------------
def compiled_residual(compiled) -> Callable[..., NDC]:
    """``F(fwd, cur, prev, eps, par)`` for a compiled model, as an ndarray.

    Unlike the rest of this module this wraps the native kernel rather than
    reimplementing it. The library evaluates residuals only from C, passing the
    cfunc address around, so no Python-level evaluator survives on
    ``CompiledModel``; the residual parity tests need one to compare against and
    this is it.
    """
    addr = compiled.construct_objective_cfunc().address
    n_eq = len(compiled.objective_eqs)

    def residual(fwd, cur, prev, eps, par):
        return residual_eval(addr, fwd, cur, prev, eps, par, n_eq)

    return residual
