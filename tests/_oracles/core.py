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


# --- core.klein --------------------------------------------------------------
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
def _symmetry_matrix(nx: int, ny: int) -> NDF:
    """SGU ``temp``: A maps the unique (j<=k) unknowns to the full stacked
    [gxx(:); hxx(:)], imposing gxx(i,j,k)=gxx(i,k,j). Row-major flat indices."""
    ngxx = ny * nx * nx
    nhxx = nx * nx * nx
    n_red_g = ny * nx * (nx + 1) // 2
    n_red_h = nx * nx * (nx + 1) // 2

    a_gxx = np.zeros((ngxx, n_red_g))
    a_hxx = np.zeros((nhxx, n_red_h))
    my = mx = 0
    for k in range(nx):
        for j in range(k, nx):  # j >= k
            for i in range(ny):
                a_gxx[i * nx * nx + j * nx + k, my] = 1.0
                a_gxx[i * nx * nx + k * nx + j, my] = 1.0
                my += 1
            for i in range(nx):
                a_hxx[i * nx * nx + j * nx + k, mx] = 1.0
                a_hxx[i * nx * nx + k * nx + j, mx] = 1.0
                mx += 1

    big = np.zeros((ngxx + nhxx, n_red_g + n_red_h))
    big[:ngxx, :n_red_g] = a_gxx
    big[ngxx:, n_red_g:] = a_hxx
    return big


def _solve_second_order_numpy(
    a: NDF, b: NDF, hessian: NDF, gx: NDF, hx: NDF, n_state: int
) -> tuple[NDF, NDF]:
    """Numpy reference for ``core.second_order.solve_second_order``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    f_xx = np.asarray(hessian, dtype=np.float64)
    gx = np.asarray(gx, dtype=np.float64)
    hx = np.asarray(hx, dtype=np.float64)

    n = a.shape[0]
    nx = int(n_state)
    ny = n - nx

    # --- adapter: our (a, b, F_xx) -> SGU's split derivatives ----------------
    fxp = a[:, :nx]
    fyp = a[:, nx:]
    fx = -b[:, :nx]
    fy = -b[:, nx:]

    xp, yp = slice(0, nx), slice(nx, n)
    x, y = slice(n, n + nx), slice(n + nx, 2 * n)
    fypyp, fypy, fypxp, fypx = (
        f_xx[:, yp, yp],
        f_xx[:, yp, y],
        f_xx[:, yp, xp],
        f_xx[:, yp, x],
    )
    fyyp, fyy, fyxp, fyx = f_xx[:, y, yp], f_xx[:, y, y], f_xx[:, y, xp], f_xx[:, y, x]
    fxpyp, fxpy, fxpxp, fxpx = (
        f_xx[:, xp, yp],
        f_xx[:, xp, y],
        f_xx[:, xp, xp],
        f_xx[:, xp, x],
    )
    fxyp, fxy, fxxp, fxx = f_xx[:, x, yp], f_xx[:, x, y], f_xx[:, x, xp], f_xx[:, x, x]

    ngxx = ny * nx * nx
    rows = n * nx * (nx + 1) // 2
    big_q = np.zeros((rows, n * nx * nx))
    q = np.zeros(rows)
    gxhx = gx @ hx  # (ny, nx)

    m = 0
    for i in range(n):
        for j in range(nx):
            for k in range(j + 1):  # k <= j
                # 1st: fyp-chain, outer gx*hx_j
                v1 = (
                    fypyp[i] @ gxhx[:, k]
                    + fypy[i] @ gx[:, k]
                    + fypxp[i] @ hx[:, k]
                    + fypx[i][:, k]
                )
                q[m] = v1 @ gxhx[:, j]

                # 2nd: gxx coeff  fyp(i,a) hx(b,j) hx(c,k)
                big_q[m, :ngxx] = np.einsum(
                    "a,b,c->abc", fyp[i], hx[:, j], hx[:, k]
                ).ravel()

                # 3rd: hxx coeff  (fyp gx)(a) at (j,k)
                h3 = np.zeros((nx, nx, nx))
                h3[:, j, k] = fyp[i] @ gx
                big_q[m, ngxx:] = h3.ravel()

                # 4th: fy-chain, outer gx_j
                v4 = (
                    fyyp[i] @ gxhx[:, k]
                    + fyy[i] @ gx[:, k]
                    + fyxp[i] @ hx[:, k]
                    + fyx[i][:, k]
                )
                q[m] += v4 @ gx[:, j]

                # 5th: gxx coeff  fy(i,a) at (j,k)
                g5 = np.zeros((ny, nx, nx))
                g5[:, j, k] = fy[i]
                big_q[m, :ngxx] += g5.ravel()

                # 6th: fxp-chain, outer hx_j
                v6 = (
                    fxpyp[i] @ gxhx[:, k]
                    + fxpy[i] @ gx[:, k]
                    + fxpxp[i] @ hx[:, k]
                    + fxpx[i][:, k]
                )
                q[m] += v6 @ hx[:, j]

                # 7th: hxx coeff  fxp(i,a) at (j,k)
                h7 = np.zeros((nx, nx, nx))
                h7[:, j, k] = fxp[i]
                big_q[m, ngxx:] += h7.ravel()

                # 8th: raw x_j second derivatives
                q[m] += (
                    fxyp[i, j] @ gxhx[:, k]
                    + fxy[i, j] @ gx[:, k]
                    + fxxp[i, j] @ hx[:, k]
                    + fxx[i, j, k]
                )
                m += 1

    sym = _symmetry_matrix(nx, ny)
    qt = big_q @ sym
    xt = -np.linalg.solve(qt, q)
    full = sym @ xt

    gxx = full[:ngxx].reshape((ny, nx, nx))
    hxx = full[ngxx:].reshape((nx, nx, nx))
    return gxx, hxx


def _solve_second_order_risk_numpy(
    a: NDF, b: NDF, hessian: NDF, gx: NDF, gxx: NDF, eta: NDF, n_state: int
) -> tuple[NDF, NDF]:
    """Numpy reference for ``core.second_order.solve_second_order_risk``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    f_xx = np.asarray(hessian, dtype=np.float64)
    gx = np.asarray(gx, dtype=np.float64)
    gxx = np.asarray(gxx, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)

    n = a.shape[0]
    nx = int(n_state)
    ny = n - nx

    fxp, fyp = a[:, :nx], a[:, nx:]
    fy = -b[:, nx:]
    xp, yp = slice(0, nx), slice(nx, n)
    fypyp = f_xx[:, yp, yp]
    fypxp = f_xx[:, yp, xp]
    fxpyp = f_xx[:, xp, yp]
    fxpxp = f_xx[:, xp, xp]

    gxe = gx @ eta  # (ny, ne)
    q_g = np.zeros((n, ny))
    q_h = np.zeros((n, nx))
    q = np.zeros(n)
    for i in range(n):
        q_h[i] = fyp[i] @ gx + fxp[i]  # 1st + 7th
        q_g[i] = fyp[i] + fy[i]  # 5th + 6th
        g4 = np.einsum("a,abc->bc", fyp[i], gxx)  # fyp . gxx over controls
        q[i] = (
            np.sum((fypyp[i] @ gxe) * gxe)  # 2nd
            + np.sum((fypxp[i] @ eta) * gxe)  # 3rd
            + np.sum((g4 @ eta) * eta)  # 4th
            + np.sum((fxpyp[i] @ gx @ eta) * eta)  # 8th
            + np.sum((fxpxp[i] @ eta) * eta)  # 9th
        )

    x = -np.linalg.solve(np.hstack([q_g, q_h]), q)
    return x[:ny], x[ny:]


def first_order_residual(a: NDF, b: NDF, gx: NDF, hx: NDF, n_state: int) -> NDF:
    """Linearized FOC ``fyp gx hx + fy gx + fxp hx + fx`` -- ~0 at the solution.
    Guards the adapter (block slicing + the ``-b`` sign) independently of gxx."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    gx = np.asarray(gx, dtype=np.float64)
    hx = np.asarray(hx, dtype=np.float64)
    nx = int(n_state)
    fxp, fyp = a[:, :nx], a[:, nx:]
    fx, fy = -b[:, :nx], -b[:, nx:]
    return fyp @ gx @ hx + fy @ gx + fxp @ hx + fx


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
