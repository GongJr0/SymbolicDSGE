"""Second-order perturbation assembly: row-major transpile of SGU's ``gxx_hxx.m``.

Given the residual's first- and second-order derivatives at the deterministic
steady state -- ``a = dF/dfwd`` and ``b = -dF/dcur`` from ``klein_preproc``, the
Hessian ``F_xx`` from ``bicomplex_hessian`` -- plus the first-order solution
(``h_x`` = KleinSolution.p, ``g_x`` = KleinSolution.f, real parts), solves the
linear system for the second-order policy tensors ``g_xx`` (controls) and
``h_xx`` (states). See Schmitt-Grohe & Uribe (JEDC 2004).

Row-major (C-order) throughout: the tensors are identical to SGU's column-major
result -- only the internal vectorization differs, chosen so the eventual native
port inherits the indexing without a C<->F translation. Validated against
SGU-original + Dynare ghxx + the first-order FOC.

Stacked arg order (matching our layout / F_x / F_xx): z = [x'; y'; x; y] with
states first (nx = n_state), controls after (ny = n - nx):
    x' -> [0:nx),  y' -> [nx:n),  x -> [n:n+nx),  y -> [n+nx:2n).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

from .._native_dispatch import FORCE_NUMBA, REQUIRE_NATIVE

NDF = NDArray[np.float64]
NDC = NDArray[np.complex128]

# Prefer the native assembly kernels; fall back to the numpy reference below when
# the extension is not built (ALWAYS_USE_NUMBA / NEVER_USE_NUMBA override -- see
# _native_dispatch). The numpy path stays the parity oracle either way.
_native_second_order: Callable[..., Any] | None
_native_second_order_risk: Callable[..., Any] | None
if FORCE_NUMBA:
    _native_second_order = None
    _native_second_order_risk = None
else:
    try:
        from .._ckernels.core import second_order as _native_second_order
        from .._ckernels.core import second_order_risk as _native_second_order_risk
    except ImportError:  # pragma: no cover - exercised only without the extension
        if REQUIRE_NATIVE:
            raise
        _native_second_order = None
        _native_second_order_risk = None


@dataclass(frozen=True)
class PerturbationSolution:
    """First- (+ optional second-) order perturbation solution.

    A superset of :class:`~SymbolicDSGE.core.klein.KleinSolution`: it carries the
    same first-order interface (``p`` = h_x, ``f`` = g_x, ``stab``, ``eig``) so it
    drops into ``SolvedModel.policy`` unchanged -- every existing first-order path
    (``sim``/``irf``/``kalman``) keeps reading ``.f``/``.p``/``.stab``. The
    second-order tensors are ``None`` at ``order == 1``:

    * ``hxx`` (nx, nx, nx), ``gxx`` (ny, nx, nx) -- the state-quadratic terms;
    * ``hss`` (nx,), ``gss`` (ny,) -- the sigma^2 risk correction.

    ``steady_state`` is the (nonlinear) expansion point the tensors are taken at.
    """

    p: NDC
    f: NDC
    stab: int
    eig: NDC
    order: int
    steady_state: NDF
    gxx: NDF | None = None
    hxx: NDF | None = None
    gss: NDF | None = None
    hss: NDF | None = None


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


def solve_second_order(
    a: NDF, b: NDF, hessian: NDF, gx: NDF, hx: NDF, n_state: int
) -> tuple[NDF, NDF]:
    """Second-order policy tensors ``(gxx, hxx)``.

    ``gxx`` is ``(ny, nx, nx)`` (controls) and ``hxx`` is ``(nx, nx, nx)``
    (states), symmetric in the last two indices. Prefers the native kernel
    (``_ckernels.core.second_order``); falls back to :func:`_solve_second_order_numpy`.
    """
    if _native_second_order is not None:
        return cast(
            "tuple[NDF, NDF]",
            _native_second_order(a, b, hessian, gx, hx, n_state),
        )
    return _solve_second_order_numpy(a, b, hessian, gx, hx, n_state)


def _solve_second_order_numpy(
    a: NDF, b: NDF, hessian: NDF, gx: NDF, hx: NDF, n_state: int
) -> tuple[NDF, NDF]:
    """Numpy reference for :func:`solve_second_order` (parity oracle + fallback)."""
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


def solve_second_order_risk(
    a: NDF, b: NDF, hessian: NDF, gx: NDF, gxx: NDF, eta: NDF, n_state: int
) -> tuple[NDF, NDF]:
    """Risk correction ``(g_ss, h_ss)`` -- row-major transpile of SGU ``gss_hss.m``.

    ``eta`` is the shock loading (nx x ne): ``x' = h(x) + eta @ eps``, ``eps ~
    N(0, I)`` (so ``eta @ eta.T`` is the state innovation covariance). Prefers the
    native kernel (``_ckernels.core.second_order_risk``); falls back to
    :func:`_solve_second_order_risk_numpy`.
    """
    if _native_second_order_risk is not None:
        return cast(
            "tuple[NDF, NDF]",
            _native_second_order_risk(a, b, hessian, gx, gxx, eta, n_state),
        )
    return _solve_second_order_risk_numpy(a, b, hessian, gx, gxx, eta, n_state)


def _solve_second_order_risk_numpy(
    a: NDF, b: NDF, hessian: NDF, gx: NDF, gxx: NDF, eta: NDF, n_state: int
) -> tuple[NDF, NDF]:
    """Numpy reference for :func:`solve_second_order_risk`.

    Only the forward-forward Hessian blocks enter, since the risk term comes from
    the shock's effect on ``(y', x')``. Solves ``[Qg Qh] [g_ss; h_ss] = -q``. The
    ``diag/sum`` idioms in the MATLAB are traces: ``sum(diag(M' N)) = sum(M*N)``.
    """
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
