from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import complex128, float64
from numpy.typing import NDArray

from .._ckernels.core import (
    steady_state_newton,
    klein_postprocess,
    klein_preprocess,
    klein_qz,
)

NDF = NDArray[float64]
NDC = NDArray[complex128]


@dataclass(slots=True, frozen=True)
class KleinSolution:
    """Solution of ``a E[y_{t+1}] = b y_t``: ``u_t = f s_t``, ``s_{t+1} = p s_t``.

    ``p``/``f`` are complex (the imaginary parts are ~1e-16 roundoff from the
    complex Schur form; the caller collapses them via ``real_if_close``). Stored
    as ``SolvedModel.policy``; downstream reads only ``.f`` and ``.stab``.

    ``steady_state`` is the Newton-resolved steady state the solve linearized at
    (the seed after convergence), so second-order and measurement callers reuse
    it instead of re-solving.
    """

    p: NDC
    f: NDC
    stab: int
    eig: NDC
    steady_state: NDF
    order: int = 1


@dataclass(slots=True, frozen=True)
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


def klein_solve(
    residual_cfunc: Any,
    params: NDF,
    ss_seed: NDF,
    n_states: int,
    *,
    log_linear: bool = False,
) -> KleinSolution:
    """First-order Klein solve of the compiled model at ``params``.

    ``residual_cfunc`` is the compiled residual as a numba @cfunc
    (``construct_objective_cfunc()``); it drives the complex-step preproc in C.
    ``ss_seed`` seeds a Newton solve of ``F(ss, ss) = 0``; the solve linearizes
    at the resolved steady state, which the returned :class:`KleinSolution`
    carries in ``steady_state``.
    """
    # Klein requires a square pencil: n_eq == n_var == len(ss_seed).
    n_eq = np.asarray(ss_seed).shape[0]
    ss, _ = steady_state_newton(residual_cfunc.address, ss_seed, params)

    a, b = klein_preprocess(residual_cfunc.address, ss, params, n_eq, log_linear)
    # Native QZ (LAPACK zgges via the scipy cython_lapack pointer, ordered by the
    # Klein 'ouc' criterion) — bit-for-bit equal to the former
    # ordqz(a, b, sort="ouc", output="complex")[0, 1, 5].
    s, t, z = klein_qz(a, b)
    f, p, stab, eig = klein_postprocess(
        np.asarray(s, dtype=complex128),
        np.asarray(t, dtype=complex128),
        np.asarray(z, dtype=complex128),
        n_states,
    )
    return KleinSolution(
        p=p, f=f, stab=stab, eig=eig, steady_state=np.asarray(ss, dtype=float64)
    )
