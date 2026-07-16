from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import complex128, float64
from numpy.typing import NDArray

from .._ckernels.core import (
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
    """

    p: NDC
    f: NDC
    stab: int
    eig: NDC
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
    steady_state: NDF,
    n_states: int,
    *,
    log_linear: bool = False,
) -> KleinSolution:
    """First-order Klein solve of the compiled model at ``(params, steady_state)``.

    ``residual_cfunc`` is the compiled residual as a numba @cfunc
    (``construct_objective_cfunc()``); it drives the complex-step preproc in C.
    Returns a :class:`KleinSolution` with complex ``p``/``f``.
    """
    # Klein requires a square pencil: n_eq == n_var == len(steady_state).
    n_eq = np.asarray(steady_state).shape[0]
    a, b = klein_preprocess(
        residual_cfunc.address, steady_state, params, n_eq, log_linear
    )
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
    return KleinSolution(p=p, f=f, stab=stab, eig=eig)
