from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numpy import complex128, float64
from numpy.typing import NDArray

from .._ckernels.core import klein_solve1

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
    it instead of re-solving. ``a``/``b`` are the pencil taken there, carried for
    the same reason: rebuilding it costs another complex-step Jacobian sweep.

    ``A``/``B`` are the assembled state space, which the solve produces on the
    way to ``p``/``f`` rather than as a separate step.
    """

    p: NDC
    f: NDC
    stab: int
    eig: NDC
    steady_state: NDF
    a: NDF
    b: NDF
    A: NDF
    B: NDF
    order: int = 1


@dataclass(slots=True, frozen=True)
class PerturbationSolution:
    """First- (+ optional second-) order perturbation solution.

    Carries the same first-order interface as
    :class:`~SymbolicDSGE.core.klein.KleinSolution`
    (``p`` = h_x, ``f`` = g_x, ``stab``, ``eig``, ``steady_state``) so it
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
    gxx: NDF
    hxx: NDF
    gss: NDF
    hss: NDF


def klein_solve(
    residual_cfunc: Any,
    params: NDF,
    ss_seed: NDF,
    n_states: int,
    *,
    n_exog: int = 0,
) -> KleinSolution:
    """First-order Klein solve of the compiled model at ``params``.

    ``residual_cfunc`` is the compiled residual as a numba @cfunc
    (``construct_objective_cfunc()``); it drives the complex-step preproc in C.
    ``ss_seed`` seeds a Newton solve of ``F(ss, ss) = 0``; the solve linearizes
    at the resolved steady state, which the returned :class:`KleinSolution`
    carries in ``steady_state``.

    One native call runs the whole solve (steady state, pencil, QZ, post-proc,
    state space) under a single GIL release, so ``n_exog`` is needed here to size
    the ``B`` block. A nonzero ``stab`` returns normally; the caller decides
    whether to raise.
    """
    try:
        ss, a, b, f, p, stab, eig, A, B = klein_solve1(
            residual_cfunc.address, ss_seed, params, n_states, n_exog
        )
    except ValueError as exc:
        # The kernel reports the factor it could not invert. This is the first
        # frame that knows the factors came from a model someone wrote, so the
        # dating that fails the same way is named here rather than there.
        if "Blanchard-Kahn" not in str(exc):
            raise
        raise ValueError(
            f"{exc} An equation shifted forward in time fails this way too: "
            f"the compiler lifts lags into states of its own, so a process "
            f"belongs in its natural form `v(t) = rho*v(t-1) + e` rather than "
            f"`v(t+1) = rho*v(t) + e`."
        ) from exc
    return KleinSolution(
        p=p, f=f, stab=stab, eig=eig, steady_state=ss, a=a, b=b, A=A, B=B
    )
