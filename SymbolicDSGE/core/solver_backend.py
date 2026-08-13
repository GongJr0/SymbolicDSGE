from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from numpy import complex128, float64, int8, int64
from numpy.typing import NDArray

from .._ckernels.core import klein_solve1, sgu_klein_solve2
from .._ckernels.occbin import occbin_solve1

NDF = NDArray[float64]
NDC = NDArray[complex128]


@dataclass(frozen=True)
class BaseSolution:
    """Base class for attributes shared across all solution methods.

    :attr:`steady_state` is the Newton-resolved expansion point the solution is linearized at.
    :attr:`stab` is the stability indicator: -1 = not enough stable eigenvalues, 0 = exactly right, 1 = too many stable eigenvalues.
    :attr:`eig` is the array of eigenvalues of the linearized system.
    :attr:`order` is the order of the solution: 1 = first-order, 2 = second-order, etc.
    """

    steady_state: NDF
    stab: int
    eig: NDC
    order: int


@dataclass(frozen=True)
class FirstOrderSolution(BaseSolution):
    """First order solution of ``a E[y_{t+1}] = b y_t``: ``u_t = f s_t``, ``s_{t+1} = p s_t``.

    ``p``/``f`` are real: the Schur form they come from is complex, but its
    imaginary parts are roundoff on a real pencil and the native solve projects
    them once. ``eig`` stays complex.

    :attr:`p`/``f`` are the policy and transition matrices.
    :attr:`A`/``B`` are the first-order state space. x_{t+1} = A x_t + B e_{t+1}

    """

    p: NDF
    f: NDF
    A: NDF
    B: NDF


@dataclass(frozen=True)
class SecondOrderSolution(FirstOrderSolution):
    """Second-order (SGU) solution: the first-order rule plus the corrections
    taken at the same expansion point.

    :attr:`gx` (ny, nx)/``hx`` (nx, nx) are ``f``/``p`` under SGU's names.
    :attr:`gxx` (ny, nx, nx)/``hxx`` (nx, nx, nx) are the state-quadratic terms.
    :attr:`gss` (ny,)/``hss`` (nx,) are the sigma^2 risk correction.
    """

    gxx: NDF
    hxx: NDF
    gss: NDF
    hss: NDF

    @property
    def hx(self) -> NDF:
        """``p`` in SGU's notation: the state transition matrix."""
        return self.p

    @property
    def gx(self) -> NDF:
        """``f`` in SGU's notation: the policy matrix."""
        return self.f


@dataclass(frozen=True)
class PiecewiseSolution(BaseSolution):
    """Piecewise-linear (OccBin) solution: everything a parameter draw fixes.

    The piecewise policy itself is path dependent, ``y_t = P_t y_{t-1} + D_t``,
    so its rules are built per date against a guessed regime sequence rather
    than once. What a draw does fix is the pencil of every regime, since all of
    them linearize at the same reference steady state. :attr:`steady_state`,
    ``stab`` and ``eig`` are that reference regime's, and so is :attr:`ref`.

    :attr:`a`/``b``/``c`` (n_regime, n_var, n_var) and ``d``
    (n_regime, n_var, n_exog) are the regime pencils
    ``a^r E[y_{t+1}] = b^r y_t + c^r y_{t-1} + d^r eps_t - cst^r``, indexed by
    binding bitmask, slot 0 the reference. ``cst`` (n_regime, n_var) is the
    constraint's whole mechanism and is zero at slot 0.
    :attr:`ghx_ref` (n_var, n_state) is the reference regime's whole rule, which
    the recursion takes as its terminal condition: past the horizon the guess is
    relaxed, so the model is its own unconstrained self.
    :attr:`ref` is that same reference regime as an ordinary first-order
    solution, so its ``A``/``B`` describe the model with the constraints ignored
    and ``ghx_ref`` is ``ref.p`` stacked over ``ref.f``.
    """

    a: NDF
    b: NDF
    c: NDF
    d: NDF
    cst: NDF
    ghx_ref: NDF
    ref: FirstOrderSolution


@contextmanager
def _bk_dating_hint() -> Generator[None]:
    """Name the model-authoring mistake behind a Blanchard-Kahn failure.

    The kernel reports the factor it could not invert. This is the first frame
    that knows the factors came from a model someone wrote, so the dating that
    fails the same way is named here rather than there.
    """
    try:
        yield
    except ValueError as exc:
        if "Blanchard-Kahn" not in str(exc):
            raise
        raise ValueError(
            f"{exc} An equation shifted forward in time fails this way too: "
            f"the compiler lifts lags into states of its own, so a process "
            f"belongs in its natural form `v(t) = rho*v(t-1) + e` rather than "
            f"`v(t+1) = rho*v(t) + e`."
        ) from exc


def klein_solve(
    residual_cfunc: Any,
    params: NDF,
    ss_seed: NDF,
    incidence: NDArray[int8],
    n_states: int,
    *,
    n_exog: int = 0,
) -> FirstOrderSolution:
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
    with _bk_dating_hint():
        ss, f, p, stab, eig, A, B = klein_solve1(
            residual_cfunc.address, ss_seed, params, incidence, n_states, n_exog
        )
    return FirstOrderSolution(
        steady_state=ss, stab=stab, eig=eig, order=1, p=p, f=f, A=A, B=B
    )


def sgu_solve(
    residual_cfunc: Any,
    bc_residual_cfunc: Any,
    params: NDF,
    ss_seed: NDF,
    incidence: NDArray[int8],
    n_states: int,
    eta: NDF,
    *,
    n_exog: int = 0,
) -> SecondOrderSolution:
    """Second-order (SGU) solve of the compiled model at ``params``.

    The first-order half is :func:`klein_solve`. ``bc_residual_cfunc``
    is the bicomplex residual (``construct_objective_cfunc_bicomplex()``) that
    drives the Hessian sweep, and ``eta`` is the ``(n_states, n_exog)`` shock
    loading the risk correction integrates against.

    One native call runs both orders under a single GIL release. A nonzero ``stab``
    returns normally; the caller decides whether to raise.

    ``A``/``B`` come back beside the solution. They are the
    first-order state space, which ``SolvedModel`` already exposes directly.
    """
    with _bk_dating_hint():
        ss, f, p, stab, eig, gxx, hxx, gss, hss, A, B = sgu_klein_solve2(
            residual_cfunc.address,
            bc_residual_cfunc.address,
            ss_seed,
            params,
            incidence,
            n_states,
            eta,
            n_exog,
        )
    return SecondOrderSolution(
        steady_state=ss,
        stab=stab,
        eig=eig,
        order=2,
        p=p,
        f=f,
        A=A,
        B=B,
        gxx=gxx,
        hxx=hxx,
        gss=gss,
        hss=hss,
    )


def piecewise_solve(
    residual_cfunc: Any,
    pencil_addrs: Sequence[int],
    rows: Sequence[NDArray[int64]],
    params: NDF,
    ss_seed: NDF,
    incidence: NDArray[int8],
    n_states: int,
    n_constraint: int,
    *,
    n_exog: int = 0,
) -> PiecewiseSolution:
    """Piecewise-linear (OccBin) solve of the compiled model at ``params``.

    The reference regime is an ordinary Klein solve, and every other regime is
    the pencil it linearized at with that regime's rows replaced. Both halves
    run in one native call, so the reference pencil never crosses back into
    Python between them.

    ``pencil_addrs`` and ``rows`` are indexed by binding bitmask and dense over
    ``0..2 ** n_constraint - 1``; slot 0 is the reference and carries address 0
    and no rows. A nonzero ``stab`` returns normally and is the reference
    regime's: a binding regime alone is routinely indeterminate, which is not an
    error, so only the reference verdict means anything.
    """
    with _bk_dating_hint():
        ss, ghx, stab, eig, A, B, a, b, c, d, cst = occbin_solve1(
            residual_cfunc.address,
            ss_seed,
            params,
            incidence,
            n_states,
            pencil_addrs,
            rows,
            n_constraint,
            n_exog,
        )
    return PiecewiseSolution(
        steady_state=ss,
        stab=stab,
        eig=eig,
        order=1,
        a=a,
        b=b,
        c=c,
        d=d,
        cst=cst,
        ghx_ref=ghx,
        # p and f are views into ghx: states lead the canonical order, so the
        # stack the recursion wants is already the two blocks end to end.
        ref=FirstOrderSolution(
            steady_state=ss,
            stab=stab,
            eig=eig,
            order=1,
            p=ghx[:n_states],
            f=ghx[n_states:],
            A=A,
            B=B,
        ),
    )
