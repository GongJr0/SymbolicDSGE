"""Backend native entry points and data structures for the solver."""

from __future__ import annotations

from typing import ClassVar
from dataclasses import dataclass
from enum import IntEnum, unique

from numpy import array2string, complex128, empty, float64, int64
from numpy.typing import NDArray

from .._ckernels.core import INC_LAG, INC_LEAD, klein_solve1, sgu_klein_solve2
from .._ckernels.core._solve_errors import SolveStatus
from .._ckernels.occbin import occbin_solve1
from .compiled_model import CompiledModel, _shock_covariance

NDF = NDArray[float64]
NDC = NDArray[complex128]


# Solve failures. One class per message shape rather than per status: what a
# failure can report is fixed by how far the solve got, which is also what the
# templates below compose.

_CODE = "Code: {status.name} ({status.value})\nDescription: {description}"
_CLASSIFICATION = (
    "\nState space: {n_state} state(s) {states}, {n_jump} jump(s) {jumps}{mixed}"
)
_EIGENVALUES = "\nGeneralized eigenvalues: {eig}"
_SECOND_ORDER = "\nThe first-order rule was obtained; the correction over it was not."


class SolveError(RuntimeError):
    """A native solve that produced no usable rule.

    :attr:`status` is the kernel's verdict. The remaining fields are whatever the
    failing stage had produced, and each subclass renders the ones it has.
    """

    template: ClassVar[str]

    def __init__(
        self,
        status: SolveStatus,
        description: str,
        states: tuple[str, ...],
        jumps: tuple[str, ...],
        mixed: tuple[str, ...],
        eig: NDC,
    ) -> None:
        self.status = status
        self.states = states
        self.jumps = jumps
        self.mixed = mixed
        self.eig = eig
        super().__init__(
            self.template.format(
                status=status,
                description=description,
                n_state=len(states or ()),
                n_jump=len(jumps or ()),
                states=states,
                jumps=jumps,
                mixed=(
                    f", {mixed} occurring at both dates and counted in each"
                    if mixed
                    else ""
                ),
                eig=(
                    array2string(eig, precision=4, max_line_width=72)
                    if eig is not None
                    else ""
                ),
            )
        )


class SteadyStateError(SolveError):
    """The steady state did not resolve, which happens before anything else runs."""

    template = _CODE


class DecompositionError(SolveError):
    """The decomposition the rule is read from failed in LAPACK."""

    template = _CODE + _CLASSIFICATION


class DecisionRuleError(DecompositionError):
    """The rule could not be read off a decomposition that succeeded."""

    template = _CODE + _CLASSIFICATION + _EIGENVALUES


class SecondOrderError(DecisionRuleError):
    """The quadratic terms or the risk correction failed over a first-order rule."""

    template = _CODE + _CLASSIFICATION + _EIGENVALUES + _SECOND_ORDER


_FAILURES: dict[SolveStatus, tuple[type[SolveError], str]] = {
    SolveStatus.NEWTON_SINGULAR: (
        SteadyStateError,
        "The steady-state Jacobian is singular at this parameter point.",
    ),
    SolveStatus.NEWTON_NO_CONVERGE: (
        SteadyStateError,
        "The steady state did not converge within the kernel's iteration budget, "
        "or the residual went non-finite.",
    ),
    SolveStatus.QR: (
        DecompositionError,
        "The rotation separating `t`-only variables failed.",
    ),
    SolveStatus.QZ: (
        DecompositionError,
        "The generalized Schur decomposition failed.",
    ),
    SolveStatus.RANK_FAIL: (
        DecisionRuleError,
        "Some combination of the model's states does not enter the solution.",
    ),
    SolveStatus.INFINITE_ROOT: (
        DecisionRuleError,
        "One of the roots taken as stable is infinite.",
    ),
    SolveStatus.NO_STABLE_SOLUTION: (
        DecisionRuleError,
        "Fewer stable eigenvalues than states: the model cannot yield a stable solution.",
    ),
    SolveStatus.STATIC_SINGULAR: (
        DecisionRuleError,
        "The subsystem of variables occurring only at 't' is singular.",
    ),
    SolveStatus.SHOCK_SINGULAR: (
        DecisionRuleError,
        "The shock loading cannot be recovered: matrix inversion failed "
        "due to singularity.",
    ),
    SolveStatus.SECOND_ORDER_SINGULAR: (
        SecondOrderError,
        "Quadratic terms cannot be recovered: second-order system is singular.",
    ),
    SolveStatus.SECOND_ORDER_RISK: (
        SecondOrderError,
        "Risk-correction terms cannot be recovered: risk system is singular.",
    ),
}


def raise_solve_error(
    status: int,
    *,
    states: tuple[str, ...],
    jumps: tuple[str, ...],
    mixed: tuple[str, ...],
    eig: NDC,
) -> None:
    """Raise the failure a nonzero solve status names.

    A status the table does not carry means the kernel grew a code, which the
    enum reports before this gets the chance.
    """
    resolved = SolveStatus(status)
    cls, description = _FAILURES[resolved]
    raise cls(resolved, description, states, jumps, mixed, eig)


@unique
class BKStatus(IntEnum):
    """Blanchard-Kahn stability indicator."""

    DETERMINATE = 0
    INDETERMINATE = 1

    @property
    def message(self) -> str:
        """Human-readable message for the stability indicator."""
        match self:
            case BKStatus.DETERMINATE:
                return "The solution for this specification is unique and stable."
            case BKStatus.INDETERMINATE:
                return (
                    "There are more forward-looking variables than unstable eigenvalues. "
                    "This solution is one of multiple that exist for this specification."
                )


@dataclass(frozen=True, repr=False)
class BaseSolution:
    """Base class for attributes shared across all solution methods.

    :attr:`steady_state` is the Newton-resolved expansion point the solution is linearized at.
    :attr:`stab` is the stability indicator: `0` = determinate, `1` = indeterminate but stable.
    :attr:`eig` is the array of eigenvalues of the linearized system.
    :attr:`order` is the order of the solution: 1 = first-order, 2 = second-order, etc.
    """

    steady_state: NDF
    stab: BKStatus
    eig: NDC
    order: int

    @property
    def is_determinate(self) -> bool:
        """Whether the solution is determinate (unique and stable)."""
        return self.stab == BKStatus.DETERMINATE


@dataclass(frozen=True, repr=False)
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


@dataclass(frozen=True, repr=False)
class SecondOrderSolution(FirstOrderSolution):
    """Second-order solution: the first-order rule plus the corrections taken at
    the same expansion point.

    :attr:`gx` (ny, nx)/``hx`` (nx, nx) are ``f``/``p`` under the perturbation
    notation. The quadratic terms split by which pair of arguments they weigh,
    the shocks having their own now that they are not states:
    :attr:`gxx` (ny, nx, nx)/``hxx`` (nx, nx, nx) over the states,
    :attr:`gxu` (ny, nx, ne)/``hxu`` (nx, nx, ne) over a state and a shock,
    :attr:`guu` (ny, ne, ne)/``huu`` (nx, ne, ne) over the shocks, and
    :attr:`gss` (ny,)/``hss`` (nx,) the sigma^2 risk correction.
    """

    gxx: NDF
    hxx: NDF
    gxu: NDF
    hxu: NDF
    guu: NDF
    huu: NDF
    gss: NDF
    hss: NDF

    @property
    def hx(self) -> NDF:
        """``p`` in the perturbation notation: the state transition matrix."""
        return self.p

    @property
    def gx(self) -> NDF:
        """``f`` in the perturbation notation: the policy matrix."""
        return self.f


@dataclass(frozen=True, repr=False)
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

    @property
    def A(self) -> NDF:
        """The reference regime's state transition matrix."""
        return self.ref.A

    @property
    def B(self) -> NDF:
        """The reference regime's shock impact matrix."""
        return self.ref.B


def _classify(
    compiled: CompiledModel,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """States, jumps, and the variables counted in both, by incidence bits."""
    states: list[str] = []
    jumps: list[str] = []
    mixed: list[str] = []
    for name, bits in zip(compiled.var_names, compiled._incidence):
        if bits & INC_LAG:
            states.append(name)
        if bits & INC_LEAD:
            jumps.append(name)
        if bits & INC_LAG and bits & INC_LEAD:
            mixed.append(name)
    return tuple(states), tuple(jumps), tuple(mixed)


def _raise_for(compiled: CompiledModel, status: int, eig: NDC) -> None:
    """Raise what a nonzero solve status names, with this model's classification."""
    states, jumps, mixed = _classify(compiled)
    raise_solve_error(status, states=states, jumps=jumps, mixed=mixed, eig=eig)


def klein_solve(
    compiled: CompiledModel, params: NDF, ss_seed: NDF
) -> FirstOrderSolution:
    """First-order Klein solve of ``compiled`` at ``params``.

    ``ss_seed`` seeds a Newton solve of ``F(ss, ss) = 0``; the solve linearizes at
    the resolved steady state, which the returned solution carries in
    ``steady_state``. One native call runs the whole solve under a single GIL
    release. A nonzero ``stab`` returns normally and the caller decides whether
    indeterminacy is fatal; anything the kernel rejects raises here.
    """
    err, ss, f, p, stab, eig, A, B = klein_solve1(
        compiled.construct_objective_cfunc().address,
        ss_seed,
        params,
        compiled._incidence,
        compiled.n_state,
        compiled.n_exog,
    )
    if err:
        _raise_for(compiled, err, eig)
    return FirstOrderSolution(
        steady_state=ss, stab=BKStatus(stab), eig=eig, order=1, p=p, f=f, A=A, B=B
    )


def sgu_solve(
    compiled: CompiledModel, params: NDF, ss_seed: NDF
) -> SecondOrderSolution:
    """Second-order solve of ``compiled`` at ``params``.

    The first-order half is :func:`klein_solve`. The bicomplex residual drives the
    Hessian sweep and the shock covariance is what the risk correction integrates
    against, both taken from ``compiled``. One native call runs both orders.

    ``A``/``B`` come back beside the solution. They are the first-order state
    space, which ``SolvedModel`` already exposes directly.
    """
    (
        err,
        ss,
        f,
        p,
        stab,
        eig,
        gxx,
        hxx,
        gxu,
        hxu,
        guu,
        huu,
        gss,
        hss,
        A,
        B,
    ) = sgu_klein_solve2(
        compiled.construct_objective_cfunc().address,
        compiled.construct_objective_cfunc_bicomplex().address,
        ss_seed,
        params,
        _shock_covariance(compiled),
        compiled._incidence,
        compiled.n_state,
        compiled.n_exog,
    )
    if err:
        _raise_for(compiled, err, eig)
    return SecondOrderSolution(
        steady_state=ss,
        stab=BKStatus(stab),
        eig=eig,
        order=2,
        p=p,
        f=f,
        A=A,
        B=B,
        gxx=gxx,
        hxx=hxx,
        gxu=gxu,
        hxu=hxu,
        guu=guu,
        huu=huu,
        gss=gss,
        hss=hss,
    )


def piecewise_solve(
    compiled: CompiledModel, params: NDF, ss_seed: NDF
) -> PiecewiseSolution:
    """Piecewise-linear (OccBin) solve of ``compiled`` at ``params``.

    The reference regime is an ordinary Klein solve, and every other regime is the
    pencil it linearized at with that regime's rows replaced. Both halves run in
    one native call, so the reference pencil never crosses back into Python
    between them.

    The regime table is indexed by binding bitmask and dense over
    ``0..2 ** n_constraint - 1``. Slot 0 is the reference and carries address 0
    and no rows. A nonzero ``stab`` is the reference regime's: a binding regime
    alone is routinely indeterminate, which is not an error.
    """
    pencil = compiled.construct_regime_pencil_func()
    if pencil is None:
        raise ValueError("Piecewise solve needs a model with constraints.")

    n_constraint = len(compiled.constraint_names)
    n_regime = 1 << n_constraint
    addrs = [0] + [pencil.address(m) for m in range(1, n_regime)]
    rows = [empty(0, dtype=int64)] + [pencil.rows[m] for m in range(1, n_regime)]

    n_state = compiled.n_state
    err, ss, ghx, stab, eig, A, B, a, b, c, d, cst = occbin_solve1(
        compiled.construct_objective_cfunc().address,
        ss_seed,
        params,
        compiled._incidence,
        n_state,
        addrs,
        rows,
        n_constraint,
        compiled.n_exog,
    )
    if err:
        _raise_for(compiled, err, eig)
    return PiecewiseSolution(
        steady_state=ss,
        stab=BKStatus(stab),
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
            stab=BKStatus(stab),
            eig=eig,
            order=1,
            p=ghx[:n_state],
            f=ghx[n_state:],
            A=A,
            B=B,
        ),
    )
