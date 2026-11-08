"""Solver for DSGE model specifications. Handles compilation, solution, and has estimation shortcuts."""

from __future__ import annotations

import warnings

import sympy as sp
from sympy import Symbol, Expr
from sympy.logic.boolalg import Boolean
from typing import Any, Sequence
from types import MappingProxyType

import numpy as np
from numpy import float64, ndarray
from numpy.typing import NDArray


from .config import ModelConfig, TIME
from .resolvers import VariableLayout, resolve_ss_seed
from .compiled_model import (
    CompiledModel,
    RegimeBlock,
)
from sympy.core.function import AppliedUndef

from .desugar import (
    DesugarResult,
    _call_offset,
    _offset_expressions,
    desugar_model,
)
from .solved_model import (
    SolvedModel,
    FirstOrderSolvedModel,
    PiecewiseSolvedModel,
    SecondOrderSolvedModel,
)
from .solver_backend import klein_solve, piecewise_solve, sgu_solve

from ..linearization.view import PointLinearView

NDF = NDArray[float64]
ND = NDArray


def compile_model(config: ModelConfig) -> CompiledModel:
    """Compile a parsed Model config into numerical representation ready for solving.

    Parameters
    ----------
    config : ModelConfig
        Model specification to compile.

    Returns
    -------
    CompiledModel
        The compiled model holding attrs and methods necessary for all subsequent operations.

    """
    if isinstance(config, PointLinearView):  # type: ignore[unreachable]
        raise TypeError(
            "A point-linearized model is not accepted as solver input. "
            "Linearizing a levels model about a point in parameter space changes the "
            "interpretation of any likelihood-related operation under varying parameters. "
            "Linearization already occurs numerically in the solver. You do not need to "
            "pre-linearize a specification symbolically. If you intended to use the model "
            "linearized at one constant point, pass `PointLinearView.to_config(). "
        )

    # Convert model to {-1, 0, 1} offsets with auxiliary variables for lags and leads.
    desugared = desugar_model(config)
    t = TIME
    ordered_variables = desugared.variables

    # Convert model to minimization problem
    residuals = [eq.lhs - eq.rhs for eq in desugared.model.values()]  # pyright: ignore

    name_to_func = {v.__name__: v for v in ordered_variables}

    layout = _infer_variable_layout(desugared, config.shocks)
    var_order = layout.canonical_names

    var_funcs = [name_to_func[name] for name in var_order]
    idx = layout.idx

    # Substitutions
    cur_syms = [Symbol(f"cur_{n}") for n in var_order]
    fwd_syms = [Symbol(f"fwd_{n}") for n in var_order]
    prev_syms = [Symbol(f"prev_{n}") for n in var_order]

    # Keys are exact applied calls and no key occurs in any value, so
    # xreplace's single structural pass is correct here and far cheaper
    # than the semantic matching subs performs.
    subs_map = {}
    for f, cur, fwd, prev in zip(var_funcs, cur_syms, fwd_syms, prev_syms):
        subs_map[f(t)] = cur  # pyright: ignore
        subs_map[f(t + 1)] = fwd  # pyright: ignore
        subs_map[f(t - 1)] = prev  # pyright: ignore

    # Shocks stay in the residual: they reach the pencil as the innovation block
    compiled_numeric: tuple[Expr, ...] = tuple(o.xreplace(subs_map) for o in residuals)

    constraint_names, constraint_exprs = _compile_constraints(
        config, subs_map, set(idx.keys()), t
    )
    regimes = _compile_regimes(
        desugared,
        config.shocks,
        constraint_names,
        subs_map,
        cur_syms,
        fwd_syms,
        prev_syms,
        t,
    )

    observable_exprs = tuple(
        expr.xreplace(subs_map) for expr in desugared.observables.values()
    )
    # Flat row-major (n_obs, n_var) jacobian; printed to a native cfunc on
    # demand via CompiledModel.construct_measurement_jacobian_cfunc.
    measurement_jacobian_eqs: tuple[Expr, ...] = tuple(
        sp.diff(expr, cur_sym) for expr in observable_exprs for cur_sym in cur_syms
    )
    params_names = tuple(p.name for p in config.calibration.parameters)
    return CompiledModel(
        name=config.name,
        calib_params=params_names,
        layout=layout,
        var_names=var_order,
        idx=MappingProxyType(idx),
        objective_eqs=compiled_numeric,
        observable_names=tuple(v.name for v in config.observables),
        observable_eqs=observable_exprs,  # pyright: ignore
        measurement_jacobian_eqs=measurement_jacobian_eqs,
        constraint_names=constraint_names,
        constraint_exprs=constraint_exprs,
        regimes=MappingProxyType(regimes),
        config_token=config.token,
    )


def solve_model(
    config: ModelConfig,
    *,
    compiled: CompiledModel | None = None,
    parameters: dict[str, float] | None = None,
    ss_seed: list[float] | ndarray | dict[str, float] | None = None,
    order: int = 1,
) -> SolvedModel:
    """Solve ``compiled`` at a parameter point to the requested order.

    Parameters
    ----------
    config : ModelConfig
        Model specification to solve.
    compiled : CompiledModel | None
        Optional pre-compiled model carrying the residual, the incidence, and the
        calibration the solve reads.
    parameters : dict[str, float] | None
        Mapping of {name: value} to update parameters with. If
        None, the model's own calibration is used.
    ss_seed : list[float] | ndarray | dict[str, float] | None
        Initial guess of the steady state Newton solver. List or array in
        declaration order or mapping of {name: guess} for the model's
        declared variables.
    order : int
        1 for the Klein linear solve, 2 to add the second-order tensors and
        the sigma^2 risk correction. Those terms are zero for a symbolically
        linearized model, whose expansion is already exact at first order.

    Returns
    -------
    SolvedModel
        Policy is a ``FirstOrderSolution`` at ``order=1`` and a
        ``SecondOrderSolution`` at ``order=2``. A model with constraints
        returns a ``PiecewiseSolution`` instead, solved by OccBin at first
        order. ``A``/``B`` are the first-order transition in every case.

    Warns
    -----
    UserWarning
        The solution is indeterminate: there are more forward-looking
        variables than unstable eigenvalues, and the rule returned is one
        of many that solve the model.

    Raises
    ------
    ValueError
        - ``order`` is neither 1 nor 2.
        - ``ss_seed`` is the wrong length or names a variable the model does not have.
        - ``compiled`` is passed but is not compatible with ``config``.
    KeyError
        Unknown parameter names in ``parameters``.
    NotImplementedError
        ``order=2`` was asked of a model with constraints.
    SolveError
        The kernel rejected the model. The subclass and its message report
        how far the solve got and what it found.

    """
    if isinstance(config, PointLinearView):  # type: ignore[unreachable]
        raise TypeError(
            "A point-linearized model is not accepted as solver input. "
            "Linearizing a levels model about a point in parameter space changes the "
            "interpretation of any likelihood-related operation under varying parameters. "
            "Linearization already occurs numerically in the solver. You do not need to "
            "pre-linearize a specification symbolically. If you intended to use the model "
            "linearized at one constant point, pass `PointLinearView.to_config()."
        )

    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}.")

    if compiled is not None:
        if compiled.config_token != config.token:
            raise ValueError(
                "The compiled model and config passed to `solve_model` are not compatible. "
                "The config could have been mutated in a way that changed the model's structure after the compiled model was created. "
                "Confirm the arguments are the ones you intended to pass. If they are, recompile the model or pass the config by itself."
            )
    else:
        compiled = compile_model(config)

    piecewise = bool(compiled.constraint_names)
    if piecewise and order == 2:
        raise NotImplementedError(
            "A model with constraints is solved with OccBin piecewise linearization, "
            "order=2 is not supported in such a case. Use order=1 or remove the "
            "constraints from the model specification."
        )

    # New config + lock
    conf = config.with_parameters(parameters or {})
    conf.lock()

    params = conf.calibration.parameters
    seed = resolve_ss_seed(conf, compiled.layout, ss_seed=ss_seed, params=params)
    param_vec = np.fromiter(params.values(), dtype=float64, count=compiled.n_par)

    sol: SolvedModel[Any]

    if piecewise:
        sol = PiecewiseSolvedModel(
            config=conf,
            compiled=compiled,
            policy=piecewise_solve(compiled, param_vec, seed),
        )
    elif order == 2:
        sol = SecondOrderSolvedModel(
            config=conf,
            compiled=compiled,
            policy=sgu_solve(conf, compiled, param_vec, seed),
        )
    else:
        sol = FirstOrderSolvedModel(
            config=conf,
            compiled=compiled,
            policy=klein_solve(compiled, param_vec, seed),
        )

    if not sol.is_determinate:
        stab = sol.policy.stab
        warnings.warn(
            f"Blanchard-Kahn condition violated. {stab.name} ({stab.value}): {stab.message}",
            UserWarning,
            stacklevel=2,
        )
    return sol


def _compile_constraints(
    conf: ModelConfig,
    subs_map: dict[Any, Symbol],
    declared: set[str],
    t: Symbol,
) -> tuple[tuple[str, ...], tuple[Boolean, ...]]:
    """Regime conditions in declaration order, bind then relax per constraint.

    Declaration order is the regime bit order; the ``frozenset`` regime keys
    carry none. Conditions are contemporaneous, so no lag shift applies and
    they are never simplified.
    """
    constraints = conf.equations.constraints
    if not constraints:
        return (), ()

    names = tuple(constraints)
    exprs: list[Boolean] = []
    for name in names:
        constraint = constraints[name]
        for kind, cond in (("bind", constraint.bind), ("relax", constraint.relax)):
            for call in cond.atoms(AppliedUndef):
                info = _call_offset(call, declared, t)
                if info is not None and info[1] != 0:
                    raise ValueError(
                        f"{kind.capitalize()}ing condition of constraint '{name}' reads {info[0]} at "
                        f"t{info[1]:+d}. Regime conditions must be contemporaneous."
                    )
            exprs.append(cond.subs(subs_map))
    return names, tuple(exprs)


def _compile_regimes(
    desugared: DesugarResult,
    shocks: Sequence[Symbol],
    constraint_names: tuple[str, ...],
    subs_map: dict[Any, Symbol],
    cur_syms: list[Symbol],
    fwd_syms: list[Symbol],
    prev_syms: list[Symbol],
    t: Symbol,
) -> dict[int, RegimeBlock]:
    """Regime blocks keyed by the bitmask of their binding constraints.

    Each regime is the reference model with its replacements overlaid. Every
    replacement target is a declared model equation, so the merge keeps
    reference equation order and every regime pencil stays row-aligned with
    the reference.

    ``rows`` records which reference rows the regime replaced, so the native
    assembly can patch those rows into a copy of the reference pencil instead
    of sweeping the whole regime. ``jac_a``/``jac_b``/``jac_c`` are those rows'
    pencil blocks, flat row-major ``(len(rows), n_var)``, and ``jac_d`` is the
    shock block, ``(len(rows), n_exog)``; ``constants`` is those rows' residual
    at the expansion point, ``(len(rows),)``.
    """
    regimes = desugared.regimes
    if not regimes:
        return {}

    bit = {name: i for i, name in enumerate(constraint_names)}
    shock_syms = list(shocks)

    # At the expansion point every date holds the same vector and the
    # innovation is zero, so the pencil blocks fold onto the cur symbols.
    at_point: dict[Any, Any] = dict(zip(fwd_syms, cur_syms))
    at_point.update(zip(prev_syms, cur_syms))
    at_point.update({shock: 0.0 for shock in shocks})

    compiled: dict[int, RegimeBlock] = {}
    for key, replacements in regimes.items():
        label = ", ".join(sorted(key))
        # An unknown target would append instead of overwrite, silently
        # lengthening the regime and misplacing every row index after it.
        unknown = set(replacements) - set(desugared.model)
        if unknown:
            raise ValueError(
                f"Regime '{label}' replaces equations the model does not "
                f"declare: {sorted(unknown)}."
            )

        residuals: list[Expr] = []
        rows: list[int] = []
        merged = {**desugared.model, **replacements}
        for row, (name, eq) in enumerate(merged.items()):
            resid = eq.lhs - eq.rhs  # pyright: ignore
            residuals.append(resid.xreplace(subs_map))
            if name in replacements:
                rows.append(row)

        # Pencils are taken at the expansion point, where fwd and cur are the
        # same vector; differentiating before that fold is what keeps a and b
        # distinct. b carries klein_preproc's sign on the cur sweep.
        replaced = [residuals[row] for row in rows]
        jac_a = tuple(
            sp.diff(resid, sym).xreplace(at_point)
            for resid in replaced
            for sym in fwd_syms
        )
        jac_b = tuple(
            (-sp.diff(resid, sym)).xreplace(at_point)
            for resid in replaced
            for sym in cur_syms
        )
        jac_c = tuple(
            (-sp.diff(resid, sym)).xreplace(at_point)
            for resid in replaced
            for sym in prev_syms
        )
        jac_d = tuple(
            (-sp.diff(resid, sym)).xreplace(at_point)
            for resid in replaced
            for sym in shock_syms
        )
        # The regime's own residual there, unnegated where b is negated:
        # a E[y+] = b y - c. Zero on unreplaced rows, so only these are kept.
        constants = tuple(resid.subs(at_point) for resid in replaced)  # pyright: ignore

        compiled[sum(1 << bit[name] for name in key)] = RegimeBlock(
            residuals=tuple(residuals),
            rows=tuple(rows),
            jac_a=jac_a,  # pyright: ignore
            jac_b=jac_b,  # pyright: ignore
            jac_c=jac_c,  # pyright: ignore
            jac_d=jac_d,  # pyright: ignore
            constants=constants,
        )
    return compiled


def _coerce_variable_name(var: Any) -> str:
    if isinstance(var, str):
        return var
    if hasattr(var, "__name__"):
        return str(var.__name__)
    if hasattr(var, "name"):
        return str(var.name)
    if hasattr(var, "func") and hasattr(var.func, "__name__"):
        return str(var.func.__name__)
    return str(var)


def _lagged_names(desugared: DesugarResult) -> set[str]:
    """Declared variables occurring at ``t-1`` anywhere in the model.

    Scanned over the reference equations, every regime and the observables
    together, so a regime that lags something the reference does not still
    finds that variable in the state block. Read symbolically rather than off
    a Jacobian: a calibration that happened to zero a coefficient would
    otherwise resize the state vector between draws.
    """
    declared = {v.__name__ for v in desugared.variables}
    lagged: set[str] = set()
    for expr in _offset_expressions(
        desugared.model, desugared.regimes, desugared.observables
    ):
        for call in expr.atoms(AppliedUndef):
            info = _call_offset(call, declared, TIME)
            if info is not None and info[1] < 0:
                lagged.add(info[0])
    return lagged


def _infer_variable_layout(
    desugared: DesugarResult,
    shocks: Sequence[Symbol],
) -> VariableLayout:
    """Canonical layout of a desugared model.

    A variable is predetermined when it occurs at ``t-1``. Everything else
    is a control.

    Canonical order is states then controls: the pencil's own decision-rule
    ordering is derived natively from the incidence, but ``A``/``B``, ``x0``,
    ``P0`` and the filter all read the state block as a prefix.

    Shocks are not variables here. They reach the residual as innovations,
    so ``n_exog`` counts them and the state space takes its loading from the
    shock jacobian.
    """
    # ``conf`` is desugared, so its variable list carries the minted names
    # too. The canonical split spans all of them including aux variables.
    # while ``declared_names`` keeps only the user's own, filtered by name
    # rather than sliced off the tail.
    all_names = tuple(v.__name__ for v in desugared.variables)
    minted = frozenset(g.name for g in desugared.generated)
    declared_names = tuple(n for n in all_names if n not in minted)

    lagged = _lagged_names(desugared)
    state_names = tuple(name for name in all_names if name in lagged)
    control_names = tuple(name for name in all_names if name not in lagged)

    canonical_names = (*state_names, *control_names)
    idx = {name: i for i, name in enumerate(canonical_names)}

    # Shock columns follow declaration order. A shock reaches the residual
    # as a bare symbol and may drive any number of equations, so which
    # variables it moves is the shock jacobian's answer, not a declared one.
    shock_names = tuple(shock.name for shock in shocks)

    return VariableLayout(
        n_var=len(all_names),
        n_declared=len(declared_names),
        n_generated=len(desugared.generated),
        n_exog=len(shock_names),
        n_state=len(state_names),
        n_ctrl=len(control_names),
        declared_names=declared_names,
        generated_names=tuple(g.name for g in desugared.generated),
        canonical_names=canonical_names,
        state_names=state_names,
        control_names=control_names,
        idx=MappingProxyType(idx),
        aux_origin=MappingProxyType({g.name: g.origin for g in desugared.generated}),
        shock_names=shock_names,
        shock_idx=MappingProxyType({name: i for i, name in enumerate(shock_names)}),
    )
