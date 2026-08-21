import warnings
from dataclasses import replace

import sympy as sp
from sympy import Symbol, Function, Expr, Basic
from sympy.logic.boolalg import Boolean
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy import float64, asarray, ndarray
from numpy.typing import NDArray

import pandas as pd

from .config import ModelConfig, SymbolGetterDict
from .compiled_model import CompiledModel, VariableLayout, RegimeBlock
from sympy.core.function import AppliedUndef

from .desugar import (
    GeneratedVariable,
    _call_offset,
    _offset_expressions,
    desugar_model,
)
from .linearization import linearize_model
from .solved_model import (
    SolvedModel,
    FirstOrderSolvedModel,
    PiecewiseSolvedModel,
    SecondOrderSolvedModel,
)
from .solver_backend import (
    FirstOrderSolution,
    PiecewiseSolution,
    SecondOrderSolution,
    klein_solve,
    piecewise_solve,
    sgu_solve,
)

if TYPE_CHECKING:
    from ..estimation.estimator import Estimator
from ..kalman.config import KalmanConfig

NDF = NDArray[float64]
ND = NDArray


class DSGESolver:
    def __init__(
        self, model_config: ModelConfig, kalman_config: KalmanConfig | None = None
    ) -> None:
        self.model_config = model_config
        self.kalman_config = kalman_config
        self.t = sp.Symbol("t", integer=True)

    def compile(
        self,
        *,
        variable_order: Sequence[Function | str] | None = None,
        params_order: list[str] | None = None,
        linearize: bool = False,
    ) -> CompiledModel:

        conf = self.model_config
        if linearize and not conf.symbolically_linearized:
            conf = linearize_model(conf)

        # Convert model to {-1, 0, 1} offsets with auxiliary variables for lags and leads.
        desugared = desugar_model(conf)
        conf = desugared.config
        kalman_conf = self.kalman_config
        t = self.t
        ordered_variables = conf.variables.variables

        # Convert model to minimization problem
        residuals = [
            eq.lhs - eq.rhs for eq in conf.equations.model.values()  # pyright: ignore
        ]

        name_to_func = {v.__name__: v for v in ordered_variables}

        layout = self._infer_variable_layout(conf, desugared.generated, variable_order)
        var_order = list(layout.canonical_names)

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

        if not params_order:
            params_order = [p.name for p in conf.parameters]

        name_to_param = {p.name: p for p in conf.parameters}
        p_missing = [p for p in params_order if p not in name_to_param]
        if p_missing:
            raise ValueError(f"params_order contains unknown parameters: {p_missing}")
        params = [name_to_param[name] for name in params_order]

        # Shocks stay in the residual: they reach the pencil as the innovation block
        compiled_numeric: list[Expr] = [o.xreplace(subs_map) for o in residuals]

        constraint_names, constraint_exprs = self._compile_constraints(
            conf, subs_map, set(idx.keys()), t
        )
        regimes = self._compile_regimes(
            conf,
            constraint_names,
            subs_map,
            cur_syms,
            fwd_syms,
            prev_syms,
            t,
        )

        observable_exprs = [
            expr.xreplace(subs_map) for expr in conf.equations.observable.values()
        ]
        # Flat row-major (n_obs, n_var) jacobian; printed to a native cfunc on
        # demand via CompiledModel.construct_observable_jacobian_cfunc.
        observable_jacobian_eqs: list[Expr] = [
            sp.diff(expr, cur_sym) for expr in observable_exprs for cur_sym in cur_syms
        ]

        return CompiledModel(
            config=conf,
            kalman=kalman_conf,
            cur_syms=cur_syms,
            layout=layout,
            var_names=var_order,
            calib_params=params,
            idx=idx,
            objective_eqs=compiled_numeric,
            observable_names=[v.name for v in conf.observables],
            observable_eqs=observable_exprs,
            observable_jacobian_eqs=observable_jacobian_eqs,
            constraint_names=constraint_names,
            constraint_exprs=constraint_exprs,
            regimes=regimes,
        )

    def _compile_constraints(
        self,
        conf: ModelConfig,
        subs_map: dict[Any, Symbol],
        declared: set[str],
        t: Symbol,
    ) -> tuple[tuple[str, ...], list[Boolean]]:
        """Regime conditions in declaration order, bind then relax per constraint.

        Declaration order is the regime bit order; the ``frozenset`` regime keys
        carry none. Conditions are contemporaneous, so no lag shift applies and
        they are never simplified.
        """
        constraints = conf.equations.constraint
        if not constraints:
            return (), []

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
        return names, exprs

    def _compile_regimes(
        self,
        conf: ModelConfig,
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
        regimes = conf.equations.regime
        if not regimes:
            return {}

        bit = {name: i for i, name in enumerate(constraint_names)}
        shock_syms = list(conf.shocks)
        # At the expansion point every date holds the same vector and the
        # innovation is zero, so the pencil blocks fold onto the cur symbols.
        at_point: dict[Any, Any] = dict(zip(fwd_syms, cur_syms))
        at_point.update(zip(prev_syms, cur_syms))
        at_point.update({shock: 0.0 for shock in conf.shocks})

        compiled: dict[int, RegimeBlock] = {}
        for key, replacements in regimes.items():
            label = ", ".join(sorted(key))
            # An unknown target would append instead of overwrite, silently
            # lengthening the regime and misplacing every row index after it.
            unknown = set(replacements) - set(conf.equations.model)
            if unknown:
                raise ValueError(
                    f"Regime '{label}' replaces equations the model does not "
                    f"declare: {sorted(unknown)}."
                )

            residuals: list[Expr] = []
            rows: list[int] = []
            merged = {**conf.equations.model, **replacements}
            for row, (name, eq) in enumerate(merged.items()):
                resid = eq.lhs - eq.rhs  # pyright: ignore
                residuals.append(resid.xreplace(subs_map))
                if name in replacements:
                    rows.append(row)

            # Pencils are taken at the expansion point, where fwd and cur are the
            # same vector; differentiating before that fold is what keeps a and b
            # distinct. b carries klein_preproc's sign on the cur sweep.
            replaced = [residuals[row] for row in rows]
            jac_a = [
                sp.diff(resid, sym).subs(at_point)  # pyright: ignore
                for resid in replaced
                for sym in fwd_syms
            ]
            jac_b = [
                (-sp.diff(resid, sym)).subs(at_point)  # pyright: ignore
                for resid in replaced
                for sym in cur_syms
            ]
            jac_c = [
                (-sp.diff(resid, sym)).subs(at_point)  # pyright: ignore
                for resid in replaced
                for sym in prev_syms
            ]
            jac_d = [
                (-sp.diff(resid, sym)).subs(at_point)  # pyright: ignore
                for resid in replaced
                for sym in shock_syms
            ]
            # The regime's own residual there, unnegated where b is negated:
            # a E[y+] = b y - c. Zero on unreplaced rows, so only these are kept.
            constants = [resid.subs(at_point) for resid in replaced]  # pyright: ignore

            compiled[sum(1 << bit[name] for name in key)] = RegimeBlock(
                residuals=residuals,
                rows=rows,
                jac_a=jac_a,
                jac_b=jac_b,
                jac_c=jac_c,
                jac_d=jac_d,
                constants=constants,
            )
        return compiled

    @staticmethod
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

    @staticmethod
    def _lagged_names(conf: ModelConfig) -> set[str]:
        """Declared variables occurring at ``t-1`` anywhere in the model.

        Scanned over the reference equations, every regime and the observables
        together, so a regime that lags something the reference does not still
        finds that variable in the state block. Read symbolically rather than off
        a Jacobian: a calibration that happened to zero a coefficient would
        otherwise resize the state vector between draws.
        """
        t = Symbol("t", integer=True)
        declared = {v.__name__ for v in conf.variables.variables}
        lagged: set[str] = set()
        for expr in _offset_expressions(conf):
            for call in expr.atoms(AppliedUndef):
                info = _call_offset(call, declared, t)
                if info is not None and info[1] < 0:
                    lagged.add(info[0])
        return lagged

    def _infer_variable_layout(
        self,
        conf: ModelConfig,
        generated: tuple[GeneratedVariable, ...],
        variable_order: Sequence[Function | str] | None = None,
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
        all_names = tuple(v.__name__ for v in conf.variables.variables)
        minted = frozenset(g.name for g in generated)
        declared_names = tuple(n for n in all_names if n not in minted)

        lagged = self._lagged_names(conf)
        state_names = tuple(name for name in all_names if name in lagged)
        control_names = tuple(name for name in all_names if name not in lagged)

        if variable_order is not None:
            state_names, control_names = self._resolve_variable_order(
                variable_order,
                state_names,
                control_names,
                frozenset(g.name for g in generated),
            )

        canonical_names = (*state_names, *control_names)
        idx = {name: i for i, name in enumerate(canonical_names)}

        # Shock columns follow declaration order. A shock reaches the residual
        # as a bare symbol and may drive any number of equations, so which
        # variables it moves is the shock jacobian's answer, not a declared one.
        shock_names = tuple(shock.name for shock in conf.shocks)

        return VariableLayout(
            n_var=len(all_names),
            n_declared=len(declared_names),
            n_generated=len(generated),
            n_exog=len(shock_names),
            n_state=len(state_names),
            n_ctrl=len(control_names),
            declared_names=declared_names,
            generated_names=tuple(g.name for g in generated),
            canonical_names=canonical_names,
            state_names=state_names,
            control_names=control_names,
            idx=idx,
            aux_origin={g.name: g.origin for g in generated},
            shock_names=shock_names,
            shock_idx={name: i for i, name in enumerate(shock_names)},
        )

    @staticmethod
    def _resolve_p0(
        P0: ND, layout: VariableLayout, generated: tuple[GeneratedVariable, ...]
    ) -> ND:
        """Widen the parse-time ``P0`` over the generated variables, then permute.

        ``P0`` is assembled in declared variable order at parse time, before the
        compiler mints anything. A lag aux is its origin some periods back, so it
        takes the origin's variance. The permutation into canonical order is a
        lossless reindex.
        """
        n_declared = layout.n_declared
        if P0.shape != (n_declared, n_declared):
            raise ValueError(
                f"P0 has shape {P0.shape}, expected ({n_declared}, {n_declared}) "
                f"for the model's declared variables."
            )

        full = (*layout.declared_names, *layout.generated_names)
        declared_idx = {name: i for i, name in enumerate(full)}
        wide = np.eye(len(full), dtype=float64)
        wide[:n_declared, :n_declared] = P0
        for g in generated:
            origin = declared_idx[g.origin]
            wide[declared_idx[g.name], declared_idx[g.name]] = P0[origin, origin]

        perm = [declared_idx[name] for name in layout.canonical_names]
        return wide[np.ix_(perm, perm)]

    @staticmethod
    def _resolve_variable_order(
        variable_order: Sequence[Function | str],
        state_names: tuple[str, ...],
        control_names: tuple[str, ...],
        generated: frozenset[str] = frozenset(),
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Coerce and validate an explicit order, and split it back into blocks.

        An order names every variable the model declares, each exactly once, and
        leads with the states. Which variables those are follows from the model,
        so an order chooses positions within a block rather than membership of
        one, and it is free within each. Compiler-minted lags take the end of the
        state block, where the default order leaves them.
        """
        var_order = tuple(DSGESolver._coerce_variable_name(v) for v in variable_order)
        if len(set(var_order)) != len(var_order):
            raise ValueError("variable_order contains duplicate variables.")

        minted = tuple(name for name in state_names if name in generated)
        declared_states = tuple(name for name in state_names if name not in generated)
        declared = set(declared_states) | set(control_names)
        if set(var_order) != declared:
            unknown = sorted(set(var_order) - declared)
            missing = sorted(declared - set(var_order))
            raise ValueError(
                "variable_order must name every declared model variable exactly "
                f"once. Unknown: {unknown}. Missing: {missing}. Compiler-generated "
                "lags are placed with the states and must not appear."
            )

        n_state = len(declared_states)
        if set(var_order[:n_state]) != set(declared_states):
            raise ValueError(
                "variable_order must lead with the model's states. Expected its "
                f"first {n_state} entries to be {list(declared_states)} in any "
                f"order, got {list(var_order[:n_state])}."
            )
        return (*var_order[:n_state], *minted), var_order[n_state:]

    def solve(
        self,
        compiled: CompiledModel,
        *,
        parameters: dict[str, float] | None = None,
        ss_seed: list[float] | ndarray | dict[str, float] | None = None,
        order: int = 1,
        raise_on_bk_violation: bool = True,
    ) -> SolvedModel:
        """Solve the model to first (``order=1``) or second (``order=2``) order.

        ``order=1`` is the Klein linear solve (policy is a
        ``FirstOrderSolution``). ``order=2`` additionally computes the
        second-order tensors and the sigma^2 risk correction (policy is a
        ``SecondOrderSolution``); it requires the native extension and a
        nonlinear steady state (see ``_solve_second_order``). The state-space
        ``A``/``B`` are the first-order transition in both cases.

        When ``raise_on_bk_violation`` is ``False`` a Klein stability/uniqueness
        failure warns instead of raising, so batch callers (e.g. an estimation
        search) can tally the failure and continue.
        """
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}.")

        piecewise = bool(compiled.constraint_names)
        if piecewise and order == 2:
            raise NotImplementedError(
                "A model with constraints is solved with OccBin piecewise linearization, "
                "order=2 is not supported in such a case. Use order=1 or remove the "
                "constraints from the model specification."
            )

        conf = compiled.config
        seed = self._resolve_ss_seed(ss_seed, compiled)

        if parameters is None:
            param_vec = np.array(
                [conf.calibration.parameters[p.name] for p in compiled.calib_params],
                dtype=float64,
            )
        else:
            param_vec = np.array(
                [parameters[p.name] for p in compiled.calib_params], dtype=float64
            )

        if piecewise:
            return self._solve_piecewise(
                compiled, param_vec, seed, raise_on_bk_violation
            )
        if order == 2:
            return self._solve_second_order(
                compiled, param_vec, seed, raise_on_bk_violation
            )
        return self._solve_first_order(compiled, param_vec, seed, raise_on_bk_violation)

    @staticmethod
    def _resolve_ss_seed(
        ss_seed: list[float | float64] | ndarray | dict[str, float | float64] | None,
        compiled: CompiledModel,
    ) -> NDF:
        """Newton seed for the steady state, in canonical variable order.

        Resolved per variable: an explicit ``ss_seed`` if it names the variable,
        else the config's ``ss_seed`` expression, else zero. Newton resolves
        ``F(ss, ss) = 0`` from here, so a gap model seeds at 0 and converges in
        one step while a level model that declares its seed seeds itself.

        An explicit seed names the model's own variables, as a mapping or as a
        dense sequence in declaration order. Compiler-minted variables are not
        addressable: at a steady state every date coincides, so an aux equals
        its origin identically and has no seed of its own to give.
        """
        layout = compiled.layout
        declared = layout.declared_names

        seed = DSGESolver._configured_ss_seed(compiled)
        if ss_seed is None:
            return seed

        if isinstance(ss_seed, Mapping):
            by_name = {str(name): float(value) for name, value in ss_seed.items()}
            minted = sorted(set(by_name) & set(layout.generated_names))
            if minted:
                raise ValueError(
                    f"ss_seed names compiler-minted variable(s) {minted}, which "
                    f"take their origin's seed. Seed the origin instead."
                )
            unknown = sorted(set(by_name) - set(declared))
            if unknown:
                raise ValueError(
                    f"ss_seed names variables the model does not have: {unknown}. "
                    f"Model variables: {list(declared)}."
                )
        else:
            values = asarray(ss_seed, dtype=float64)
            if values.shape != (len(declared),):
                raise ValueError(
                    f"ss_seed has shape {values.shape}, expected "
                    f"({len(declared)},) for the model's variables in declaration "
                    f"order: {list(declared)}. Name the variables in a dict to "
                    f"seed only some of them."
                )
            by_name = dict(zip(declared, (float(v) for v in values)))

        # An aux follows its origin here too, so seeding the origin seeds the
        # whole chain off it.
        for i, name in enumerate(compiled.var_names):
            origin = layout.aux_origin.get(name, name)
            if origin in by_name:
                seed[i] = by_name[origin]
        return seed

    @staticmethod
    def _configured_ss_seed(compiled: CompiledModel) -> NDF:
        """The model's own symbolic ``ss_seed``, evaluated in canonical order.

        A variable that declares none seeds at zero.
        """
        conf = compiled.config
        name_to_func = {v.__name__: v for v in conf.variables.variables}
        params = conf.calibration.parameters
        ss = np.zeros(compiled.n_var, dtype=float64)
        for i, name in enumerate(compiled.var_names):
            expr = conf.variables.ss_seed[name_to_func[name]]
            if expr is None:
                continue
            val = sp.simplify(sp.sympify(expr).subs(params))
            try:
                ss[i] = float(val)
            except TypeError as exc:
                raise ValueError(
                    f"ss_seed for '{name}' did not evaluate to a number: {val}"
                ) from exc
        return ss

    @staticmethod
    def _raise_or_warn_stability_error(stab: int, *, should_raise: bool = True) -> None:
        """Raise or warn on a Klein stability/uniqueness violation."""
        if stab == 0:
            return
        msg = f"Klein stability/uniqueness condition violated (stab={stab})."
        if should_raise:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    def _solve_first_order(
        self,
        compiled: CompiledModel,
        param_vec: NDF,
        seed: NDF,
        raise_on_bk_violation: bool = True,
    ) -> SolvedModel[FirstOrderSolution]:
        """First-order (Klein) solve."""
        sol = klein_solve(
            compiled.construct_objective_cfunc(),
            param_vec,
            seed,
            compiled._incidence,
            compiled.n_state,
            n_exog=compiled.n_exog,
        )
        self._raise_or_warn_stability_error(
            sol.stab, should_raise=raise_on_bk_violation
        )
        return FirstOrderSolvedModel(compiled=compiled, policy=sol)

    def _solve_second_order(
        self,
        compiled: CompiledModel,
        param_vec: NDF,
        seed: NDF,
        raise_on_bk_violation: bool = True,
    ) -> SolvedModel[SecondOrderSolution]:
        """Second-order solve. Runs the Klein first order (which Newton-resolves
        the steady state from ``seed``), sweeps the bicomplex Hessian at that
        steady state, and assembles the quadratic blocks over the states, the
        shocks and their cross plus the risk correction into a
        :class:`SecondOrderSolution`. Requires the native extension."""

        pert = sgu_solve(
            compiled.construct_objective_cfunc(),
            compiled.construct_objective_cfunc_bicomplex(),
            param_vec,
            seed,
            self._build_Q(compiled),
            compiled._incidence,
            compiled.n_state,
            n_exog=compiled.n_exog,
        )
        self._raise_or_warn_stability_error(
            pert.stab, should_raise=raise_on_bk_violation
        )
        # p/f are the first-order solution unchanged, so its state space stands.
        return SecondOrderSolvedModel(compiled=compiled, policy=pert)

    def _solve_piecewise(
        self,
        compiled: CompiledModel,
        param_vec: NDF,
        seed: NDF,
        raise_on_bk_violation: bool = True,
    ) -> SolvedModel[PiecewiseSolution]:
        """Piecewise-linear (OccBin) solve: the reference regime and every pencil.

        A draw fixes one pencil per binding combination, all linearized at the
        same reference steady state, so the whole table is built here and the
        per-date guess-and-verify happens in ``sim``.
        """
        pencil = compiled.construct_regime_pencil_func()
        if pencil is None:  # pragma: no cover - solve() gates on the same thing
            raise ValueError("Piecewise solve needs a model with constraints.")

        n_constraint = len(compiled.constraint_names)
        # Slot 0 is the reference regime: no cfunc, no replaced rows, so the
        # kernel fills it with the pencil the reference solve linearized at.
        addrs = [0] + [pencil.address(m) for m in range(1, 1 << n_constraint)]
        rows = [np.empty(0, dtype=np.int64)] + [
            pencil.rows[m] for m in range(1, 1 << n_constraint)
        ]

        sol = piecewise_solve(
            compiled.construct_objective_cfunc(),
            addrs,
            rows,
            param_vec,
            seed,
            compiled._incidence,
            compiled.n_state,
            n_constraint,
            n_exog=compiled.n_exog,
        )

        # Only the reference regime has to be determinate.
        self._raise_or_warn_stability_error(
            sol.stab, should_raise=raise_on_bk_violation
        )
        return PiecewiseSolvedModel(compiled=compiled, policy=sol)

    @staticmethod
    def _build_Q(compiled: CompiledModel) -> NDF:
        """The shock covariance, ``(n_exog, n_exog)``.

        Stds scale it and correlations fill its off-diagonals. It crosses the
        boundary whole rather than as a factor: the risk correction integrates
        against the covariance itself, and the filters read it as ``Q``.
        """
        conf = compiled.config
        n_exog = compiled.n_exog
        eta = np.zeros((n_exog, n_exog), dtype=float64)
        if n_exog == 0:
            return eta

        params = conf.calibration.parameters
        shock_std = conf.calibration.shock_std
        shock_corr = conf.calibration.shock_corr
        innovations = list(conf.shocks)

        stds = np.empty(n_exog, dtype=float64)
        for i, innov in enumerate(innovations):
            sig_sym = shock_std.get(innov)
            stds[i] = (
                float(params[sig_sym]) if sig_sym in params else 1.0  # pyright: ignore
            )
        corr = np.eye(n_exog, dtype=float64)
        for i in range(n_exog):
            for j in range(i + 1, n_exog):
                c_sym = shock_corr[innovations[i], innovations[j]]
                cij = (
                    float(params[c_sym])
                    if (c_sym is not None and c_sym in params)
                    else 0.0
                )
                corr[i, j] = corr[j, i] = cij

        return corr * np.outer(stds, stds)

    def _estimator(
        self,
        *,
        compiled: CompiledModel,
        y: NDArray | pd.DataFrame,
        observables: list[str] | None = None,
        filter_mode: str = "linear",
        estimated_params: list[str] | None = None,
        priors: Mapping[str, Any] | None = None,
        ss_seed: list[float] | NDArray | dict[str, float] | None = None,
        x0: NDArray | None = None,
        P0: NDArray | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool = True,
        R: NDArray | None = None,
    ) -> "Estimator":
        # Lazy import prevents a solver->estimation->solver import cycle.
        from ..estimation import Estimator

        return Estimator(
            solver=self,
            compiled=compiled,
            y=y,
            observables=observables,
            filter_mode=filter_mode,
            estimated_params=estimated_params,
            priors=priors,
            ss_seed=(
                asarray(ss_seed, dtype=float64)
                if isinstance(ss_seed, list)
                else ss_seed
            ),
            x0=x0,
            P0=P0,
            jitter=jitter,
            symmetrize=symmetrize,
            R=R,
        )

    @staticmethod
    def _sync_calibration_with_params(
        compiled: CompiledModel, params: Mapping[str, float64]
    ) -> None:
        calib = compiled.config.calibration.parameters
        for key in list(calib.keys()):
            name = key if isinstance(key, str) else getattr(key, "name", None)
            if name is not None and name in params:
                calib[key] = float64(params[name])

    def estimate(
        self,
        *,
        compiled: CompiledModel,
        y: NDArray | pd.DataFrame,
        method: str = "mle",
        theta0: NDArray | Mapping[str, float] | None = None,
        observables: list[str] | None = None,
        filter_mode: str = "linear",
        estimated_params: list[str] | None = None,
        priors: Mapping[str, Any] | None = None,
        ss_seed: list[float] | NDArray | dict[str, float] | None = None,
        x0: NDArray | None = None,
        P0: NDArray | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool = True,
        R: NDArray | None = None,
        **method_kwargs: Any,
    ) -> Any:
        est = self._estimator(
            compiled=compiled,
            y=y,
            observables=observables,
            filter_mode=filter_mode,
            estimated_params=estimated_params,
            priors=priors,
            ss_seed=(
                asarray(ss_seed, dtype=float64)
                if isinstance(ss_seed, list)
                else ss_seed
            ),
            x0=x0,
            P0=P0,
            jitter=jitter,
            symmetrize=symmetrize,
            R=R,
        )

        init = est.resolve_theta0(theta0)

        method_norm = method.lower()
        if method_norm == "mle":
            return est.mle(theta0=init, **method_kwargs)
        if method_norm == "map":
            return est.map(theta0=init, **method_kwargs)
        if method_norm == "mcmc":
            return est.mcmc(theta0=init, **method_kwargs)
        raise ValueError("method must be one of {'mle', 'map', 'mcmc'}.")

    def estimate_and_solve(
        self,
        *,
        compiled: CompiledModel,
        y: NDArray | pd.DataFrame,
        method: str = "mle",
        theta0: NDArray | Mapping[str, float] | None = None,
        posterior_point: str = "mean",
        observables: list[str] | None = None,
        filter_mode: str = "linear",
        estimated_params: list[str] | None = None,
        priors: Mapping[str, Any] | None = None,
        ss_seed: list[float] | NDArray | dict[str, float] | None = None,
        x0: NDArray | None = None,
        P0: NDArray | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool = True,
        R: NDArray | None = None,
        **method_kwargs: Any,
    ) -> tuple[Any, SolvedModel]:

        ss_seed = (
            np.asarray(ss_seed, dtype=float64) if isinstance(ss_seed, list) else ss_seed
        )
        est = self._estimator(
            compiled=compiled,
            y=y,
            observables=observables,
            filter_mode=filter_mode,
            estimated_params=estimated_params,
            priors=priors,
            ss_seed=ss_seed,
            x0=x0,
            P0=P0,
            jitter=jitter,
            symmetrize=symmetrize,
            R=R,
        )

        init = est.resolve_theta0(theta0)

        method_norm = method.lower()
        result: Any
        if method_norm == "mle":
            result = est.mle(theta0=init, **method_kwargs)
            solve_params = result.theta
        elif method_norm == "map":
            result = est.map(theta0=init, **method_kwargs)
            solve_params = result.theta
        elif method_norm == "mcmc":
            result = est.mcmc(theta0=init, **method_kwargs)
            if posterior_point == "mean":
                theta_star = asarray(result.samples.mean(axis=0), dtype=float64)
            elif posterior_point == "last":
                theta_star = asarray(result.samples[-1], dtype=float64)
            elif (posterior_point == "map") or (posterior_point == "mode"):
                idx = int(np.argmax(result.logpost_trace))
                theta_star = asarray(result.samples[idx], dtype=float64)
            else:
                raise ValueError(
                    "posterior_point must be one of {'mean', 'last', 'map', 'mode'}."
                )
            # The round trip projects a posterior summary back onto the
            # transforms, which matters for a mean that lands off a correlation
            # block's manifold; the estimated names are read back off it.
            projected = est.theta_to_params(est.params_to_theta(theta_star))
            solve_params = {name: projected[name] for name in est.param_names}
        else:
            raise ValueError("method must be one of {'mle', 'map', 'mcmc'}.")

        # Sync writes the estimated values into the calibration, which is where
        # `solve` reads its full parameter vector from; passing them again would
        # only re-supply what was just installed.
        self._sync_calibration_with_params(compiled, solve_params)
        solved = self.solve(
            compiled=compiled,
            ss_seed=ss_seed,
        )
        return result, solved
