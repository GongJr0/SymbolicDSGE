import warnings
from dataclasses import replace

import sympy as sp
from sympy import Symbol, Function, Expr, Basic
from sympy.logic.boolalg import Boolean
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy import float64, complex128, asarray, ndarray
from numpy.typing import NDArray

import pandas as pd

from .config import ModelConfig, SymbolGetterDict
from .compiled_model import CompiledModel, VariableLayout, RegimeBlock
from .desugar import GeneratedKind, GeneratedVariable, desugar_model
from .linearization import linearize_model
from .solved_model import SolvedModel
from .solver_backend import KleinSolution, PerturbationSolution, klein_solve
from .._ckernels.core import (
    assemble_state_space,
    second_order,
    second_order_risk,
    bicomplex_hessian,
    klein_preprocess,
)

if TYPE_CHECKING:
    from ..estimation.estimator import Estimator
from ..kalman.config import KalmanConfig

NDF = NDArray[float64]
ND = NDArray


class DSGESolver:
    def __init__(self, model_config: ModelConfig, kalman_config: KalmanConfig) -> None:
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
        # Lags and shocks become variables of their own here, so everything below
        # this line sees a two-date model whose states are exactly the variables
        # the compiler minted.
        desugared = desugar_model(conf)
        conf = desugared.config
        kalman_conf = self.kalman_config
        t = self.t
        ordered_variables = conf.variables.variables

        # Convert model to minimization problem
        residuals = [
            sp.simplify(eq.lhs - eq.rhs)  # pyright: ignore
            for eq in conf.equations.model.values()
        ]

        name_to_func = {v.__name__: v for v in ordered_variables}

        layout = self._infer_variable_layout(conf, desugared.generated, variable_order)
        var_order = list(layout.canonical_names)

        var_funcs = [name_to_func[name] for name in var_order]
        idx = dict(layout.idx)

        if kalman_conf is not None:
            kalman_conf = replace(
                kalman_conf,
                P0=self._resolve_p0(kalman_conf.P0, layout, desugared.generated),
            )

        for i, residual in enumerate(residuals):
            bad = self._bad_time_offsets(residual, var_funcs, t)
            if bad:
                raise ValueError(
                    f"Equation {i} has bad time offsets {bad}. Only offsets of 0 and 1 are allowed."
                )

        # Substitutions
        cur_syms = [Symbol(f"cur_{n}") for n in var_order]
        fwd_syms = [Symbol(f"fwd_{n}") for n in var_order]

        subs_map = {}
        for _, f, cur, fwd in zip(var_order, var_funcs, cur_syms, fwd_syms):

            subs_map[f(t)] = cur  # pyright: ignore
            subs_map[f(t + 1)] = fwd  # pyright: ignore

        if not params_order:
            params_order = [p.name for p in conf.parameters]

        name_to_param = {p.name: p for p in conf.parameters}
        p_missing = [p for p in params_order if p not in name_to_param]
        if p_missing:
            raise ValueError(f"params_order contains unknown parameters: {p_missing}")
        params = [name_to_param[name] for name in params_order]

        compiled: list[Expr] = [sp.simplify(o.subs(subs_map)) for o in residuals]
        shock_zero_subs = {shock: 0.0 for shock in conf.shock_map.keys()}
        compiled_numeric: list[Expr] = [
            sp.simplify(expr.subs(shock_zero_subs))  # pyright: ignore
            for expr in compiled
        ]

        constraint_names, constraint_exprs = self._compile_constraints(
            conf, subs_map, var_funcs, t
        )
        regimes = self._compile_regimes(
            conf, constraint_names, subs_map, cur_syms, fwd_syms, var_funcs, t
        )

        observable_exprs = [
            sp.simplify(expr.subs(subs_map))
            for expr in conf.equations.observable.values()
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
            n_state=layout.n_state,
            n_exog=layout.n_exog,
            constraint_names=constraint_names,
            constraint_exprs=constraint_exprs,
            regimes=regimes,
        )

    def _compile_constraints(
        self,
        conf: ModelConfig,
        subs_map: dict[Any, Symbol],
        var_funcs: list[Function],
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
                bad = self._bad_time_offsets(cond, var_funcs, t, allowed={0})
                if bad:
                    raise ValueError(
                        f"Condition '{kind}' of constraint '{name}' has bad time "
                        f"offsets {sorted(bad)}. Constraint conditions may only "
                        f"reference contemporaneous variables."
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
        var_funcs: list[Function],
        t: Symbol,
    ) -> dict[int, RegimeBlock]:
        """Regime blocks keyed by the bitmask of their binding constraints.

        Each regime is the reference model with its replacements overlaid. Every
        replacement target is a declared model equation, so the merge keeps
        reference equation order and every regime pencil stays row-aligned with
        the reference. Shocks are zeroed as they are on the reference residual.

        ``rows`` records which reference rows the regime replaced, so the native
        assembly can patch those rows into a copy of the reference pencil instead
        of sweeping the whole regime. ``jac_a``/``jac_b`` are those rows' pencil
        blocks, flat row-major ``(len(rows), n_var)``.
        """
        regimes = conf.equations.regime
        if not regimes:
            return {}

        bit = {name: i for i, name in enumerate(constraint_names)}
        shock_zero_subs = {shock: 0.0 for shock in conf.shock_map}
        fwd_to_cur = dict(zip(fwd_syms, cur_syms))

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
                resid = sp.simplify(eq.lhs - eq.rhs)  # pyright: ignore
                bad = self._bad_time_offsets(resid, var_funcs, t)
                if bad:
                    raise ValueError(
                        f"Equation '{name}' in regime '{label}' has bad time "
                        f"offsets {sorted(bad)}. Only offsets of 0 and 1 are allowed."
                    )
                residuals.append(
                    sp.simplify(
                        resid.subs(subs_map).subs(shock_zero_subs)  # pyright: ignore
                    )
                )
                if name in replacements:
                    rows.append(row)

            # Pencils are taken at the expansion point, where fwd and cur are the
            # same vector; differentiating before that fold is what keeps the two
            # blocks distinct. b carries klein_preproc's sign on the cur sweep.
            replaced = [residuals[row] for row in rows]
            jac_a = [
                sp.diff(resid, sym).subs(fwd_to_cur)  # pyright: ignore
                for resid in replaced
                for sym in fwd_syms
            ]
            jac_b = [
                (-sp.diff(resid, sym)).subs(fwd_to_cur)  # pyright: ignore
                for resid in replaced
                for sym in cur_syms
            ]

            compiled[sum(1 << bit[name] for name in key)] = RegimeBlock(
                residuals=residuals, rows=rows, jac_a=jac_a, jac_b=jac_b
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

    def _infer_variable_layout(
        self,
        conf: ModelConfig,
        generated: tuple[GeneratedVariable, ...],
        variable_order: Sequence[Function | str] | None = None,
    ) -> VariableLayout:
        """Canonical layout of a desugared model.

        Desugaring lifts every lag and every shock into a variable of its own, and
        after it the predetermined variables are exactly those: a lag aux is
        pinned at t-1 by its defining equation and a shock state carries the
        innovation. Everything the model itself declares reads as a static
        function of them, so it lands in the control block.

        Canonical order is shock states, lag states, controls, which keeps the
        shocked states leading as the impact block and ``eta`` require.
        """
        declared_names = tuple(v.__name__ for v in conf.variables.variables)
        generated_names = {g.name for g in generated}

        exo_state_names = tuple(
            g.name for g in generated if g.kind is GeneratedKind.SHOCK
        )
        endo_state_names = tuple(
            g.name for g in generated if g.kind is GeneratedKind.LAG
        )
        model_names = tuple(
            name for name in declared_names if name not in generated_names
        )

        control_names = (
            model_names
            if variable_order is None
            else self._resolve_variable_order(variable_order, model_names)
        )
        canonical_names = (*exo_state_names, *endo_state_names, *control_names)
        idx = {name: i for i, name in enumerate(canonical_names)}

        # Shock columns follow shock_map order, which is the order the shock
        # states were minted in. Only the keys are read: the lift keys on the
        # shock symbol, so a shock enters any number of equations and the target
        # a shock declares names nothing the compiler needs.
        shock_names = tuple(shock.name for shock in conf.shock_map)

        return VariableLayout(
            declared_names=declared_names,
            canonical_names=canonical_names,
            exo_state_names=exo_state_names,
            endo_state_names=endo_state_names,
            control_names=control_names,
            n_exog=len(exo_state_names),
            n_state=len(exo_state_names) + len(endo_state_names),
            idx=idx,
            generated={g.name: idx[g.name] for g in generated},
            lag_origin={
                g.name: g.origin for g in generated if g.kind is GeneratedKind.LAG
            },
            shock_names=shock_names,
            shock_idx={name: i for i, name in enumerate(shock_names)},
        )

    @staticmethod
    def _resolve_p0(
        P0: ND, layout: VariableLayout, generated: tuple[GeneratedVariable, ...]
    ) -> ND:
        """Widen the parse-time ``P0`` over the generated variables, then permute.

        ``P0`` is assembled in declared variable order at parse time, before the
        compiler mints anything. A lag aux is its origin one period back, so it
        takes the origin's variance; a shock state takes the 1.0 an unspecified
        entry would have carried. The permutation into canonical order is a
        lossless reindex.
        """
        n_declared = len(layout.declared_names) - len(generated)
        if P0.shape != (n_declared, n_declared):
            raise ValueError(
                f"P0 has shape {P0.shape}, expected ({n_declared}, {n_declared}) "
                f"for the model's declared variables."
            )

        declared_idx = {name: i for i, name in enumerate(layout.declared_names)}
        wide = np.eye(len(layout.declared_names), dtype=float64)
        wide[:n_declared, :n_declared] = P0
        for g in generated:
            if g.kind is GeneratedKind.LAG:
                origin = declared_idx[g.origin]
                wide[declared_idx[g.name], declared_idx[g.name]] = P0[origin, origin]

        perm = [declared_idx[name] for name in layout.canonical_names]
        return wide[np.ix_(perm, perm)]

    @staticmethod
    def _resolve_variable_order(
        variable_order: Sequence[Function | str],
        model_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Coerce and validate an explicit order for the control block.

        The states are the compiler's own variables and always lead the canonical
        order, so an explicit order names the variables the model declares, each
        exactly once.
        """
        var_order = tuple(DSGESolver._coerce_variable_name(v) for v in variable_order)
        if len(set(var_order)) != len(var_order):
            raise ValueError("variable_order contains duplicate variables.")

        declared = set(model_names)
        if set(var_order) != declared:
            unknown = sorted(set(var_order) - declared)
            missing = sorted(declared - set(var_order))
            raise ValueError(
                "variable_order must name every declared model variable exactly "
                f"once. Unknown: {unknown}. Missing: {missing}. Compiler-generated "
                "states lead the canonical order and must not appear."
            )
        return var_order

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

        ``order=1`` is the Klein linear solve (policy is a ``KleinSolution``).
        ``order=2`` additionally computes the SGU second-order tensors and the
        sigma^2 risk correction (policy is a ``PerturbationSolution``); it requires
        the native extension and a nonlinear steady state (see ``_solve_second_order``).
        The state-space ``A``/``B`` are the first-order transition in both cases.

        When ``raise_on_bk_violation`` is ``False`` a Klein stability/uniqueness
        failure warns instead of raising, so batch callers (e.g. an estimation
        search) can tally the failure and continue.
        """
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}.")

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

        if order == 2:
            return self._solve_second_order(
                compiled, param_vec, seed, raise_on_bk_violation
            )
        return self._solve_first_order(compiled, param_vec, seed, raise_on_bk_violation)

    @staticmethod
    def _resolve_ss_seed(
        ss_seed: list[float] | ndarray | dict[str, float] | None,
        compiled: CompiledModel,
    ) -> NDF:
        """Newton seed for the steady state, in canonical variable order.

        Priority: an explicit ``ss_seed`` > the model's configured symbolic
        ``ss_seed`` > zeros. Newton resolves ``F(ss, ss) = 0`` from here, so a
        gap model (ss = 0) seeds at 0 and converges in one step, while a level
        model that declares its seed in the config seeds itself.

        A seed is written over the declared variables, and the generated block is
        derived the way ``desugar`` derives the configured one. A compiled-length
        array is taken as canonical order already, which is what feeds a previous
        solve's steady state back in.
        """
        layout = compiled.layout
        n_var = len(compiled.var_names)
        n_declared = len(layout.declared_names) - len(layout.generated)
        declared = layout.declared_names[:n_declared]

        if ss_seed is not None:
            if isinstance(ss_seed, dict):
                return DSGESolver._widen_ss_seed(ss_seed, compiled)
            seed = asarray(ss_seed, dtype=float64)
            # Newton reads this buffer natively, so a short one is an
            # out-of-bounds read rather than an exception.
            if seed.shape == (n_var,):
                return seed
            if seed.shape != (n_declared,):
                raise ValueError(
                    f"ss_seed has shape {seed.shape}, expected ({n_declared},) for "
                    f"the declared variables {list(declared)} or ({n_var},) for the "
                    f"compiled variables {list(compiled.var_names)}."
                )
            return DSGESolver._widen_ss_seed(dict(zip(declared, seed)), compiled)

        conf = compiled.config
        name_to_func = {v.__name__: v for v in conf.variables.variables}
        params = conf.calibration.parameters
        ss = np.zeros(n_var, dtype=float64)
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
    def _widen_ss_seed(by_name: Mapping[str, float], compiled: CompiledModel) -> NDF:
        """Scatter a seed keyed by variable name into canonical order.

        A lag aux tracks its origin exactly, so it shares the origin's expansion
        point; a shock state has no steady state of its own and starts at 0. An
        entry naming a generated variable outright still wins, so a seed read
        back off the compiled layout round-trips.
        """
        lag_origin = compiled.layout.lag_origin
        return np.array(
            [
                by_name.get(name, by_name.get(lag_origin.get(name, name), 0.0))
                for name in compiled.var_names
            ],
            dtype=float64,
        )

    @staticmethod
    def _assemble_state_space(
        sol: KleinSolution | PerturbationSolution, compiled: CompiledModel
    ) -> tuple[ND, ND]:
        """First-order state space: X_t = [states; controls], x_{t+1} = p x_t (+
        shocks), controls_t = f x_t. Shocks hit only the first n_exog states."""
        p = np.asarray(sol.p, dtype=complex128)
        f = np.asarray(sol.f, dtype=complex128)
        n_s = compiled.n_state
        n_u = len(compiled.var_names) - n_s
        n_exo = compiled.n_exog
        if n_exo > n_s:
            raise ValueError(f"n_exog ({n_exo}) cannot exceed n_state ({n_s}).")

        return assemble_state_space(p, f, n_s, n_u, n_exo)

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
    ) -> SolvedModel:
        """First-order (Klein) solve."""
        sol = klein_solve(
            compiled.construct_objective_cfunc(),
            param_vec,
            seed,
            compiled.n_state,
        )
        self._raise_or_warn_stability_error(
            sol.stab, should_raise=raise_on_bk_violation
        )
        A, B = self._assemble_state_space(sol, compiled)

        return SolvedModel(compiled=compiled, policy=sol, A=A, B=B)

    def _solve_second_order(
        self,
        compiled: CompiledModel,
        param_vec: NDF,
        seed: NDF,
        raise_on_bk_violation: bool = True,
    ) -> SolvedModel:
        """Second-order (SGU) solve. Runs the Klein first order (which Newton-
        resolves the steady state from ``seed``), sweeps the bicomplex Hessian at
        that steady state, and assembles ``g_xx``/``h_xx`` + the ``g_ss``/``h_ss``
        risk correction into a :class:`PerturbationSolution`. Requires the native
        extension."""

        n_eq = len(compiled.var_names)
        n_state = compiled.n_state
        cf = compiled.construct_objective_cfunc()
        cf_bc = compiled.construct_objective_cfunc_bicomplex()

        sol = klein_solve(cf, param_vec, seed, n_state)
        self._raise_or_warn_stability_error(
            sol.stab, should_raise=raise_on_bk_violation
        )
        ss = sol.steady_state
        gx, hx = np.real(sol.f), np.real(sol.p)

        a, b = klein_preprocess(cf.address, ss, param_vec, n_eq, False)
        f_xx = bicomplex_hessian(cf_bc.address, ss, param_vec, n_eq)
        gxx, hxx = second_order(a, b, f_xx, gx, hx, n_state)
        eta = self._build_eta(compiled)
        gss, hss = second_order_risk(a, b, f_xx, gx, gxx, eta, n_state)

        pert = PerturbationSolution(
            p=sol.p,
            f=sol.f,
            stab=sol.stab,
            eig=sol.eig,
            order=2,
            steady_state=ss,
            gxx=gxx,
            hxx=hxx,
            gss=gss,
            hss=hss,
        )
        A, B = self._assemble_state_space(pert, compiled)
        return SolvedModel(compiled=compiled, policy=pert, A=A, B=B)

    @staticmethod
    def _build_eta(compiled: CompiledModel) -> NDF:
        """Shock loading ``eta`` (nx x n_exog): ``eta @ eta.T`` is the state
        innovation covariance. Stds fill the exog-state rows; correlations enter via
        the Cholesky of the exog-shock covariance."""
        conf = compiled.config
        n_state = compiled.n_state
        n_exog = compiled.n_exog
        eta = np.zeros((n_state, n_exog), dtype=float64)
        if n_exog == 0:
            return eta

        params = conf.calibration.parameters
        shock_std = conf.calibration.shock_std
        shock_corr = conf.calibration.shock_corr
        innovations = list(conf.shock_map)

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

        cov = corr * np.outer(stds, stds)
        eta[:n_exog, :] = np.linalg.cholesky(cov)
        return eta

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
        symmetrize: bool | None = None,
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
        symmetrize: bool | None = None,
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
        symmetrize: bool | None = None,
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
            solve_params = est.theta_to_params(est.params_to_theta(theta_star))
        else:
            raise ValueError("method must be one of {'mle', 'map', 'mcmc'}.")

        self._sync_calibration_with_params(compiled, solve_params)
        solved = self.solve(
            compiled=compiled,
            parameters={k: float(v) for k, v in solve_params.items()},
            ss_seed=ss_seed,
        )
        return result, solved

    @staticmethod
    def _bad_time_offsets(
        expr: Basic,
        var_funcs: list[Function],
        t: Symbol,
        allowed: set[int] | None = None,
    ) -> set[int]:
        allowed = {0, 1} if allowed is None else allowed
        bad: set[int] = set()

        for call in expr.atoms(sp.Function):
            if (
                call.func not in [vf.func for vf in var_funcs]
                and call.func not in var_funcs
            ):
                continue

            if not call.args:
                continue

            arg0 = call.args[0]
            if arg0.free_symbols and t in arg0.free_symbols:
                k = sp.simplify(arg0 - t)
                if k.is_integer:
                    kk = int(k)
                    if kk not in allowed:
                        bad.add(kk)
        return bad
