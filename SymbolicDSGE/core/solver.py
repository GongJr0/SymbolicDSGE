import warnings
from dataclasses import replace

import sympy as sp
from sympy import Symbol, Function, Expr
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy import float64, complex128, asarray, ndarray, real_if_close
from numpy.typing import NDArray

import pandas as pd

from .config import ModelConfig, SymbolGetterDict
from .compiled_model import CompiledModel, VariableLayout
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
        kalman_conf = self.kalman_config
        t = self.t
        ordered_variables = conf.variables.variables

        # Convert model to minimization problem
        obj = [
            sp.simplify(eq.lhs - eq.rhs)  # pyright: ignore
            for eq in conf.equations.model
        ]

        shifted = [self._offset_lags(o, t) for o in obj]

        name_to_func = {v.__name__: v for v in ordered_variables}

        layout = self._infer_variable_layout(conf, variable_order)
        var_order = list(layout.canonical_names)

        var_funcs = [name_to_func[name] for name in var_order]
        idx = dict(layout.idx)

        # P0 was assembled in declared variable order at parse time; permute it
        # into canonical order so it aligns with the compiled state vector. P0 is
        # diagonal, so this is a lossless entry permutation (a no-op when the
        # declared order already matches canonical).
        if kalman_conf is not None:
            declared_idx = {name: i for i, name in enumerate(layout.declared_names)}
            perm = [declared_idx[name] for name in layout.canonical_names]
            kalman_conf = replace(kalman_conf, P0=kalman_conf.P0[np.ix_(perm, perm)])

        for i, obj in enumerate(shifted):
            bad = self._bad_time_offsets(obj, var_funcs, t)
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

        compiled: list[Expr] = [sp.simplify(o.subs(subs_map)) for o in shifted]
        shock_zero_subs = {shock: 0.0 for shock in conf.shock_map.keys()}
        compiled_numeric: list[Expr] = [
            sp.simplify(expr.subs(shock_zero_subs))  # pyright: ignore
            for expr in compiled
        ]

        shifted_obs = [
            self._offset_lags(expr, t) for expr in conf.equations.observable.values()
        ]
        observable_exprs = [sp.simplify(expr.subs(subs_map)) for expr in shifted_obs]
        symbolic_jacobian = (
            sp.Matrix(
                [
                    [sp.diff(expr, cur_sym) for cur_sym in cur_syms]
                    for expr in observable_exprs
                ]
            )
            if observable_exprs
            else sp.zeros(0, len(cur_syms))
        )
        # Flat row-major (n_obs, n_var) jacobian; printed to a native cfunc on
        # demand via CompiledModel.construct_observable_jacobian_cfunc.
        observable_jacobian_eqs = list(symbolic_jacobian)  # pyright: ignore

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
        )

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
        variable_order: Sequence[Function | str] | None = None,
    ) -> VariableLayout:
        declared_names = tuple(v.__name__ for v in conf.variables.variables)
        declared_set = set(declared_names)
        t = self.t

        shock_targets = [
            self._coerce_variable_name(target) for target in conf.shock_map.values()
        ]
        unknown_targets = [name for name in shock_targets if name not in declared_set]
        if unknown_targets:
            raise ValueError(
                "shock_map targets unknown model variable(s): "
                + ", ".join(unknown_targets)
            )
        if len(set(shock_targets)) != len(shock_targets):
            raise ValueError(
                "shock_map contains multiple shocks for the same variable."
            )

        shocked = set(shock_targets)
        exo_state_names = tuple(name for name in declared_names if name in shocked)

        state_candidates: set[str] = set()
        for eq in conf.equations.model:
            lhs_info = self._function_call_offset(eq.lhs, declared_set, t)
            if lhs_info is not None:
                name, offset = lhs_info
                if offset == 1:
                    state_candidates.add(name)

            expr = sp.simplify(eq.lhs - eq.rhs)  # pyright: ignore
            for call in expr.atoms(sp.Function):
                call_info = self._function_call_offset(call, declared_set, t)
                if call_info is None:
                    continue
                name, offset = call_info
                if offset == -1:
                    state_candidates.add(name)

        endo_state_names = tuple(
            name
            for name in declared_names
            if name in state_candidates and name not in shocked
        )
        state_names = (*exo_state_names, *endo_state_names)
        control_names = tuple(
            name for name in declared_names if name not in set(state_names)
        )
        n_exog = len(exo_state_names)
        n_state = len(state_names)

        if variable_order is None:
            canonical_names: tuple[str, ...] = (*state_names, *control_names)
        else:
            canonical_names = self._resolve_variable_order(
                variable_order,
                declared_names=declared_names,
                exo_state_names=exo_state_names,
                endo_state_names=endo_state_names,
                n_exog=n_exog,
                n_state=n_state,
            )

        return VariableLayout(
            declared_names=declared_names,
            canonical_names=canonical_names,
            exo_state_names=tuple(canonical_names[:n_exog]),
            endo_state_names=tuple(canonical_names[n_exog:n_state]),
            control_names=tuple(canonical_names[n_state:]),
            n_exog=n_exog,
            n_state=n_state,
            idx={name: i for i, name in enumerate(canonical_names)},
        )

    @staticmethod
    def _function_call_offset(
        expr: Any,
        declared_names: set[str],
        t: Symbol,
    ) -> tuple[str, int] | None:
        if not hasattr(expr, "func") or not hasattr(expr.func, "__name__"):
            return None
        name = str(expr.func.__name__)
        if name not in declared_names or not getattr(expr, "args", None):
            return None

        arg0 = expr.args[0]
        if t not in getattr(arg0, "free_symbols", set()):
            return None
        offset = sp.simplify(arg0 - t)
        if offset.is_Integer or offset.is_integer is True:
            return name, int(offset)
        return None

    @staticmethod
    def _resolve_variable_order(
        variable_order: Sequence[Function | str],
        *,
        declared_names: tuple[str, ...],
        exo_state_names: tuple[str, ...],
        endo_state_names: tuple[str, ...],
        n_exog: int,
        n_state: int,
    ) -> tuple[str, ...]:
        """Coerce, validate, and return an explicit ``variable_order``.

        The order must name every model variable exactly once and stay compatible
        with the inferred state layout: its first ``n_exog`` entries are the shocked
        states and its first ``n_state`` entries are the states. The controls and the
        within-block ordering are free.
        """
        var_order = [DSGESolver._coerce_variable_name(v) for v in variable_order]
        declared = set(declared_names)
        unknown = [name for name in var_order if name not in declared]
        if unknown:
            raise ValueError(
                f"The following variables in var_order do not exist in the model: {unknown}"
            )
        if len(set(var_order)) != len(var_order):
            raise ValueError("variable_order contains duplicate variables.")
        if set(var_order) != declared:
            raise ValueError("variable_order must contain every model variable.")

        if set(var_order[:n_exog]) != set(exo_state_names):
            raise ValueError(
                "variable_order is incompatible with inferred state layout. "
                f"Expected first n_exog variables to be shocked states "
                f"{list(exo_state_names)}."
            )

        expected_states = (*exo_state_names, *endo_state_names)
        if set(var_order[:n_state]) != set(expected_states):
            raise ValueError(
                "variable_order is incompatible with inferred state layout. "
                f"Expected first n_state variables to be states "
                f"{list(expected_states)}."
            )
        return tuple(var_order)

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

        Priority: an explicit ``ss_seed`` (a dict is scattered into canonical
        order, missing entries 0) > the model's configured symbolic steady state
        > zeros. Newton resolves ``F(ss, ss) = 0`` from here, so a gap model
        (ss = 0) seeds at 0 and converges in one step, while a level model that
        declares its steady state in the config seeds itself.
        """
        if ss_seed is not None:
            if isinstance(ss_seed, dict):
                return np.array(
                    [ss_seed.get(vn, 0.0) for vn in compiled.var_names],
                    dtype=float64,
                )
            return asarray(ss_seed, dtype=float64)

        conf = compiled.config
        name_to_func = {v.__name__: v for v in conf.variables.variables}
        params = conf.calibration.parameters
        ss = np.zeros(len(compiled.var_names), dtype=float64)
        for i, name in enumerate(compiled.var_names):
            expr = conf.variables.steady_state[name_to_func[name]]
            if expr is None:
                continue
            val = sp.simplify(sp.sympify(expr).subs(params))
            try:
                ss[i] = float(val)
            except TypeError as exc:
                raise ValueError(
                    f"Steady state for '{name}' did not evaluate to a number: {val}"
                ) from exc
        return ss

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
        rev: SymbolGetterDict = SymbolGetterDict(
            {v: k for k, v in conf.shock_map.items()}  # variable -> innovation
        )
        innovations = [rev[vname] for vname in compiled.var_names[:n_exog]]

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
    def _min_time_offset(expr: Expr, t: Symbol) -> int:
        offs = []
        for call in expr.atoms(Function):
            if not call.args:
                continue

            arg0 = call.args[0]
            if arg0.free_symbols and t in arg0.free_symbols:
                k = sp.simplify(arg0 - t)
                if k.is_Integer:
                    offs.append(int(k))
        return min(offs) if offs else 0

    def _offset_lags(self, obj: Expr, t: Symbol) -> Expr:
        min_off = self._min_time_offset(obj, t)

        if min_off < 0:
            return sp.simplify(obj.subs(t, t - min_off))
        return obj

    @staticmethod
    def _bad_time_offsets(expr: Expr, var_funcs: list[Function], t: Symbol) -> set[int]:
        allowed = {0, 1}
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
