import warnings
import sympy as sp
from sympy import Symbol, Function, Expr
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence
from textwrap import dedent

import numpy as np
from numpy import float64, complex128, asarray, ndarray, real_if_close
from numpy.typing import NDArray

from numba import njit
from numba import types as nb_typ
from numba.core import errors as nb_err

import pandas as pd
from scipy.optimize import root

from .klein import klein_solve
from .config import ModelConfig, SymbolGetterDict
from .compiled_model import CompiledModel, VariableLayout
from .linearization import linearize_model
from .solved_model import SolvedModel
from .second_order import (
    PerturbationSolution,
    solve_second_order,
    solve_second_order_risk,
)

# Second-order needs the native bicomplex-Hessian / complex-step drivers; there is
# no numba fallback for the Hessian sweep (order=2 raises if the extension is absent).
# The steady-state Newton is native-preferred with a scipy.optimize.root fallback.
_bicomplex_hessian: Callable[..., Any] | None
_klein_preprocess: Callable[..., Any] | None
_steady_state_newton: Callable[..., Any] | None
try:
    from .._ckernels.core import bicomplex_hessian as _bicomplex_hessian
    from .._ckernels.core import klein_preprocess as _klein_preprocess
    from .._ckernels.core import steady_state_newton as _steady_state_newton
except ImportError:  # pragma: no cover - exercised only without the extension
    _bicomplex_hessian = None
    _klein_preprocess = None
    _steady_state_newton = None

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
        n_state: int | None = None,
        n_exog: int | None = None,
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
        inferred_layout = self._infer_variable_layout(conf)
        resolved_n_exog = self._resolve_layout_count(
            provided=n_exog,
            inferred=inferred_layout.n_exog,
            name="n_exog",
        )
        resolved_n_state = self._resolve_layout_count(
            provided=n_state,
            inferred=inferred_layout.n_state,
            name="n_state",
        )

        if variable_order is None:
            var_order = list(inferred_layout.canonical_names)
        else:
            var_order = [self._coerce_variable_name(v) for v in variable_order]
            self._validate_explicit_variable_order(
                var_order=var_order,
                inferred_layout=inferred_layout,
                n_exog=resolved_n_exog,
                n_state=resolved_n_state,
            )

        missing = [v for v in var_order if v not in name_to_func]
        if missing:
            raise ValueError(
                f"The following variables in var_order do not exist in the model: {missing}"
            )
        if len(set(var_order)) != len(var_order):
            raise ValueError("variable_order contains duplicate variables.")

        layout = VariableLayout(
            declared_names=inferred_layout.declared_names,
            canonical_names=tuple(var_order),
            exo_state_names=tuple(var_order[:resolved_n_exog]),
            endo_state_names=tuple(var_order[resolved_n_exog:resolved_n_state]),
            control_names=tuple(var_order[resolved_n_state:]),
            n_exog=resolved_n_exog,
            n_state=resolved_n_state,
            idx={name: i for i, name in enumerate(var_order)},
        )

        var_funcs = [name_to_func[name] for name in var_order]
        idx = dict(layout.idx)

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
        observable_funcs = [
            njit(sp.lambdify([*cur_syms, *params], expr, modules="numpy"))
            for expr in observable_exprs
        ]

        if observable_exprs:
            symbolic_jacobian = sp.Matrix(
                [
                    [sp.diff(expr, cur_sym) for cur_sym in cur_syms]
                    for expr in observable_exprs
                ]
            )
        else:
            symbolic_jacobian = sp.zeros(0, len(cur_syms))

        jac_scalars = list(symbolic_jacobian)  # pyright: ignore
        jac_scalar_funcs = tuple(
            njit(sp.lambdify([*cur_syms, *params], scalar, modules="numpy"))
            for scalar in jac_scalars
        )

        n_obs, n_vars = symbolic_jacobian.shape

        arg_names = [f"var{i}" for i in range(len(cur_syms))] + [
            f"param{i}" for i in range(len(params))
        ]
        args_str = ", ".join(f"{name}: float64" for name in arg_names)
        call_args_str = ", ".join(arg_names)

        func_bindings = "\n".join(
            f"    f{k} = jac_scalar_funcs[{k}]" for k in range(len(jac_scalar_funcs))
        )

        body_lines = []
        k = 0
        for i in range(n_obs):
            for j in range(n_vars):
                body_lines.append(f"    J[{i}, {j}] = f{k}({call_args_str})")
                k += 1

        jac_str = dedent(
            f"""
def jacobian_func({args_str}) -> NDF:
    J = np.empty(({n_obs}, {n_vars}), dtype=float64)
{func_bindings}
{chr(10).join(body_lines)}
    return J
"""
        )

        ns: dict[str, Any] = {
            "jac_scalar_funcs": jac_scalar_funcs,
            "np": np,
            "float64": float64,
            "NDF": NDF,
        }
        exec(jac_str, ns)
        jacobian_func = njit(ns["jacobian_func"])

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=nb_err.NumbaExperimentalFeatureWarning
            )
            jacobian_func.compile(  # pyright: ignore
                tuple(nb_typ.float64 for _ in arg_names)
            )

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
            observable_funcs=observable_funcs,
            observable_jacobian=jacobian_func,
            observable_jacobian_funcs=jac_scalar_funcs,
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

    @staticmethod
    def _resolve_layout_count(
        *,
        provided: int | None,
        inferred: int,
        name: str,
    ) -> int:
        if provided is None:
            return inferred
        provided_int = int(provided)
        if provided_int != inferred:
            raise ValueError(
                f"{name}={provided_int} does not match inferred {name}={inferred}."
            )
        return provided_int

    def _infer_variable_layout(self, conf: ModelConfig) -> VariableLayout:
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
        canonical_names = (*state_names, *control_names)

        return VariableLayout(
            declared_names=declared_names,
            canonical_names=canonical_names,
            exo_state_names=exo_state_names,
            endo_state_names=endo_state_names,
            control_names=control_names,
            n_exog=len(exo_state_names),
            n_state=len(state_names),
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
    def _validate_explicit_variable_order(
        *,
        var_order: Sequence[str],
        inferred_layout: VariableLayout,
        n_exog: int,
        n_state: int,
    ) -> None:
        declared = set(inferred_layout.declared_names)
        unknown = [name for name in var_order if name not in declared]
        if unknown:
            raise ValueError(
                f"The following variables in var_order do not exist in the model: {unknown}"
            )
        if len(set(var_order)) != len(var_order):
            raise ValueError("variable_order contains duplicate variables.")
        if set(var_order) != declared:
            raise ValueError("variable_order must contain every model variable.")

        expected_exo = set(inferred_layout.exo_state_names)
        got_exo = set(var_order[:n_exog])
        if got_exo != expected_exo:
            raise ValueError(
                "variable_order is incompatible with inferred state layout. "
                f"Expected first n_exog variables to be shocked states "
                f"{list(inferred_layout.exo_state_names)}."
            )

        expected_states = set(
            (*inferred_layout.exo_state_names, *inferred_layout.endo_state_names)
        )
        got_states = set(var_order[:n_state])
        if got_states != expected_states:
            raise ValueError(
                "variable_order is incompatible with inferred state layout. "
                f"Expected first n_state variables to be states "
                f"{list((*inferred_layout.exo_state_names, *inferred_layout.endo_state_names))}."
            )

    def solve(
        self,
        compiled: CompiledModel,
        *,
        parameters: dict[str, float] | None = None,
        steady_state: list[float] | ndarray | dict[str, float] | None = None,
        order: int = 1,
    ) -> SolvedModel:
        """Solve the model to first (``order=1``) or second (``order=2``) order.

        ``order=1`` is the Klein linear solve (policy is a ``KleinSolution``).
        ``order=2`` additionally computes the SGU second-order tensors and the
        sigma^2 risk correction (policy is a ``PerturbationSolution``); it requires
        the native extension and a nonlinear steady state (see ``_solve_second_order``).
        The state-space ``A``/``B`` are the first-order transition in both cases.
        """
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}.")
        conf = self.model_config

        if parameters is None:
            params: dict[str, float64] = {
                p.name: float64(conf.calibration.parameters[p])
                for p in conf.parameters
                if p in conf.calibration.parameters
            }
        else:
            params = {p: float64(v) for p, v in parameters.items()}

        param_vec = np.ascontiguousarray(
            np.array([params[p.name] for p in compiled.calib_params], dtype=float64)
        )
        given_ss = self._coerce_steady_state(steady_state, compiled)

        if order == 2:
            return self._solve_second_order(compiled, param_vec, given_ss)

        ss = (
            given_ss
            if given_ss is not None
            else np.zeros(len(compiled.var_names), dtype=float64)
        )
        sol = klein_solve(
            compiled.construct_objective_vector_func(),
            param_vec,
            ss,
            compiled.n_state,
            residual_cfunc=compiled.construct_objective_cfunc(),
        )
        if sol.stab != 0:
            raise ValueError(
                f"Klein stability/uniqueness condition violated (stab={sol.stab})."
            )
        A, B = self._assemble_state_space(sol, compiled)
        return SolvedModel(compiled=compiled, policy=sol, A=A, B=B)

    @staticmethod
    def _coerce_steady_state(
        steady_state: list[float] | ndarray | dict[str, float] | None,
        compiled: CompiledModel,
    ) -> NDF | None:
        """User-supplied steady state -> canonical-order vector (or ``None``)."""
        if steady_state is None:
            return None
        if isinstance(steady_state, dict):
            return np.array(
                [steady_state.get(vn, 0.0) for vn in compiled.var_names], dtype=float64
            )
        return asarray(steady_state, dtype=float64)

    @staticmethod
    def _assemble_state_space(sol: Any, compiled: CompiledModel) -> tuple[ND, ND]:
        """First-order state space: X_t = [states; controls], x_{t+1} = p x_t (+
        shocks), controls_t = f x_t. Shocks hit only the first n_exog states."""
        p = np.asarray(sol.p, dtype=complex128)
        f = np.asarray(sol.f, dtype=complex128)
        n_s = compiled.n_state
        n_u = len(compiled.var_names) - n_s
        n_exo = compiled.n_exog
        if n_exo > n_s:
            raise ValueError(f"n_exog ({n_exo}) cannot exceed n_state ({n_s}).")

        A = real_if_close(
            np.block([[p, np.zeros((n_s, n_u))], [f @ p, np.zeros((n_u, n_u))]])
        )
        B_state = np.vstack(
            [
                np.eye(n_exo, dtype=float64),
                np.zeros((n_s - n_exo, n_exo), dtype=float64),
            ]
        )
        B = real_if_close(np.vstack([B_state, f @ B_state]))
        return A, B

    def _solve_second_order(
        self, compiled: CompiledModel, param_vec: NDF, given_ss: NDF | None
    ) -> SolvedModel:
        """Second-order (SGU) solve. Resolves + validates the nonlinear steady
        state, runs the Klein first order, sweeps the bicomplex Hessian, and
        assembles ``g_xx``/``h_xx`` + the ``g_ss``/``h_ss`` risk correction into a
        :class:`PerturbationSolution`. Requires the native extension."""
        if _bicomplex_hessian is None or _klein_preprocess is None:
            raise RuntimeError(
                "order=2 requires the native _ckernels extension (bicomplex Hessian "
                "sweep); it is not available in this build."
            )
        n_eq = len(compiled.var_names)
        n_state = compiled.n_state
        eq = compiled.construct_objective_vector_func()
        cf = compiled.construct_objective_cfunc()
        cf_bc = compiled.construct_objective_cfunc_bicomplex()

        seed = (
            given_ss
            if given_ss is not None
            else self._resolve_config_steady_state(compiled)
        )
        ss = self._checked_steady_state(eq, cf, param_vec, seed)

        sol = klein_solve(eq, param_vec, ss, n_state, residual_cfunc=cf)
        if sol.stab != 0:
            raise ValueError(
                f"Klein stability/uniqueness condition violated (stab={sol.stab})."
            )
        gx, hx = np.real(sol.f), np.real(sol.p)

        a, b = _klein_preprocess(cf.address, ss, param_vec, n_eq, False)
        f_xx = _bicomplex_hessian(cf_bc.address, ss, param_vec, n_eq)
        gxx, hxx = solve_second_order(a, b, f_xx, gx, hx, n_state)
        eta = self._build_eta(compiled)
        gss, hss = solve_second_order_risk(a, b, f_xx, gx, gxx, eta, n_state)

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
    def _resolve_config_steady_state(compiled: CompiledModel) -> NDF:
        """Evaluate each variable's symbolic ``steady_state`` (None -> 0) against
        the calibration -> a canonical-order vector."""
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
    def _checked_steady_state(
        eq: Callable[..., NDArray], cf: Any, param_vec: NDF, seed: NDF
    ) -> NDF:
        """Solve ``F(ss, ss) = 0`` from ``seed`` and cross-check against it: a
        configured steady state that disagrees with the solved one (beyond
        tolerance) is a modeling error, so raise rather than expand at a bad point.

        Prefers the native Newton (``klein_preproc`` Jacobian + f64 LU); falls back
        to ``scipy.optimize.root`` when the extension is absent."""
        seed = np.ascontiguousarray(seed, dtype=float64)
        if _steady_state_newton is not None and cf is not None:
            ss, _iters = _steady_state_newton(cf.address, seed, param_vec)
            solved = np.asarray(ss, dtype=float64)
        else:
            par_c = param_vec.astype(complex128)

            def residual(x: NDF) -> NDF:
                xc = np.ascontiguousarray(x.astype(complex128))
                return np.real(eq(xc, xc, par_c)).astype(float64)

            result = root(residual, seed, method="hybr")
            if not result.success:
                raise ValueError(
                    "Steady-state solve did not converge from the configured seed "
                    f"{np.asarray(seed)}: {result.message}"
                )
            solved = np.asarray(result.x, dtype=float64)
        if not np.allclose(solved, seed, rtol=1e-6, atol=1e-8):
            diff = float(np.max(np.abs(solved - seed)))
            raise ValueError(
                "Configured steady state disagrees with the numerically solved "
                f"steady state (max abs diff {diff:.3e}): configured="
                f"{np.asarray(seed)}, solved={solved}. Fix the model's steady_state "
                "entries (or the steady_state= argument) so F(ss, ss) = 0 holds."
            )
        return solved

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
            stds[i] = float(params[sig_sym]) if sig_sym in params else 1.0
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

    @staticmethod
    def _validate_prior_initial_guess(
        *,
        priors: Mapping[str, Any] | None,
        initial_params: Mapping[str, float64],
    ) -> None:
        if priors is None:
            return

        invalid: list[str] = []
        for name, prior in priors.items():
            if name not in initial_params:
                continue
            val = float64(initial_params[name])
            try:
                if hasattr(prior, "transform"):
                    z = float64(getattr(prior, "transform").safe_forward(val))
                else:
                    z = val
                prior.logpdf(z)
            except (
                Exception
            ) as exc:  # pragma: no cover - exact exception type is prior-dependent
                invalid.append(f"{name}={val} ({type(exc).__name__}: {exc})")

        if invalid:
            msg = (
                "Initial calibration values are incompatible with the provided priors "
                "or their transforms: " + ", ".join(invalid)
            )
            raise ValueError(msg)

    @staticmethod
    def _theta0_to_array(
        est: "Estimator", theta0: NDArray | Mapping[str, float] | None
    ) -> NDArray:
        if theta0 is None:
            return est.theta0()

        if isinstance(theta0, Mapping):
            missing = [name for name in est.param_names if name not in theta0]
            if missing:
                raise ValueError(
                    f"theta0 dictionary is missing estimated parameters: {missing}"
                )
            unknown = [k for k in theta0.keys() if k not in est.param_names]
            if unknown:
                raise ValueError(f"theta0 dictionary has unknown parameters: {unknown}")
            return est.params_to_theta(
                {name: float64(theta0[name]) for name in est.param_names}
            )

        return asarray(theta0, dtype=float64)

    def _estimator(
        self,
        *,
        compiled: CompiledModel,
        y: NDArray | pd.DataFrame,
        observables: list[str] | None = None,
        estimated_params: list[str] | None = None,
        priors: Mapping[str, Any] | None = None,
        steady_state: list[float] | NDArray | dict[str, float] | None = None,
        x0: NDArray | None = None,
        p0_mode: str | None = None,
        p0_scale: float | float64 | None = None,
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
            estimated_params=estimated_params,
            priors=priors,
            steady_state=(
                asarray(steady_state, dtype=float64)
                if isinstance(steady_state, list)
                else steady_state
            ),
            x0=x0,
            p0_mode=p0_mode,
            p0_scale=p0_scale,
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
        estimated_params: list[str] | None = None,
        priors: Mapping[str, Any] | None = None,
        steady_state: list[float] | NDArray | dict[str, float] | None = None,
        x0: NDArray | None = None,
        p0_mode: str | None = None,
        p0_scale: float | float64 | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool | None = None,
        R: NDArray | None = None,
        **method_kwargs: Any,
    ) -> Any:
        est = self._estimator(
            compiled=compiled,
            y=y,
            observables=observables,
            estimated_params=estimated_params,
            priors=priors,
            steady_state=(
                asarray(steady_state, dtype=float64)
                if isinstance(steady_state, list)
                else steady_state
            ),
            x0=x0,
            p0_mode=p0_mode,
            p0_scale=p0_scale,
            jitter=jitter,
            symmetrize=symmetrize,
            R=R,
        )

        init = self._theta0_to_array(est, theta0)
        self._validate_prior_initial_guess(
            priors=est.priors,
            initial_params=est.theta_to_params(init),
        )
        if (
            R is None
            and hasattr(compiled, "var_names")
            and hasattr(compiled, "observable_names")
            and hasattr(compiled, "kalman")
        ):
            from ..estimation import backend as est_backend

            est.R = est_backend.estimate_R(
                solver=self,
                compiled=compiled,
                y=y,
                params=est.theta_to_params(init),
                observables=observables,
                steady_state=(
                    asarray(steady_state, dtype=float64)
                    if isinstance(steady_state, list)
                    else steady_state
                ),
                x0=x0,
                p0_mode=p0_mode,
                p0_scale=p0_scale,
                jitter=jitter,
                symmetrize=symmetrize,
            )

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
        estimated_params: list[str] | None = None,
        priors: Mapping[str, Any] | None = None,
        steady_state: list[float] | NDArray | dict[str, float] | None = None,
        x0: NDArray | None = None,
        p0_mode: str | None = None,
        p0_scale: float | float64 | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool | None = None,
        R: NDArray | None = None,
        **method_kwargs: Any,
    ) -> tuple[Any, SolvedModel]:

        steady_state = (
            np.asarray(steady_state, dtype=float64)
            if isinstance(steady_state, list)
            else steady_state
        )
        est = self._estimator(
            compiled=compiled,
            y=y,
            observables=observables,
            estimated_params=estimated_params,
            priors=priors,
            steady_state=steady_state,
            x0=x0,
            p0_mode=p0_mode,
            p0_scale=p0_scale,
            jitter=jitter,
            symmetrize=symmetrize,
            R=R,
        )

        init = self._theta0_to_array(est, theta0)
        self._validate_prior_initial_guess(
            priors=est.priors,
            initial_params=est.theta_to_params(init),
        )
        if (
            R is None
            and hasattr(compiled, "var_names")
            and hasattr(compiled, "observable_names")
            and hasattr(compiled, "kalman")
        ):
            from ..estimation import backend as est_backend

            est.R = est_backend.estimate_R(
                solver=self,
                compiled=compiled,
                y=y,
                params=est.theta_to_params(init),
                observables=observables,
                steady_state=steady_state,
                x0=x0,
                p0_mode=p0_mode,
                p0_scale=p0_scale,
                jitter=jitter,
                symmetrize=symmetrize,
            )

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
            steady_state=steady_state,
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
