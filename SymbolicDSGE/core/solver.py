from logging import warning
import warnings
import sympy as sp
from sympy import Symbol, Function, Expr
from typing import TYPE_CHECKING, Any, Mapping
from textwrap import dedent

import numpy as np
from numpy import float64, complex128, asarray, ndarray, real_if_close
from numpy.typing import NDArray

import numba
from numba import njit

import pandas as pd  # fuck linearsolve

from .. import _linearsolve as linearsolve
from .config import ModelConfig
from .compiled_model import CompiledModel
from .linearization import linearize_model
from .solved_model import SolvedModel

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
        variable_order: list[Function] | None = None,
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

        # Deterministic var order
        if not variable_order:
            var_order: list[str] = [
                v.func.__name__ if hasattr(v, "func") else v.__name__
                for v in ordered_variables
            ]
            var_order = [v.__name__ for v in ordered_variables]
        else:
            var_order = [  # pyright: ignore
                v.__name__ if hasattr(v, "func") else v for v in variable_order
            ]

        name_to_func = {v.__name__: v for v in ordered_variables}
        missing = [v for v in var_order if v not in name_to_func]
        if missing:
            raise ValueError(
                f"The following variables in var_order do not exist in the model: {missing}"
            )

        var_funcs = [name_to_func[name] for name in var_order]
        idx = {name: i for i, name in enumerate(var_order)}

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
        for name, f, cur, fwd in zip(var_order, var_funcs, cur_syms, fwd_syms):

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
            sp.simplify(expr.subs(shock_zero_subs)) for expr in compiled
        ]

        lambda_args = [*fwd_syms, *cur_syms, *params]
        objective_scalar_funcs = [
            njit(sp.lambdify(lambda_args, c, modules="numpy")) for c in compiled_numeric
        ]

        if n_state is None or n_exog is None:
            raise ValueError(
                "For linearsolve backend you must provide n_state and n_exog explicitly."
            )

        shifted_obs = [
            self._offset_lags(expr, t) for expr in conf.equations.observable.values()
        ]
        observable_exprs = [sp.simplify(expr.subs(subs_map)) for expr in shifted_obs]
        observable_funcs = [
            njit(sp.lambdify([*cur_syms, *params], expr, modules="numpy"))
            for expr in observable_exprs
        ]

        symbolic_jacobian: sp.Matrix = conf.equations.obs_jacobian
        variables = [ordered_variables[idx[name]] for name in var_order]

        jac_scalars = list(symbolic_jacobian)
        jac_scalar_funcs = tuple(
            njit(sp.lambdify([*variables, *params], scalar, modules="numpy"))
            for scalar in jac_scalars
        )

        n_obs, n_vars = symbolic_jacobian.shape

        arg_names = [f"var{i}" for i in range(len(variables))] + [
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
                "ignore", category=numba.errors.NumbaExperimentalFeatureWarning
            )
            jacobian_func.compile(tuple(numba.float64 for _ in arg_names))

        return CompiledModel(
            config=conf,
            kalman=kalman_conf,
            cur_syms=cur_syms,
            var_names=var_order,
            calib_params=params,
            idx=idx,
            objective_eqs=compiled_numeric,
            objective_funcs=objective_scalar_funcs,
            observable_names=[v.name for v in conf.observables],
            observable_eqs=observable_exprs,
            observable_funcs=observable_funcs,
            observable_jacobian=jacobian_func,
            observable_jacobian_funcs=jac_scalar_funcs,
            n_state=int(n_state),
            n_exog=int(n_exog),
        )

    def solve(
        self,
        compiled: CompiledModel,
        *,
        parameters: dict[str, float] | None = None,
        steady_state: ndarray | dict[str, float] | None = None,
    ) -> SolvedModel:
        conf = self.model_config

        if parameters is None:
            params: dict[str, float64] = {
                p.name: float64(conf.calibration.parameters[p])
                for p in conf.parameters
                if p in conf.calibration.parameters
            }
        else:
            params = {p: float64(v) for p, v in parameters.items()}

        if steady_state is None:
            ss = np.zeros(len(compiled.var_names), dtype=float64)
        elif isinstance(steady_state, dict):
            ss = np.array(
                [steady_state.get(vn, 0.0) for vn in compiled.var_names], dtype=float64
            )
        else:
            ss = asarray(steady_state, dtype=float64)

        mdl = linearsolve.model(
            equations=compiled.equations,
            variables=compiled.var_names,
            parameters=np.ascontiguousarray(
                np.array(
                    [params[p.name] for p in compiled.calib_params], dtype=complex128
                )
            ),
            parameter_names=[p.name for p in compiled.calib_params],
            n_states=compiled.n_state,
            n_exo_states=compiled.n_exog,
        )
        setattr(mdl, "_equations_numeric", compiled.construct_objective_vector_func())
        setattr(
            mdl,
            "_parameter_array",
            np.ascontiguousarray(
                np.array([params[p.name] for p in compiled.calib_params], dtype=float64)
            ),
        )

        mdl.set_ss(ss)
        mdl.approximate_and_solve()

        # Extract solution matrices (linearsolve uses .gx, .hx style in some versions, keep flexible)
        # Common conventions in linear RE solvers:
        # x_{t+1} = hx x_t + eta eps_{t+1}
        # y_t = gx x_t

        p = np.asarray(mdl.p, dtype=complex128)
        f = np.asarray(mdl.f, dtype=complex128)

        n_s = compiled.n_state
        n_u = len(compiled.var_names) - n_s
        n_exo = compiled.n_exog  # number of shocked states (must be <= n_s)

        if n_exo > n_s:
            raise ValueError(f"n_exog ({n_exo}) cannot exceed n_state ({n_s}).")

        # Build full transition for X_t = [states_t; controls_t]
        A = real_if_close(
            np.block(
                [
                    [p, np.zeros((n_s, n_u))],
                    [f @ p, np.zeros((n_u, n_u))],
                ]
            )
        )
        # Shocks hit only the first n_exo states with identity.
        B_state = np.vstack(
            [
                np.eye(n_exo, dtype=float64),
                np.zeros((n_s - n_exo, n_exo), dtype=float64),
            ]
        )
        B = real_if_close(
            np.vstack(
                [
                    B_state,
                    f @ B_state,
                ]
            )
        )

        if getattr(mdl, "stab", 0) != 0:
            raise ValueError(
                f"Klein stability/uniqueness condition violated (stab={mdl.stab})."
            )

        return SolvedModel(
            compiled=compiled,
            policy=mdl,
            A=A,
            B=B,
        )

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
        steady_state: NDArray | dict[str, float] | None = None,
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
            steady_state=steady_state,
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
        steady_state: NDArray | dict[str, float] | None = None,
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
            if steady_state is not None
            else None
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
