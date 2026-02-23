import sympy as sp
from sympy import Symbol, Function, Expr

import numpy as np
from numpy import float64, complex128, asarray, ndarray, real_if_close
from numpy.typing import NDArray

from numba import njit

import pandas as pd  # fuck linearsolve
import linearsolve


from .config import ModelConfig
from .compiled_model import CompiledModel
from .solved_model import SolvedModel
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
    ) -> CompiledModel:

        conf = self.model_config
        kalman_conf = self.kalman_config
        t = self.t

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
                for v in conf.variables
            ]
            var_order = [v.__name__ for v in conf.variables]
        else:
            var_order = [  # pyright: ignore
                v.__name__ if hasattr(v, "func") else v for v in variable_order
            ]

        name_to_func = {v.__name__: v for v in conf.variables}
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

        lambda_args = [*fwd_syms, *cur_syms, *params]
        funcs = [sp.lambdify(lambda_args, c, modules="numpy") for c in compiled]

        def equations(
            fwd: ndarray, cur: ndarray, par: dict[str, float] | ndarray
        ) -> ndarray:
            fwd = np.asarray(fwd, dtype=complex128)
            cur = np.asarray(cur, dtype=complex128)

            if isinstance(par, dict):
                par_vec = np.array([par[p.name] for p in params], dtype=complex128)
            else:
                par_vec = np.asarray(par, dtype=complex128)
                if par_vec.shape[0] != len(params):
                    raise ValueError(
                        f"Parameter vector length {par_vec.shape[0]} != {len(params)}"
                    )

            vals = [f(*fwd, *cur, *par_vec) for f in funcs]
            return np.asarray(vals)

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
        # Recompile measurement functions with params substituted (No param passthrough to downstream EKF)
        variables = [conf.variables[idx[name]] for name in var_order]

        jac_lambda = sp.lambdify(
            [*variables, *params],
            symbolic_jacobian,
            modules="numpy",
        )

        return CompiledModel(
            config=conf,
            kalman=kalman_conf,
            cur_syms=cur_syms,
            var_names=var_order,
            calib_params=params,
            idx=idx,
            objective_eqs=compiled,
            objective_funcs=funcs,
            equations=equations,
            observable_names=[v.name for v in conf.observables],
            observable_eqs=observable_exprs,
            observable_funcs=observable_funcs,
            observable_jacobian=njit(jac_lambda),
            n_state=int(n_state),
            n_exog=int(n_exog),
        )

    def solve(
        self,
        compiled: CompiledModel,
        *,
        parameters: dict[str, float] | None = None,
        steady_state: ndarray | dict[str, float] | None = None,
        log_linear: bool = False,
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

        def _eqs(
            fwd: ndarray, cur: ndarray, par: dict[str, float] | ndarray
        ) -> ndarray:
            return compiled.equations(fwd, cur, par)

        mdl = linearsolve.model(
            equations=_eqs,
            variables=compiled.var_names,
            parameters=pd.Series(params, dtype=complex128),
            n_states=compiled.n_state,
            n_exo_states=compiled.n_exog,
        )

        mdl.set_ss(ss)
        mdl.approximate_and_solve(log_linear=log_linear)

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
