import sympy as sp
from sympy import Symbol, Function, Expr

import numpy as np
from numpy import float64, complex128, asarray, ndarray, real_if_close
from numpy.typing import NDArray

import pandas as pd  # fuck linearsolve
import linearsolve

from dataclasses import dataclass, asdict
from typing import Callable, Any, Union, Tuple, TypedDict, Iterable

import matplotlib.pyplot as plt

from .model_config import ModelConfig

NDF = NDArray[float64]
ND = NDArray


class MeasurementSpec(TypedDict):
    lin: dict[str, float | float64]
    const: list[float | float64 | str]


@dataclass(frozen=True)
class CompiledModel:
    config: ModelConfig

    var_names: list[str]
    idx: dict[str, int]

    objective_eqs: list[Expr]
    objective_funcs: list[Callable]
    equations: Callable[[Any, Any, Any], ND]

    observable_names: list[str]
    observable_eqs: list[Expr]
    observable_funcs: list[Callable]

    n_state: int
    n_exog: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SolvedModel:
    compiled: CompiledModel
    policy: Any
    A: ndarray
    B: ndarray

    def sim(
        self,
        T: int,
        shocks: dict[
            str, Union[Callable[[float | list[list[float]]], NDF], NDF]
        ] = None,
        shock_scale: float = 1.0,
        x0: ndarray = None,
        observables: bool = False,
    ) -> dict[str, NDF]:
        """
        Simulate the solved DSGE model over T periods.
        Parameters
        ----------
        T : int
            Number of time periods to simulate.

        shocks : dict[str, Union[Callable[[float], ndarray], ndarray]], optional
            A dictionary mapping shock variable names to either a callable that generates

        shock_scale : float, optional
            A scaling factor applied to all shocks.

        x0 : ndarray, optional
            Initial state vector. If None, defaults to zero vector.

        observables : bool, optional
            If True, compute and include observable variables in the output.

        Returns
        -------

        dict[str, ndarray]
            A dictionary mapping variable names to their simulated time series.
        """

        conf = self.compiled.config
        n = self.A.shape[0]

        if x0 is None:
            x0 = np.zeros((n,))
        x0 = asarray(x0, dtype=float64)

        shock_mat = np.zeros(
            (T, self.B.shape[1]), dtype=float
        )  # Deterministic if no shocks

        if shocks is not None:
            shock_list = self._shock_unpack(shocks)
            for idx, shock_vals in shock_list:
                if shock_vals.shape[0] != T:
                    raise ValueError(
                        f"Shock array for variable index {idx} must have length {T}."
                    )
                shock_mat[:, idx] = shock_scale * shock_vals

        X = np.zeros((T + 1, n), dtype=float64)
        X[0, :] = x0

        for t in range(T):
            X[t + 1] = self.A @ X[t] + self.B @ shock_mat[t]

        out = {name: X[:, self.compiled.idx[name]] for name in self.compiled.var_names}
        out["_X"] = X  # Include full state matrix for reference

        if observables:
            Y = np.zeros((T + 1, len(self.compiled.observable_names)), dtype=float64)
            for i, func in enumerate(self.compiled.observable_funcs):
                for t in range(T + 1):
                    cur = X[t, :]
                    params = [
                        p for p in self.compiled.config.calibration.parameters.values()
                    ]
                    Y[t, i] = func(*cur, *params)

            for i, name in enumerate(self.compiled.observable_names):
                out[name] = Y[:, i]

        return out

    def irf(
        self, shocks: list[str], T: int, scale: float = 1.0, observables: bool = False
    ) -> dict[str, NDF]:
        """
        Compute impulse response functions for specified shocks over T periods.
        Parameters
        ----------
        shocks : list[str]
            List of shock variable names to apply the impulse to.

        T : int
            Number of time periods to simulate.

        scale : float, optional
            Scaling factor for the initial shock.

        observables : bool, optional
            If True, include observable variables in the output.

        Returns
        -------
        dict[str, ndarray]
            A dictionary mapping variable names to their impulse response time series.
        """

        if not shocks:
            raise ValueError("At least one shock must be specified for IRF.")
        if not all(
            s in self.compiled.var_names[: self.compiled.n_exog] for s in shocks
        ):
            raise ValueError("Shocked variable not found in exogenous model variables.")
        conf = self.compiled.config

        shock_spec = {}
        for s in shocks:
            sym = Symbol(f"sig_{s}")
            sig = conf.calibration.parameters.get(sym, 1.0)
            arr = np.zeros((T,), dtype=float64)
            arr[0] = sig
            shock_spec[s] = arr

        return self.sim(
            T,
            shocks=shock_spec,  # type: ignore
            shock_scale=scale,
            x0=np.zeros((self.A.shape[0],), dtype=float64),
            observables=observables,
        )

    def transition_plot(
        self, T: int, shocks: list[str], scale: float = 1.0, observables: bool = False
    ) -> None:
        """
        Plot impulse response functions for specified shocks over T periods.
        Parameters
        ----------
        T : int
            Number of time periods to simulate.

        shocks : list[str]
            List of shock variable names to apply the impulse to.

        scale : float, optional
            Scaling factor for the initial shock.

        observables : bool, optional
            If True, include observable variables in the plots.

        Returns
        -------
        None
        """

        transitions = self.irf(shocks=shocks, T=T, scale=scale, observables=observables)
        obs_vars = [v.name for v in self.compiled.config.observables]
        transitions.pop("_X", None)

        n_vars = len(transitions)
        fig_square = np.ceil(np.sqrt(n_vars))

        fig, ax = plt.subplots(
            int(fig_square), int(fig_square), figsize=(4 * fig_square, 3 * fig_square)
        )  # 4:3 aspect ratio
        ax = ax.flatten()
        time = np.arange(T + 1)  # +1 for initial state

        # Remove unused axes
        nplots = len(transitions)
        while nplots < len(ax):
            fig.delaxes(ax[-1])
            ax = ax[:-1]

        for i, (var, series) in enumerate(transitions.items()):
            title_kwargs = {}
            if var in obs_vars:
                title_kwargs = {"color": "blue", "style": "italic"}
            elif var in shocks:
                title_kwargs = (
                    {"color": "red", "weight": "bold"} if var in shocks else {}
                )

            ax[i].plot(time, series)
            ax[i].set_title(var, **title_kwargs)
            ax[i].set_xlabel("Time")
            ax[i].set_ylabel(rf"{var}")
            ax[i].grid(color="black", linestyle=":", alpha=0.33)
        plt.suptitle("Impulse Response Functions")
        plt.tight_layout()
        plt.show()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the SolvedModel dataclass to a dictionary.
        Returns
        -------
        dict[str, Any]
            A dictionary representation of the SolvedModel.
        """
        return asdict(self)

    def _build_cov(self, vars: Iterable[str]) -> NDF:
        sig = np.array([self._get_param(f"sig_{v}", 1.0) for v in vars])
        rho = np.array(
            [[self._get_param(f"rho_{v1}{v2}", 0.0) for v2 in vars] for v1 in vars]
        )
        cov = rho * np.outer(sig, sig)
        return cov

    def _shock_unpack(
        self, shocks: dict[str, NDF | Callable[[float | list[list[float]]], NDF]]
    ) -> list[Tuple[int, NDF]]:
        for name, shock in shocks.items():
            if "," in name:
                # Multi-Var
                multi_names = [n.strip() for n in name.split(",")]
                # All names must be exogenous and grouped names must not be re-allocated as univariate shocks
                assert all(
                    (n in self.compiled.var_names[: self.compiled.n_exog])
                    and (n not in shocks)
                    for n in multi_names
                )
                indices = [self.compiled.idx[n] for n in multi_names]

                if isinstance(shock, ndarray):
                    assert shock.shape[1] == len(
                        multi_names
                    ), f"Shock array for {name} must have shape (T, {len(multi_names)})"
                    return list(
                        zip(indices, [shock[:, i] for i in range(shock.shape[1])])
                    )

                elif callable(shock):
                    sigs = [self._get_param(f"sig_{n}", 1.0) for n in multi_names]
                    rhos = [
                        self._get_rho(n1, n2, 0.0)
                        for n1 in multi_names
                        for n2 in multi_names
                    ]
                    corr = np.array(rhos).reshape((len(multi_names), len(multi_names)))
                    cov = corr * np.outer(sigs, sigs)
                    mv_mat = shock(cov)
                    return list(
                        zip(indices, [mv_mat[:, i] for i in range(mv_mat.shape[1])])
                    )
                else:
                    raise TypeError(
                        f"Shock for {name} must be a callable or ndarray, got {type(shock)}."
                    )
            else:
                # Uni-Var
                if name not in self.compiled.var_names[: self.compiled.n_exog]:
                    raise ValueError(
                        f"Shock variable {name} not found in exogenous model variables."
                    )
                idx = self.compiled.idx[name]
                if callable(shock):
                    sig = self._get_param(f"sig_{name}", 1.0)
                    shock_vals = shock(sig)
                    return [(idx, shock_vals)]
                elif isinstance(shock, ndarray):
                    shock_vals = asarray(shock, dtype=float64)
                    return [(idx, shock_vals)]
                else:
                    raise TypeError(
                        f"Shock for {name} must be a callable or ndarray, got {type(shock)}."
                    )
        raise ValueError("No shocks provided.")

    def _get_rho(self, var1: str, var2: str, default: float = 0.0) -> float:
        """
        Retrieve the correlation coefficient between two variables from the calibration parameters.
        Parameters
        ----------
        var1 : str
            The name of the first variable.
        var2 : str
            The name of the second variable.
        default : float, optional
            The default value to return if the correlation parameter is not found.
        Returns
        -------
        float
            The correlation coefficient between var1 and var2.
        """
        params = self.compiled.config.calibration.parameters
        if var1 == var2:
            return 1.0

        sym = Symbol(f"rho_{var1}{var2}")
        if sym in params:
            return float64(params[sym])
        sym_rev = Symbol(f"rho_{var2}{var1}")
        if sym_rev in params:
            return float64(params[sym_rev])
        return float64(default)

    def _get_param(self, name: str, default: float = None) -> float:
        """
        Retrieve a parameter value by name from the calibration parameters.
        Parameters
        ----------
        name : str
            The name of the parameter to retrieve.
        default : float, optional
            The default value to return if the parameter is not found.
        Returns
        -------
        float
            The value of the parameter.
        """
        params = self.compiled.config.calibration.parameters
        sym = Symbol(name)
        if sym in params:
            return float64(params[sym])
        elif default is not None:
            return float64(default)
        raise KeyError(f"Parameter '{name}' not found in calibration parameters.")

    def _build_measurement(
        self, spec: dict[str, MeasurementSpec]
    ) -> Tuple[NDF, NDF, list[str]]:
        n = self.A.shape[0]
        obs_names = list(spec.keys())
        m = len(obs_names)

        C = np.zeros((m, n), dtype=float64)
        d = np.zeros((m,), dtype=float64)

        for i, obs in enumerate(obs_names):
            row: MeasurementSpec = spec[obs]
            lin = row.get("lin", {})
            const = row.get("const", [])
            for varname, coef in lin.items():
                j = self.compiled.idx.get(varname)
                if j is None:
                    raise KeyError(
                        f"Variable '{varname}' not found in model variables."
                    )
                C[i, j] += float64(coef)

            for c in const:
                if isinstance(c, str):
                    d[i] += self._get_param(c)
                else:
                    d[i] += float64(c)
        return C, d, obs_names


class DSGESolver:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.t = sp.Symbol("t", integer=True)

    def compile(
        self,
        *,
        variable_order: list[Function] = None,
        n_state: int = None,
        n_exog: int = None,
        params_order: list[str] = None,
    ) -> CompiledModel:

        conf = self.model_config
        t = self.t

        # Convert model to minimization problem
        obj = [sp.simplify(eq.lhs - eq.rhs) for eq in conf.equations.model]

        shifted = [self._offset_lags(o, t) for o in obj]

        # Deterministic var order
        if not variable_order:
            var_order: list[str] = [
                v.func.__name__ if hasattr(v, "func") else v.__name__
                for v in conf.variables
            ]
            var_order = [v.__name__ for v in conf.variables]
        else:
            var_order = [
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

            subs_map[f(t)] = cur  # noqa
            subs_map[f(t + 1)] = fwd  # noqa

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
            sp.lambdify([*cur_syms, *params], expr, modules="numpy")
            for expr in observable_exprs
        ]

        return CompiledModel(
            config=conf,
            var_names=var_order,
            idx=idx,
            objective_eqs=compiled,
            objective_funcs=funcs,
            equations=equations,
            observable_names=[v.name for v in conf.observables],
            observable_eqs=observable_exprs,
            observable_funcs=observable_funcs,
            n_state=int(n_state),
            n_exog=int(n_exog),
        )

    def solve(
        self,
        compiled: CompiledModel,
        *,
        parameters: dict[str, float] = None,
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
                pass

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
