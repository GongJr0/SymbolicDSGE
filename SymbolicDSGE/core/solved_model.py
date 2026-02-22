from dataclasses import dataclass, asdict
from typing import Any, Callable, Tuple, Union, Literal, TypedDict
import textwrap


import numpy as np
from numpy import ndarray, float64, asarray
from numpy.typing import NDArray

from numba import njit
import pandas as pd
from sympy import Symbol

import matplotlib.pyplot as plt

from .compiled_model import CompiledModel
from .config import ModelConfig, SymbolGetterDict
from ..kalman.config import KalmanConfig
from ..kalman.interface import KalmanInterface
from ..kalman.filter import FilterResult

_JIT_CACHE: dict[int, Callable] = {}
ND = NDArray
NDF = NDArray[float64]


class MeasurementSpec(TypedDict):
    lin: dict[str, float | float64]
    const: list[float | float64 | str]


@dataclass(frozen=True)
class SolvedModel:
    compiled: CompiledModel
    policy: Any
    A: ndarray
    B: ndarray

    @property
    def config(self) -> ModelConfig:
        return self.compiled.config

    @property
    def kalman_config(self) -> KalmanConfig | None:
        return self.compiled.kalman

    def sim(
        self,
        T: int,
        shocks: (
            dict[str, Union[Callable[[float | list[list[float]]], NDF], NDF]] | None
        ) = None,
        shock_scale: float = 1.0,
        x0: ndarray | None = None,
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
        n = self.A.shape[0]

        if x0 is None:
            x0 = np.zeros((n,))
        x0 = asarray(x0, dtype=float64)
        n_state = self.compiled.n_state
        x0[n_state:] = x0[:n_state] @ np.real_if_close(self.policy.f.T)

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
            is_affine = self.config.equations.obs_is_affine
            if all(is_affine.values()):
                C, d = self._build_C_d_from_obs(self.compiled.observable_names)
                Y = X @ C.T + d
                for i, name in enumerate(self.compiled.observable_names):
                    out[name] = Y[:, i]
            else:
                Y = self._non_affine_measurement(self.compiled.observable_names, X)
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
        rev: SymbolGetterDict[Symbol, Symbol] = SymbolGetterDict(
            {v: k for k, v in self.config.shock_map.items()}
        )  # variable -> innovation
        sig_map = conf.calibration.shock_std
        for s in shocks:
            sym = rev[s]
            sig_sym = sig_map.get(sym)
            sig = conf.calibration.parameters.get(sig_sym, 1.0)  # pyright: ignore
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

    def _shock_unpack(
        self, shocks: dict[str, NDF | Callable[[float | list[list[float]]], NDF]]
    ) -> list[Tuple[int, NDF]]:
        out: list[Tuple[int, NDF]] = []

        conf = self.compiled.config
        reverse_shock_map: SymbolGetterDict[Symbol, Symbol] = SymbolGetterDict(
            {v: k for k, v in conf.shock_map.items()}
        )
        shock_stds = conf.calibration.shock_std

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
                perm = np.argsort(indices)
                multi_names_sorted = [multi_names[i] for i in perm]
                indices_sorted = [indices[i] for i in perm]

                if isinstance(shock, ndarray):
                    assert shock.shape[1] == len(
                        multi_names
                    ), f"Shock array for {name} must have shape (T, {len(multi_names)})"
                    shock_sorted = shock[:, perm]
                    out.extend(
                        zip(
                            indices_sorted,
                            [shock_sorted[:, i] for i in range(shock_sorted.shape[1])],
                        )
                    )

                elif callable(shock):
                    shock_syms = [reverse_shock_map[n] for n in multi_names_sorted]
                    sig_params = [shock_stds[sym] for sym in shock_syms]
                    sigs = [self._get_param(sig, 1.0) for sig in sig_params]
                    rhos = [
                        self._get_rho(n1, n2, 0.0)
                        for n1 in shock_syms
                        for n2 in shock_syms
                    ]
                    corr = np.array(rhos).reshape(
                        (len(multi_names_sorted), len(multi_names_sorted))
                    )
                    cov = corr * np.outer(sigs, sigs)

                    mv_mat = shock(cov)  # pyright: ignore
                    if mv_mat.shape[1] != len(multi_names):
                        raise ValueError(
                            f"Shock callable for {name} must return array with shape (T, {len(multi_names)})"
                        )
                    out.extend(
                        zip(
                            indices_sorted,
                            [mv_mat[:, i] for i in range(mv_mat.shape[1])],
                        )
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
                    sym = reverse_shock_map[name]
                    sig_param = shock_stds[sym]
                    sig = self._get_param(sig_param, 1.0)

                    shock_vals = shock(sig)
                    out.append((idx, shock_vals))
                elif isinstance(shock, ndarray):
                    shock_vals = asarray(shock, dtype=float64)
                    out.append((idx, shock_vals))
                else:
                    raise TypeError(
                        f"Shock for {name} must be a callable or ndarray, got {type(shock)}."
                    )
        return out

    def _get_rho(
        self, var1: str | Symbol, var2: str | Symbol, default: float = 0.0
    ) -> float:
        """
        Retrieve the correlation coefficient between two variables from the calibration parameters.
        Parameters
        ----------
        var1 : str | Symbol
            The name of the first variable.
        var2 : str | Symbol
            The name of the second variable.
        default : float, optional
            The default value to return if the correlation parameter is not found.
        Returns
        -------
        float
            The correlation coefficient between var1 and var2.
        """
        if var1 == var2:
            return 1.0

        conf = self.compiled.config.calibration
        corrs = conf.shock_corr

        corr = corrs[var1, var2]  # pyright: ignore # Overloaded __getitem__
        if corr is not None:
            return self._get_param(corr, default=default)

        return float64(default)

    def _get_param(self, name: str | Symbol, default: float | None = None) -> float:
        """
        Retrieve a parameter value by name from the calibration parameters.
        Parameters
        ----------
        name : str | Symbol
            The name of the parameter to retrieve.
        default : float, optional
            The default value to return if the parameter is not found.
        Returns
        -------
        float
            The value of the parameter.
        """
        params = self.compiled.config.calibration.parameters
        sym = Symbol(name) if isinstance(name, str) else name
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

    def _build_C_d_from_obs(
        self,
        y_names: list[str],
    ) -> Tuple[NDF, NDF]:

        obs_expr = dict(
            zip(self.compiled.observable_names, self.compiled.observable_eqs)
        )
        param_subs = {
            p: float64(v)
            for p, v in self.compiled.config.calibration.parameters.items()
        }

        m = len(y_names)
        n = self.A.shape[0]

        C = np.zeros((m, n), dtype=float64)
        d = np.zeros((m,), dtype=float64)

        zero_subs = {s: 0.0 for s in self.compiled.cur_syms}

        for i, y in enumerate(y_names):
            expr = obs_expr[y]

            # Constants
            d_i = expr.subs(zero_subs).subs(param_subs)  # pyright: ignore
            d[i] = float64(d_i)

            # Linear Coefficients
            for j, sym in enumerate(self.compiled.cur_syms):
                a = expr.coeff(sym)
                if a != 0:
                    a = a.subs(param_subs)  # pyright: ignore
                    C[i, j] = float64(a)

        return C, d

    @staticmethod
    def _make_jit_measurement(K: int):  # type: ignore # (Types getting misinterpreted with no static known arg count)
        f = _JIT_CACHE.get(K)
        if f is not None:
            return f

        # build source code with explicit call func(r[0],...,r[K-1])
        args = ", ".join([f"r[{j}]" for j in range(K)])
        src = f"""
        def _jit_measurement(func, args_mat):
            n_obs = args_mat.shape[0]
            out = np.empty(n_obs, dtype=np.float64)
            for i in range(n_obs):
                r = args_mat[i]
                out[i] = func({args})
            return out
        """
        src = textwrap.dedent(src)
        ns = {"np": np}
        exec(src, ns)
        f = njit(ns["_jit_measurement"])
        _JIT_CACHE[K] = f
        return f

    def _non_affine_measurement(
        self,
        y_names: list[str],
        state: NDF,
    ) -> NDF:
        obs_funcs = self.compiled.observable_funcs

        # Arguments in form [*cur_vars, *params]
        params = self.compiled.config.calibration.parameters
        param_vals = [
            float64(params[p]) for p in self.config.calibration.parameters
        ]  # Ensure dtype + order

        # Assume state is in canonical order of cur_syms (Checked at compile time)
        n_obs = len(y_names)
        T = state.shape[0]
        args_mat = np.empty(
            (T, len(self.compiled.cur_syms) + len(param_vals)), dtype=float64
        )
        param_block = np.ones((T, len(param_vals)), dtype=float64) * np.array(
            param_vals, dtype=float64
        )
        args_mat[:, : state.shape[1]] = state
        args_mat[:, state.shape[1] :] = param_block

        K = args_mat.shape[1]
        _jit_measurement = self._make_jit_measurement(K)

        out = np.empty((T, n_obs), dtype=float64)
        for i, y in enumerate(y_names):
            func = obs_funcs[self.compiled.observable_names.index(y)]
            out[:, i] = _jit_measurement(func, args_mat)  # pyright: ignore
        return out

    def kalman(
        self,
        y: NDF | pd.DataFrame,
        filter_mode: Literal["linear", "extended"] = "linear",
        *,
        observables: list[str] | None = None,
        x0: NDF | None = None,
        p0_mode: Literal["diag", "eye"] | None = None,
        p0_scale: float | float64 | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool | None = None,
        return_shocks: bool = False,
        _debug: bool = False,
    ) -> FilterResult:

        params = asarray(
            [self.config.calibration.parameters[p] for p in self.compiled.calib_params],
            dtype=float64,
        )

        h_func: Callable[..., NDF] | None = None
        H_jac: Callable[..., NDF] | None = None

        if filter_mode == "extended":
            h_func = self.compiled.construct_measurement_vector_func()
            H_jac = self.compiled.observable_jacobian

        ki = KalmanInterface(
            model=self,
            filter_mode=filter_mode,
            observables=observables,
            y=y,
            h_func=h_func,
            H_jac=H_jac,
            calib_params=params,
            p0_mode=p0_mode,
            p0_scale=p0_scale,
            jitter=jitter,
            symmetrize=symmetrize,
            return_shocks=return_shocks,
        )

        run = ki.filter(x0=x0, _debug=_debug)
        if _debug:
            print(ki._debug_info)
        return run
