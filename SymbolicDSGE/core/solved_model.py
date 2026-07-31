from dataclasses import dataclass, asdict
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Tuple,
    Union,
    Literal,
    TypedDict,
    Mapping,
    cast,
)


import numpy as np
from numpy import ndarray, float64, asarray
from numpy.typing import NDArray

import pandas as pd
from sympy import Symbol

import matplotlib.pyplot as plt

from .shock_generators import Shock, _gaussian_factor
from .shock_plan import ShockPlan, ShockPlanEntry, validate_shock_targets
from .solver_backend import KleinSolution, PerturbationSolution

from .compiled_model import CompiledModel
from .config import ModelConfig, SymbolGetterDict
from .._ckernels.core import (
    affine_observations_into,
    measurement_path,
    simulate_linear_states_into,
    simulate_second_order_pruned,
)
from ..kalman.config import KalmanConfig
from ..kalman.interface import KalmanInterface, _KFMatrices
from ..kalman.filter import (
    FilterRawResult,
    FilterResult,
    UnscentedFilterRawResult,
    UnscentedFilterResult,
    _filter_result_from_raw,
    _unscented_filter_result_from_raw,
)

if TYPE_CHECKING:
    from ..regression.sr.config import TemplateConfig
    from ..regression.sr.fit_result import FitResult
    from ..regression.sr.model_defaults import PySRParams
    from ..regression.sr.model_parametrizer import ModelParametrizer

ND = NDArray
NDF = NDArray[float64]


def _require_horizon(T: int | None, name: str) -> int:
    """A live ``Shock`` resolves its family against a horizon; demand one."""
    if T is None:
        raise ValueError(
            f"Shock spec {name!r} is a live Shock, so resolving it needs a "
            "horizon T. Pass T, or materialize the spec into a draw closure first."
        )
    return T


def _load_sr_fit_dependencies() -> tuple[type, type]:
    from ..regression.sr.model_parametrizer import ModelParametrizer
    from ..regression.sr.sr_interface import SRInterface

    return ModelParametrizer, SRInterface


class MeasurementSpec(TypedDict):
    lin: dict[str, float | float64]
    const: list[float | float64 | str]


@dataclass(frozen=True)
class SolvedModel:
    compiled: CompiledModel
    policy: KleinSolution | PerturbationSolution
    A: ndarray
    B: ndarray

    def __post_init__(self) -> None:
        if self.policy.order not in SIM_FUNC_DISPATCH:
            raise ValueError(
                f"Simulation for solution order {self.policy.order} is not implemented."
            )

    def sim(
        self,
        T: int,
        shocks: Mapping[str, Shock | Callable[[float | NDF], NDF] | NDF] | None = None,
        shock_scale: float = 1.0,
        x0: list[float] | ndarray | None = None,
        observables: bool = False,
    ) -> dict[str, NDF]:
        """
        Simulate the solved DSGE model over T periods.
        Parameters
        ----------
        T : int
            Number of time periods to simulate.

        shocks : Mapping[str, Shock | Callable[[float], ndarray] | ndarray], optional
            Maps each exogenous variable name to its shock. A ``"a,b"`` key is a
            joint (multivar) shock over those variables. Each value may be a
            :class:`Shock` distribution spec (materialized into a ``T``-horizon
            draw here), a ``callable`` taking the shock scale and returning a
            ``(T,)``/``(T, k)`` array, or a raw ndarray path of that shape. When
            ``None``, all shocks are zero.

        shock_scale : float, optional
            A scaling factor applied to all shocks.

        x0 : list[float] | ndarray, optional
            Initial state vector. If None, defaults to zero vector.

        observables : bool, optional
            If True, compute and include observable variables in the output.

        Returns
        -------

        dict[str, ndarray]
            A dictionary mapping variable names to their simulated time series.
        """
        X = self._simulate_state_matrix(
            T=T,
            shocks=shocks,
            shock_scale=shock_scale,
            x0=asarray(x0, dtype=float64) if x0 is not None else None,
        )

        out = {name: X[:, self.compiled.idx[name]] for name in self.compiled.var_names}
        out["_X"] = X  # Include full state matrix for reference

        if observables:
            Y = self._simulate_observable_matrix(X, drop_initial=False)
            for i, name in enumerate(self.compiled.observable_names):
                out[name] = Y[:, i]

        return out

    def _simulation_initial_state(self, x0: ndarray | None = None) -> NDF:
        n = self.A.shape[0]
        n_state = self.compiled.n_state
        if x0 is None:
            x0_arr = np.zeros((n,), dtype=float64)
        else:
            raw = asarray(x0, dtype=float64)
            if raw.shape[0] == n:
                x0_arr = raw.copy()
            elif raw.shape[0] == n_state:
                x0_arr = np.zeros((n,), dtype=float64)
                x0_arr[:n_state] = raw
            else:
                raise ValueError(
                    f"x0 must have length {n_state} or {n}, got {raw.shape[0]}."
                )
        x0_arr[n_state:] = x0_arr[:n_state] @ np.real_if_close(self.policy.f.T)
        return x0_arr

    @staticmethod
    def _materialize_shocks(
        shocks: Mapping[str, Shock | Callable[[float | NDF], NDF] | NDF],
        T: int,
    ) -> dict[str, Callable[[float | NDF], NDF] | NDF]:
        """Resolve any ``Shock`` specs into their ``T``-horizon draw closures.

        Live callables and raw arrays pass through untouched. Callers that hold
        the specs themselves can hand them to ``_resolve_shock_plan`` directly,
        which resolves the family without going through a closure.
        """
        return {
            name: shock.shock_generator(T) if isinstance(shock, Shock) else shock
            for name, shock in shocks.items()
        }

    def _simulation_shock_matrix(
        self,
        T: int,
        shocks: (
            Mapping[str, Shock | Union[Callable[[float | NDF], NDF], NDF]] | None
        ) = None,
        shock_scale: float = 1.0,
    ) -> NDF:
        if shocks is None:
            return np.zeros((T, self.compiled.n_exog), dtype=float64)
        return self._resolve_shock_plan(shocks, T).matrix(T, shock_scale)

    def _simulate_state_matrix(
        self,
        T: int,
        shocks: (
            Mapping[str, Shock | Union[Callable[[float | NDF], NDF], NDF]] | None
        ) = None,
        shock_scale: float = 1.0,
        x0: ndarray | None = None,
    ) -> NDF:
        return SIM_FUNC_DISPATCH[self.policy.order](self, T, shocks, shock_scale, x0)

    def _simulate_observable_matrix(
        self,
        states: NDF,
        *,
        drop_initial: bool = False,
    ) -> NDF:
        start = 1 if drop_initial else 0
        y_names = self.compiled.observable_names
        is_affine = self.config.equations.obs_is_affine
        if all(is_affine.values()):
            C, d = self._build_C_d_from_obs(y_names)
            Y = np.empty((states.shape[0] - start, len(y_names)), dtype=float64)
            affine_observations_into(states, C, d, Y)
            return Y

        Y = self._non_affine_measurement(y_names, states)
        return np.ascontiguousarray(Y[start:], dtype=float64)

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

        out = self.sim(
            T,
            shocks=shock_spec,
            shock_scale=scale,
            x0=None,
            observables=observables,
        )
        if self.policy.order != 2:
            return out

        baseline = self.sim(
            T,
            shocks=None,
            shock_scale=scale,
            x0=None,
            observables=observables,
        )
        return {key: value - baseline[key] for key, value in out.items()}

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
        time = np.arange(T)

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

    def serve(
        self,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        open_browser: bool = True,
    ) -> None:
        """Launch the SymbolicDSGE web playground with this model preloaded.

        Serves the bundled UI and opens a browser, with this model loaded as
        the ``reference`` model. Requires the optional UI dependencies::

            pip install 'SymbolicDSGE[ui]'

        Parameters
        ----------
        host, port:
            Bind address; ``port`` defaults to an available port.
        open_browser:
            Whether to open a browser window automatically.
        """
        try:
            from ..ui.serve import serve_from
        except ImportError as exc:  # pragma: no cover - exercised without [ui]
            raise ImportError(
                "The SymbolicDSGE UI extra is required for .serve(). "
                "Install it with: pip install 'SymbolicDSGE[ui]'"
            ) from exc

        serve_from(
            source=self,
            host=host,
            port=port,
            open_browser=open_browser,
        )

    def to_bundle_builder(
        self,
        *,
        yaml_text: str | None = None,
        role: str = "reference",
        compile_kwargs: Mapping[str, Any] | None = None,
        solve_kwargs: Mapping[str, Any] | None = None,
        created_by: str | None = None,
    ) -> "Any":
        """Return a :class:`BundleBuilder` pre-seeded with this model's YAML.

        Chain estimation/MC/simulation members and call ``.write(path)``::

            solved.to_bundle_builder().add_estimation(spec, ...).write("out.sdsge")

        ``yaml_text`` overrides the YAML embedded in the bundle; if not given,
        the source YAML retained on :attr:`compiled.config.source_yaml` is used.
        Raises :class:`ValueError` when neither is available (e.g. for models
        built programmatically without parsing a YAML).
        """
        from ..bundle.builder import BundleBuilder

        yaml = yaml_text if yaml_text is not None else self.compiled.config.source_yaml
        if yaml is None:
            raise ValueError(
                "Cannot create a .sdsge bundle: this model has no source YAML "
                "attached. Pass yaml_text=... explicitly, or load the model via "
                "ModelParser(path) / ModelParser.from_string(text) so the "
                "source is retained on compiled.config.source_yaml."
            )
        return BundleBuilder(created_by=created_by).add_model(
            role,
            yaml,
            compile_kwargs=compile_kwargs,
            solve_kwargs=solve_kwargs,
        )

    def save_sdsge(
        self,
        path: "str | Any",
        *,
        yaml_text: str | None = None,
        role: str = "reference",
        compile_kwargs: Mapping[str, Any] | None = None,
        solve_kwargs: Mapping[str, Any] | None = None,
    ) -> "Any":
        """Write a model-only ``.sdsge`` bundle at ``path``.

        Shortcut for ``self.to_bundle_builder(...).write(path)``. For bundles
        that also carry estimation / Monte-Carlo / simulation members, call
        :meth:`to_bundle_builder` directly and chain the additions.
        """
        return self.to_bundle_builder(
            yaml_text=yaml_text,
            role=role,
            compile_kwargs=compile_kwargs,
            solve_kwargs=solve_kwargs,
        ).write(path)

    def _resolve_shock_plan(
        self,
        shocks: Mapping[str, Shock | NDF | Callable[[float | NDF], NDF]],
        T: int | None = None,
    ) -> ShockPlan:
        """Resolve a shock spec against this model, once.

        Everything here depends only on the model and the spec: target column
        indices, the canonical order of a grouped entry, the standard deviations
        and correlations read off the calibration, the assembled covariance, and
        its Cholesky factor. None of it depends on the seed, so a caller drawing
        many paths from one spec resolves a plan and redraws from it.

        ``T`` is required only when the mapping carries live :class:`Shock`
        specs, which resolve their distribution family against a horizon.
        """
        conf = self.compiled.config
        reverse_shock_map: SymbolGetterDict[Symbol, Symbol] = SymbolGetterDict(
            {v: k for k, v in conf.shock_map.items()}
        )
        shock_stds = conf.calibration.shock_std

        exog_names = list(self.compiled.var_names[: self.compiled.n_exog])
        validate_shock_targets(list(shocks), exog_names)

        entries: list[ShockPlanEntry] = []
        seeded_count = 0

        for name, shock in shocks.items():
            if isinstance(shock, Shock) and shock.seed is not None:
                seeded_count += 1

            if "," in name:
                multi_names = [n.strip() for n in name.split(",")]
                indices = [self.compiled.idx[n] for n in multi_names]
                perm = np.argsort(indices)
                multi_names_sorted = [multi_names[i] for i in perm]
                indices_sorted = tuple(indices[i] for i in perm)

                if isinstance(shock, ndarray):
                    assert shock.shape[1] == len(
                        multi_names
                    ), f"Shock array for {name} must have shape (T, {len(multi_names)})"
                    entries.append(
                        ShockPlanEntry(
                            key=name,
                            indices=indices_sorted,
                            multivar=True,
                            values=asarray(shock[:, perm], dtype=float64),
                        )
                    )
                    continue

                if not isinstance(shock, Shock) and not callable(shock):
                    raise TypeError(
                        f"Shock for {name} must be a callable or ndarray, got {type(shock)}."
                    )

                shock_syms = [reverse_shock_map[n] for n in multi_names_sorted]
                sig_params = [shock_stds[sym] for sym in shock_syms]
                sigs = [self._get_param(sig, 1.0) for sig in sig_params]
                rhos = [
                    self._get_rho(n1, n2, 0.0) for n1 in shock_syms for n2 in shock_syms
                ]
                corr = np.array(rhos).reshape(
                    (len(multi_names_sorted), len(multi_names_sorted))
                )
                cov = corr * np.outer(sigs, sigs)

                if isinstance(shock, Shock):
                    entries.append(
                        ShockPlanEntry(
                            key=name,
                            indices=indices_sorted,
                            multivar=True,
                            scale=cov,
                            factor=_gaussian_factor(cov),
                            draw=shock.draw_fn(_require_horizon(T, name)),
                            base_seed=(None if shock.seed is None else int(shock.seed)),
                            spec=shock,
                        )
                    )
                else:
                    entries.append(
                        ShockPlanEntry(
                            key=name,
                            indices=indices_sorted,
                            multivar=True,
                            scale=cov,
                            func=shock,
                        )
                    )
                continue

            # Uni-Var (target validity already checked by validate_shock_targets)
            idx = (self.compiled.idx[name],)
            if isinstance(shock, ndarray):
                entries.append(
                    ShockPlanEntry(
                        key=name,
                        indices=idx,
                        multivar=False,
                        values=asarray(shock, dtype=float64),
                    )
                )
                continue

            if not isinstance(shock, Shock) and not callable(shock):
                raise TypeError(
                    f"Shock for {name} must be a callable or ndarray, got {type(shock)}."
                )

            sym = reverse_shock_map[name]
            sig = self._get_param(shock_stds[sym], 1.0)

            if isinstance(shock, Shock):
                entries.append(
                    ShockPlanEntry(
                        key=name,
                        indices=idx,
                        multivar=False,
                        scale=sig,
                        draw=shock.draw_fn(_require_horizon(T, name)),
                        base_seed=None if shock.seed is None else int(shock.seed),
                        spec=shock,
                    )
                )
            else:
                entries.append(
                    ShockPlanEntry(
                        key=name,
                        indices=idx,
                        multivar=False,
                        scale=sig,
                        func=shock,
                    )
                )

        return ShockPlan(
            entries=tuple(entries),
            n_exog=self.compiled.n_exog,
            seeded_count=seeded_count,
        )

    def _shock_unpack(
        self, shocks: Mapping[str, NDF | Callable[[float | NDF], NDF]]
    ) -> list[Tuple[int, NDF]]:
        """Resolve a spec and draw it once, as ``(exogenous index, column)``."""
        return self._resolve_shock_plan(shocks).unpack()

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
        key = (tuple(y_names), self._calibration_fingerprint())
        hit = self._cd_cache.get(key)
        if hit is not None:
            return hit

        result = self.compiled.build_affine_measurement_matrices(
            self.compiled.config.calibration.parameters,
            y_names,
            self.policy.steady_state,
        )
        self._cd_cache[key] = result
        return result

    def _non_affine_measurement(
        self,
        y_names: list[str],
        state: NDF,
    ) -> NDF:
        # ``state`` is (T, n_var) in cur_syms canonical order (checked at compile).
        params = self.compiled.config.calibration.parameters
        param_vals = np.array(
            [float64(params[p]) for p in self.compiled.calib_params],
            dtype=float64,
        )

        # The measurement cfunc emits observables sorted by model index; map its
        # output columns back to the caller's y_names order.
        obs_sorted = self.compiled._normalize_observables(y_names)
        meas_addr = self.compiled.construct_measurement_cfunc(y_names).address
        raw = measurement_path(meas_addr, state, param_vals, len(obs_sorted))

        pos = {name: j for j, name in enumerate(obs_sorted)}
        perm = [pos[name] for name in y_names]
        return raw[:, perm]

    def kalman(
        self,
        y: NDF | pd.DataFrame,
        filter_mode: Literal["linear", "extended", "unscented"] = "linear",
        *,
        observables: list[str] | None = None,
        x0: NDF | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool | None = None,
        return_shocks: bool = False,
        P0: NDF | None = None,
        R: NDF | None = None,
        _debug: bool = False,
    ) -> FilterResult | UnscentedFilterResult:
        raw = self._kalman_raw(
            y=y,
            filter_mode=filter_mode,
            observables=observables,
            x0=x0,
            jitter=jitter,
            symmetrize=symmetrize,
            return_shocks=return_shocks,
            P0=P0,
            R=R,
            _debug=_debug,
        )
        if isinstance(raw, UnscentedFilterRawResult):
            return _unscented_filter_result_from_raw(raw)
        return _filter_result_from_raw(raw)

    def _kalman_raw(
        self,
        y: NDF | pd.DataFrame,
        filter_mode: Literal["linear", "extended", "unscented"] = "linear",
        *,
        observables: list[str] | None = None,
        x0: NDF | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool | None = None,
        return_shocks: bool = False,
        P0: NDF | None = None,
        R: NDF | None = None,
        _debug: bool = False,
    ) -> FilterRawResult | UnscentedFilterRawResult:
        params = asarray(
            [self.config.calibration.parameters[p] for p in self.compiled.calib_params],
            dtype=float64,
        )

        meas_addr: int | None = None
        jac_addr: int | None = None

        if filter_mode in {"extended", "unscented"}:
            obs_idx = {name: i for i, name in enumerate(self.compiled.observable_names)}
            if observables is None:
                selected_obs = list(self.compiled.observable_names)
            else:
                selected_obs = list(observables)
            selected_obs = sorted(selected_obs, key=lambda name: obs_idx[name])

            meas_addr = self.compiled.construct_measurement_cfunc(selected_obs).address
            jac_addr = self.compiled.construct_observable_jacobian_cfunc(
                selected_obs
            ).address

        if filter_mode == "unscented":
            if return_shocks:
                raise ValueError(
                    "return_shocks is not supported for unscented filtering."
                )
            if self.policy.order != 2:
                raise ValueError(
                    "Unscented Kalman Filter requires a second order solution."
                )
        ki = KalmanInterface(
            model=self,
            filter_mode=filter_mode,
            observables=observables,
            y=y,
            P0=P0,
            R=R,
            meas_addr=meas_addr,
            jac_addr=jac_addr,
            calib_params=params,
            jitter=jitter,
            symmetrize=symmetrize,
            return_shocks=return_shocks,
        )

        run = ki.filter_raw(x0=x0, _debug=_debug)
        if _debug:
            print(ki._debug_info)
        return run

    def fit_kf(
        self,
        y: NDF | pd.DataFrame,
        observable: str,
        template_config: "TemplateConfig | None" = None,
        sr_params: "PySRParams | None" = None,
        variables: list[str] | None = None,
        parametrizer: "ModelParametrizer | None" = None,
    ) -> "FitResult":
        if parametrizer is None:
            if template_config is None or sr_params is None:
                raise ValueError(
                    "Provide either a pre-built parametrizer or both template_config and sr_params."
                )
            ModelParametrizer, SRInterface = _load_sr_fit_dependencies()
            parametrizer = ModelParametrizer(
                variables or self.compiled.var_names,
                sr_params,
                template_config,
            )
        elif variables is not None and set(variables) != set(
            parametrizer.variable_names
        ):
            raise ValueError(
                "Provided variables do not match the parametrizer's variable names."
            )
        else:
            _, SRInterface = _load_sr_fit_dependencies()

        interface = SRInterface(
            model=self,
            obs_name=observable,
            parametrizer=parametrizer,
        )

        return cast("FitResult", interface.fit_to_kf(y))

    def _calibration_fingerprint(self) -> int:
        """Hashable snapshot of calibration parameter values."""

        params = self.config.calibration.parameters
        return hash((tuple(params.keys()), tuple(float(v) for v in params.values())))

    def _kf_cache_get(self, key: tuple) -> _KFMatrices | None:
        """Cached Kalman matrices for ``key``, or ``None`` on miss."""

        return self._kf_cache.get(key)

    def _kf_cache_put(self, key: tuple, matrices: _KFMatrices) -> None:
        """Store Kalman matrices for ``key`` in the cache."""

        self._kf_cache[key] = matrices

    def clear_kf_cache(self) -> None:
        """Drop cached Kalman matrices."""
        self._kf_cache.clear()
        self._cd_cache.clear()

    @property
    def config(self) -> ModelConfig:
        return self.compiled.config

    @property
    def kalman_config(self) -> KalmanConfig | None:
        return self.compiled.kalman

    @cached_property
    def _cd_cache(self) -> dict[tuple, Tuple[NDF, NDF]]:
        return {}

    @cached_property
    def _kf_cache(self) -> dict[tuple, _KFMatrices]:
        return {}


SimFn = Callable[
    [
        SolvedModel,
        int,
        Mapping[str, Shock | Union[Callable[[float | NDF], NDF], NDF]] | None,
        float,
        ndarray | None,
    ],
    NDF,
]


def _simulate_order1(
    model: SolvedModel,
    T: int,
    shocks: Mapping[str, Shock | Union[Callable[[float | NDF], NDF], NDF]] | None,
    shock_scale: float,
    x0: ndarray | None,
) -> NDF:
    x0_arr = model._simulation_initial_state(x0)
    shock_mat = model._simulation_shock_matrix(
        T=T,
        shocks=shocks,
        shock_scale=shock_scale,
    )
    X = np.empty((T, model.A.shape[0]), dtype=float64)
    simulate_linear_states_into(
        asarray(model.A, dtype=float64),
        asarray(model.B, dtype=float64),
        x0_arr,
        shock_mat,
        X,
    )
    return X


def _policy_array(policy: Any, name: str) -> NDF:
    value = getattr(policy, name, None)
    if value is None:
        raise ValueError(f"Second order simulation requires policy.{name}.")
    return asarray(value, dtype=float64)


def _simulate_order2(
    model: SolvedModel,
    T: int,
    shocks: Mapping[str, Shock | Union[Callable[[float | NDF], NDF], NDF]] | None,
    shock_scale: float,
    x0: ndarray | None,
) -> NDF:
    n_state = model.compiled.n_state
    n = model.A.shape[0]
    ny = n - n_state
    policy = model.policy
    ss = _policy_array(policy, "steady_state")
    ss_state = ss[:n_state]

    if x0 is None:
        x0_state = ss_state
    else:
        x0_state = model._simulation_initial_state(x0)[:n_state]
    x0_dev = asarray(x0_state - ss_state, dtype=float64)
    shock_mat = model._simulation_shock_matrix(
        T=T,
        shocks=shocks,
        shock_scale=shock_scale,
    )

    x_path, y_path = simulate_second_order_pruned(
        asarray(np.real_if_close(policy.p), dtype=float64),
        asarray(np.real_if_close(policy.f), dtype=float64),
        asarray(model.B[:n_state, :], dtype=float64),
        _policy_array(policy, "hxx"),
        _policy_array(policy, "gxx"),
        _policy_array(policy, "hss"),
        _policy_array(policy, "gss"),
        x0_dev,
        shock_mat,
    )

    X = np.empty((T, n), dtype=float64)
    X[:, :n_state] = x_path + ss_state
    if ny > 0:
        X[:, n_state:] = y_path + ss[n_state:]
    return X


SIM_FUNC_DISPATCH: dict[int, SimFn] = {
    1: _simulate_order1,
    2: _simulate_order2,
}
