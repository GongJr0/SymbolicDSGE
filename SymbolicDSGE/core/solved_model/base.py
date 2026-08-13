from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Tuple,
    TypeVar,
    Generic,
    Union,
    Literal,
    Mapping,
    cast,
)


import numpy as np
from numpy import ndarray, float64, asarray
from numpy.typing import NDArray

import pandas as pd
from sympy import Symbol

import matplotlib.pyplot as plt


from . import export, measurement

from ..shock_generators import Shock
from ..solver_backend import BaseSolution
from ..sim_result import StatePath, SimResult

from ..compiled_model import CompiledModel
from ..config import ModelConfig
from ...kalman.config import KalmanConfig
from ...kalman.interface import KalmanInterface, _KFMatrices
from ...kalman.filter import (
    FilterRawResult,
    FilterResult,
    UnscentedFilterRawResult,
    UnscentedFilterResult,
    _filter_result_from_raw,
    _unscented_filter_result_from_raw,
)

if TYPE_CHECKING:
    from ...regression.sr.config import TemplateConfig
    from ...regression.sr.fit_result import FitResult
    from ...regression.sr.model_defaults import PySRParams
    from ...regression.sr.model_parametrizer import ModelParametrizer

ND = NDArray
NDF = NDArray[float64]
Policy = TypeVar("Policy", bound=BaseSolution)


def _load_sr_fit_dependencies() -> tuple[type, type]:
    from ...regression.sr.model_parametrizer import ModelParametrizer
    from ...regression.sr.sr_interface import SRInterface

    return ModelParametrizer, SRInterface


class SolvedModel(ABC, Generic[Policy]):
    def __init__(self, compiled: CompiledModel, policy: Policy) -> None:
        self.compiled = compiled
        self.policy = policy

    @abstractmethod
    def _simulate_state_matrix(
        self,
        T: int,
        shocks: (
            Mapping[str, Shock | Union[Callable[[float | NDF], NDF], NDF]] | None
        ) = None,
        shock_scale: float = 1.0,
        x0: list[float] | ndarray | None = None,
    ) -> StatePath: ...

    def sim(
        self,
        T: int,
        shocks: Mapping[str, Shock | Callable[[float | NDF], NDF] | NDF] | None = None,
        shock_scale: float = 1.0,
        x0: list[float] | ndarray | None = None,
        observables: bool = False,
    ) -> SimResult:
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
            Initial state, in levels, of length ``n_state`` or ``n_var``. If
            None, the model starts at its steady state.

        observables : bool, optional
            If True, compute and include observable variables in the output.

        Returns
        -------
        SimResult
            The simulated path in levels, with each variable's series available
            by name.
        """
        path = self._simulate_state_matrix(
            T=T,
            shocks=shocks,
            shock_scale=shock_scale,
            x0=x0,
        )
        return self._assemble_simulation(path, observables=observables)

    def irf(
        self, shocks: list[str], T: int, scale: float = 1.0, observables: bool = False
    ) -> SimResult:
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
        unknown = [s for s in shocks if s not in self.compiled.shock_names]
        if unknown:
            raise ValueError(
                f"Unknown shock(s) {unknown}. Model shocks: "
                f"{list(self.compiled.shock_names)}."
            )
        conf = self.compiled.config

        shock_spec = {}
        sig_map = conf.calibration.shock_std
        for s in shocks:
            sig_sym = sig_map.get(Symbol(s))
            sig = conf.calibration.parameters.get(sig_sym, 1.0)  # pyright: ignore
            arr = np.zeros((T,), dtype=float64)
            arr[0] = sig
            shock_spec[s] = arr

        X, regimes, diagnostics = self._simulate_state_matrix(
            T=T, shocks=shock_spec, shock_scale=scale, x0=None
        )
        base_X, _, _ = self._simulate_state_matrix(
            T,
            shocks=None,
            shock_scale=scale,
            x0=None,
        )

        y = None
        if observables:
            y = self._simulate_observable_matrix(X, drop_initial=False)
            y -= self._simulate_observable_matrix(base_X, drop_initial=False)

        X -= base_X

        # The regimes are the shocked run's: the baseline's say what binds with
        # no shock, which is not what the response is about.
        return SimResult(
            var_names=self.compiled.var_names,
            X=X,
            observable_names=self.compiled.observable_names if observables else (),
            y=y,
            _regimes=regimes,
            _diagnostics=diagnostics,
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

        tr = self.irf(shocks=shocks, T=T, scale=scale, observables=observables)
        obs_vars = [v.name for v in self.compiled.config.observables]

        n_vars = tr.X.shape[1] + (tr.y.shape[1] if tr.y is not None else 0)
        fig_square = np.ceil(np.sqrt(n_vars))

        fig, ax = plt.subplots(
            int(fig_square), int(fig_square), figsize=(4 * fig_square, 3 * fig_square)
        )  # 4:3 aspect ratio
        ax = ax.flatten()
        time = np.arange(T)

        # Remove unused axes
        while n_vars < len(ax):
            fig.delaxes(ax[-1])
            ax = ax[:-1]

        vars = tr.states | (tr.observables if tr.y is not None else {})
        for i, (var, series) in enumerate(vars.items()):
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
        export.serve(self, host=host, port=port, open_browser=open_browser)

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
        return export.to_bundle_builder(
            self.compiled,
            yaml_text=yaml_text,
            role=role,
            compile_kwargs=compile_kwargs,
            solve_kwargs=solve_kwargs,
            created_by=created_by,
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

    def _simulation_initial_state(self, x0: list[float] | ndarray | None = None) -> NDF:
        """``x0`` in levels, converted to deviations and widened to the full ``n_var`` layout.

        A caller states an initial condition the way it reads the result, so
        ``x0`` arrives in levels and everything downstream runs in deviations.
        ``None`` is the steady state, which is zero once converted.

        A policy-dependent projection happens, therefore the caller must pass
        the relevant policty matrix: ``policy.f`` for a perturbation solve, or
        ``f_ref`` for a piecewise one. Everything else is the model's
        variable layout.
        """
        n = self.compiled.n_var
        n_state = self.compiled.n_state
        if x0 is None:
            x0_arr = np.zeros((n,), dtype=float64)
        else:
            raw = asarray(x0, dtype=float64)
            if raw.shape[0] == n:
                x0_arr = raw - self.policy.steady_state
            elif raw.shape[0] == n_state:
                x0_arr = np.zeros((n,), dtype=float64)
                x0_arr[:n_state] = raw - self.policy.steady_state[:n_state]
            else:
                raise ValueError(
                    f"x0 must have length {n_state} or {n}, got {raw.shape[0]}."
                )
        x0_arr[n_state:] = 0.0
        return x0_arr

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
            return measurement.affine_path(states, C, d, len(y_names), start)

        Y = measurement.non_affine_measurement(
            self.compiled, y_names, states + self.policy.steady_state
        )
        return np.ascontiguousarray(Y[start:], dtype=float64)

    def _assemble_simulation(self, path: StatePath, observables: bool) -> SimResult:
        X, regimes, diagnostics = path

        y = None
        if observables:
            y = self._simulate_observable_matrix(X, drop_initial=False)

        X += self.policy.steady_state  # add ss; user sees levels.

        return SimResult(
            var_names=self.compiled.var_names,
            X=X,
            observable_names=self.compiled.observable_names if observables else (),
            y=y,
            _regimes=regimes,
            _diagnostics=diagnostics,
        )

    def _build_C_d_from_obs(
        self,
        y_names: list[str],
    ) -> Tuple[NDF, NDF]:
        """``(C, d)`` for ``y_names``, memoized against the calibration."""
        key = (tuple(y_names), self.config.calibration.fingerprint())
        hit = self._cd_cache.get(key)
        if hit is not None:
            return hit

        result = measurement.build_C_d_from_obs(
            self.compiled, y_names, self.policy.steady_state
        )
        self._cd_cache[key] = result
        return result

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
