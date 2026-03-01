from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from numpy import asarray, float64
from numpy.typing import NDArray
from scipy import optimize

from ..bayesian.priors import Prior
from ..bayesian.transforms.identity import Identity
from ..bayesian.transforms.transform import Transform
from ..core.compiled_model import CompiledModel
from ..core.solver import DSGESolver
from . import backend
from .results import MCMCResult, OptimizationResult

NDF = NDArray[np.float64]


class Estimator:
    """
    Estimation interface exposing three public methods:
    - maximum likelihood estimation (`mle`)
    - maximum a posteriori estimation (`map`)
    - adaptive random-walk Metropolis MCMC (`mcmc`)
    """

    @staticmethod
    def make_prior(
        *,
        distribution: str,
        parameters: dict[str, Any],
        transform: str,
        transform_kwargs: dict[str, Any] | None = None,
    ) -> Prior:
        from ..bayesian.priors import make_prior as _make_prior

        return _make_prior(
            distribution=distribution,
            parameters=parameters,
            transform=transform,
            transform_kwargs=transform_kwargs,
        )

    def __init__(
        self,
        *,
        solver: DSGESolver,
        compiled: CompiledModel,
        y: NDF | pd.DataFrame,
        observables: list[str] | None = None,
        estimated_params: Sequence[str] | None = None,
        priors: Mapping[str, Prior] | None = None,
        steady_state: NDF | dict[str, float] | None = None,
        log_linear: bool = False,
        x0: NDF | None = None,
        p0_mode: str | None = None,
        p0_scale: float | float64 | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool | None = None,
        R: NDF | None = None,
    ) -> None:
        self.solver = solver
        self.compiled = compiled
        self.y = y
        self.observables = observables
        self.filter_mode = backend.infer_filter_mode(compiled, observables)

        self.priors = dict(priors) if priors is not None else None

        self.steady_state = steady_state
        self.log_linear = bool(log_linear)
        self.x0 = x0
        self.p0_mode = p0_mode
        self.p0_scale = p0_scale
        self.jitter = jitter
        self.symmetrize = symmetrize
        self.R = R

        self._base_params = backend.extract_base_params(compiled)
        if estimated_params is None:
            default_names = [backend._name_of(p) for p in compiled.calib_params]
            self.param_names = list(default_names)
        else:
            self.param_names = list(estimated_params)

        unknown = [p for p in self.param_names if p not in self._base_params]
        if unknown:
            raise ValueError(
                f"Unknown estimated parameters {unknown}. "
                f"Known calibration parameters: {list(self._base_params.keys())}"
            )
        self._param_index = {name: i for i, name in enumerate(self.param_names)}
        identity = Identity()
        self._param_transforms: dict[str, Transform] = {}
        for name in self.param_names:
            tr: Transform = identity
            if self.priors is not None and name in self.priors:
                prior_obj = self.priors[name]
                if hasattr(prior_obj, "transform"):
                    tr = cast(Transform, getattr(prior_obj, "transform"))
            self._param_transforms[name] = tr
        self._warning_signal_count = 0

    def theta0(self) -> NDF:
        constrained = asarray(
            [self._base_params[name] for name in self.param_names],
            dtype=float64,
        )
        return self.params_to_theta(constrained)

    def params_to_theta(self, params: Mapping[str, float] | NDF) -> NDF:
        if isinstance(params, Mapping):
            missing = [name for name in self.param_names if name not in params]
            if missing:
                raise ValueError(
                    f"Parameter mapping is missing estimated parameters: {missing}"
                )
            vals = asarray(
                [float64(params[name]) for name in self.param_names], dtype=float64
            )
        else:
            vals = asarray(params, dtype=float64)
            if vals.ndim != 1:
                raise ValueError("params array must be 1D.")
            if vals.shape[0] != len(self.param_names):
                raise ValueError(
                    f"params length {vals.shape[0]} does not match estimated parameter count {len(self.param_names)}."
                )
        out = np.empty_like(vals, dtype=float64)
        for i, name in enumerate(self.param_names):
            out[i] = float64(self._param_transforms[name].forward(float64(vals[i])))
        return out

    def theta_to_params(self, theta: NDF) -> dict[str, float64]:
        theta = asarray(theta, dtype=float64)
        if theta.ndim != 1:
            raise ValueError("theta must be a 1D array.")
        if theta.shape[0] != len(self.param_names):
            raise ValueError(
                f"theta length {theta.shape[0]} does not match estimated parameter count {len(self.param_names)}."
            )
        full = dict(self._base_params)
        for i, name in enumerate(self.param_names):
            full[name] = float64(
                self._param_transforms[name].inverse(float64(theta[i]))
            )
        return full

    def _loglik_from_params(
        self, params: Mapping[str, float64], R_override: NDF | None = None
    ) -> float64:
        return backend.evaluate_loglik(
            solver=self.solver,
            compiled=self.compiled,
            y=self.y,
            params=params,
            filter_mode=self.filter_mode,
            observables=self.observables,
            steady_state=self.steady_state,
            log_linear=self.log_linear,
            x0=self.x0,
            p0_mode=self.p0_mode,
            p0_scale=self.p0_scale,
            jitter=self.jitter,
            symmetrize=self.symmetrize,
            R=(self.R if R_override is None else R_override),
        )

    def _effective_observables(self) -> list[str]:
        canon = list(self.compiled.observable_names)
        canon_idx = {name: i for i, name in enumerate(canon)}
        kalman = self.compiled.kalman

        if self.observables is None:
            if kalman is not None and getattr(kalman, "y_names", None):
                obs_given = list(kalman.y_names)
            else:
                obs_given = list(canon)
        else:
            obs_given = list(self.observables)

        missing = [n for n in obs_given if n not in canon_idx]
        if missing:
            raise ValueError(f"Unknown observables not in compiled model: {missing}")
        return sorted(obs_given, key=lambda n: canon_idx[n])

    def loglik(self, theta: NDF) -> float64:
        params = self.theta_to_params(theta)
        return self._loglik_from_params(params)

    def logprior(self, theta: NDF) -> float64:
        if self.priors is None:
            return float64(0.0)
        theta = asarray(theta, dtype=float64)
        if theta.ndim != 1:
            raise ValueError("theta must be a 1D array.")
        if theta.shape[0] != len(self.param_names):
            raise ValueError(
                f"theta length {theta.shape[0]} does not match estimated parameter count {len(self.param_names)}."
            )
        lp = float64(0.0)
        for name, prior in self.priors.items():
            if name in self._param_index:
                z = float64(theta[self._param_index[name]])
            elif name in self._base_params:
                x0 = float64(self._base_params[name])
                if hasattr(prior, "transform"):
                    z = float64(
                        cast(Transform, getattr(prior, "transform")).forward(x0)
                    )
                else:
                    z = x0
            else:
                raise KeyError(f"Prior specified for unknown parameter '{name}'.")
            lp += float64(prior.logpdf(z))
        return lp

    def logpost(self, theta: NDF) -> float64:
        return float64(self.loglik(theta) + self.logprior(theta))

    def _logpost_with_overrides(
        self,
        theta: NDF,
        *,
        params_override: Mapping[str, float64] | None = None,
        R_override: NDF | None = None,
    ) -> float64:
        params = (
            params_override
            if params_override is not None
            else self.theta_to_params(theta)
        )
        return float64(
            self._loglik_from_params(params, R_override) + self.logprior(theta)
        )

    @staticmethod
    def _count_stdout_warning_signals(text: str) -> int:
        return sum(
            1 for line in text.splitlines() if line.strip().startswith("Warning")
        )

    def _eval_with_warning_capture(
        self, fn: Callable[[NDF], float64], theta: NDF
    ) -> tuple[float64, int]:
        stream = io.StringIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with redirect_stdout(stream):
                val = float64(fn(theta))
        n_signals = int(
            len(caught) + self._count_stdout_warning_signals(stream.getvalue())
        )
        return val, n_signals

    def _reset_search_warning_count(self) -> None:
        self._warning_signal_count = 0

    def _report_search_warning_count(self, kind: str) -> None:
        print(
            f"[Estimator:{kind}] warning signals encountered during search: {self._warning_signal_count}"
        )

    def _safe_loglik(self, theta: NDF) -> float64:
        try:
            ll, n_signals = self._eval_with_warning_capture(self.loglik, theta)
            self._warning_signal_count += n_signals
            if n_signals > 0 or not np.isfinite(ll):
                return float64(-np.inf)
            return float64(ll)
        except BaseException:
            return float64(-np.inf)

    def _safe_logpost(self, theta: NDF) -> float64:
        try:
            lp, n_signals = self._eval_with_warning_capture(
                lambda th: self._logpost_with_overrides(th), theta
            )
            self._warning_signal_count += n_signals
            if n_signals > 0 or not np.isfinite(lp):
                return float64(-np.inf)
            return float64(lp)
        except BaseException:
            return float64(-np.inf)

    def _safe_logprior(self, theta: NDF) -> float64:
        try:
            lp = float64(self.logprior(theta))
            if not np.isfinite(lp):
                return float64(-np.inf)
            return lp
        except BaseException:
            return float64(-np.inf)

    def _pack_opt_result(
        self, kind: str, res: optimize.OptimizeResult
    ) -> OptimizationResult:
        x = asarray(res.x, dtype=float64)
        theta = self.theta_to_params(x)

        ll = self._safe_loglik(x)
        lpr = self._safe_logprior(x)
        lpo = (
            float64(ll + lpr)
            if np.isfinite(ll) and np.isfinite(lpr)
            else float64(-np.inf)
        )

        return OptimizationResult(
            kind=kind,
            x=x,
            theta=theta,
            success=bool(res.success),
            message=str(res.message),
            fun=float64(res.fun),
            loglik=float64(ll),
            logprior=float64(lpr),
            logpost=float64(lpo),
            nfev=int(res.nfev),
            nit=(int(res.nit) if hasattr(res, "nit") and res.nit is not None else None),
            raw=res,
        )

    def mle(
        self,
        *,
        theta0: NDF | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
        method: str = "L-BFGS-B",
        options: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        self._reset_search_warning_count()
        init = self.theta0() if theta0 is None else asarray(theta0, dtype=float64)

        def obj(theta: NDF) -> float64:
            ll = self._safe_loglik(theta)
            if not np.isfinite(ll):
                return float64(np.inf)
            return float64(-ll)

        minimize = cast(Any, optimize.minimize)
        res = cast(
            optimize.OptimizeResult,
            minimize(
                obj,
                x0=init,
                method=method,
                bounds=bounds,
                options=(dict(options) if options is not None else None),
            ),
        )
        out = self._pack_opt_result("mle", res)
        self._report_search_warning_count("mle")
        return out

    def map(
        self,
        *,
        theta0: NDF | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
        method: str = "L-BFGS-B",
        options: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        if self.priors is None:
            raise ValueError("MAP requires priors. No priors were provided.")

        self._reset_search_warning_count()
        init = self.theta0() if theta0 is None else asarray(theta0, dtype=float64)

        def obj(theta: NDF) -> float64:
            lp = self._safe_logpost(theta)
            if not np.isfinite(lp):
                return float64(np.inf)
            return float64(-lp)

        minimize = cast(Any, optimize.minimize)
        res = cast(
            optimize.OptimizeResult,
            minimize(
                obj,
                x0=init,
                method=method,
                bounds=bounds,
                options=(dict(options) if options is not None else None),
            ),
        )
        out = self._pack_opt_result("map", res)
        self._report_search_warning_count("map")
        return out

    def mcmc(
        self,
        *,
        n_draws: int,
        burn_in: int = 1000,
        thin: int = 1,
        theta0: NDF | None = None,
        random_state: int | np.random.Generator | None = None,
        adapt: bool = True,
        adapt_start: int = 100,
        adapt_interval: int = 25,
        proposal_scale: float = 0.1,
        adapt_epsilon: float = 1e-8,
        update_R_in_iterations: bool = False,
    ) -> MCMCResult:
        if n_draws <= 0:
            raise ValueError("n_draws must be positive.")
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if thin <= 0:
            raise ValueError("thin must be positive.")
        if self.priors is None:
            raise ValueError("MCMC requires priors to define a posterior.")
        self._reset_search_warning_count()

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        current = self.theta0() if theta0 is None else asarray(theta0, dtype=float64)
        d = current.shape[0]
        if d == 0:
            raise ValueError("No estimated parameters were provided.")

        total_steps = burn_in + n_draws * thin
        cov = (float64(proposal_scale) ** 2) * np.eye(d, dtype=float64)
        scale = float64((2.38**2) / d)

        dynamic_R_enabled = False
        dynamic_obs: list[str] | None = None
        kalman = getattr(self.compiled, "kalman", None)
        if update_R_in_iterations and kalman is not None:
            r_param_names = list(getattr(kalman, "R_param_names", None) or [])
            if getattr(kalman, "R_builder", None) is not None and any(
                n in self._param_index for n in r_param_names
            ):
                dynamic_R_enabled = True
                dynamic_obs = self._effective_observables()

        def _safe_logpost_chain(theta: NDF) -> float64:
            if not dynamic_R_enabled:
                return self._safe_logpost(theta)
            try:
                params = self.theta_to_params(theta)
                R_iter = backend.build_R_from_config_params(
                    compiled=self.compiled,
                    kalman=self.compiled.kalman,
                    observables=cast(list[str], dynamic_obs),
                    params=params,
                )
                lp, n_signals = self._eval_with_warning_capture(
                    lambda th: self._logpost_with_overrides(
                        th,
                        params_override=params,
                        R_override=R_iter,
                    ),
                    theta,
                )
                self._warning_signal_count += n_signals
                if n_signals > 0 or not np.isfinite(lp):
                    return float64(-np.inf)
                return float64(lp)
            except BaseException:
                return float64(-np.inf)

        cur_lp = _safe_logpost_chain(current)
        accepted = 0

        history = np.empty((total_steps, d), dtype=float64)
        lp_trace = np.empty((total_steps,), dtype=float64)
        kept = np.empty((n_draws, d), dtype=float64)
        kept_lp = np.empty((n_draws,), dtype=float64)

        keep_i = 0
        eye_d = np.eye(d, dtype=float64)
        t0 = perf_counter()

        for t in range(total_steps):
            prop = rng.multivariate_normal(current, cov)
            prop_lp = _safe_logpost_chain(prop)

            if np.isfinite(prop_lp):
                log_alpha = prop_lp - cur_lp
                if np.log(rng.random()) < log_alpha:
                    current = prop
                    cur_lp = prop_lp
                    accepted += 1

            history[t] = current
            lp_trace[t] = cur_lp

            if (
                adapt
                and t < burn_in
                and t >= adapt_start
                and (t + 1) % adapt_interval == 0
            ):
                hist = history[: t + 1]
                if d == 1:
                    var = (
                        np.var(hist[:, 0], ddof=1)
                        if hist.shape[0] > 1
                        else float64(1.0)
                    )
                    cov = np.array([[scale * var + adapt_epsilon]], dtype=float64)
                else:
                    emp = np.cov(hist.T, ddof=1) if hist.shape[0] > 1 else eye_d
                    cov = scale * (asarray(emp, dtype=float64) + adapt_epsilon * eye_d)

            if t >= burn_in and (t - burn_in) % thin == 0:
                kept[keep_i] = current
                kept_lp[keep_i] = cur_lp
                keep_i += 1
        elapsed = max(perf_counter() - t0, np.finfo(float).eps)

        kept_params = np.empty_like(kept, dtype=float64)
        for i in range(n_draws):
            p = self.theta_to_params(kept[i])
            for j, name in enumerate(self.param_names):
                kept_params[i, j] = float64(p[name])

        print(
            f"MCMC sampling concluded in {elapsed:.2f} seconds with {float64(total_steps / elapsed):<10.2f} iterations per second."
        )

        out = MCMCResult(
            param_names=list(self.param_names),
            samples=kept_params,
            logpost_trace=kept_lp,
            accept_rate=float64(accepted / total_steps),
            n_draws=n_draws,
            burn_in=burn_in,
            thin=thin,
        )
        self._report_search_warning_count("mcmc")
        return out
