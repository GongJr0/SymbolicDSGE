"""Estimator interface for DSGE model estimations, including likelihood and bayesian methods."""

from __future__ import annotations

import warnings
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from numpy import asarray, float64
from numpy.typing import NDArray

from ..bayesian.priors import Prior
from ..core.compiled_model import CompiledModel

from .._ckernels.estimation import (
    run_estimation,
    run_mcmc,
    loglik,
    logprior,
    logpost,
)

from .results import MCMCResult, MLEResult, MAPResult, OptimizationResult
from .spec import EstimatorSpec, EstimatorParams, _coerce_ss_seed

from . import backend as b
from . import resolvers as r
from .backend import EstimDTO, estim_context

NDF = NDArray[np.float64]


class Estimator:
    """Estimation interface exposing three public methods.

    - maximum likelihood estimation (`mle`)
    - maximum a posteriori estimation (`map`)
    - adaptive random-walk Metropolis MCMC (`mcmc`)

    Parameters
    ----------
    compiled : CompiledModel
        Compiled model object containing the target model's information and configuration.
    y : NDF | pd.DataFrame
        Observed data for the estimation procedure. Array in declared observable order or a DataFrame with columns matching the observable names.
    observables : Sequence[str] | None
        Declared observable names in the order they appear in (array) data. If none, all observables in the model are used in the order they were declared in the config.
    filter_mode : str
        Filter mode to use for the estimation procedure. One of 'linear', 'extended', or 'unscented'.
    estimated_params : Sequence[str] | Mapping[str, Prior] | None
        Names of the parameters to be estimated, or a mapping of parameter names to their `Prior`s.
        MAP and MCMC require priors; MLE ignores any provided priors. `None` uses all parameters and no priors.
    ss_seed : Sequence[float] | NDF | Mapping[str, float] | None
        Initial guess for the Newton steady state solver.
    jitter : float | float64 | None
        Jitter to add when a cholesky decomposition fails in the Kalman filter. If none, no jitter is added.
    symmetrize : bool
        Whether to symmetrize the covariance matrices in the Kalman filter.
    joseph_cov : bool
        Whether to use the Joseph form of the covariance update in the Kalman filter.
    R : NDF | None
        Observation covariance matrix. If none, the model's Kalman configuration must provide a symbolic or scalar R.
    P0 : NDF | None
        Initial state covariance matrix for the Kalman filter. If none, the model's stationary state covariance is used.

    Attributes
    ----------
    compiled : CompiledModel
        The compiled model being estimated.
    y : NDF | pd.DataFrame
        The observed data.
    observables : Sequence[str] | None
        The resolved observable names.
    estimated_params : Sequence[str] | None
        The parameter names under estimation.
    priors : Mapping[str, Prior] | None
        The configured priors.
    ss_seed : Sequence[float] | NDF | Mapping[str, float] | None
        The steady-state solver seed.
    R : NDF | None
        The observation covariance override.
    P0 : NDF | None
        The initial state covariance override.
    kalman : KalmanConfig | None
        The compiled model's filter configuration.
    param_names : list[str]
        Estimated parameter names in theta order.
    """

    def __init__(
        self,
        *,
        compiled: CompiledModel,
        y: NDF | pd.DataFrame,
        observables: Sequence[str] | None = None,
        filter_mode: str = "linear",
        estimated_params: Sequence[str] | Mapping[str, Prior] | None = None,
        ss_seed: Sequence[float] | NDF | Mapping[str, float] | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool = True,
        joseph_cov: bool = False,
        R: NDF | None = None,
        P0: NDF | None = None,
    ) -> None:
        self.estimated_params, self.priors = r.get_names_and_priors(
            estimated_params, compiled
        )
        self.compiled = compiled
        if compiled.kalman is None and R is None:
            raise ValueError(
                "R must be provided in symbolic or scalar form, either through the "
                "model's Kalman configuration or as a parameter override."
            )

        self.kalman = compiled.kalman
        self.observables = observables
        self.y = y

        self.ss_seed = ss_seed
        self.R = R
        self.P0 = P0
        self._prepared_filter = b.prepare_filter_run(
            compiled=compiled,
            y=y,
            observables=observables,
            filter_mode=filter_mode,
            jitter=jitter,
            symmetrize=symmetrize,
            joseph_cov=bool(joseph_cov),
            P0=P0,
        )

        r.check_parameters(compiled, self.estimated_params)
        r.check_estimated_versus_constant_R(
            self.estimated_params,
            self.kalman,
            self.R,
        )

        self._param_index, self._matrix_blocks = r.resolve_theta_layout(
            self.estimated_params,
            self.priors,
            self.compiled,
            self.kalman,
            self._prepared_filter.observables,
        )
        r.assert_scalar_corr_spd_safe(
            self._param_index,
            self._matrix_blocks,
            self.compiled,
            self.kalman,
            self._prepared_filter.observables,
        )

    @property
    def param_names(self) -> list[str]:
        """Estimated parameter names in theta order."""
        return list(self._param_index)

    def to_spec(self) -> EstimatorSpec:
        """Produce a bundle-format spec from an :class:`Estimator` instance, suitable for serialization and later reconstruction.

        Returns
        -------
        EstimatorSpec
            Bundle-ready spec class.

        """
        priors = {name: prior.to_spec() for name, prior in (self.priors or {}).items()}

        params = EstimatorParams(
            observables=self.observables,
            filter_mode=self._prepared_filter.mode,
            P0=self.P0.tolist() if self.P0 is not None else None,
            R=self.R.tolist() if self.R is not None else None,
            estimated_params=self.estimated_params,
            priors=priors or None,
            ss_seed=_coerce_ss_seed(self.ss_seed),
            jitter=self._prepared_filter.jitter,
            symmetrize=self._prepared_filter.sym,
            joseph_cov=self._prepared_filter.joseph_cov,
        )

        if isinstance(self.y, pd.DataFrame):
            y = self.y.to_numpy().tolist()
        else:
            y = self.y.tolist()

        return EstimatorSpec(
            y=y,
            params=params,
        )

    @classmethod
    def from_spec(cls, spec: EstimatorSpec, compiled: CompiledModel) -> "Estimator":
        """Build an :class:`Estimator` instance from a bundle-format spec and a compiled model.

        Parameters
        ----------
        spec : EstimatorSpec
            Spec class containing the serialized estimator configuration and data.
        compiled : CompiledModel
            Compiled model object containing the target model's information and configuration.

        Returns
        -------
        "Estimator"
            Live :class:`Estimator` instance reconstructed from the spec and compiled model.

        """
        params = spec.params
        y = np.asarray(spec.y, dtype=float64)
        R = np.asarray(params["R"], dtype=float64) if params["R"] is not None else None
        P0 = (
            np.asarray(params["P0"], dtype=float64)
            if params["P0"] is not None
            else None
        )
        priors = {
            name: Prior.from_spec(prior_spec)
            for name, prior_spec in (params["priors"] or {}).items()
        }
        estimated_params = priors if priors else params["estimated_params"]
        return cls(
            compiled=compiled,
            y=y,
            observables=params["observables"],
            filter_mode=params["filter_mode"],
            estimated_params=estimated_params,
            ss_seed=params["ss_seed"],
            jitter=params["jitter"],
            symmetrize=params["symmetrize"],
            joseph_cov=params["joseph_cov"],
            R=R,
            P0=P0,
        )

    def loglik(self, params: Mapping[str, float] | None = None) -> float64:
        """Log-likelihood of the data given the model and parameters.

        Parameters
        ----------
        params : Mapping[str, float] | None
            Values of the estimated parameters. Omitted parameters use the calibrated values from the model.

        Returns
        -------
        float64
            Log-likelihood value of the data given the model and parameters.

        """
        ctx, theta, _, _ = self._build_call_context(
            priored=self.priors is not None, theta0=params
        )
        return loglik(ctx, theta)

    def logprior(
        self, params: Mapping[str, float] | None = None, include_logjac: bool = False
    ) -> float64:
        """Log-prior of the parameters given the specified priors.

        Parameters
        ----------
        params : Mapping[str, float] | None
            Values of the estimated parameters. Omitted parameters use the calibrated values from the model.
        include_logjac : bool
            Whether to include the log-Jacobian term from the parameter transforms in the log-prior calculation.
            MAP estimation does not sample a distribution, therefore the change RV jacobian term should not be included.
            MCMC sampling does apply prior transformations on distributions, therefore the change RV jacobian term should be included.

        Returns
        -------
        float64
            Log-prior value of the parameters given the specified priors, optionally including the log-Jacobian term.

        """
        ctx, theta, _, _ = self._build_call_context(
            priored=self.priors is not None, theta0=params
        )
        return logprior(ctx, theta, include_logjac)

    def logpost(
        self, params: Mapping[str, float] | None = None, include_logjac: bool = False
    ) -> float64:
        """Log-posterior of the parameters given the data, model, and priors.

        Parameters
        ----------
        params : Mapping[str, float] | None
            Values of the estimated parameters. Omitted parameters use the calibrated values from the model.
        include_logjac : bool
            Whether to include the log-Jacobian term from the parameter transforms in the log-posterior calculation.
            MAP estimation does not sample a distribution, therefore the change RV jacobian term should not be included.
            MCMC sampling does apply prior transformations on distributions, therefore the change RV jacobian term should be included.

        Returns
        -------
        float64
            Log-posterior value of the parameters given the data, model, and priors, optionally including the log-Jacobian term.

        """
        ctx, theta, _, _ = self._build_call_context(
            priored=self.priors is not None, theta0=params
        )
        return logpost(ctx, theta, include_logjac)

    def _report_search_warning_count(self, kind: str, n_err: int) -> None:
        print(
            f"[Estimator:{kind}] BK stability warnings encountered during search: {n_err}"
        )

    def _pack_opt_result(
        self,
        kind: str,
        res: dict[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        x = asarray(res["x"], dtype=float64)
        # res["params"] is the calib-order parameter vector at x_best. Only the
        # estimated names are this result's own; every other parameter sits at
        # the calibration it entered with and belongs to the model, not here.
        params_at_x = dict(
            zip((str(name) for name in self.compiled.calib_params), res["params"])
        )
        theta = {name: float64(params_at_x[name]) for name in self.param_names}

        vcov = res.get("vcov")
        se = res.get("se")
        common: dict[str, Any] = dict(
            x=x,
            theta=theta,
            vcov=vcov,
            se=dict(zip(self.param_names, se)) if se is not None else None,
            cov_status=int(res.get("cov_status", 0)),
            success=bool(res["success"]),
            message=str(res["message"]),
            fun=float64(res["fun"]),
            nfev=int(res["nfev"]),
            nit=int(res["nit"]),
            optimizer_config=dict(config or {}),
        )
        if kind == "mle":
            return MLEResult(**common, loglik=-res["fun"])
        if kind == "map":
            return MAPResult(**common, logpost=-res["fun"], logprior=res["logprior"])
        raise ValueError(f"unknown result kind {kind!r}")

    def _build_call_context(
        self,
        *,
        priored: bool,
        theta0: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    ) -> tuple[EstimDTO, NDF, NDF, NDF]:
        """Build one call's native inputs for the current filter mode.

        ``priored`` is the method's answer rather than ``self``'s: MLE ignores
        the priors it was given, their transforms included, so it builds the
        transform-free leg and takes the generated box with it. What the driver
        does with the ctx (minimized ``-logpost`` vs ``+logpost``) is still the
        driver's own business.
        """
        return estim_context(
            compiled=self.compiled,
            prepared=self._prepared_filter,
            param_names=self.param_names,
            param_index=self._param_index,
            matrix_blocks=self._matrix_blocks,
            priors=self.priors if priored else None,
            ss_seed=self.ss_seed,
            R_override=self.R,
            theta0=theta0,
            bounds=bounds,
        )

    def _point_estimate(
        self,
        routine: Literal["mle", "map"],
        has_priors: bool,
        jacobian: bool = False,
        theta0: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        method: Literal["L-BFGS-B", "Nelder-Mead"] = "L-BFGS-B",
        m: int = 10,
        maxiter: int = 15000,
        maxfun: int = 15000,
        maxls: int = 20,
        factr: float = 1e7,
        pgtol: float = 1e-5,
        fd_step: float = 0.0,
        xatol: float = 1e-4,
        fatol: float = 1e-4,
        cov: bool = True,
        cov_fd_step_scale: float = 1.0,
        cov_fd_absolute_floor: float = 0.1,
    ) -> OptimizationResult:

        ctx, theta, lo, hi = self._build_call_context(
            priored=has_priors, theta0=theta0, bounds=bounds
        )

        res = run_estimation(
            ctx,
            method,
            include_logjac=jacobian,
            theta0=theta,
            lo=lo,
            hi=hi,
            has_priors=has_priors,
            m=m,
            maxiter=maxiter,
            maxfun=maxfun,
            maxls=maxls,
            factr=factr,
            pgtol=pgtol,
            fd_step=fd_step,
            xatol=xatol,
            fatol=fatol,
            compute_cov=cov,
            cov_fd_step_scale=cov_fd_step_scale,
            cov_fd_absolute_floor=cov_fd_absolute_floor,
        )

        out = self._pack_opt_result(
            routine,
            res,
            config={
                "theta0": dict(theta0) if theta0 is not None else None,
                "method": method,
                "bounds": r.serialize_bounds(bounds),
                "m": m,
                "maxiter": maxiter,
                "maxfun": maxfun,
                "maxls": maxls,
                "factr": factr,
                "pgtol": pgtol,
                "fd_step": fd_step,
                "xatol": xatol,
                "fatol": fatol,
                "jacobian": jacobian,
                "cov": cov,
                "cov_fd_step_scale": cov_fd_step_scale,
                "cov_fd_absolute_floor": cov_fd_absolute_floor,
            },
        )
        self._report_search_warning_count(routine, res["bk_violations"])
        return out

    def mle(
        self,
        *,
        theta0: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        method: Literal["L-BFGS-B", "Nelder-Mead"] = "L-BFGS-B",
        m: int = 10,
        maxiter: int = 15000,
        maxfun: int = 15000,
        maxls: int = 20,
        factr: float = 1e7,
        pgtol: float = 1e-5,
        fd_step: float = 0.0,
        xatol: float = 1e-4,
        fatol: float = 1e-4,
        cov: bool = True,
        cov_fd_step_scale: float = 1.0,
        cov_fd_absolute_floor: float = 0.1,
    ) -> MLEResult:
        """Maximum likelihood estimation of the model parameters given the data and model configuration.

        Parameters
        ----------
        theta0 : NDF | Mapping[str, float] | None
            Initial guess for the parameter vector. If None, uses the model calibration.
        bounds : Mapping[str, tuple[float | None, float | None]] | None
            Bounds to restrict the parameter search space, keyed by estimated parameter name, as ``(lower, upper)`` with ``None`` for an open side. Bounds are read in parameter space and mapped through the parameter's transform, so a bound tighter than the transform's own region is the only one that binds. A member of an estimated correlation block cannot be bounded.
        method : Literal["L-BFGS-B", "Nelder-Mead"]
            Optimization method to use for the estimation. "L-BFGS-B" is a quasi-Newton method suitable for large problems, while "Nelder-Mead" is a derivative-free method.
        m : int
            L-BFGS-B memory parameter. The number of previous gradients and updates to store for approximating the inverse Hessian matrix. Larger values may improve convergence but increase memory usage.
        maxiter : int
            Maximum number of iterations for the optimization algorithm. The optimization will stop if this limit is reached.
        maxfun : int
            Maximum number of function evaluations for the optimization algorithm. The optimization will stop if this limit is reached.
        maxls : int
            Maximum number of line search steps for the optimization algorithm. The optimization will stop if this limit is reached.
        factr : float
            L-BFGS-B convergence criterion. The optimization will stop when the change in the objective function is less than ``factr * machine_epsilon``.
        pgtol : float
            L-BFGS-B convergence criterion. The optimization will stop when the projected gradient is less than ``pgtol``.
        fd_step : float
            Step size for finite difference approximation of gradients. If 0.0, the algorithm will choose an appropriate step size automatically.
        xatol : float
            Nelder-Mead convergence criterion. The optimization will stop when the change in the parameter vector is less than ``xatol``.
        fatol : float
            Nelder-Mead convergence criterion. The optimization will stop when the change in the objective function is less than ``fatol``.
        cov : bool
            Whether to compute the covariance matrix of the estimated parameters. If True, the covariance matrix will be computed using the inverse Hessian at the optimum.
        cov_fd_step_scale : float
            Scale factor for the finite difference step size used in covariance matrix estimation. A larger value may improve numerical stability but may also introduce bias.
        cov_fd_absolute_floor : float
            Absolute floor for the finite difference step size used in covariance matrix estimation. This prevents the step size from becoming too small and causing numerical issues.

        Returns
        -------
        MLEResult
            Result object containing the estimated parameters, covariance matrix, and optimization details.

        """
        if self.priors is not None:
            warnings.warn(
                "MLE will ignore any provided priors. Use MAP or MCMC for prior-informed estimation.",
                UserWarning,
            )

        return cast(
            MLEResult,
            self._point_estimate(
                routine="mle",
                has_priors=False,
                jacobian=False,
                theta0=theta0,
                bounds=bounds,
                method=method,
                m=m,
                maxiter=maxiter,
                maxfun=maxfun,
                maxls=maxls,
                factr=factr,
                pgtol=pgtol,
                fd_step=fd_step,
                xatol=xatol,
                fatol=fatol,
                cov=cov,
                cov_fd_step_scale=cov_fd_step_scale,
                cov_fd_absolute_floor=cov_fd_absolute_floor,
            ),
        )

    def map(
        self,
        *,
        theta0: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        method: Literal["L-BFGS-B", "Nelder-Mead"] = "L-BFGS-B",
        jacobian: bool = False,
        m: int = 10,
        maxiter: int = 15000,
        maxfun: int = 15000,
        maxls: int = 20,
        factr: float = 1e7,
        pgtol: float = 1e-5,
        fd_step: float = 0.0,
        xatol: float = 1e-4,
        fatol: float = 1e-4,
        cov: bool = True,
        cov_fd_step_scale: float = 1.0,
        cov_fd_absolute_floor: float = 0.1,
    ) -> MAPResult:
        """Maximum a posteriori estimation of the model parameters given the data, model configuration, and specified priors.

        Parameters
        ----------
        theta0 : NDF | Mapping[str, float] | None
            Initial guess for the parameter vector. If None, uses the model calibration.
        bounds : Mapping[str, tuple[float | None, float | None]] | None
            Bounds to restrict the parameter search space, keyed by estimated parameter name, as ``(lower, upper)`` with ``None`` for an open side. Bounds are read in parameter space and mapped through the parameter's transform, so a bound tighter than the transform's own region is the only one that binds. A member of an estimated correlation block cannot be bounded.
        method : Literal["L-BFGS-B", "Nelder-Mead"]
            Optimization method to use for the estimation. "L-BFGS-B" is a quasi-Newton method suitable for large problems, while "Nelder-Mead" is a derivative-free method.
        jacobian : bool
            Whether to include the log-Jacobian term from the parameter transforms in the log-posterior calculation. This is should be ``True`` if you MAP results and/or covariance will later feed an MCMC run, and ``False`` if you are only interested in the MAP point estimate.
        m : int
            L-BFGS-B memory parameter. The number of previous gradients and updates to store for approximating the inverse Hessian matrix. Larger values may improve convergence but increase memory usage.
        maxiter : int
            Maximum number of iterations for the optimization algorithm. The optimization will stop if this limit is reached.
        maxfun : int
            Maximum number of function evaluations for the optimization algorithm. The optimization will stop if this limit is reached.
        maxls : int
            Maximum number of line search steps for the optimization algorithm. The optimization will stop if this limit is reached.
        factr : float
            L-BFGS-B convergence criterion. The optimization will stop when the change in the objective function is less than ``factr * machine_epsilon``.
        pgtol : float
            L-BFGS-B convergence criterion. The optimization will stop when the projected gradient is less than ``pgtol``.
        fd_step : float
            Step size for finite difference approximation of gradients. If 0.0, the algorithm will choose an appropriate step size automatically.
        xatol : float
            Nelder-Mead convergence criterion. The optimization will stop when the change in the parameter vector is less than ``xatol``.
        fatol : float
            Nelder-Mead convergence criterion. The optimization will stop when the change in the objective function is less than ``fatol``.
        cov : bool
            Whether to compute the covariance matrix of the estimated parameters. If True, the covariance matrix will be computed using the inverse Hessian at the optimum.
        cov_fd_step_scale : float
            Scale factor for the finite difference step size used in covariance matrix estimation. A larger value may improve numerical stability but may also introduce bias.
        cov_fd_absolute_floor : float
            Absolute floor for the finite difference step size used in covariance matrix estimation. This prevents the step size from becoming too small and causing numerical issues.

        Returns
        -------
        MAPResult
            MAP result object containing the estimated parameters, covariance matrix, log-prior, and optimization details.

        """
        if self.priors is None:
            raise ValueError("MAP requires priors. No priors were provided.")

        return cast(
            MAPResult,
            self._point_estimate(
                routine="map",
                has_priors=True,
                jacobian=jacobian,
                theta0=theta0,
                bounds=bounds,
                method=method,
                m=m,
                maxiter=maxiter,
                maxfun=maxfun,
                maxls=maxls,
                factr=factr,
                pgtol=pgtol,
                fd_step=fd_step,
                xatol=xatol,
                fatol=fatol,
                cov=cov,
                cov_fd_step_scale=cov_fd_step_scale,
                cov_fd_absolute_floor=cov_fd_absolute_floor,
            ),
        )

    def mcmc(
        self,
        *,
        n_draws: int,
        burn_in: int = 1000,
        thin: int = 1,
        theta0: Mapping[str, float] | None = None,
        random_state: int | None = None,
        adapt: bool = True,
        adapt_start: int = 100,
        proposal_scale: float = 0.1,
        adapt_epsilon: float = 1e-8,
        compute_map: bool = True,
        map_options: dict[str, Any] | None = None,
        proposal_cov: NDF | None = None,
        cov_fd_step_scale: float = 1.0,
        cov_fd_absolute_floor: float = 0.1,
    ) -> MCMCResult:
        """Markov Chain Monte Carlo sampling of the posterior distribution over the estimated parameters.

        Parameters
        ----------
        n_draws : int
            Number of MCMC draws to generate (after burn-in and thinning).
        burn_in : int
            Number of initial draws to discard as burn-in.
        thin : int
            Thinning factor: keep every ``thin``-th draw after burn-in.
        theta0 : NDF | Mapping[str, float] | None
            Initial guess for the unconstrained parameter vector. If None, uses the model calibration.
        random_state : int | None
            Seed for the random number generator. If None, uses a random seed.
        adapt : bool
            Whether to adapt the proposal covariance, ``True`` corresponds to Haario et al. (2001) adaptive MCMC;
            ``False`` is a regular Random Walk Metropolis-Hastings sampler with fixed proposal covariance.
        adapt_start : int
            When to start adapting the proposal covariance. Must be less than ``n_draws``.
        proposal_scale : float
            Scaling factor for the proposal covariance.
        adapt_epsilon : float
            Small constant added to the diagonal of the proposal covariance during adaptation to ensure positive definiteness.
        compute_map : bool
            Whether to compute the MAP estimate to use it as initial guess and the hessian point for the (initial or fixed) proposal covariance.
        map_options : dict[str, Any] | None
            Keyword arguments to pass when calling :meth:`map` to compute the MAP estimate. ``None`` uses the default options.
            Refer to :meth:`map` for the available options.
        proposal_cov : NDF | None
            Direct, user-specified proposal covariance matrix. Cannot be used when ``compute_map=True``.
        cov_fd_step_scale : float
            Scale of the finite difference approximation step used to compute the Hessian for the proposal covariance. Used when solving for the proposal covariance from the MAP estimate. Ignored if ``proposal_cov`` is provided.
        cov_fd_absolute_floor : float
            Absolute floor for the finite difference approximation step used to compute the Hessian for the proposal covariance. Used when solving for the proposal covariance from the MAP estimate. Ignored if ``proposal_cov`` is provided.

        Returns
        -------
        MCMCResult
            Results of the MCMC sampling, including the samples, acceptance rate, and other diagnostics.

        """
        if self.priors is None:
            raise ValueError("MCMC requires priors to define a posterior.")

        rng = np.random.default_rng(random_state)

        map_bounds = None if map_options is None else map_options.get("bounds")
        ctx, theta, map_lo, map_hi = self._build_call_context(
            priored=True, theta0=theta0, bounds=map_bounds
        )

        native_map_options = dict(map_options or {})
        native_map_options.pop("bounds", None)
        native_map_options["lo"] = map_lo
        native_map_options["hi"] = map_hi

        # The chain runs entirely in native nogil code; ``rng`` (numpy's own
        # PCG64) is borrowed for the run and must outlive it, which the local
        # reference here guarantees. Timing wraps only the native call.
        t0 = perf_counter()
        out = run_mcmc(
            ctx,
            theta,
            rng,
            n_draws=n_draws,
            burn_in=burn_in,
            thin=thin,
            adapt=adapt,
            adapt_start=adapt_start,
            proposal_scale=proposal_scale,
            proposal_cov=proposal_cov,
            cov_fd_step_scale=cov_fd_step_scale,
            cov_fd_absolute_floor=cov_fd_absolute_floor,
            adapt_epsilon=adapt_epsilon,
            compute_map=compute_map,
            map_options=native_map_options,
        )
        elapsed = max(perf_counter() - t0, np.finfo(float).eps)

        total_steps = int(out["total_steps"])
        kept = out["samples"]

        print(
            f"MCMC sampling concluded in {elapsed:.2f} seconds with {float(total_steps / elapsed):.2f} iterations per second."
        )

        usr_opts = dict(map_options) if map_options is not None else None
        if usr_opts and "bounds" in usr_opts:
            usr_opts["bounds"] = r.serialize_bounds(usr_opts["bounds"])

        result = MCMCResult(
            param_names=list(self.param_names),
            samples=kept,
            logpost_trace=out["logpost_trace"],
            logjac_trace=out["logjac_trace"],
            accept_rate=float64(out["n_accepted"] / total_steps),
            n_draws=n_draws,
            burn_in=burn_in,
            thin=thin,
            sampler_config={
                "n_draws": int(n_draws),
                "burn_in": int(burn_in),
                "thin": int(thin),
                "theta0": dict(theta0) if theta0 is not None else None,
                "adapt": bool(adapt),
                "adapt_start": int(adapt_start),
                "proposal_scale": float(proposal_scale),
                "adapt_epsilon": float(adapt_epsilon),
                "compute_map": bool(compute_map),
                "map_options": usr_opts,
                "proposal_cov": (
                    proposal_cov.tolist() if proposal_cov is not None else None
                ),
                "cov_fd_step_scale": float(cov_fd_step_scale),
                "cov_fd_absolute_floor": float(cov_fd_absolute_floor),
                "random_state": (None if random_state is None else int(random_state)),
            },
        )
        self._report_search_warning_count("mcmc", out["bk_violations"])
        return result
