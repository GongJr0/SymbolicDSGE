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
from .backend import (
    EstimCall,
    MatrixPriorBlock,
    build_dto,
)

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
    estimated_params : Sequence[str] | None
        Names of the parameters to be estimated. If none, all parameters in the model are estimated.
    priors : Mapping[str, Prior] | None
        Mapping of parameter names to their prior distributions. If none, no priors are used.
        "map" and "mcmc" require priors to be specified for all estimated parameters.
        When priors are specified, estimated_params must match the names of the priors or be omitted.
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
        estimated_params: Sequence[str] | None = None,
        priors: Mapping[str, Prior] | None = None,
        ss_seed: Sequence[float] | NDF | Mapping[str, float] | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool = True,
        joseph_cov: bool = False,
        R: NDF | None = None,
        P0: NDF | None = None,
    ) -> None:
        self.estimated_params = estimated_params
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
        self._base_params = b.extract_base_params(compiled)
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

        self.priors = dict(priors) if priors is not None else None

        requested_names_raw = r.assert_only_prior_or_parameters(
            estimated_params,
            self.priors,
        )
        r.check_parameters(compiled, requested_names_raw)
        r.check_estimated_versus_constant_R(
            requested_names_raw,
            self.kalman,
            self.R,
        )

        self._param_index, self._matrix_blocks = r.resolve_theta_layout(
            requested_names_raw,
            self.priors,
            self.compiled,
            self.kalman,
            self._prepared_filter.observables,
        )

        self._matrix_member_names = {
            name
            for block in self._matrix_blocks.values()
            for name in block.member_names
        }

    @property
    def param_names(self) -> list[str]:
        """Estimated parameter names in theta order."""
        return list(self._param_index)

    def _spd_member_names(self) -> tuple[set[str], set[str]]:
        """Names of the SPD-relevant std (diagonal) and correlation (off-diagonal)
        parameters across the R and Q matrices, read straight from the parser's
        name maps.

        The two roles are kept separate because they need different constraining
        transforms: a variance wants a positivity map, a correlation a (-1, 1)
        map. Membership is deliberately independent of whether a prior exists, so
        this drives the transform defaults on the prior-free (MLE) path, not just
        the prior-gated CPC block.
        """
        std_members: set[str] = set()
        corr_members: set[str] = set()
        observed = set(self._prepared_filter.observables)
        active_shocks = set(self.compiled.shock_names)

        r_std_map = getattr(self.kalman, "R_std_param_map", None) or {}
        for obs, v in r_std_map.items():
            if v is not None and (observed is None or str(obs) in observed):
                std_members.add(v)
        r_corr_map = getattr(self.kalman, "R_corr_param_map", None) or {}
        for pair, v in r_corr_map.items():
            if v is not None and (
                observed is None or {str(x) for x in pair} <= observed
            ):
                corr_members.add(v)

        calibration = self.compiled.config.calibration
        shock_std = getattr(calibration, "shock_std", None) or {}
        for shock, sym in shock_std.items():
            if sym is not None and (
                active_shocks is None or str(shock) in active_shocks
            ):
                std_members.add(sym)
        shock_corr = getattr(calibration, "shock_corr", None) or {}
        for pair, sym in shock_corr.items():
            if sym is not None and (
                active_shocks is None or {str(s) for s in pair} <= active_shocks
            ):
                corr_members.add(sym)

        return std_members, corr_members

    def _corr_pairs_by_name(self) -> dict[str, tuple[str, frozenset[str]]]:
        """Map each named correlation parameter to ``(matrix_key, {var_a, var_b})``,
        for the joint-SPD safety gate on standalone scalar correlations.
        """
        out: dict[str, tuple[str, frozenset[str]]] = {}
        observed = set(self._prepared_filter.observables)
        active_shocks = set(self.compiled.shock_names)
        r_corr_map = getattr(self.kalman, "R_corr_param_map", None) or {}
        for pair, nm in r_corr_map.items():
            vars_ = frozenset(str(v) for v in pair)
            if nm is not None and (observed is None or vars_ <= observed):
                out[nm] = ("R_corr", vars_)
        shock_corr = getattr(self.compiled.config.calibration, "shock_corr", None) or {}
        for pair, sym in shock_corr.items():
            vars_ = frozenset(str(s) for s in pair)
            if sym is not None and (active_shocks is None or vars_ <= active_shocks):
                out[sym] = ("Q_corr", vars_)
        return out

    @staticmethod
    def _requested_param_keys(
        allowed_names: set[str],
        estimated_params: Sequence[str] | None,
        priors: Mapping[str, Prior] | None = None,
    ) -> list[str]:
        if estimated_params is not None:
            if not all(param in allowed_names for param in estimated_params):
                missing = set(estimated_params) - allowed_names
                raise ValueError(
                    f"Parameters {{{missing}}} are not estimable targets of the model: {sorted(allowed_names)}"
                )
            if not all(param in estimated_params for param in priors or {}):
                missing = set(priors or {}) - set(estimated_params)
                raise ValueError(
                    f"Priors specified for parameters {{{missing}}} which are not in the estimated parameters: {list(estimated_params)}"
                )
            return list(estimated_params)
        if priors is not None:
            if not all(param in allowed_names for param in priors):
                missing = set(priors) - allowed_names
                raise ValueError(
                    f"Parameters {{{missing}}} are not estimable targets of the model: {sorted(allowed_names)}"
                )
            return list(priors)
        raise ValueError(
            "Either estimated_params or priors must be provided to determine the requested parameters."
        )

    @staticmethod
    def _format_pairs(pairs: Sequence[tuple[str, str]]) -> str:
        return ", ".join(f"({a}, {b})" for a, b in pairs)

    @staticmethod
    def _corr_from_member_values(block: MatrixPriorBlock, values: NDF) -> NDF:
        corr = np.eye(block.K, dtype=float64)
        rows = block.positions[:, 0]
        cols = block.positions[:, 1]
        vals = np.asarray(values, dtype=float64)
        corr[rows, cols] = vals
        corr[cols, rows] = vals
        return corr

    @staticmethod
    def _block_cpc_from_corr(block: MatrixPriorBlock, corr: NDF) -> NDF:
        try:
            return b._unconstrained_from_corr(corr)
        except ValueError as exc:
            raise ValueError(
                f"Correlation values do not form a valid positive-definite "
                f"correlation matrix over {block.labels}: {exc}"
            ) from exc

    @staticmethod
    def _block_corr_from_theta(
        block: MatrixPriorBlock, theta_block: NDF
    ) -> tuple[NDF, NDF]:
        Lcorr = b._corr_chol_from_unconstrained(theta_block, block.K)
        corr = np.asarray(Lcorr @ Lcorr.T, dtype=float64)
        return corr, np.asarray(Lcorr, dtype=float64)

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
        return cls(
            compiled=compiled,
            y=y,
            observables=params["observables"],
            filter_mode=params["filter_mode"],
            estimated_params=params["estimated_params"],
            priors=priors or None,
            ss_seed=params["ss_seed"],
            jitter=params["jitter"],
            symmetrize=params["symmetrize"],
            joseph_cov=params["joseph_cov"],
            R=R,
            P0=P0,
        )

    def theta0(self) -> NDF:
        """Convert the model's base calibration to the unconstrained theta vector.

        Returns
        -------
        NDF
            Array of unconstrained initial parameter values corresponding to the model's base calibration.

        """
        constrained = asarray(
            [self._base_params[name] for name in self.param_names],
            dtype=float64,
        )
        return self.params_to_theta(constrained)

    def resolve_theta0(self, theta0: NDF | Mapping[str, float] | None) -> NDF:
        """Coerce a user ``theta0`` to the unconstrained theta vector.

        ``None`` seeds from the model calibration (:meth:`theta0`); a mapping is
        validated against the estimated parameter names and converted through
        :meth:`params_to_theta`; an array is taken as-is.
        """
        if theta0 is None:
            return self.theta0()
        if isinstance(theta0, Mapping):
            missing = [name for name in self.param_names if name not in theta0]
            if missing:
                raise ValueError(
                    f"theta0 dictionary is missing estimated parameters: {missing}"
                )
            unknown = [key for key in theta0 if key not in self.param_names]
            if unknown:
                raise ValueError(f"theta0 dictionary has unknown parameters: {unknown}")
            return self.params_to_theta(
                {name: float64(theta0[name]) for name in self.param_names}
            )
        return asarray(theta0, dtype=float64)

    def _validate_theta0(self, theta: NDF) -> None:
        """Fail fast on an initial guess the objective cannot score.

        A transform's inverse lands inside its own support, so a theta only
        arrives unusable by being non-finite itself or by saturating its
        transform: a std that overflows to infinity, or one that underflows to
        the zero its support excludes. The support is the role's, so this holds
        for an MLE start as much as a MAP one.
        """
        invalid: list[str] = []
        for i, name in enumerate(self.param_names):
            z = float64(theta[i])
            if not np.isfinite(z):
                invalid.append(f"{name}={z}")
                continue
            if name in self._matrix_member_names:
                # A block's run decodes to a valid correlation for any finite z.
                continue
            transform = self._param_transforms[name]  # type: ignore
            value = float64(transform.safe_inverse(z))
            if not np.isfinite(value) or not transform.support.contains(value):
                invalid.append(f"{name}={value}")
        if invalid:
            raise ValueError(
                "Initial guess maps to parameter values the objective cannot "
                "score: " + ", ".join(invalid)
            )

    def params_to_theta(self, params: Mapping[str, float] | NDF) -> NDF:
        """Convert a parameter mapping or array to the unconstrained theta vector.

        Parameters
        ----------
        params : Mapping[str, float] | NDF
            Mapping of {name: value} or array of parameter values in the order of ``self.param_names``.

        Returns
        -------
        NDF
            Array of unconstrained parameter values corresponding to the provided parameters.

        """
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
        handled = np.zeros((len(self.param_names),), dtype=bool)
        for block in self._matrix_blocks.values():
            corr_vals = np.asarray(vals[block.theta_slice], dtype=float64)
            corr = self._corr_from_member_values(block, corr_vals)
            out[block.theta_slice] = self._block_cpc_from_corr(block, corr)
            handled[block.theta_slice] = True

        for i, name in enumerate(self.param_names):
            if handled[i]:
                continue
            out[i] = float64(
                self._param_transforms[name].safe_forward(float64(vals[i]))  # type: ignore
            )
        return out

    def theta_to_params(self, theta: NDF) -> dict[str, float64]:
        """A theta draw as the named parameters, over the base calibration.

        A CPC block's members come off the correlation its run decodes to;
        every other estimated entry comes through its own inverse transform.
        """
        theta = asarray(theta, dtype=float64)
        if theta.ndim != 1:
            raise ValueError("theta must be a 1D array.")
        if theta.shape[0] != len(self.param_names):
            raise ValueError(
                f"theta length {theta.shape[0]} does not match estimated parameter count {len(self.param_names)}."
            )
        full = dict(self._base_params)
        handled = np.zeros((len(self.param_names),), dtype=bool)
        for block in self._matrix_blocks.values():
            theta_block = np.asarray(theta[block.theta_slice], dtype=float64)
            corr, _ = self._block_corr_from_theta(block, theta_block)
            member_vals = corr[block.positions[:, 0], block.positions[:, 1]]
            for name, val in zip(block.member_names, member_vals):
                full[name] = float64(val)
            handled[block.theta_slice] = True

        for i, name in enumerate(self.param_names):
            if handled[i]:
                continue
            full[name] = float64(
                self._param_transforms[name].safe_inverse(float64(theta[i]))  # type: ignore
            )
        return full

    def loglik(self, theta: NDF) -> float64:
        """Log-likelihood of the data given the model and parameters.

        Parameters
        ----------
        theta : NDF
            Parameter vector in the unconstrained space, corresponding to the estimated parameters of the model.

        Returns
        -------
        float64
            Log-likelihood value of the data given the model and parameters.

        """
        ctx = self._build_native_context(priored=self.priors is not None).dto
        return loglik(ctx, theta)

    def logprior(self, theta: NDF, include_logjac: bool = False) -> float64:
        """Log-prior of the parameters given the specified priors.

        Parameters
        ----------
        theta : NDF
            Parameter vector in the unconstrained space, corresponding to the estimated parameters of the model.
        include_logjac : bool
            Whether to include the log-Jacobian term from the parameter transforms in the log-prior calculation.
            MAP estimation does not sample a distribution, therefore the change RV jacobian term should not be included.
            MCMC sampling does apply prior transformations on distributions, therefore the change RV jacobian term should be included.

        Returns
        -------
        float64
            Log-prior value of the parameters given the specified priors, optionally including the log-Jacobian term.

        """
        ctx = self._build_native_context(priored=self.priors is not None).dto
        return logprior(ctx, theta, include_logjac)

    def logpost(self, theta: NDF, include_logjac: bool = False) -> float64:
        """Log-posterior of the parameters given the data, model, and priors.

        Parameters
        ----------
        theta : NDF
            Parameter vector in the unconstrained space, corresponding to the estimated parameters of the model.
        include_logjac : bool
            Whether to include the log-Jacobian term from the parameter transforms in the log-posterior calculation.
            MAP estimation does not sample a distribution, therefore the change RV jacobian term should not be included.
            MCMC sampling does apply prior transformations on distributions, therefore the change RV jacobian term should be included.

        Returns
        -------
        float64
            Log-posterior value of the parameters given the data, model, and priors, optionally including the log-Jacobian term.

        """
        ctx = self._build_native_context(priored=self.priors is not None).dto
        return logpost(ctx, theta, include_logjac)

    def _report_search_warning_count(self, kind: str, n_err: int) -> None:
        print(
            f"[Estimator:{kind}] BK stability warnings encountered during search: {n_err}"
        )

    @staticmethod
    def _serialize_bounds(
        bounds: Mapping[str, tuple[float | None, float | None]] | None,
    ) -> dict[str, list[float | None]] | None:
        if bounds is None:
            return None
        return {
            name: [
                None if lo is None else float(lo),
                None if hi is None else float(hi),
            ]
            for name, (lo, hi) in bounds.items()
        }

    @staticmethod
    def _bounds_pairs(
        lo: NDF, hi: NDF
    ) -> list[tuple[float | None, float | None]] | None:
        """The theta box as the per-slot pairs the native entry point parses.

        That parser reads a present side as a bound, which here is a finite one;
        an all-infinite box restricts nothing and passes as ``None``, so neither
        driver enters its bounded path over a box that would never bind.
        """
        low = np.isfinite(lo)
        high = np.isfinite(hi)
        if not (low.any() or high.any()):
            return None
        return [
            (
                float(lo[i]) if low[i] else None,
                float(hi[i]) if high[i] else None,
            )
            for i in range(lo.shape[0])
        ]

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

    def _build_native_context(
        self,
        *,
        priored: bool,
        theta0: NDF | Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    ) -> EstimCall:
        """Build one call's native inputs for the current filter mode.

        ``priored`` is the method's answer rather than ``self``'s: MLE ignores
        the priors it was given, their transforms included, so it builds the
        transform-free leg and takes the generated box with it. What the driver
        does with the ctx (minimized ``-logpost`` vs ``+logpost``) is still the
        driver's own business.
        """
        return build_dto(
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
        theta0: NDF | Mapping[str, float] | None = None,
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

        call = self._build_native_context(
            priored=has_priors, theta0=theta0, bounds=bounds
        )

        res = run_estimation(
            call.dto,
            method,
            include_logjac=jacobian,
            theta0=call.theta,
            bounds=self._bounds_pairs(call.lo, call.hi),
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
                "theta0": theta0.tolist() if isinstance(theta0, np.ndarray) else theta0,
                "method": method,
                "bounds": self._serialize_bounds(bounds),
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
        theta0: NDF | Mapping[str, float] | None = None,
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
        theta0: NDF | Mapping[str, float] | None = None,
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
        theta0: NDF | Mapping[str, float] | None = None,
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
        call = self._build_native_context(
            priored=True, theta0=theta0, bounds=map_bounds
        )
        current = call.theta
        if current.shape[0] == 0:
            raise ValueError("No estimated parameters were provided.")

        # The MAP seed is the only leg of the chain that takes a box, so the
        # call's box is its box.
        native_map_options = dict(map_options or {})
        if map_bounds is not None:
            native_map_options["bounds"] = self._bounds_pairs(call.lo, call.hi)

        # The chain runs entirely in native nogil code; ``rng`` (numpy's own
        # PCG64) is borrowed for the run and must outlive it, which the local
        # reference here guarantees. Timing wraps only the native call.
        t0 = perf_counter()
        out = run_mcmc(
            call.dto,
            current,
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

        # Recorded, not re-used: the run already consumed map_options, so this
        # copy exists only to carry a JSON-safe bounds shape into the config.
        recorded_map_options: dict[str, Any] | None = None
        if map_options is not None:
            recorded_map_options = dict(map_options)
            if recorded_map_options.get("bounds") is not None:
                recorded_map_options["bounds"] = self._serialize_bounds(
                    recorded_map_options["bounds"]
                )

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
                "theta0": theta0.tolist() if isinstance(theta0, np.ndarray) else theta0,
                "adapt": bool(adapt),
                "adapt_start": int(adapt_start),
                "proposal_scale": float(proposal_scale),
                "adapt_epsilon": float(adapt_epsilon),
                "compute_map": bool(compute_map),
                "map_options": recorded_map_options,
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
