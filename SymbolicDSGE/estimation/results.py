"""Optimization and MCMC result containers for DSGE model estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from functools import cached_property

from ..core.config import ModelConfig
from .spec import (
    OptimizationResultSpec,
    MLEResultSpec,
    MAPResultSpec,
    MCMCResultMeta,
    MCMCResultSpec,
)

from .._ckernels.estimation import ess, ess1d

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]


@dataclass(frozen=True)
class OptimizationResult:
    """Point estimate optimization result container.

    Attributes
    ----------
    x : NDArray[np.float64]
        Optimal parameter vector in the unconstrained space.
    theta : dict[str, float64]
        Optimal parameter vector in the constrained space, keyed by parameter names.
    success : bool
        Whether the optimization was successful.
    message : str
        Error/success message from the optimizer.
    fun : float64
        Objective function value at the optimum.
    nfev : int
        Number of function evaluations performed by the optimizer.
    nit : int | None
        Number of iterations performed by the optimizer. None if not applicable.
    optimizer_config : dict[str, Any]
        Configuration to reproduce the optimization run (optimizer method, bounds, options).
    vcov : NDArray[float64] | None
        Covariance matrix of the unconstrained parameters at the optimum, derived from the finite-difference Hessian.
        In unconstrained space; directly passable to MCMC as the proposal covariance.
    se : dict[str, float64] | None
        Standard errors of the constrained parameters, computed if covariance was requested.
    cov_status : int
        Status of the covariance computation: 0 if successful, otherwise indicates why it was not computed.

    """

    x: NDF
    theta: dict[str, float64]
    success: bool
    message: str
    fun: float64
    nfev: int
    nit: int | None
    #: Call configuration for the ``mle``/``map`` run (optimizer ``method``,
    #: ``bounds``, ``options``) — recorded so the run is reconstructable.
    optimizer_config: dict[str, Any]
    #: Asymptotic covariance at the optimum, from the finite-difference Hessian
    #: there. It describes ``x``, the unconstrained vector, so its rows and
    #: columns follow ``list(theta)`` and it is the covariance an MCMC proposal
    #: wants. ``None`` when the run was asked not to compute it; NaN throughout
    #: when it was asked and the Hessian at the optimum is not positive definite.
    vcov: NDF | None = field(default=None, kw_only=True)
    #: Standard errors keyed like ``theta`` and in the same constrained space,
    #: so ``theta[name]`` and ``se[name]`` are read together. Derived from
    #: ``vcov`` through the prior transforms' Jacobian, which is why this is not
    #: simply ``sqrt(diag(vcov))`` unless every transform is the identity.
    se: dict[str, float64] | None = field(default=None, kw_only=True)
    #: 0 when the covariance was produced, otherwise why it was not.
    cov_status: int = field(default=0, kw_only=True)

    def common_spec(self) -> OptimizationResultSpec:
        """Base spec for JSON-serialization of MLE and MAP result containers.

        Returns
        -------
        OptimizationResultSpec
            Shared components of point estimate results.

        """
        return OptimizationResultSpec(
            x=self.x.tolist(),
            theta={k: float(v) for k, v in self.theta.items()},
            success=bool(self.success),
            message=str(self.message),
            fun=float(self.fun),
            nfev=int(self.nfev),
            nit=int(self.nit) if self.nit is not None else None,
            optimizer_config=dict(self.optimizer_config),
            vcov=self.vcov.tolist() if self.vcov is not None else None,
            cov_status=int(self.cov_status),
            se=(
                {k: float(v) for k, v in self.se.items()}
                if self.se is not None
                else None
            ),
        )

    def apply_to(self, config: ModelConfig) -> ModelConfig:
        """Return a new config with the parameter values set to the point estimate.

        Parameters
        ----------
        config : ModelConfig
            Config to take as a base.
            Must contain all parameter names in the result's ``theta``.

        Returns
        -------
        ModelConfig
            New config object with updated parameters.

        """
        return config.with_parameters(self.theta)


@dataclass(frozen=True)
class MLEResult(OptimizationResult):
    """Maximum-likelihood point estimate result."""

    loglik: float64

    def to_spec(self) -> MLEResultSpec:
        """Return a JSON-serializable dict representation of the result."""
        return MLEResultSpec(**self.common_spec(), loglik=float(self.loglik))

    @classmethod
    def from_spec(cls, data: MLEResultSpec) -> MLEResult:
        """Construct a MLEResult from a JSON-serializable dict representation."""
        se = data.get("se") or {}
        return cls(
            x=np.asarray(data["x"], dtype=float64),
            theta={k: float64(v) for k, v in data["theta"].items()},
            success=bool(data["success"]),
            message=str(data["message"]),
            fun=float64(data["fun"]),
            nfev=int(data["nfev"]),
            nit=int(data["nit"]) if data["nit"] is not None else None,
            optimizer_config=dict(data.get("optimizer_config", {})),
            vcov=(
                np.asarray(data["vcov"], dtype=float64)
                if data.get("vcov") is not None
                else None
            ),
            se={k: float64(v) for k, v in se.items()} or None,
            cov_status=int(data.get("cov_status", 0)),
            loglik=float64(data["loglik"]),
        )


@dataclass(frozen=True)
class MAPResult(OptimizationResult):
    """Maximum-a-posteriori point estimate result."""

    logpost: float64
    logprior: float64

    def to_spec(self) -> MAPResultSpec:
        """Return a JSON-serializable dict representation of the result."""
        return MAPResultSpec(
            **self.common_spec(),
            logpost=float(self.logpost),
            logprior=float(self.logprior),
        )

    @classmethod
    def from_spec(cls, data: MAPResultSpec) -> MAPResult:
        """Construct a MAPResult from a JSON-serializable dict representation."""
        se = data.get("se") or {}
        return cls(
            x=np.asarray(data["x"], dtype=float64),
            theta={k: float64(v) for k, v in data["theta"].items()},
            success=bool(data["success"]),
            message=str(data["message"]),
            fun=float64(data["fun"]),
            nfev=int(data["nfev"]),
            nit=int(data["nit"]) if data["nit"] is not None else None,
            optimizer_config=dict(data.get("optimizer_config", {})),
            vcov=(
                np.asarray(data["vcov"], dtype=float64)
                if data.get("vcov") is not None
                else None
            ),
            se={k: float64(v) for k, v in se.items()} or None,
            cov_status=int(data.get("cov_status", 0)),
            logpost=float64(data["logpost"]),
            logprior=float64(data["logprior"]),
        )


@dataclass(frozen=True)
class MCMCResult:
    """Result container for a Markov Chain Monte Carlo (MCMC) posterior sampling run.

    Attributes
    ----------
    param_names : tuple[str, ...]
        Names of the parameters in the order they appear in the samples array.
    samples : NDF
        Posterior samples (in parameter space) drawn from the MCMC run, typically
        a 2D array of shape (n_draws, n_params).
    logpost_trace : NDF
        Log-posterior values corresponding to each sample in the samples array.
    logjac_trace : NDF
        Log-Jacobian values corresponding to each sample in the samples array.
    accepted_burn_in : int
        Number of proposals accepted during the burn-in phase of the MCMC run.
    accepted_in_sample : int
        Number of proposals accepted during the retained phase of the MCMC run.
    burn_in_seconds : float
        Time spent in seconds during the burn-in phase of the MCMC run.
    in_sample_seconds : float
        Time spent in seconds during the retained phase of the MCMC run.
    n_draws : int
        Total number of draws (samples) generated by the MCMC sampler.
    burn_in : int
        Discarded initial samples (burn-in period) to allow the chain to converge
        before collecting samples for inference.
    thin : int
        Thinning factor applied to the samples to reduce autocorrelation, indicating
        how many samples to skip between retained samples.
    sampler_config : dict[str, Any]
        Configuration details of the MCMC sampler, including tuning parameters and
        settings used during the run. This is recorded for reproducibility and
        unpacking it into :meth:`Estimator.mcmc` reproduces the run; provided the
        run was seeded.
    accept_rate
    accept_rate_burn_in
    accept_rate_in_sample
    ess
    ess_per_second
    mcse
    """

    param_names: tuple[str, ...]
    samples: NDF
    logpost_trace: NDF
    logjac_trace: NDF
    accepted_burn_in: int
    accepted_in_sample: int
    burn_in_seconds: float
    in_sample_seconds: float
    n_draws: int
    burn_in: int
    thin: int
    #: Sampler tuning for the ``mcmc`` run (``adapt``/``proposal_scale``/seed/…)
    #: beyond ``n_draws``/``burn_in``/``thin`` — recorded for reconstruction.
    sampler_config: dict[str, Any] = field(default_factory=dict)

    def to_spec(self) -> MCMCResultSpec:
        """Project to the scalar text metadata carried in a ``.sdsge`` bundle."""
        meta = MCMCResultMeta(
            param_names=list(self.param_names),
            accepted_burn_in=int(self.accepted_burn_in),
            accepted_in_sample=int(self.accepted_in_sample),
            burn_in_seconds=float(self.burn_in_seconds),
            in_sample_seconds=float(self.in_sample_seconds),
            n_draws=int(self.n_draws),
            burn_in=int(self.burn_in),
            thin=int(self.thin),
            sampler_config=dict(self.sampler_config),
        )

        return MCMCResultSpec(
            samples=self.samples.tolist(),
            logpost_trace=self.logpost_trace.tolist(),
            logjac_trace=self.logjac_trace.tolist(),
            meta=meta,
        )

    @classmethod
    def from_spec(cls, data: MCMCResultSpec) -> MCMCResult:
        """Produce a :class:`MCMCResult` from the bundle-format MCMC result spec.

        Parameters
        ----------
        data : MCMCResultSpec
            Spec data, typically loaded from a ``.sdsge`` file.

        Returns
        -------
        "MCMCResult"
            Result object containing MCMC samples, traces, and metadata.

        """
        meta = data.meta
        return cls(
            param_names=tuple(meta["param_names"]),
            samples=np.asarray(data.samples, dtype=float64),
            logpost_trace=np.asarray(data.logpost_trace, dtype=float64),
            logjac_trace=np.asarray(data.logjac_trace, dtype=float64),
            accepted_burn_in=int(meta["accepted_burn_in"]),
            accepted_in_sample=int(meta["accepted_in_sample"]),
            burn_in_seconds=float(meta["burn_in_seconds"]),
            in_sample_seconds=float(meta["in_sample_seconds"]),
            n_draws=int(meta["n_draws"]),
            burn_in=int(meta["burn_in"]),
            thin=int(meta["thin"]),
            sampler_config=dict(meta["sampler_config"]),
        )

    @property
    def accept_rate(self) -> float:
        # `n_draws` counts retained draws; the proposals behind them are
        # `n_draws * thin`, which is what the kernel's step total is built from.
        total_steps = self.burn_in + self.n_draws * self.thin
        return (self.accepted_burn_in + self.accepted_in_sample) / total_steps

    @property
    def accept_rate_burn_in(self) -> float:
        return self.accepted_burn_in / self.burn_in if self.burn_in > 0 else np.nan

    @property
    def accept_rate_in_sample(self) -> float:
        return self.accepted_in_sample / (self.n_draws * self.thin)

    @cached_property
    def ess(self) -> dict[str, float]:
        """Compute the effective sample size (ESS) for each parameter.

        Returns
        -------
        dict[str, float]
            Effective sample size for each parameter, keyed by parameter name.
        """
        ess_vals = ess(self.samples)
        return dict(
            zip(self.param_names, np.clip(ess_vals, 0.0, self.n_draws).tolist())
        )

    @property
    def ess_per_second(self) -> dict[str, float]:
        """Compute the effective sample size (ESS) per second for each parameter.

        The returned values are retained ess per post-burn-in seconds of the MCMC run,
        i.e. ``ess / in_sample_seconds``.

        Returns
        -------
        dict[str, float]
            Effective sample size per second for each parameter, keyed by parameter name.
        """
        return {name: v / self.in_sample_seconds for name, v in self.ess.items()}

    @property
    def mcse(self) -> dict[str, float]:
        """Compute the Monte Carlo Standard Error (MCSE) of the mean for each parameter."""
        ess = self.ess
        sd = np.std(self.samples, axis=0, ddof=1)
        return {
            name: float(sd[i] / np.sqrt(ess[name]))
            for i, name in enumerate(self.param_names)
        }

    def mcse_quantile(self, q: float, digits: int | None = None) -> dict[str, float]:
        """Compute the MCSE of the specified quantile for each parameter.

        The error is taken on the probability scale from the effective sample size
        of the indicator chain ``1{x <= x_q}``, then carried back to the parameter
        scale through the empirical quantile function, which needs no density
        estimate at ``x_q``. NaN for a parameter whose indicator chain has no
        usable ESS.

        Parameters
        ----------
        q : float
            Quantile to compute the MCSE for, strictly between 0 and 1.
        digits: int | None
            Number of decimals for rounding, `None` does not apply rounding.

        Returns
        -------
        dict[str, float]
            MCSE of the specified quantile for each parameter keyed by name.

        """
        from scipy.stats import beta

        # norm.cdf(-1) and norm.cdf(1)
        SIGMA_MASS = (0.15865525393145707, 0.8413447460685429)

        if not (0.0 < q < 1.0):
            raise ValueError("q must satisfy 0 < q < 1.")
        n_draws = self.samples.shape[0]

        out: dict[str, float] = {}
        for i, name in enumerate(self.param_names):
            chain = np.asarray(self.samples[:, i], dtype=float64)
            order = np.sort(chain)
            x_q = np.quantile(order, q, method="linear")
            # ``chain`` rather than ``order``: the indicator's ESS measures
            # autocorrelation, which only exists in draw order.
            ess_q = min(ess1d((chain <= x_q).astype(float64)), float(n_draws))
            if not np.isfinite(ess_q):
                out[name] = float(np.nan)
                continue
            lo, hi = np.asarray(
                beta.ppf(SIGMA_MASS, ess_q * q + 1.0, ess_q * (1.0 - q) + 1.0)
            )
            i_lo = int(np.clip(np.floor(lo * n_draws - 1.0), 0, n_draws - 1))
            i_hi = int(np.clip(np.ceil(hi * n_draws - 1.0), 0, n_draws - 1))
            out[name] = float((order[i_hi] - order[i_lo]) / 2.0)
        return out

    def hpd_intervals(
        self, alpha: float = 0.05, digits: int | None = None
    ) -> dict[str, tuple[float, float]]:
        """
        Compute marginal highest-posterior-density intervals for each parameter.

        The returned interval for each parameter is the shortest empirical interval
        containing approximately ``1 - alpha`` of the retained posterior draws.
        """
        alpha64 = _validate_hpd_alpha(alpha)
        n_draws = self.samples.shape[0]
        window = _hpd_window_size(n_draws, alpha64)

        out: dict[str, tuple[float, float]] = {}
        for i, name in enumerate(self.param_names):
            col = np.sort(np.asarray(self.samples[:, i], dtype=float64))
            if window == n_draws:
                low = float(col[0])
                high = float(col[-1])
            else:
                widths = col[window - 1 :] - col[: n_draws - window + 1]
                start = int(np.argmin(widths))
                low = float(col[start])
                high = float(col[start + window - 1])
            out[name] = (_nanrnd(low, digits), _nanrnd(high, digits))
        return out

    def joint_hpd_set(
        self, alpha: float = 0.05, digits: int | None = None
    ) -> tuple[NDF, NDF, float, NDI]:
        """
        Compute an empirical joint HPD set for the full parameter vector.

        The set is formed by retaining the draws with the largest log-posterior
        values until at least ``1 - alpha`` of the retained mass is covered. The
        returned tuple is ``(samples, logpost, threshold, indices)`` where
        ``threshold`` is the log-posterior cutoff and ``indices`` are positions in
        the original retained chain.
        """
        alpha64 = _validate_hpd_alpha(alpha)
        n_draws = self.samples.shape[0]

        window = _hpd_window_size(n_draws, alpha64)
        order = np.argsort(self.logpost_trace)[::-1]
        threshold = float(self.logpost_trace[order[window - 1]])
        mask = np.asarray(self.logpost_trace >= threshold, dtype=bool)
        idx = np.flatnonzero(mask).astype(np.int64, copy=False)
        return (
            _nanrndarr(np.asarray(self.samples[idx, :], dtype=float64), digits),
            _nanrndarr(np.asarray(self.logpost_trace[idx], dtype=float64), digits),
            threshold,
            idx,
        )

    def posterior_kde_plot(self, grid_points: int = 1000) -> None:
        """Produce a grid-based Kernel Density Estimate (KDE) plot for each parameter's marginal posterior.

        Parameters
        ----------
        grid_points : int
            Number of points in the grid for evaluating the KDE.
        """
        from scipy.stats import gaussian_kde
        import matplotlib.pyplot as plt

        fig_sq = np.ceil(np.sqrt(len(self.param_names)))
        fig, axes = plt.subplots(
            int(fig_sq), int(fig_sq), figsize=(4 * fig_sq, 3 * fig_sq)
        )
        ax = np.atleast_1d(axes).ravel()
        while len(ax) > len(self.param_names):
            fig.delaxes(ax[-1])
            ax = ax[:-1]

        for i, name in enumerate(self.param_names):
            col = np.asarray(self.samples[:, i], dtype=float64)
            kde = gaussian_kde(col)
            x_min, x_max = col.min(), col.max()
            x_grid = np.linspace(x_min, x_max, grid_points)
            ax[i].plot(x_grid, kde(x_grid))
            ax[i].set_title(name)
        plt.tight_layout()
        plt.show()

    def posterior_traces(self) -> None:
        """Produce trace plots for each parameter's posterior samples."""
        import matplotlib.pyplot as plt

        fig_sq = np.ceil(np.sqrt(len(self.param_names)))
        fig, axes = plt.subplots(
            int(fig_sq), int(fig_sq), figsize=(4 * fig_sq, 3 * fig_sq)
        )

        ax = axes.flatten()
        while len(ax) > len(self.param_names):
            fig.delaxes(ax[-1])
            ax = ax[:-1]

        for i, name in enumerate(self.param_names):
            col = np.asarray(self.samples[:, i], dtype=float64)
            ax[i].plot(col)
            ax[i].set_title(name)
        plt.tight_layout()
        plt.show()

    def logpost_trace_plot(self) -> None:
        """Produce the log-posterior trace plot for the MCMC run."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(self.logpost_trace)
        plt.title("Log-Posterior Trace")
        plt.xlabel("Iteration")
        plt.ylabel("Log-Posterior")
        plt.tight_layout()
        plt.show()

    def posterior_point(
        self,
        point: Literal["mean", "map", "last"] = "mean",
        *,
        digits: int | None = None,
    ) -> dict[str, float]:
        """Select a point estimate from the posterior samples.

        Parameters
        ----------
        point : Literal["mean", "map", "last"]
            Point estimate to return:
                - "mean": Return the mean of the posterior samples.
                - "map": Return the sample maximizing the log-posterior (MAP)
                - "last": Return the last sample in the posterior chain.
        digits : int | None
            Number of decimals for rounding, `None` does not apply rounding.

        Returns
        -------
        dict[str, float]
            Parameter values at the point, keyed by name.
        """
        match point:
            case "mean":
                theta_star = self.samples.mean(axis=0)
            case "map":
                idx = int(np.argmax(self.logpost_trace))
                theta_star = self.samples[idx]
            case "last":
                theta_star = self.samples[-1]
            case _:
                raise ValueError(
                    "`posterior_point` must be one of {'mean', 'map', 'last'}, got "
                    f"{point!r}."
                )
        return dict(
            zip(self.param_names, (_nanrnd(float(v), digits) for v in theta_star))
        )

    def apply_to(
        self, config: ModelConfig, point: Literal["mean", "map", "last"] = "mean"
    ) -> ModelConfig:
        """Return a new config with the parameter values set to the point estimate.

        Parameters
        ----------
        config : ModelConfig
            Config to take as a base.
            Must contain all parameter names in the result's ``param_names``.
        point : Literal["mean", "map", "last"]
            Point estimate to return:
                - "mean": Return the mean of the posterior samples.
                - "map": Return the sample maximizing the log-posterior (MAP)
                - "last": Return the last sample in the posterior chain.

        Returns
        -------
        ModelConfig
            New config object with updated parameters.

        """
        return config.with_parameters(self.posterior_point(point))


def _validate_hpd_alpha(alpha: float) -> float64:
    alpha64 = float64(alpha)
    if not (0.0 <= alpha64 < 1.0):
        raise ValueError("alpha must satisfy 0 <= alpha < 1.")
    return alpha64


def _hpd_window_size(n_draws: int, alpha: float64) -> int:
    mass = float64(1.0 - alpha)
    return max(1, min(n_draws, int(np.ceil(float(n_draws) * float(mass)))))


def _nanrnd(val: float, digits: int | None) -> float:
    if digits is None:
        return val
    return round(val, digits)


def _nanrndarr(arr: NDF, digits: int | None) -> NDF:
    if digits is None:
        return arr
    else:
        return arr.round(digits)
