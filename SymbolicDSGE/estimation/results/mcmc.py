"""MCMC result containers for DSGE model estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from functools import cached_property

import pandas as pd

from ...core.config import ModelConfig
from ..spec import (
    MCMCResultMeta,
    MCMCResultSpec,
)

from ..._ckernels.estimation import ess_rhat, hpd_intervals, mcse_quantile

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]


@dataclass(frozen=True)
class MCMCResult:
    """Result container for a Markov Chain Monte Carlo (MCMC) posterior sampling run.

    Attributes
    ----------
    param_names : tuple[str, ...]
        Names of the parameters in the order they appear in the samples array.
    samples : NDArray[float64]
        Posterior samples (in parameter space) drawn from the MCMC run.
        3D array of shape (n_chains, n_draws, n_params).
    logpost_trace : NDArray[float64]
        Log-posterior values in shape (n_chains, n_draws) corresponding
        to each sample in the samples array.
    logjac_trace : NDArray[float64]
        Log-Jacobian values in shape (n_chains, n_draws) corresponding
        to each sample in the samples array.
    accepted_burn_in : NDArray[int64]
        Number of proposals accepted per-chain during the burn-in
        phase of the MCMC run.
    accepted_in_sample : NDArray[int64]
        Number of proposals accepted per-chain during the retained
        phase of the MCMC run.
    wall_seconds : float
        Wall clock time of the MCMC sampler, excludes initialization time.
    init_seconds : float
        Initialization time before the draws start, includes MAP and inverse
        Hessian computations where applicable.
    burn_in_seconds : NDArray[float64]
        Time spent in seconds per-chain during the burn-in phase of the MCMC run.
    in_sample_seconds : NDArray[float64]
        Time spent in seconds per-chain during the retained phase of the MCMC run.
    n_chains : int
        Number of independent MCMC chains the result contains.
    n_draws : int
        Number of retained draws (samples) per chain.
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
    split_rhat
    """

    param_names: tuple[str, ...]
    samples: NDF
    logpost_trace: NDF
    logjac_trace: NDF
    accepted_burn_in: NDI
    accepted_in_sample: NDI
    wall_seconds: float
    init_seconds: float
    burn_in_seconds: NDF
    in_sample_seconds: NDF
    n_chains: int
    n_draws: int
    burn_in: int
    thin: int
    sampler_config: dict[str, Any] = field(default_factory=dict)

    def to_spec(self) -> MCMCResultSpec:
        """Project to the scalar text metadata carried in a ``.sdsge`` bundle."""
        meta = MCMCResultMeta(
            param_names=list(self.param_names),
            accepted_burn_in=self.accepted_burn_in.tolist(),
            accepted_in_sample=self.accepted_in_sample.tolist(),
            wall_seconds=float(self.wall_seconds),
            init_seconds=float(self.init_seconds),
            burn_in_seconds=self.burn_in_seconds.tolist(),
            in_sample_seconds=self.in_sample_seconds.tolist(),
            n_chains=int(self.n_chains),
            n_draws=int(self.n_draws),
            burn_in=int(self.burn_in),
            thin=int(self.thin),
            sampler_config=dict(self.sampler_config),
        )

        return MCMCResultSpec(
            # samples are (n_chains, n_draws, n_params), lp and lj are (n_chains, n_draws).
            # Reshape versus ravel matches the convention of each and produces a n_chains * n_draws
            # major axis.
            samples=self.samples.reshape(self.n_chains * self.n_draws, -1).tolist(),
            logpost_trace=self.logpost_trace.ravel().tolist(),
            logjac_trace=self.logjac_trace.ravel().tolist(),
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
        n_chains = int(meta["n_chains"])
        n_draws = int(meta["n_draws"])
        return cls(
            param_names=tuple(meta["param_names"]),
            samples=np.asarray(data.samples, dtype=float64).reshape(
                n_chains, n_draws, -1
            ),
            logpost_trace=np.asarray(data.logpost_trace, dtype=float64).reshape(
                n_chains, n_draws
            ),
            logjac_trace=np.asarray(data.logjac_trace, dtype=float64).reshape(
                n_chains, n_draws
            ),
            accepted_burn_in=np.asarray(meta["accepted_burn_in"], dtype=np.int64),
            accepted_in_sample=np.asarray(meta["accepted_in_sample"], dtype=np.int64),
            wall_seconds=float(meta["wall_seconds"]),
            init_seconds=float(meta["init_seconds"]),
            burn_in_seconds=np.asarray(meta["burn_in_seconds"], dtype=float64),
            in_sample_seconds=np.asarray(meta["in_sample_seconds"], dtype=float64),
            n_chains=int(meta["n_chains"]),
            n_draws=int(meta["n_draws"]),
            burn_in=int(meta["burn_in"]),
            thin=int(meta["thin"]),
            sampler_config=dict(meta["sampler_config"]),
        )

    @cached_property
    def __ess_rhat(self) -> tuple[NDF, NDF, NDI]:
        return ess_rhat(self.samples)

    @property
    def accept_rate(self) -> list[float]:
        total = self.burn_in + self.n_draws * self.thin
        return cast(
            list[float],
            ((self.accepted_burn_in + self.accepted_in_sample) / total).tolist(),
        )

    @property
    def accept_rate_burn_in(self) -> list[float]:
        return (
            (self.accepted_burn_in / self.burn_in).tolist()
            if self.burn_in > 0
            else [np.nan] * self.n_chains
        )

    @property
    def accept_rate_in_sample(self) -> list[float]:
        return cast(
            list[float],
            (self.accepted_in_sample / (self.n_draws * self.thin)).tolist(),
        )

    @property
    def split_rhat(self) -> dict[str, float]:
        """Compute the split-R-hat convergence diagnostic for each parameter.

        Returns
        -------
        dict[str, float]
            Split-R-hat values for each parameter, keyed by parameter name.
        """
        _, rhat, _ = self.__ess_rhat
        return dict(zip(self.param_names, rhat.tolist()))

    @property
    def ess(self) -> dict[str, float]:
        """Compute the effective sample size (ESS) for each parameter.

        Returns
        -------
        dict[str, float]
            Effective sample size for each parameter, keyed by parameter name.
        """
        ess, _, status = self.__ess_rhat
        ess = np.where(status == 0, ess, np.nan)

        return dict(
            zip(
                self.param_names,
                np.clip(ess, 0.0, self.n_chains * self.n_draws).tolist(),
            )
        )

    @property
    def ess_per_second(self) -> dict[str, float]:
        """Compute the effective sample size (ESS) per second for each parameter.

        The returned values are retained ess per sampling second of the MCMC run,
        i.e. ``ess / wall_seconds``.

        Returns
        -------
        dict[str, float]
            Effective sample size per second for each parameter, keyed by parameter name.
        """
        return {name: v / self.wall_seconds for name, v in self.ess.items()}

    @property
    def mcse(self) -> dict[str, float]:
        """Compute the Monte Carlo Standard Error (MCSE) of the mean for each parameter."""
        ess = self.ess
        sd = np.std(self.samples.reshape(-1, len(self.param_names)), axis=0, ddof=1)
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
        mcse = mcse_quantile(self.samples, q)
        return {
            name: _nanrnd(float(mcse[i]), digits)
            for i, name in enumerate(self.param_names)
        }

    def hpd_intervals(
        self, alpha: float = 0.05, digits: int | None = None
    ) -> dict[str, tuple[float, float]]:
        """
        Compute marginal highest-posterior-density intervals for each parameter.

        The returned interval for each parameter is the shortest empirical interval
        containing approximately ``1 - alpha`` of the retained posterior draws.
        """
        lo, hi = hpd_intervals(self.samples, alpha)
        return {
            name: (_nanrnd(float(lo[i]), digits), _nanrnd(float(hi[i]), digits))
            for i, name in enumerate(self.param_names)
        }

    def posterior_kde_plot(self, grid_points: int = 1000) -> None:
        """Produce a grid-based Kernel Density Estimate (KDE) plot for each parameter's marginal posterior.

        Parameters
        ----------
        grid_points : int
            Number of points in the grid for evaluating the KDE.
        """
        from scipy.stats import gaussian_kde
        import matplotlib.pyplot as plt

        samples = self.samples.reshape(-1, len(self.param_names))

        fig_sq = np.ceil(np.sqrt(len(self.param_names)))
        fig, axes = plt.subplots(
            int(fig_sq), int(fig_sq), figsize=(4 * fig_sq, 3 * fig_sq)
        )
        ax = np.atleast_1d(axes).ravel()
        while len(ax) > len(self.param_names):
            fig.delaxes(ax[-1])
            ax = ax[:-1]

        for i, name in enumerate(self.param_names):
            col = np.asarray(samples[:, i], dtype=float64)
            kde = gaussian_kde(col)
            x_min, x_max = col.min(), col.max()
            x_grid = np.linspace(x_min, x_max, grid_points)
            ax[i].plot(x_grid, kde(x_grid))
            ax[i].set_title(name)
        plt.tight_layout()
        plt.show()

    def logpost_trace_plot(self) -> None:
        """Produce the log-posterior trace plot for the MCMC run."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            self.n_chains, 1, figsize=(8, 3 * self.n_chains), sharex=True
        )
        ax = np.atleast_1d(axes).ravel()
        for i in range(self.n_chains):
            ax[i].set_title(f"Chain {i}")
            ax[i].plot(self.logpost_trace[i])
            ax[i].set_ylabel("Log-Posterior")

        plt.suptitle("Log-Posterior Trace")
        plt.xlabel("Draw")
        plt.tight_layout()
        plt.show()

    def posterior_point(
        self,
        point: Literal["mean", "map"] = "mean",
        digits: int | None = None,
    ) -> dict[str, float]:
        """Select a point estimate from the posterior samples.

        Parameters
        ----------
        point : Literal["mean", "map"]
            Point estimate to return:
                - "mean": Return the mean of the posterior samples.
                - "map": Return the sample maximizing the log-posterior (MAP)
        digits : int | None
            Number of decimals for rounding, `None` does not apply rounding.

        Returns
        -------
        dict[str, float]
            Parameter values at the point, keyed by name.
        """
        samples = self.samples.reshape(-1, len(self.param_names))
        match point:
            case "mean":
                theta_star = samples.mean(axis=0)
            case "map":
                idx = int(np.argmax(self.logpost_trace))
                theta_star = samples[idx]

            case _:
                raise ValueError(
                    "`posterior_point` must be one of {'mean', 'map'}, got "
                    f"{point!r}."
                )
        return dict(
            zip(self.param_names, (_nanrnd(float(v), digits) for v in theta_star))
        )

    def apply_to(
        self, config: ModelConfig, point: Literal["mean", "map"] = "mean"
    ) -> ModelConfig:
        """Return a new config with the parameter values set to the point estimate.

        Parameters
        ----------
        config : ModelConfig
            Config to take as a base.
            Must contain all parameter names in the result's ``param_names``.
        point : Literal["mean", "map"]
            Point estimate to return:
                - "mean": Return the mean of the posterior samples.
                - "map": Return the sample maximizing the log-posterior (MAP)

        Returns
        -------
        ModelConfig
            New config object with updated parameters.

        """
        return config.with_parameters(self.posterior_point(point))

    def summary(self, alpha: float = 0.05, digits: int | None = None) -> pd.DataFrame:
        """Produce a summary table of posterior statistics for each parameter.

        Parameters
        ----------
        alpha : float
            Significance level for the highest-posterior-density (HPD) intervals.
        digits : int | None
            Number of decimals for rounding, `None` does not apply rounding.

        Returns
        -------
        pd.DataFrame
            Summary table containing mean, standard deviation, split R-hat, ESS, MCSE,
            and HPD intervals for each parameter.
        """
        summary_df = pd.DataFrame(
            {
                "mean": self.posterior_point(point="mean"),
                "std": dict(
                    zip(
                        self.param_names,
                        np.std(
                            self.samples.reshape(-1, len(self.param_names)),
                            axis=0,
                            ddof=1,
                        ),
                    )
                ),
                "split-R\u0302": self.split_rhat,
                "ess": self.ess,
                "ess/s": self.ess_per_second,
                "mcse": self.mcse,
                f"hpd (\u03b1={alpha:.3f})": self.hpd_intervals(
                    alpha=alpha, digits=digits
                ),
            }
        )
        if digits is not None:
            summary_df = summary_df.round(digits)
        return summary_df


def _nanrnd(val: float, digits: int | None) -> float:
    if digits is None:
        return val
    return round(val, digits)
