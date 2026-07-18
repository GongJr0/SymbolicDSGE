from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import float64
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .spec import MCMCResultMeta, OptimizationResultMeta

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]


@dataclass(frozen=True)
class OptimizationResult:
    kind: str
    x: NDF
    theta: dict[str, float64]
    success: bool
    message: str
    fun: float64
    loglik: float64
    logprior: float64
    logpost: float64
    nfev: int
    nit: int | None
    #: Call configuration for the ``mle``/``map`` run (optimizer ``method``,
    #: ``bounds``, ``options``) — recorded so the run is reconstructable.
    optimizer_config: dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> "OptimizationResultMeta":
        """Project to the text-only metadata carried in a ``.sdsge`` bundle.

        Drops the flat ``x`` vector; ``theta`` carries the same point estimate
        by parameter name. The ``optimizer_config`` (method/bounds/options) is
        preserved.
        """
        from .spec import OptimizationResultMeta

        return OptimizationResultMeta(
            kind=self.kind,
            theta={str(k): float(v) for k, v in self.theta.items()},
            success=bool(self.success),
            message=str(self.message),
            fun=float(self.fun),
            loglik=float(self.loglik),
            logprior=float(self.logprior),
            logpost=float(self.logpost),
            nfev=int(self.nfev),
            nit=None if self.nit is None else int(self.nit),
            optimizer_config=dict(self.optimizer_config),
        )


@dataclass(frozen=True)
class MCMCResult:
    param_names: list[str]
    samples: NDF
    logpost_trace: NDF
    accept_rate: float64
    n_draws: int
    burn_in: int
    thin: int
    #: Sampler tuning for the ``mcmc`` run (``adapt``/``proposal_scale``/seed/…)
    #: beyond ``n_draws``/``burn_in``/``thin`` — recorded for reconstruction.
    sampler_config: dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> "MCMCResultMeta":
        """Project to the scalar text metadata carried in a ``.sdsge`` bundle.

        Bulk ``samples`` / ``logpost_trace`` are not included — pair this with
        :meth:`posterior_arrays` when bundling so they ride a sibling member.
        The ``sampler_config`` is preserved.
        """
        from .spec import MCMCResultMeta

        return MCMCResultMeta(
            param_names=list(self.param_names),
            accept_rate=float(self.accept_rate),
            n_draws=int(self.n_draws),
            burn_in=int(self.burn_in),
            thin=int(self.thin),
            sampler_config=dict(self.sampler_config),
        )

    def posterior_arrays(self) -> dict[str, NDF]:
        """Bulk posterior columns keyed for :func:`SymbolicDSGE.bundle.trace_to_json`.

        ``{"samples": (n_draws, n_params), "logpost": (n_draws,)}`` — the shape
        :class:`BundleBuilder` expects as ``posterior`` and the loader returns in
        ``LoadedEstimation.posterior``.
        """
        return {"samples": self.samples, "logpost": self.logpost_trace}

    @staticmethod
    def _validate_hpd_alpha(alpha: float) -> float64:
        alpha64 = float64(alpha)
        if not (0.0 <= alpha64 < 1.0):
            raise ValueError("alpha must satisfy 0 <= alpha < 1.")
        return alpha64

    def _validate_samples(self) -> tuple[int, int]:
        if self.samples.ndim != 2:
            raise ValueError("samples must be a 2D array of shape (n_draws, n_params).")
        n_draws, n_params = self.samples.shape
        if n_draws == 0:
            raise ValueError("samples is empty.")
        if len(self.param_names) != n_params:
            raise ValueError(
                "param_names length does not match the number of sample columns."
            )
        return n_draws, n_params

    @staticmethod
    def _hpd_window_size(n_draws: int, alpha: float64) -> int:
        mass = float64(1.0 - alpha)
        return max(1, min(n_draws, int(np.ceil(float(n_draws) * float(mass)))))

    def hpd_intervals(
        self, alpha: float = 0.05, n_digits: int = 3
    ) -> dict[str, tuple[float64, float64]]:
        """
        Compute marginal highest-posterior-density intervals for each parameter.

        The returned interval for each parameter is the shortest empirical interval
        containing approximately ``1 - alpha`` of the retained posterior draws.
        """

        alpha64 = self._validate_hpd_alpha(alpha)
        n_draws, _ = self._validate_samples()
        window = self._hpd_window_size(n_draws, alpha64)

        out: dict[str, tuple[float64, float64]] = {}
        for i, name in enumerate(self.param_names):
            col = np.sort(np.asarray(self.samples[:, i], dtype=float64))
            if window == n_draws:
                low = float64(col[0])
                high = float64(col[-1])
            else:
                widths = col[window - 1 :] - col[: n_draws - window + 1]
                start = int(np.argmin(widths))
                low = float64(col[start])
                high = float64(col[start + window - 1])
            out[name] = (low.round(n_digits), high.round(n_digits))
        return out

    def joint_hpd_set(
        self, alpha: float = 0.05, n_digits: int = 3
    ) -> tuple[NDF, NDF, float64, NDI]:
        """
        Compute an empirical joint HPD set for the full parameter vector.

        The set is formed by retaining the draws with the largest log-posterior
        values until at least ``1 - alpha`` of the retained mass is covered. The
        returned tuple is ``(samples, logpost, threshold, indices)`` where
        ``threshold`` is the log-posterior cutoff and ``indices`` are positions in
        the original retained chain.
        """

        alpha64 = self._validate_hpd_alpha(alpha)
        n_draws, _ = self._validate_samples()
        if self.logpost_trace.ndim != 1 or self.logpost_trace.shape[0] != n_draws:
            raise ValueError(
                "logpost_trace must be a 1D array with one entry per retained draw."
            )

        window = self._hpd_window_size(n_draws, alpha64)
        order = np.argsort(self.logpost_trace)[::-1]
        threshold = float64(self.logpost_trace[order[window - 1]])
        mask = np.asarray(self.logpost_trace >= threshold, dtype=bool)
        idx = np.flatnonzero(mask).astype(np.int64, copy=False)
        return (
            np.asarray(self.samples[idx, :], dtype=float64).round(n_digits),
            np.asarray(self.logpost_trace[idx], dtype=float64).round(n_digits),
            threshold,
            idx,
        )

    def posterior_kde_plot(self, grid_points: int = 1000) -> None:
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
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(self.logpost_trace)
        plt.title("Log-Posterior Trace")
        plt.xlabel("Iteration")
        plt.ylabel("Log-Posterior")
        plt.tight_layout()
        plt.show()
