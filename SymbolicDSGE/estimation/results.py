from dataclasses import dataclass

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

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
    raw: OptimizeResult


@dataclass(frozen=True)
class MCMCResult:
    param_names: list[str]
    samples: NDF
    logpost_trace: NDF
    accept_rate: float64
    n_draws: int
    burn_in: int
    thin: int

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

    def hpd_intervals(self, alpha: float = 0.05) -> dict[str, tuple[float64, float64]]:
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
            out[name] = (low, high)
        return out

    def joint_hpd_set(self, alpha: float = 0.05) -> tuple[NDF, NDF, float64, NDI]:
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
            np.asarray(self.samples[idx, :], dtype=float64),
            np.asarray(self.logpost_trace[idx], dtype=float64),
            threshold,
            idx,
        )

    def posterior_kde_plot(self) -> None:
        from scipy.stats import gaussian_kde
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
            kde = gaussian_kde(col)
            x_min, x_max = col.min(), col.max()
            x_grid = np.linspace(x_min, x_max, 1000)
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
