from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping

from numpy import float64
from numpy.typing import NDArray

from SymbolicDSGE.core.solved_model.base import SolvedModel
from .base import Filter, ResultT
from .filter import (
    FilterResult,
    UnscentedFilterResult,
    run_linear,
    run_extended,
    run_unscented,
)
from .resolvers import (
    FilterMode,
    _resolve_linear_args,
    _resolve_extended_args,
    _resolve_unscented_args,
)


@dataclass
class GaussianFilter(Filter[ResultT]):
    """Base class for Gaussian filters."""

    jitter: float = 0.0
    symmetrize: bool = True


@dataclass
class LinearKalman(GaussianFilter[FilterResult]):
    mode: ClassVar[FilterMode] = FilterMode.LINEAR
    jospeh_cov: bool = True
    return_shocks: bool = False

    def filter(
        self,
        model: SolvedModel,
        y: NDArray[float64],
        observables: list[str] | tuple[str] | None = None,
        R: NDArray[float64] | None = None,
        x0: Mapping[str, float] | None = None,
        P0: NDArray[float64] | None = None,
    ) -> FilterResult:
        """Run a linear Kalman filter on the given model and data.

        Parameters
        ----------
        model : SolvedModel
            Model to filter.
        y : NDArray[float64] | pd.DataFrame
            Observed data as an array or DataFrame.
            If an array is provided, the order of `observables` should match the order of the columns.
            If a DataFrame is provided, the columns should name the observables.
        observables : list[str] | tuple[str] | None
            Model observables that the data corresponds to.
            If `None`, all observables are used in the order they are defined in the model.
        R : NDArray[float64] | None
            Measurement noise covariance matrix.
            If `None`, and the model's parameterization does not declare a `calibration.measurement_noise` block,
            zeros are used.
        x0 : Mapping[str, float] | None
            Initial state vector.
            If `None`, the model's steady state is used.
        P0 : NDArray[float64] | None
            Initial state covariance matrix.
            If `None` the model's stationary covariance is used.

        Returns
        -------
        FilterResult
            Result of the filtration process.
            Includes filtered/predicted states, observables, and covariances, as well as optional shock estimates and the log-likelihood.

        """
        return run_linear(
            **_resolve_linear_args(
                model,
                y,
                observables,
                x0=x0,
                P0=P0,
                R=R,
                jitter=self.jitter,
                symmetrize=self.symmetrize,
                joseph_cov=self.jospeh_cov,
                return_shocks=self.return_shocks,
            )
        )


@dataclass
class ExtendedKalman(GaussianFilter[FilterResult]):
    mode: ClassVar[FilterMode] = FilterMode.EXTENDED
    jospeh_cov: bool = True
    return_shocks: bool = False

    def filter(
        self,
        model: SolvedModel,
        y: NDArray[float64],
        observables: list[str] | tuple[str] | None = None,
        R: NDArray[float64] | None = None,
        x0: Mapping[str, float] | None = None,
        P0: NDArray[float64] | None = None,
    ) -> FilterResult:
        """Run an extended Kalman filter on the given model and data.

        Parameters
        ----------
        model : SolvedModel
            Model to filter.
        y : NDArray[float64] | pd.DataFrame
            Observed data as an array or DataFrame.
            If an array is provided, the order of `observables` should match the order of the columns.
            If a DataFrame is provided, the columns should name the observables.
        observables : list[str] | tuple[str] | None
            Model observables that the data corresponds to.
            If `None`, all observables are used in the order they are defined in the model.
        R : NDArray[float64] | None
            Measurement noise covariance matrix.
            If `None`, and the model's parameterization does not declare a `calibration.measurement_noise` block,
            zeros are used.
        x0 : Mapping[str, float] | None
            Initial state vector.
            If `None`, the model's steady state is used.
        P0 : NDArray[float64] | None
            Initial state covariance matrix.
            If `None` the model's stationary covariance is used.

        Returns
        -------
        FilterResult
            Result of the filtration process.
            Includes filtered/predicted states, observables, and covariances, as well as optional shock estimates and the log-likelihood.

        """
        return run_extended(
            **_resolve_extended_args(
                model,
                y,
                observables,
                x0=x0,
                P0=P0,
                R=R,
                jitter=self.jitter,
                symmetrize=self.symmetrize,
                joseph_cov=self.jospeh_cov,
                return_shocks=self.return_shocks,
            )
        )


@dataclass
class UnscentedKalman(GaussianFilter[UnscentedFilterResult]):
    mode: ClassVar[FilterMode] = FilterMode.UNSCENTED

    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 1.0

    def filter(
        self,
        model: SolvedModel,
        y: NDArray[float64],
        observables: list[str] | tuple[str] | None = None,
        R: NDArray[float64] | None = None,
        x0: Mapping[str, float] | None = None,
        P0: NDArray[float64] | None = None,
    ) -> UnscentedFilterResult:
        """Run an extended Kalman filter on the given model and data.

        Parameters
        ----------
        model : SolvedModel
            Model to filter.
        y : NDArray[float64] | pd.DataFrame
            Observed data as an array or DataFrame.
            If an array is provided, the order of `observables` should match the order of the columns.
            If a DataFrame is provided, the columns should name the observables.
        observables : list[str] | tuple[str] | None
            Model observables that the data corresponds to.
            If `None`, all observables are used in the order they are defined in the model.
        R : NDArray[float64] | None
            Measurement noise covariance matrix.
            If `None`, and the model's parameterization does not declare a `calibration.measurement_noise` block,
            zeros are used.
        x0 : Mapping[str, float] | None
            Initial state vector.
            If `None`, the model's steady state is used.
        P0 : NDArray[float64] | None
            Initial state covariance matrix.
            If `None` the model's stationary covariance is used.
        alpha : float
            Spread of the sigma points around the mean.
        beta : float
            Controls the weight of the central sigma point in covariance calculations.
            2.0 is optimal for Gaussian distributions.
        kappa : float
            Secondary scaling parameter.

        Returns
        -------
        UnscentedFilterResult
            Result of the filtration process.
            Includes filtered/predicted states, observables, and covariances, as well as optional shock estimates and the log-likelihood.

        """
        return run_unscented(
            **_resolve_unscented_args(
                model,
                y,
                observables,
                x0=x0,
                P0=P0,
                R=R,
                jitter=self.jitter,
                symmetrize=self.symmetrize,
                alpha=self.alpha,
                beta=self.beta,
                kappa=self.kappa,
            )
        )
