"""Kalman filter configuration for filter specifications."""

from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass(frozen=True)
class KalmanStateSpace:
    """State-space representation for a Kalman filter.

    Attributes
    ----------
    A : NDArray[float64]
        Transition matrix for the state vector.
    B : NDArray[float64]
        Shock loading matrix for the state vector.
    C : NDArray[float64]
        Measurement coefficient matrix.
    d : NDArray[float64]
        Measurement intercept vector.
    Q : NDArray[float64]
        Shock covariance matrix for the state vector.
    y_names : list[str]
        Observable names corresponding to the rows of ``C`` and ``d``.
    eps_names : list[str]
        Shock names corresponding to the columns of ``B`` and ``Q``.
    x_names : list[str]
        State names corresponding to the rows of ``A`` and ``B``.

    """

    A: NDArray
    B: NDArray
    C: NDArray
    d: NDArray
    Q: NDArray

    y_names: list[str]
    eps_names: list[str]
    x_names: list[str]
