from numpy import float64, int64
from numpy.typing import NDArray

_F64 = NDArray[float64]
_I64 = NDArray[int64]

SDSGE_ESS_OK: int = ...
SDSGE_ESS_EBADSHAPE: int = ...
SDSGE_ESS_EUDEFVARIANCE: int = ...
SDSGE_ESS_ENODECAY: int = ...
SDSGE_ESS_ENONPOSTAU: int = ...

def ess_rhat(x: _F64) -> tuple[_F64, _F64, _I64]:
    """Compute the ESS and R-hat per column for a 3D sample of (n_chains, n_draws, n_params).

    Returns status and writes NaNs to outputs affected by a given error.
    R-hat can be returned while ESS fails.

    Parameters
    ----------
    x : _F64
        Sample values to compute the ESS and R-hat for.

    Returns
    -------
    tuple[_F64, _F64, _I64]
        3-tuple of lists of floats: ESS, R-hat, and status code
        per column of the input sample.
    """
    ...

def ess_quantile(x: _F64, q: float) -> _F64:
    """Compute the ESS of a given quantile per column for a 3D sample of (n_chains, n_draws, n_params).

    Writes NaNs to outputs affected by an error.

    Parameters
    ----------
    x : _F64
        Sample values to compute the ESS for.
    q : float
        Quantile to compute the ESS for, must be in the open interval (0, 1).

    Returns
    -------
    _F64
        ESS values in parameter order.
    """
    ...

def hpd_intervals(x: _F64, alpha: float) -> tuple[_F64, _F64]:
    """
    Compute the highest posterior density (HPD) intervals for a given set of samples.

    Parameters
    ----------
    x : _F64
        A 3D array of shape (n_chains, n_samples, n_parameters) containing the samples.
    alpha : float
        The significance level for the HPD interval.

    Returns
    -------
    tuple[_F64, _F64]
        Lower and upper bounds as arrays in parameter order.
    """
    ...

def mcse_quantile(x: _F64, q: float) -> _F64:
    """Compute the Monte Carlo standard error (MCSE) of the given quantile for each parameter.

    Parameters
    ----------
    x : _F64
        A 3D array of shape (n_chains, n_samples, n_parameters) containing the samples.
    q : float
        Quantile to compute the MCSE for, must be in the open interval (0, 1).

    Returns
    -------
    _F64
        MCSE values in parameter order.

    """
    ...

def marginal_kde(
    x: _F64, grid_points: int, bandwidth: str | float = "scott"
) -> tuple[tuple[_F64, _F64], ...]:
    """Compute the 1D Gaussian KDE of each parameter over its pooled chains.

    Parameters
    ----------
    x : _F64
        A 3D array of shape (n_chains, n_samples, n_parameters) containing the samples.
    grid_points : int
        Number of evenly spaced points spanning each parameter's sample range,
        at least 2. The resulting grid is equivalent to a ``np.linspace`` call.
    bandwidth : str | float
        ``"scott"``, ``"silverman"``, or a positive factor on the sample standard
        deviation.

    Returns
    -------
    tuple[tuple[_F64, _F64], ...]
        A tuple of (grid, density) per parameter, both shaped (grid_points,).
    """
    ...
