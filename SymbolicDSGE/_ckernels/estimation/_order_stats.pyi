from numpy import float64
from numpy.typing import NDArray

_F64 = NDArray[float64]

def hpd_intervals(samples: _F64, alpha: float) -> tuple[_F64, _F64]:
    """
    Compute the highest posterior density (HPD) intervals for a given set of samples.

    Parameters
    ----------
    samples : _F64
        A 3D array of shape (n_chains, n_samples, n_parameters) containing the samples.
    alpha : float
        The significance level for the HPD interval.

    Returns
    -------
    tuple[_F64, _F64]
        Lower and upper bounds as arrays in parameter order.
    """
    ...
