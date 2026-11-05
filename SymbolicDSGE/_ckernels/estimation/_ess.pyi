from numpy import float64
from numpy.typing import NDArray

_F64 = NDArray[float64]

def ess1d(x: _F64) -> float:
    """Compute the effective sample size of a 1D sample.

    Does not raise, returns NaN when any raise condition is met.

    Parameters
    ----------
    x : NDArray[float64]
        Input sample, 1D array of floats.

    Returns
    -------
    float
        ESS of the input sample.

    """
    ...

def ess(x: _F64) -> list[float]:
    """Compute the ESS per column of a 2D sample.

    Does not raise, returns NaNs for columns that meet any raise condition.

    Parameters
    ----------
    x : _F64
        Sample values to compute the ESS for.

    Returns
    -------
    list[float]
        Array of ESS values, one per column of the input sample.

    """
    ...

def ess1d_kernel_probe(x: _F64) -> tuple[int, float]:
    """C kernel forwarding function, retuns (rc, ess) for a 1D sample.

    Meant for testing, ess1d() and ess() should be used in library code.

    Parameters
    ----------
    x : NDArray[float64]
        Input sample, 1D array of floats.

    Returns
    -------
    tuple[int, float]
        Return code and ESS for the input sample.

    """
    ...
