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
