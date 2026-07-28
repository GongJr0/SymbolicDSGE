from numpy import float64, int64
from numpy.typing import NDArray

_F64 = NDArray[float64]

def ljung_box_runner(
    x: _F64,
    lags: int,
    z_scratch: _F64,
    acorr_scratch: _F64,
) -> tuple[float64, int]: ...
def jarque_bera_runner(x: _F64) -> tuple[float64, int]: ...
def wald_runner(
    mean: _F64,
    target: _F64,
    omega: _F64,
    n: int,
    dev_scratch: _F64,
    factor_scratch: _F64,
    pivot_scratch: NDArray[int64],
    solved_scratch: _F64,
) -> tuple[float64, int]: ...
def breusch_pagan_runner(
    residuals: _F64,
    X_aug: _F64,
    robust: bool,
    arena: _F64,
) -> tuple[float64, int]: ...
def breusch_godfrey_runner(
    residuals: _F64,
    X: _F64,
    lags: int,
    arena: _F64,
) -> tuple[float64, int]: ...
def chow_runner(
    y: _F64,
    X: _F64,
    t_break: int,
    arena: _F64,
) -> tuple[float64, int]: ...
def cusum_runner(y: _F64, X: _F64, arena: _F64) -> tuple[float64, int]: ...
def cusumsq_runner(y: _F64, X: _F64, arena: _F64) -> tuple[float64, int]: ...
