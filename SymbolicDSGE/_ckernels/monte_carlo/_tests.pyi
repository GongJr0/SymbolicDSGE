from numpy import float64
from numpy.typing import NDArray

_F64 = NDArray[float64]

def ljung_box_fit(
    x: _F64,
    lags: int,
    z_scratch: _F64,
    acorr_scratch: _F64,
) -> tuple[float64, int]: ...
def jarque_bera_fit(x: _F64) -> tuple[float64, int]: ...
def breusch_pagan_fit(
    residuals: _F64,
    X_aug: _F64,
    robust: bool,
    arena: _F64,
) -> tuple[float64, int]: ...
def breusch_godfrey_fit(
    residuals: _F64,
    X: _F64,
    lags: int,
    arena: _F64,
) -> tuple[float64, int]: ...
def chow_fit(
    y: _F64,
    X: _F64,
    t_break: int,
    arena: _F64,
) -> tuple[float64, int]: ...
def cusum_fit(y: _F64, X: _F64, arena: _F64) -> tuple[float64, int]: ...
def cusumsq_fit(y: _F64, X: _F64, arena: _F64) -> tuple[float64, int]: ...
