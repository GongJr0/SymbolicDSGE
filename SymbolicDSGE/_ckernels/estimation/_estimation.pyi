"""Type stubs for the compiled ``_estimation`` extension.

The native composer carries no inspectable type information, so these signatures
exist solely for the LSP / mypy. They must stay in sync with ``_estimation.pyx``;
the parity tests guard the runtime behavior, not this stub. ``run_estimation`` is
the production optimizer driver; the ``obj_*_base`` entries are the n_theta == 0
parity harnesses.
"""

from typing import Any, Sequence

import numpy as np
from numpy import float64
from numpy.typing import NDArray

_F64 = NDArray[float64]

def obj_linear_base(
    residual_addr: int,
    meas_addr: int,
    jac_addr: int,
    n_state: int,
    n_exog: int,
    n_obs: int,
    ss_seed: _F64,
    base_calib: _F64,
    Q: _F64,
    R: _F64,
    y: _F64,
    P0: _F64,
    jitter: float,
    symmetrize: int,
) -> tuple[float, int]: ...
def obj_extended_base(
    residual_addr: int,
    meas_addr: int,
    jac_addr: int,
    n_state: int,
    n_exog: int,
    n_obs: int,
    ss_seed: _F64,
    base_calib: _F64,
    Q: _F64,
    R: _F64,
    y: _F64,
    P0: _F64,
    jitter: float,
    symmetrize: int,
) -> tuple[float, int]: ...
def obj_unscented_base(
    residual_addr: int,
    bc_residual_addr: int,
    meas_addr: int,
    n_state: int,
    n_exog: int,
    n_obs: int,
    ss_seed: _F64,
    base_calib: _F64,
    Q: _F64,
    R: _F64,
    y: _F64,
    P0: _F64,
    jitter: float,
    symmetrize: int,
    alpha: float = ...,
    beta: float = ...,
    kappa: float = ...,
) -> tuple[float, int]: ...
def run_estimation(
    ctx_dto: Any,
    mode: str,
    method: str,
    theta0: _F64,
    bounds: Sequence[tuple[float | None, float | None]] | None = ...,
    has_priors: int = ...,
    m: int = ...,
    maxiter: int = ...,
    maxfun: int = ...,
    maxls: int = ...,
    factr: float = ...,
    pgtol: float = ...,
    fd_step: float = ...,
    xatol: float = ...,
    fatol: float = ...,
) -> dict[str, Any]: ...
def run_mcmc(
    ctx_dto: Any,
    mode: str,
    theta0: _F64,
    rng: np.random.Generator,
    n_draws: int,
    burn_in: int = ...,
    thin: int = ...,
    adapt: int = ...,
    adapt_start: int = ...,
    adapt_interval: int = ...,
    proposal_scale: float = ...,
    adapt_epsilon: float = ...,
    hessian_fd_step_scale: float = ...,
    hessian_fd_absolute_floor: float = ...,
    map_method: str = ...,
    map_bounds: Sequence[tuple[float | None, float | None]] | None = ...,
    map_m: int = ...,
    map_maxiter: int = ...,
    map_maxfun: int = ...,
    map_maxls: int = ...,
    map_factr: float = ...,
    map_pgtol: float = ...,
    map_fd_step: float = ...,
    map_xatol: float = ...,
    map_fatol: float = ...,
) -> dict[str, Any]: ...

class NativeLogpost:
    def __init__(self, ctx_dto: Any, mode: str) -> None: ...
    def loglik(self, theta: _F64) -> float: ...
    def logpost(self, theta: _F64) -> float: ...
