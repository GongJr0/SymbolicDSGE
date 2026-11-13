"""Type stubs for the compiled ``_estimation`` extension.

The native composer carries no inspectable type information, so these signatures
exist solely for the LSP / mypy. They must stay in sync with ``_estimation.pyx``;
the parity tests guard the runtime behavior, not this stub. ``run_estimation`` is
the production optimizer driver; the ``obj_*_base`` entries are the n_theta == 0
parity harnesses.
"""

from typing import Any

import numpy as np
from numpy import float64
from numpy.typing import NDArray

_F64 = NDArray[float64]

def run_estimation(
    ctx_dto: Any,
    method: str,
    theta0: _F64,
    lo: _F64,
    hi: _F64,
    has_priors: bool,
    include_logjac: bool,
    m: int,
    maxiter: int,
    maxfun: int,
    maxls: int,
    factr: float,
    pgtol: float,
    fd_step: float,
    xatol: float,
    fatol: float,
    compute_cov: bool,
    cov_fd_step_scale: float,
    cov_fd_absolute_floor: float,
) -> dict[str, Any]: ...
def run_mcmc(
    ctx_dto: Any,
    theta0: _F64,
    seed: np.uint64,
    n_draws: int,
    burn_in: int,
    thin: int,
    adapt: bool,
    adapt_start: int,
    adapt_end: int,
    adapt_interval: int,
    proposal_scale: float,
    proposal_cov: _F64 | None,
    cov_fd_step_scale: float,
    cov_fd_absolute_floor: float,
    adapt_epsilon: float,
    compute_map: bool,
    map_options: dict[str, Any] | None,
) -> dict[str, Any]: ...
def loglik(ctx_dto: Any, theta: _F64) -> float64: ...
def logprior(ctx_dto: Any, theta: _F64, jacobian: bool = False) -> float64: ...
def logpost(ctx_dto: Any, theta: _F64, jacobian: bool = False) -> float64: ...
