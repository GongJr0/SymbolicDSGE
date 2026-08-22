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

def run_estimation(
    ctx_dto: Any,
    mode: str,
    method: str,
    theta0: _F64,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    has_priors: bool = False,
    include_logjac: bool = False,
    m: int = 10,
    maxiter: int = 15000,
    maxfun: int = 15000,
    maxls: int = 20,
    factr: float = 1e7,
    pgtol: float = 1e-5,
    fd_step: float = 0.0,
    xatol: float = 1e-4,
    fatol: float = 1e-4,
    compute_cov: bool = True,
    cov_fd_step_scale: float = 1.0,
    cov_fd_absolute_floor: float = 0.1,
) -> dict[str, Any]: ...
def run_mcmc(
    ctx_dto: Any,
    mode: str,
    theta0: _F64,
    rng: np.random.Generator,
    n_draws: int,
    burn_in: int = 1000,
    thin: int = 1,
    adapt: bool = True,
    adapt_start: int = 100,
    proposal_scale: float = 0.1,
    proposal_cov: _F64 | None = None,
    cov_fd_step_scale: float = 1.0,
    cov_fd_absolute_floor: float = 0.1,
    adapt_epsilon: float = 1e-8,
    compute_map: bool = True,
    map_options: dict[str, Any] | None = None,
) -> dict[str, Any]: ...

class NativeLogpost:
    def __init__(self, ctx_dto: Any, mode: str) -> None: ...
    def loglik(self, theta: _F64) -> float: ...
    def logpost(self, theta: _F64) -> float: ...
