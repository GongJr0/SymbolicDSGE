"""Native estimation kernels: the packed log-prior program and the correlation
covariance transforms.

If the extension is not built, importing this module raises ``ImportError``.
"""

from ._prior_program import (
    cov_from_unconstrained,
    unconstrained_from_corr_chol,
)
from ._estimation import (
    run_estimation,
    run_mcmc,
    loglik,
    logprior,
    logpost,
)

from ._ess import ess_rhat
from ._order_stats import hpd_intervals

__all__ = [
    "run_mcmc",
    "run_estimation",
    "loglik",
    "logprior",
    "logpost",
    "cov_from_unconstrained",
    "unconstrained_from_corr_chol",
    "ess_rhat",
    "hpd_intervals",
]
