"""Native estimation kernels: the packed log-prior program and the correlation
covariance transforms.

Re-exports the compiled ``_prior_program`` extension. The estimation stack is
native-only; if the extension is not built, importing this module raises
``ImportError``.
"""

from ._prior_program import (
    cov_from_unconstrained,
    dist_logpdf,
    lkj_chol_logjac,
    lkj_chol_logpdf_from_z,
    logprior,
    logprior_program,
    transform_inverse_and_logjac,
    unconstrained_from_corr_chol,
)
from ._estimation import run_estimation, run_mcmc, loglik, logpost

__all__ = [
    "run_mcmc",
    "run_estimation",
    "loglik",
    "logpost",
    "cov_from_unconstrained",
    "dist_logpdf",
    "lkj_chol_logjac",
    "lkj_chol_logpdf_from_z",
    "logprior",
    "logprior_program",
    "transform_inverse_and_logjac",
    "unconstrained_from_corr_chol",
]
