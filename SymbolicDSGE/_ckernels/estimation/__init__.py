"""Native estimation kernels (the packed log-prior program).

Re-exports the compiled ``_prior_program`` extension. If it is not built,
importing this module raises ``ImportError`` and the consumer
(``SymbolicDSGE.estimation.prior_program``) falls back to its numba kernel.
"""

from ._prior_program import (
    dist_logpdf as dist_logpdf,
    lkj_chol_logjac as lkj_chol_logjac,
    lkj_chol_logpdf_from_z as lkj_chol_logpdf_from_z,
    logprior_program as logprior_program,
    transform_inverse_and_logjac as transform_inverse_and_logjac,
)
