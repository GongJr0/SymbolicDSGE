"""Native diagnostic-test kernels (Breusch-Godfrey, Breusch-Pagan, Chow, CUSUM).

Re-exports the compiled ``_diag`` extension. If it is not built, importing this
module raises ``ImportError`` and the consumers (``SymbolicDSGE._diag_tests``)
fall back to their numba kernels.
"""

from ._diag import (
    FALLBACK as FALLBACK,
    bg_stat as bg_stat,
    bp_aux as bp_aux,
    chow_stat as chow_stat,
    cusum_series as cusum_series,
    cusum_stat as cusum_stat,
    cusumsq_stat as cusumsq_stat,
    hac_estimator_matmul as hac_estimator_matmul,
    recursive_residuals as recursive_residuals,
)
