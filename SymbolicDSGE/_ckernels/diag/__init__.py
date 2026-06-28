"""Native diagnostic-test kernels (Breusch-Godfrey, Breusch-Pagan, Chow, CUSUM).

Re-exports the compiled ``_diag`` extension. If it is not built, importing this
module raises ``ImportError`` and the consumers (``SymbolicDSGE._diag_tests``)
fall back to their numba kernels.
"""

from ._diag import (
    FALLBACK as FALLBACK,
    acorr as acorr,
    bg_stat as bg_stat,
    bp_aux as bp_aux,
    chow_stat as chow_stat,
    cusum_series as cusum_series,
    cusum_stat as cusum_stat,
    cusumsq_stat as cusumsq_stat,
    fill_centered_ax0 as fill_centered_ax0,
    fill_mean_ax0 as fill_mean_ax0,
    fill_symmetric_target_vec as fill_symmetric_target_vec,
    hac_estimator_matmul as hac_estimator_matmul,
    lb_stat as lb_stat,
    recursive_residuals as recursive_residuals,
    symmetric_outer_prod_2dim as symmetric_outer_prod_2dim,
    wald_stat_from_mean_and_cov as wald_stat_from_mean_and_cov,
)
