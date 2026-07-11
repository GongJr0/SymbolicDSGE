"""Native diagnostic-test kernels (Breusch-Godfrey, Breusch-Pagan, Chow, CUSUM).

Re-exports the compiled ``_diag`` extension. If it is not built, importing this
module raises ``ImportError`` and the consumers (``SymbolicDSGE._diag_tests``)
fall back to their numba kernels.
"""

from ._diag import (
    FALLBACK as FALLBACK,
    acorr,
    bg_stat,
    bp_aux,
    chow_stat,
    cusum_series,
    cusum_stat,
    cusumsq_stat,
    fill_centered_ax0,
    fill_mean_ax0,
    fill_symmetric_target_vec,
    hac_estimator_matmul,
    jb_stat,
    lb_stat,
    recursive_residuals,
    symmetric_outer_prod_2dim,
    wald_stat_from_mean_and_cov,
)

__all__ = [
    "FALLBACK",
    "acorr",
    "bg_stat",
    "bp_aux",
    "chow_stat",
    "cusum_series",
    "cusum_stat",
    "cusumsq_stat",
    "fill_centered_ax0",
    "fill_mean_ax0",
    "fill_symmetric_target_vec",
    "hac_estimator_matmul",
    "jb_stat",
    "lb_stat",
    "recursive_residuals",
    "symmetric_outer_prod_2dim",
    "wald_stat_from_mean_and_cov",
]
