"""Native regression kernels (the ridge normal-equation solve + grid search).

Re-exports the compiled ``_regression`` extension. If it is not built, importing
this module raises ``ImportError`` and the consumer (``SymbolicDSGE.regression``)
falls back to its numba kernels.
"""

from ._regression import (
    chol_solve_L2 as chol_solve_L2,
    lars_lasso_gram as lars_lasso_gram,
    lasso_gram_cd as lasso_gram_cd,
    lasso_path_eval as lasso_path_eval,
    ridge_grid_search as ridge_grid_search,
)
