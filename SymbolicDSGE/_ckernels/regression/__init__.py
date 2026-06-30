"""Native regression kernels (the ridge normal-equation solve + grid search).

Re-exports the compiled ``_regression`` extension. If it is not built, importing
this module raises ``ImportError`` and the consumer (``SymbolicDSGE.regression``)
falls back to its numba kernels.
"""

from ._regression import (
    chol_solve_L2 as chol_solve_L2,
    elastic_net_active_dof as elastic_net_active_dof,
    elastic_net_gram_cd as elastic_net_gram_cd,
    elastic_net_gram_cd_path as elastic_net_gram_cd_path,
    lars_lasso_gram as lars_lasso_gram,
    lasso_gram_cd as lasso_gram_cd,
    lasso_path_eval as lasso_path_eval,
    ols_chol_solve as ols_chol_solve,
    ridge_grid_search as ridge_grid_search,
)
