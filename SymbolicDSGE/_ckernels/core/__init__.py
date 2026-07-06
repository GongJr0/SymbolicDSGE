"""Native core kernels (state-space simulation, affine observations).

Re-exports the compiled ``_core`` extension. If the extension is not built,
importing this module raises ``ImportError`` and the consumer
(``SymbolicDSGE.core.simulation``) falls back to its numba kernels.
"""

from ._core import (
    affine_observations_into as affine_observations_into,
    bicomplex_hessian as bicomplex_hessian,
    klein_postprocess as klein_postprocess,
    klein_preprocess as klein_preprocess,
    residual_path as residual_path,
    second_order as second_order,
    second_order_risk as second_order_risk,
    simulate_linear_states_into as simulate_linear_states_into,
    simulate_second_order_pruned as simulate_second_order_pruned,
    steady_state_newton as steady_state_newton,
)
