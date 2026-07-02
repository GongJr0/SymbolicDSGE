"""Native core kernels (state-space simulation, affine observations).

Re-exports the compiled ``_core`` extension. If the extension is not built,
importing this module raises ``ImportError`` and the consumer
(``SymbolicDSGE.core.simulation``) falls back to its numba kernels.
"""

from ._core import (
    affine_observations_into as affine_observations_into,
    klein_postprocess as klein_postprocess,
    klein_preprocess as klein_preprocess,
    simulate_linear_states_into as simulate_linear_states_into,
)
