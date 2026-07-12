"""Native core kernels (state-space simulation, affine observations, Klein/SGU
perturbation drivers).

Re-exports the compiled ``_core`` extension, which is mandatory: if it is not
built, importing this module (and the library) raises ``ImportError``.
"""

from ._core import (
    affine_observations_into,
    bicomplex_hessian,
    jacobian_eval,
    klein_postprocess,
    klein_preprocess,
    measurement_eval,
    measurement_path,
    residual_eval,
    residual_path,
    second_order,
    second_order_risk,
    simulate_linear_states_into,
    simulate_second_order_pruned,
    steady_state_newton,
)

__all__ = [
    "affine_observations_into",
    "bicomplex_hessian",
    "jacobian_eval",
    "klein_postprocess",
    "klein_preprocess",
    "measurement_eval",
    "measurement_path",
    "residual_eval",
    "residual_path",
    "second_order",
    "second_order_risk",
    "simulate_linear_states_into",
    "simulate_second_order_pruned",
    "steady_state_newton",
]
