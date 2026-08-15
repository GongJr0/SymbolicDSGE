"""Native core kernels (state-space simulation, affine observations, Klein and
second-order perturbation drivers).

Re-exports the compiled ``_core`` extension, which is mandatory: if it is not
built, importing this module (and the library) raises ``ImportError``.
"""

from ._core import (
    INC_CUR,
    INC_LAG,
    INC_LEAD,
    assemble_transition,
    affine_observations_into,
    bicomplex_hessian,
    jacobian_eval,
    klein_postprocess,
    klein_preprocess,
    klein_qz,
    klein_solve1,
    measurement_eval,
    measurement_path,
    pencil_dim,
    residual_eval,
    residual_path,
    second_order,
    sgu_klein_solve2,
    simulate_linear_states_into,
    simulate_second_order_pruned,
    steady_state_newton,
)

__all__ = [
    "INC_CUR",
    "INC_LAG",
    "INC_LEAD",
    "assemble_transition",
    "affine_observations_into",
    "bicomplex_hessian",
    "jacobian_eval",
    "klein_postprocess",
    "klein_preprocess",
    "klein_qz",
    "klein_solve1",
    "measurement_eval",
    "measurement_path",
    "pencil_dim",
    "residual_eval",
    "residual_path",
    "second_order",
    "sgu_klein_solve2",
    "simulate_linear_states_into",
    "simulate_second_order_pruned",
    "steady_state_newton",
]
