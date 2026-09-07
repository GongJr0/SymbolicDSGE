"""Building the measurement equation of a compiled model.

Only the steady state comes from the policy, and every solution kind has one,
so nothing here is specific to how the model was solved.
"""

from __future__ import annotations

from typing import Tuple, Sequence

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..._ckernels.core import affine_observations_into, measurement_path
from ..compiled_model import CompiledModel

NDF = NDArray[float64]


def build_C_d_from_obs(
    compiled: CompiledModel, y_names: Sequence[str], ss: NDF
) -> Tuple[NDF, NDF]:
    """``(C, d)`` for the declared observables, linearized at ``ss``."""
    return compiled.build_affine_measurement_matrices(
        compiled.config.calibration.parameters,
        y_names,
        ss,
    )


def non_affine_measurement(
    compiled: CompiledModel,
    y_names: list[str],
    state: NDF,
) -> NDF:
    """Observables evaluated along a path, for a non-affine measurement."""
    # ``state`` is (T, n_var) in cur_syms canonical order (checked at compile).
    params = compiled.config.calibration.parameters
    param_vals = np.array(
        [float64(params[p]) for p in compiled.calib_params],
        dtype=float64,
    )

    # The measurement cfunc emits observables sorted by model index; map its
    # output columns back to the caller's y_names order.
    obs_sorted = compiled._normalize_observables(y_names)
    meas_addr = compiled.construct_measurement_cfunc(y_names).address
    raw = measurement_path(meas_addr, state, param_vals, len(obs_sorted))

    pos = {name: j for j, name in enumerate(obs_sorted)}
    perm = [pos[name] for name in y_names]
    return raw[:, perm]


def affine_path(states: NDF, C: NDF, d: NDF, n_obs: int, start: int) -> NDF:
    """Observables along a path from an affine measurement."""
    Y = np.empty((states.shape[0] - start, n_obs), dtype=float64)
    affine_observations_into(states, C, d, Y)
    return Y
