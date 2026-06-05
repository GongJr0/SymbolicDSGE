from numba import njit
import numpy as np
from numpy.typing import NDArray

NDF = NDArray[np.float64]


@njit(cache=True)
def simulate_linear_states_into(
    A: NDF,
    B: NDF,
    x0: NDF,
    shock_mat: NDF,
    out: NDF,
) -> None:
    T = shock_mat.shape[0]
    n = A.shape[0]
    k = B.shape[1]

    for i in range(n):
        out[0, i] = x0[i]

    for t in range(T):
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i, j] * out[t, j]
            for j in range(k):
                s += B[i, j] * shock_mat[t, j]
            out[t + 1, i] = s


@njit(cache=True)
def affine_observations_into(
    states: NDF,
    C: NDF,
    d: NDF,
    state_start: int,
    out: NDF,
) -> None:
    T = out.shape[0]
    m = C.shape[0]
    n = C.shape[1]

    for t in range(T):
        state_row = state_start + t
        for i in range(m):
            s = d[i]
            for j in range(n):
                s += C[i, j] * states[state_row, j]
            out[t, i] = s
