"""Type stubs for the compiled ``_core`` extension.

The native kernels carry no inspectable type information (the type checker never
parses ``_core.pyx`` nor introspects the compiled object), so these signatures
exist solely to give the LSP and mypy the shapes of the exported functions. They
must stay in sync with ``_core.pyx`` / ``core.c`` and the reference oracles in
``tests/_oracles/core.py``; the parity tests guard the runtime behavior, not this
stub.
"""

from numpy import complex128, float64
from numpy.typing import NDArray

_F64 = NDArray[float64]
_C128 = NDArray[complex128]

def simulate_linear_states_into(
    A: _F64,
    B: _F64,
    x0: _F64,
    shock_mat: _F64,
    out: _F64,
) -> None:
    """out[(T+1, n)] <- linear state recursion. Mirrors the numba kernel."""

def affine_observations_into(
    states: _F64,
    C: _F64,
    d: _F64,
    state_start: int,
    out: _F64,
) -> None:
    """out[(T, m)] <- d + C @ states[state_start + t]. Mirrors the numba kernel."""

def simulate_second_order_pruned(
    hx: _F64,
    gx: _F64,
    bx: _F64,
    hxx: _F64,
    gxx: _F64,
    hss: _F64,
    gss: _F64,
    x0: _F64,
    shock_mat: _F64,
) -> tuple[_F64, _F64]:
    """Pruned second order simulation. Returns split state and jump paths."""

def klein_postprocess(
    s: _C128,
    t: _C128,
    z: _C128,
    n_states: int,
) -> tuple[_C128, _C128, int, _C128]:
    """(f, p, stab, eig) from the ordered Schur factors."""

def spike_drive(fn_addr: int, a: _C128, b: _C128, out: _C128) -> None:
    """Stage-0 (#248): call a numba @cfunc (by ``.address``) from native C, nogil."""

def klein_preprocess(
    residual_addr: int,
    steady_state: _F64,
    params: _F64,
    n_eq: int,
    log_linear: bool,
) -> tuple[_F64, _F64]:
    """Complex-step (a, b) from a residual @cfunc address. Native twin of
    ``klein._approximate_system_numeric``."""

def steady_state_newton(
    residual_addr: int,
    seed: _F64,
    params: _F64,
    max_iter: int = ...,
    tol: float = ...,
) -> tuple[_F64, int]:
    """Newton solve of F(ss, ss) = 0 from a residual @cfunc address; returns
    (ss, iters). Jacobian a - b via klein_preproc, step via f64 LU."""

def second_order(
    a: _F64,
    b: _F64,
    f_xx: _F64,
    gx: _F64,
    hx: _F64,
    n_state: int,
) -> tuple[_F64, _F64]:
    """SGU second-order tensors (gxx, hxx) -- native twin of
    core.second_order.solve_second_order."""

def second_order_risk(
    a: _F64,
    b: _F64,
    f_xx: _F64,
    gx: _F64,
    gxx: _F64,
    eta: _F64,
    n_state: int,
) -> tuple[_F64, _F64]:
    """Sigma^2 risk correction (gss, hss) -- native twin of
    core.second_order.solve_second_order_risk."""

def residual_path(
    residual_addr: int,
    cur_states: _C128,
    fwd_states: _C128,
    params: _C128,
    n_eq: int,
) -> _F64:
    """Real residual matrix (n_steps, n_eq) from a residual @cfunc over a path."""

def residual_eval(
    residual_addr: int,
    fwd: _C128,
    cur: _C128,
    params: _C128,
    n_eq: int,
) -> _C128:
    """Complex residual vector (n_eq,) from a residual @cfunc address at a single
    (fwd, cur, par) point. Native twin of the old numba objective vector func."""

def measurement_eval(
    meas_addr: int,
    vars: _F64,
    par: _F64,
    n_obs: int,
) -> _F64:
    """Measurement vector (n_obs,) from a measurement @cfunc address at a single
    (vars, par) point. Native twin of the old numba observable funcs."""

def jacobian_eval(
    jac_addr: int,
    vars: _F64,
    par: _F64,
    n_obs: int,
    n_var: int,
) -> _F64:
    """Observable jacobian (n_obs, n_var) from a jacobian @cfunc address at a
    single (vars, par) point."""

def measurement_path(
    meas_addr: int,
    states: _F64,
    par: _F64,
    n_obs: int,
) -> _F64:
    """Measurement matrix (T, n_obs) from a measurement @cfunc over a (T, n_var)
    state path."""

def bicomplex_hessian(
    residual_addr: int,
    steady_state: _F64,
    params: _F64,
    n_eq: int,
    step: float = ...,
) -> _F64:
    """Residual Hessian (n_eq, 2*n_var, 2*n_var) via the bicomplex step."""

# --- bicomplex (bc256) primitives -------------------------------------------
# A bc256 crosses the boundary as the 4-tuple (real, i, j, ij).
_BC = tuple[float, float, float, float]

def bc_add(x: _BC, y: _BC) -> _BC: ...
def bc_sub(x: _BC, y: _BC) -> _BC: ...
def bc_neg(x: _BC) -> _BC: ...
def bc_mul(x: _BC, y: _BC) -> _BC: ...
def bc_div(x: _BC, y: _BC) -> _BC: ...
def bc_real_scale(x: _BC, s: float) -> _BC: ...
def bc_i_conj(x: _BC) -> _BC: ...
def bc_j_conj(x: _BC) -> _BC: ...
def bc_conj(x: _BC) -> _BC: ...
def bc_exp(x: _BC) -> _BC: ...
def bc_log(x: _BC) -> _BC: ...
def bc_spow(x: _BC, p: float) -> _BC: ...
def bc_ipow(X: _BC, p: int) -> _BC: ...
def bc_sqrt(x: _BC) -> _BC: ...
def c_sqrt(z: tuple[float, float]) -> tuple[float, float]: ...
def bc_cpow(x: _BC, y: _BC) -> _BC: ...
def bc_accessors(x: _BC) -> _BC: ...
def bc_proj(x: _BC) -> tuple[float, float, float, float]: ...
def bc_reconst(p: tuple[float, float, float, float]) -> _BC: ...
