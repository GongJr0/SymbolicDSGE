"""Type stubs for the compiled ``_core`` extension.

The native kernels carry no inspectable type information (the type checker never
parses ``_core.pyx`` nor introspects the compiled object), so these signatures
exist solely to give the LSP and mypy the shapes of the exported functions. They
must stay in sync with ``_core.pyx`` / ``core.c`` and the reference oracles in
``tests/_oracles/core.py``; the parity tests guard the runtime behavior, not this
stub.
"""

from numpy import complex128, float64, int8
from numpy.typing import NDArray

_F64 = NDArray[float64]
_C128 = NDArray[complex128]
_I8 = NDArray[int8]

#: ``SDSGE_INC_*`` from ``pencil.h``, re-exported so the compiler and the solve
#: share one definition of the incidence bits.
INC_LAG: int
INC_CUR: int
INC_LEAD: int

def assemble_transition(
    p: _F64,
    f: _F64,
    n_state: int,
    n_control: int,
) -> _F64:
    """A <- the first-order transition from (p, f). The shock loading is the
    pencil stage's own output, not assembled from these."""

def simulate_linear_states_into(
    A: _F64,
    B: _F64,
    x0: _F64,
    shock_mat: _F64,
    out: _F64,
    steady_state: _F64 | None = ...,
) -> None:
    """out[(T, n)] <- linear state recursion. ``x0`` and the recursion are
    deviations; ``steady_state`` denominates the written rows in levels."""

def affine_observations_into(
    states: _F64,
    C: _F64,
    d: _F64,
    out: _F64,
) -> None:
    """out[(T, m)] <- d + C @ states[t]. Mirrors the numba kernel."""

def simulate_second_order_pruned(
    hx: _F64,
    gx: _F64,
    bu: _F64,
    hxx: _F64,
    gxx: _F64,
    hxu: _F64,
    gxu: _F64,
    huu: _F64,
    guu: _F64,
    hss: _F64,
    gss: _F64,
    x0: _F64,
    shock_mat: _F64,
    steady_state: _F64 | None = ...,
) -> _F64:
    """Pruned second order simulation. Returns the stacked variable path. ``bu``
    spans every variable: a control responds to an innovation contemporaneously.
    ``steady_state`` denominates the returned rows in levels."""

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
    n_exog: int,
) -> tuple[_F64, _F64, _F64, _F64]:
    """Complex-step (a, b, c, d) from a residual @cfunc address, so the system
    reads ``a y' = b y + c y_prev + d eps``. ``d`` is (n_eq, n_exog), the rest
    (n_eq, n_var)."""

def pencil_dim(incidence: _I8, n_var: int) -> int:
    """Pencil size ``ndynamic + n_both`` from an incidence. Not bounded by
    n_var: a mixed lag/lead variable needs a companion row."""

def klein_qz(a: _F64 | _C128, b: _F64 | _C128) -> tuple[_C128, _C128, _C128]:
    """Ordered generalized Schur (QZ) with the Klein 'ouc' ordering via LAPACK
    zgges. Returns (s, t, z) == scipy.linalg.ordqz(a, b, sort='ouc',
    output='complex')[0, 1, 5]."""

def steady_state_newton(
    residual_addr: int,
    seed: _F64,
    params: _F64,
    n_exog: int,
    max_iter: int = ...,
    tol: float = ...,
) -> tuple[_F64, int]:
    """Newton solve of F(ss, ss, ss) = 0 at a zero innovation from a residual
    @cfunc address; returns (ss, iters). Jacobian a - b - c via klein_preproc,
    step via f64 LU."""

def klein_solve1(
    residual_addr: int,
    seed: _F64,
    params: _F64,
    incidence: _I8,
    n_state: int,
    n_exog: int = ...,
) -> tuple[_F64, _F64, _F64, int, _C128, _F64, _F64]:
    """(ss, f, p, stab, eig, A, B) <- one-shot first-order Klein solve.

    Fuses steady_state_newton, klein_preprocess, klein_qz, klein_postprocess and
    assemble_transition into one GIL release. ``f``/``p`` are real; the Schur
    form's imaginary parts are roundoff on a real pencil. ``stab`` is reported,
    not raised on."""

def sgu_klein_solve2(
    residual_addr: int,
    bc_residual_addr: int,
    seed: _F64,
    params: _F64,
    Q: _F64,
    incidence: _I8,
    n_state: int,
    n_exog: int = ...,
) -> tuple[
    _F64,
    _F64,
    _F64,
    int,
    _C128,
    _F64,
    _F64,
    _F64,
    _F64,
    _F64,
    _F64,
    _F64,
    _F64,
    _F64,
    _F64,
]:
    """(ss, f, p, stab, eig, gxx, hxx, gxu, hxu, guu, huu, gss, hss, A, B) <-
    one-shot second-order solve.

    klein_solve1 plus bicomplex_hessian and second_order in one GIL release. The
    pencil and the residual Hessian stay native. ``Q`` is the (n_exog, n_exog)
    shock covariance, which the risk correction integrates against. ``stab`` is
    reported, not raised on."""

def second_order(
    a: _F64,
    b: _F64,
    f_xx: _F64,
    gx: _F64,
    hx: _F64,
    bu: _F64,
    Q: _F64,
    n_state: int,
) -> tuple[_F64, _F64, _F64, _F64, _F64, _F64, _F64, _F64]:
    """Second-order policy tensors (gxx, hxx, gxu, hxu, guu, huu, gss, hss) --
    native twin of core.second_order.solve_second_order."""

def residual_path(
    residual_addr: int,
    cur_states: _C128,
    fwd_states: _C128,
    prev_states: _C128,
    shocks: _C128,
    params: _C128,
    n_eq: int,
) -> _F64:
    """Real residual matrix (n_steps, n_eq) from a residual @cfunc over a path.
    ``prev_states`` is (n_steps, n_var) and ``shocks`` (n_steps, n_exog)."""

def residual_eval(
    residual_addr: int,
    fwd: _C128,
    cur: _C128,
    prev: _C128,
    eps: _C128,
    params: _C128,
    n_eq: int,
) -> _C128:
    """Complex residual vector (n_eq,) from a residual @cfunc address at a single
    (fwd, cur, prev, eps, par) point."""

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
    n_exog: int,
    n_eq: int,
) -> _F64:
    """Residual Hessian (n_eq, nz, nz) via the bicomplex step, over
    ``z = (lag, cur, lead, eps)`` with ``nz = 3*n_var + n_exog``.

    Spanning every date and the innovations is what lets the chain rule contract
    it to state space in one step."""

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
