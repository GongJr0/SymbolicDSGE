from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..core.solved_model import SolvedModel
    from ..core.solver_backend import PerturbationSolution

import numpy as np
import pandas as pd
import sympy as sp
from numpy import asarray, float64
from numpy.typing import NDArray
from sympy import Symbol

from .._ckernels.core import measurement_eval, jacobian_eval
from .._ckernels.estimation import (
    cov_from_unconstrained,
    unconstrained_from_corr_chol,
)
from ..bayesian.priors import Prior
from ..core.compiled_model import CompiledModel
from ..core.config import SymbolGetterDict
from ..core.solver import DSGESolver
from ..kalman.config import KalmanConfig, make_R
from ..kalman.filter import KalmanFilter
from ..kalman.interface import _resolve_P0
from ..kalman.validator import FilterMode

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]


@dataclass(frozen=True)
class PreparedFilterRun:
    observables: list[str]
    y_reordered: NDF
    mode: str
    meas_addr: int
    jac_addr: int
    P0: NDF | None
    kf_jitter: float64
    kf_sym: bool


# ---------------------------------------------------------------------------
# Native estimation context DTOs (issue #330).
#
# Struct-shaped mirrors of the C context in
# ``_ckernels/estimation/estimation.h``: one dataclass per C struct, fields in
# struct order so the C header reads as the checklist. The Python producer fills
# these once per run (all name->index resolution and table flattening); the
# Cython composer maps them field-for-field onto the C structs, allocates the
# scratch buffers, and makes the single native call.
#
# Contract at this seam:
#   * Python pins DTYPE only. Every int64 array is ``np.int64`` and every
#     float64 array is ``np.float64``; contiguity is NOT guaranteed here.
#   * Cython enforces C-contiguity at the transmission layer (the deterministic
#     final cast), pins the arrays, and holds the keepalive across the ``nogil``
#     call. No defensive re-cast on the Python side.
#   * Count fields on the C structs (``n_scalars``, ``n_pairs``, ``n_scalar``,
#     ``n_blocks``) are NOT carried here: the composer derives each from its
#     array length, so length is the single source of truth and a count/array
#     mismatch is unrepresentable.
#   * Scratch buffers (``solve1``/``solve2``, ``Q``/``R``/``corr_*``/``std_*``,
#     ``params``, linear ``C``/``d``) have no Python source; the composer
#     allocates them from ``dims`` and they are intentionally absent from these
#     DTOs.
#   * Native outputs (``bk_violations``, the result struct) are absent too.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PyDims:
    """Mirror of ``sdsge_dims``: model and data dimensions (all i64)."""

    n_theta: int  # estimated params
    n_var: int  # nx + ny (pencil / filter dim)
    n_state: int  # nx
    n_ctrl: int  # ny
    n_exog: int  # k
    n_obs: int  # m
    n_par: int  # calib params
    T: int  # observations


def get_dims(compiled: CompiledModel, estimated_params: list[str], y: NDF) -> PyDims:
    n_var = len(compiled.var_names)
    return PyDims(
        n_theta=len(estimated_params),
        n_var=n_var,
        n_state=compiled.n_state,
        n_ctrl=n_var - compiled.n_state,
        n_exog=compiled.n_exog,
        n_obs=y.shape[1],
        n_par=len(compiled.calib_params),
        T=y.shape[0],
    )


@dataclass(frozen=True, slots=True)
class PyScalarScatter:
    """Mirror of ``sdsge_scalar_scatter``: one estimated scalar's theta->params
    scatter. ``transform_params`` is a ``np.float64`` array of length
    ``SDSGE_N_TRANSFORM_PARAMS``."""

    theta_idx: int
    param_slot: int
    transform_code: int
    transform_params: NDF


@dataclass(frozen=True, slots=True)
class PyParamMap:
    """Mirror of ``sdsge_param_map``: theta->params resolution tables.

    ``scalars`` is the array-of-structs the C ``scalars`` pointer addresses
    (``n_scalars`` = ``len(scalars)``). ``base_params`` and every slot index
    (``scalars`` ``param_slot``, and the cov specs' ``std_slots``/``pair_slot``)
    are in ``calib_params`` order, so ``params`` doubles as the residual argument
    vector with no gather step."""

    base_params: NDF  # n_par, calib_params order
    scalars: list[PyScalarScatter]  # n_scalars


@dataclass(frozen=True, slots=True)
class PyCovSpec:
    """Mirror of ``sdsge_cov_spec``: a Q or R covariance build spec.

    ``is_constant`` picks a loop-invariant ``constant`` (K*K, resolved once in
    prep) over the per-eval rebuild. When rebuilt, ``std_slots`` gives the K
    diagonal param slots; the correlation comes either from a CPC block
    (``corr_from_block`` with ``block_theta_off``/``block_theta_len`` into theta)
    or from the ``pair_i``/``pair_j``/``pair_slot`` triples
    (``n_pairs`` = ``len(pair_i)``)."""

    is_constant: bool
    constant: NDF | None  # K*K, or None
    K: int  # n_exog (Q) or n_obs (R)
    std_slots: NDI  # K
    corr_from_block: bool
    block_theta_off: int
    block_theta_len: int
    pair_i: NDI  # n_pairs
    pair_j: NDI  # n_pairs
    pair_slot: NDI  # n_pairs


@dataclass(frozen=True, slots=True)
class PyPriorTables:
    """Mirror of ``sdsge_prior_tables``: packed log-prior program arguments.

    ``has_prior`` gates the whole block. Scalar columns run to ``n_scalar`` =
    ``len(scalar_indices)``; ``scalar_dist_params`` is flattened n_scalar*5 and
    ``scalar_transform_params`` n_scalar*3. Matrix (CPC/LKJ) block columns run to
    ``n_blocks`` = ``len(matrix_offsets)``."""

    has_prior: bool
    scalar_indices: NDI  # n_scalar
    scalar_dist_codes: NDI  # n_scalar
    scalar_transform_codes: NDI  # n_scalar
    scalar_dist_params: NDF  # n_scalar*5
    scalar_transform_params: NDF  # n_scalar*3
    matrix_offsets: NDI  # n_blocks
    matrix_dims: NDI  # n_blocks
    matrix_lengths: NDI  # n_blocks
    matrix_etas: NDF  # n_blocks
    matrix_log_constants: NDF  # n_blocks


@dataclass(frozen=True, slots=True)
class PyObjCommon:
    """Mirror of ``sdsge_obj_common``: the mode-independent objective inputs.

    Runtime addresses arrive as ``int`` (cfunc ``.address`` / capsule pointer);
    ``zgges`` is absent because the composer pulls it from the scipy cython_lapack
    capsule, not from Python. The scratch fields on the C struct (``params``,
    ``Q``, ``R``, ``corr_q``, ``corr_r``, ``std_q``, ``std_r``) and the
    ``bk_violations`` output are composer-owned and omitted here."""

    dims: PyDims

    residual_addr: int
    bc_residual_addr: int  # bicomplex-Hessian residual; 0 when unused (linear)
    meas_addr: int
    jac_addr: int

    ss_seed: NDF  # n_var: Newton seed for the steady state
    log_linear: bool

    y: NDF  # T*n_obs
    P0: NDF  # n_var*n_var; UKF 2*n_state square
    x0: NDF | None  # n_var, or None
    jitter: float
    symmetrize: bool

    pmap: PyParamMap
    q_spec: PyCovSpec
    r_spec: PyCovSpec
    prior: PyPriorTables


@dataclass(frozen=True, slots=True)
class PyLinearContext:
    """Mirror of ``sdsge_linear_ctx``. The ``solve1`` buffers and the ``C``/``d``
    measurement-linearization outputs are composer-allocated scratch, so this
    wrapper adds no Python-provided fields beyond ``base``."""

    base: PyObjCommon


@dataclass(frozen=True, slots=True)
class PyExtendedContext:
    """Mirror of ``sdsge_extended_ctx``. ``solve1`` is composer-allocated scratch;
    no Python-provided fields beyond ``base``."""

    base: PyObjCommon


@dataclass(frozen=True, slots=True)
class PyUnscentedContext:
    """Mirror of ``sdsge_unscented_ctx``. ``solve1``/``solve2`` are
    composer-allocated scratch. ``z0`` is the Python-provided initial augmented
    state ``[x0_state; 0]`` of shape ``(2*n_state,)`` (the user's first-order
    ``x0``, given as ``n_state`` or full ``n_var`` and sliced to the state block;
    the tail is zeroed). ``alpha``/``beta``/``kappa`` are the UKF tuning scalars."""

    base: PyObjCommon
    z0: NDF  # 2*n_state
    alpha: float
    beta: float
    kappa: float


def extract_base_params(compiled: CompiledModel) -> dict[str, float64]:
    params = compiled.config.calibration.parameters
    return {str(k): float64(v) for k, v in params.items()}


def build_full_params(
    base_params: Mapping[str, float64],
    estimated_names: Sequence[str],
    theta: NDF,
) -> dict[str, float64]:
    if theta.ndim != 1:
        raise ValueError("theta must be a 1D array.")
    if len(theta) != len(estimated_names):
        raise ValueError(
            f"theta length {len(theta)} does not match estimated parameter count {len(estimated_names)}."
        )
    full = dict(base_params)
    for i, name in enumerate(estimated_names):
        full[name] = float64(theta[i])
    return full


def build_calib_param_vector(
    compiled: CompiledModel,
    params: Mapping[str, float64],
) -> NDF:
    names = [str(p) for p in compiled.calib_params]
    return asarray([float64(params[name]) for name in names], dtype=float64)


def reorder_observables(
    compiled: CompiledModel,
    observables: list[str] | None,
    y: NDF | pd.DataFrame,
) -> tuple[list[str], NDF]:
    canon = compiled.observable_names
    canon_idx = {name: i for i, name in enumerate(canon)}

    if observables is None:
        obs_given = list(canon)
    else:
        obs_given = list(observables)

    if len(obs_given) == 0:
        raise ValueError("Observable list is empty.")
    if len(set(obs_given)) != len(obs_given):
        raise ValueError("Observable list contains duplicates.")

    missing = [n for n in obs_given if n not in canon_idx]
    if missing:
        raise ValueError(f"Unknown observables not in compiled model: {missing}")

    obs_canonical = sorted(obs_given, key=lambda n: canon_idx[n])

    if isinstance(y, pd.DataFrame):
        missing_cols = [n for n in obs_given if n not in y.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing observable columns: {missing_cols}")
        # copy=True: pandas can hand back a read-only view under copy-on-write,
        # which the UKF hot loop (writable memoryview) rejects.
        y_reordered = y.loc[:, obs_canonical].to_numpy(dtype=float64, copy=True)
    else:
        y_arr = asarray(y, dtype=float64)
        if y_arr.ndim != 2:
            raise ValueError(
                f"Observation data must be 2D. Shape (T,m) expected, got {y_arr.shape}."
            )
        _, m = y_arr.shape
        if m != len(obs_given):
            raise ValueError(
                f"y has {m} columns but observable list has {len(obs_given)} names."
            )
        pos_in_given = {name: j for j, name in enumerate(obs_given)}
        y_reordered = y_arr[:, [pos_in_given[name] for name in obs_canonical]]

    if np.isnan(y_reordered).any():
        raise ValueError("Observation data contains NaN values.")

    return obs_canonical, y_reordered


def build_R(
    compiled: CompiledModel,
    kalman: KalmanConfig,
    observables: list[str],
    params: Mapping[str, float64],
    *,
    R_override: NDF | None = None,
) -> NDF:
    """Assemble the measurement covariance for a likelihood eval, mirroring
    :func:`build_Q`. Priority: a user-supplied ``R_override`` wins (validated to
    the observable count); else, if the config carries parser-generated
    std/correlation maps, R is rebuilt from the current ``params`` every eval
    exactly as Q is; else a fixed ``kalman.R`` (a directly-configured constant
    with no named params) is sliced to the observables as-is."""
    if R_override is not None:
        R = asarray(R_override, dtype=float64)
        m = len(observables)
        if R.shape != (m, m):
            raise ValueError(f"Provided R has shape {R.shape}, expected ({m}, {m}).")
        return R

    if kalman.R_std_param_map is not None:
        return build_R_from_config_params(
            compiled=compiled, kalman=kalman, observables=observables, params=params
        )

    if kalman.R is None:
        raise ValueError("R is not available. Provide `R` or a KalmanConfig with R.")
    obs_idx = {name: i for i, name in enumerate(compiled.observable_names)}
    mat_idx = [obs_idx[name] for name in observables]
    return asarray(kalman.R[np.ix_(mat_idx, mat_idx)], dtype=float64)


def build_Q(
    compiled: CompiledModel,
    params: Mapping[str, float64],
    *,
    corr: NDF | None = None,
) -> NDF:
    shock_map = compiled.config.shock_map
    shock_std = compiled.config.calibration.shock_std
    shock_corr = compiled.config.calibration.shock_corr

    exogs = compiled.var_names[: compiled.n_exog]
    rev: SymbolGetterDict[Symbol, Symbol] = SymbolGetterDict(
        {exo: shock for shock, exo in shock_map.items()}
    )
    shocks = [rev[exo] for exo in exogs]

    stds = asarray([float64(params[shock_std[s].name]) for s in shocks], dtype=float64)

    # When an LKJ block already materialized the shock correlation matrix (in exog
    # order) it is passed in directly, so we skip the name-keyed re-gather. Without
    # a block the correlations live in ``params`` as named scalars (fixed
    # calibration or plain estimated params) and are assembled here.
    if corr is None:
        corr = np.eye(len(exogs), dtype=float64)
        n = len(stds)
        for i in range(n):
            for j in range(i + 1, n):
                pair = frozenset({shocks[i], shocks[j]})
                corr_sym = shock_corr.get(pair, None)
                corr_ij = (
                    float64(params[corr_sym.name]) if corr_sym is not None else 0.0
                )
                corr[i, j] = corr_ij
                corr[j, i] = corr_ij
    return np.outer(stds, stds) * corr


def build_Q_symbolic(compiled: CompiledModel) -> sp.Matrix:
    shock_map = compiled.config.shock_map
    shock_std = compiled.config.calibration.shock_std
    shock_corr = compiled.config.calibration.shock_corr

    exogs = compiled.var_names[: compiled.n_exog]
    rev: SymbolGetterDict[Symbol, Symbol] = SymbolGetterDict(
        {exo: shock for shock, exo in shock_map.items()}
    )
    shocks = [rev[exo] for exo in exogs]

    stds = sp.Matrix([shock_std[s] for s in shocks])
    corr = sp.eye(len(exogs))

    n = len(stds)
    for i in range(n):
        for j in range(i + 1, n):
            pair = frozenset({shocks[i], shocks[j]})
            corr_sym = shock_corr.get(pair, None)
            corr_ij = corr_sym if corr_sym is not None else 0.0
            corr[i, j] = corr_ij
            corr[j, i] = corr_ij
    return (stds * stds.T).multiply_elementwise(corr)


def build_C_d_from_cfunc(
    meas_addr: int,
    jac_addr: int,
    ss: NDF,
    calib_params: NDF,
    n_obs: int,
) -> tuple[NDF, NDF]:
    n_var = ss.shape[0]
    d = measurement_eval(meas_addr, ss, calib_params, n_obs)
    C = jacobian_eval(jac_addr, ss, calib_params, n_obs, n_var)
    return C, d


def resolve_filter_options(
    jitter: float | float64 | None,
    symmetrize: bool | None,
) -> tuple[float64, bool]:
    kf_jitter = float64(0.0) if jitter is None else float64(jitter)
    kf_sym = False if symmetrize is None else bool(symmetrize)
    return kf_jitter, kf_sym


def prepare_filter_run(
    *,
    compiled: CompiledModel,
    kalman: KalmanConfig,
    y: NDF | pd.DataFrame,
    observables: list[str] | None,
    filter_mode: str,
    jitter: float | float64 | None,
    symmetrize: bool | None,
    P0: NDF | None = None,
) -> PreparedFilterRun:
    obs, y_reordered = reorder_observables(compiled, observables, y)
    mode = filter_mode

    kf_jitter, kf_sym = resolve_filter_options(jitter, symmetrize)

    return PreparedFilterRun(
        observables=obs,
        y_reordered=y_reordered,
        mode=mode,
        meas_addr=compiled.construct_measurement_cfunc(obs).address,
        jac_addr=compiled.construct_observable_jacobian_cfunc(obs).address,
        P0=_resolve_P0(
            FilterMode(mode),
            compiled.n_state,
            kalman.P0 if P0 is None else P0,
        ),
        kf_jitter=kf_jitter,
        kf_sym=kf_sym,
    )


def build_R_from_config_params(
    *,
    compiled: CompiledModel,
    kalman: KalmanConfig | None,
    observables: list[str],
    params: Mapping[str, float64],
) -> NDF:
    if kalman is None:
        raise ValueError("KalmanConfig is required to build R from config parameters.")
    std_map = kalman.R_std_param_map
    corr_map = kalman.R_corr_param_map
    if std_map is None:
        raise ValueError("KalmanConfig does not expose named R parameter metadata.")

    def _param(name: str) -> float64:
        if name not in params:
            raise KeyError(f"Missing R parameter '{name}' in params.")
        return float64(params[name])

    all_obs = compiled.observable_names
    y_syms = [Symbol(name) for name in all_obs]
    std_vals = {Symbol(name): _param(std_map[name]) for name in all_obs}
    corr_vals = {
        frozenset(Symbol(n) for n in pair): _param(param_name)
        for pair, param_name in (corr_map or {}).items()
        if param_name is not None
    }
    R_full = make_R(y_syms, std_vals, corr_vals)

    obs_idx = {name: i for i, name in enumerate(all_obs)}
    mat_idx = [obs_idx[name] for name in observables]
    return asarray(R_full[np.ix_(mat_idx, mat_idx)], dtype=float64)


def _get_solution(
    *,
    solver: DSGESolver,
    compiled: CompiledModel,
    params: Mapping[str, float64],
    mode: str,
    ss_seed: NDF | dict[str, float] | None,
    raise_on_bk_violation: bool = True,
) -> Any:
    """Solve the model to the order the filter mode requires.

    Unscented filtering consumes the second-order policy tensors (``order=2``);
    the linear and extended filters use the first-order ``A``/``B`` (``order=1``).
    The single ``mode -> order`` authority, so it cannot drift from the field
    reads in :func:`_prepare_filter_loglik`. ``raise_on_bk_violation`` reaches the
    solver unchanged: ``False`` for the warning-counted search path, ``True`` for
    the one-shot R estimators (which catch the raise and fall back to diagonal R).
    """
    return solver.solve(
        compiled=compiled,
        order=2 if mode == "unscented" else 1,
        parameters={k: float(v) for k, v in params.items()},
        ss_seed=ss_seed,
        raise_on_bk_violation=raise_on_bk_violation,
    )


def _prepare_filter_loglik(
    *,
    sol: SolvedModel,
    prepared: PreparedFilterRun,
    Q: NDF,
    calib_params: NDF,
    x0: NDF | None,
    raise_on_error: bool,
) -> Callable[[NDF], float64]:
    """Bind a prepared filter run to a closure ``R -> loglik``.

    The single filter-mode dispatch: builds the linear measurement ``(C, d)``
    once (hoisted out of any R-optimization loop), picks the matching
    ``KalmanFilter`` entry point, and raises on an unknown mode. The returned
    closure runs that filter for a given ``R`` and returns its log-likelihood.
    ``raise_on_error`` reaches the filter unchanged: ``False`` for the
    warning-counted search path, ``True`` for the one-shot R estimators.
    """
    mode = prepared.mode
    common: dict[str, Any] = dict(
        Q=Q,
        y=prepared.y_reordered,
        P0=prepared.P0,
        jitter=float(prepared.kf_jitter),
        symmetrize=prepared.kf_sym,
        _store_history=False,
        _raise_on_error=raise_on_error,
    )
    run_filter: Callable[..., Any]
    if mode == "linear":
        C, d = build_C_d_from_cfunc(
            prepared.meas_addr,
            prepared.jac_addr,
            sol.policy.steady_state,
            calib_params,
            prepared.y_reordered.shape[1],
        )
        run_filter = KalmanFilter.run_raw
        mode_args: dict[str, Any] = {
            "A": sol.A,
            "B": sol.B,
            "C": C,
            "d": d,
            "x0": x0,
            "return_shocks": False,
        }
    elif mode == "extended":
        run_filter = KalmanFilter.run_extended_raw
        mode_args = {
            "A": sol.A,
            "B": sol.B,
            "meas_addr": prepared.meas_addr,
            "jac_addr": prepared.jac_addr,
            "calib_params": calib_params,
            "compute_y_filt": False,
            "x0": x0,
            "return_shocks": False,
        }
    elif mode == "unscented":
        # hx is (n_state, n_state); recover n_state from it so bx and the
        # augmented z0 don't need `compiled` threaded in.
        pol = cast("PerturbationSolution", sol.policy)
        n_state = sol.policy.p.shape[0]
        if x0 is None:
            x0_state = np.zeros((n_state,), dtype=float64)
        else:
            raw = asarray(x0, dtype=float64)
            x0_state = raw[:n_state] if raw.shape[0] != n_state else raw
        z0 = np.zeros((2 * n_state,), dtype=float64)
        z0[:n_state] = x0_state
        run_filter = KalmanFilter.run_unscented_raw
        mode_args = {
            "meas_addr": prepared.meas_addr,
            "hx": pol.p,
            "gx": pol.f,
            "bx": asarray(sol.B[:n_state, :], dtype=float64),
            "hxx": pol.hxx,
            "gxx": pol.gxx,
            "hss": pol.hss,
            "gss": pol.gss,
            "steady_state": pol.steady_state,
            "calib_params": calib_params,
            "z0": z0,
        }
    else:
        raise ValueError(f"Unrecognized filter_mode: {mode!r}")

    def loglik_of_R(R: NDF) -> float64:
        run = run_filter(R=R, **mode_args, **common)
        return float64(run.loglik)

    return loglik_of_R


def evaluate_loglik(
    *,
    solver: DSGESolver,
    compiled: CompiledModel,
    kalman: KalmanConfig,
    y: NDF | pd.DataFrame,
    params: Mapping[str, float64],
    filter_mode: str,
    observables: list[str] | None,
    ss_seed: NDF | dict[str, float] | None,
    x0: NDF | None,
    jitter: float | float64 | None,
    symmetrize: bool | None,
    R: NDF | None,
    P0: NDF | None = None,
    prepared: PreparedFilterRun | None = None,
    q_corr: NDF | None = None,
) -> float64:
    prepared_run = (
        prepared
        if prepared is not None
        else prepare_filter_run(
            compiled=compiled,
            kalman=kalman,
            y=y,
            observables=observables,
            filter_mode=filter_mode,
            jitter=jitter,
            symmetrize=symmetrize,
            P0=P0,
        )
    )
    sol = _get_solution(
        solver=solver,
        compiled=compiled,
        params=params,
        mode=prepared_run.mode,
        ss_seed=ss_seed,
        raise_on_bk_violation=False,
    )
    Q = build_Q(compiled, params, corr=q_corr)
    R_mat = build_R(compiled, kalman, prepared_run.observables, params, R_override=R)
    calib_params = build_calib_param_vector(compiled, params)
    loglik_of_R = _prepare_filter_loglik(
        sol=sol,
        prepared=prepared_run,
        Q=Q,
        calib_params=calib_params,
        x0=x0,
        raise_on_error=False,
    )
    return loglik_of_R(R_mat)


def _corr_chol_from_unconstrained(z: NDF, K: int) -> NDF:
    """Map unconstrained z in R^(K(K-1)/2) -> valid corr Cholesky factor."""
    expected = (K * (K - 1)) // 2
    if z.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} unconstrained CPC elements, got {z.shape[0]}."
        )
    # std = ones -> the returned covariance is the correlation; L is its factor.
    _, L = cov_from_unconstrained(z, np.ones(K, dtype=float64))
    return L


def _unconstrained_from_corr_chol(L: NDF) -> NDF:
    L = asarray(L, dtype=float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("Input must be a square lower-triangular correlation factor.")
    if not np.allclose(L, np.tril(L), atol=1e-12, rtol=0.0):
        raise ValueError("Input must be lower triangular.")
    if np.any(np.diag(L) <= 0.0):
        raise ValueError("Diagonal of a correlation Cholesky factor must be positive.")
    for i in range(L.shape[0]):
        row = L[i, : i + 1]
        if not np.allclose(np.dot(row, row), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError(
                "Each row of a correlation Cholesky factor must have unit norm."
            )
    return unconstrained_from_corr_chol(L)


def _unconstrained_from_corr(corr: NDF) -> NDF:
    corr = asarray(corr, dtype=float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    if not np.allclose(corr, corr.T, atol=1e-10, rtol=0.0):
        raise ValueError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(corr), np.ones(corr.shape[0]), atol=1e-10, rtol=0.0):
        raise ValueError("Correlation matrix must have unit diagonal.")
    try:
        L = np.linalg.cholesky(corr).astype(float64)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Correlation matrix must be positive definite.") from exc
    return _unconstrained_from_corr_chol(L)


def evaluate_logprior(
    params: Mapping[str, float64],
    priors: Mapping[str, Prior] | None,
) -> float64:
    if priors is None:
        return float64(0.0)
    lp = float64(0.0)
    for name, prior in priors.items():
        if name not in params:
            raise KeyError(f"Prior specified for unknown parameter '{name}'.")
        lp += float64(prior.logpdf(float64(params[name])))
    return lp
