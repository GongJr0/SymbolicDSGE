"""Backend native entry points and estimation kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    NamedTuple,
    Literal,
    Any,
    Mapping,
    Sequence,
)


import numpy as np
import pandas as pd
from numpy import asarray, float64
from numpy.typing import NDArray

from .._ckernels.estimation import (
    cov_from_unconstrained,
    unconstrained_from_corr_chol,
)
from ..bayesian.priors import Prior
from .prior_program import (
    _pack_transform,
    build_packed_logprior,
    N_TRANSFORM_PARAMS,
    PyPriorTables,
)
from ..core.compiled_model import (
    CompiledModel,
    _shock_covariance,
    _measurement_covariance,
)
from ..core.solver import DSGESolver
from ..kalman.resolvers import FilterMode, _resolve_P0

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]


class MatrixPriorBlock(NamedTuple):
    """Minimal per-matrix LKJ metadata.

    ``positions`` is an ``(n_members, 2)`` int array of ``(row, col)`` targets in
    the correlation matrix, parallel to ``member_names``. The block's members
    occupy a contiguous theta run, so ``theta_slice`` (a plain ``slice``) covers
    them all: ``theta[theta_slice]`` is the block's unconstrained z with no gather.
    Resolution yields a partial block (``theta_slice`` empty, ``prior`` ``None``);
    it is completed via ``_replace`` once the theta layout is known. The reserved
    matrix key ("R_corr"/"Q_corr") is the dict key the block is stored under in
    ``_matrix_blocks``, so it is not repeated on the block itself.
    """

    K: int
    labels: list[str]
    member_names: list[str]
    positions: NDArray[np.int64]
    theta_slice: slice
    prior: Prior | None


@dataclass(frozen=True)
class FilterDTO:
    """Composed filter run inputs, ready for native evaluation.

    Attributes
    ----------
    observables : list[str]
        List of observables in the order they are passed to the filter.
    y_reordered : NDArray[float64]
        Observed data, reordered to match ``observables``.
    mode : str
        Filter mode; one of "linear", "extended", or "unscented".
    meas_addr : int
        Pointer to the measurement evaluation callable (``numba.cfunc`` address).
    jac_addr : int
        Pointer to the measurement jacobian evaluation callable (``numba.cfunc`` address).
    P0 : NDArray[float64] | None
        State covariance initialization matrix, or None if not provided.
    jitter : float64
        Jitter to add to covariance matrices when cholesky decomposition fails.
    sym : bool
        Whether to symmetrize covariance matrices in the Kalman kernels.
    joseph_cov : bool
        Whether to use the Joseph form for covariance updates in the Kalman filter.
    alpha : float | None
        Alpha parameter for the unscented Kalman filter, or None if not applicable.
    beta : float | None
        Beta parameter for the unscented Kalman filter, or None if not applicable.
    kappa : float | None
        Kappa parameter for the unscented Kalman filter, or None if not applicable.
    T : int
        Number of time steps in the observation data.
    n_obs : int
        Number of observables (columns) in the observation data.

    """

    observables: list[str]
    y_reordered: NDF
    mode: str
    meas_addr: int
    jac_addr: int
    P0: NDF | None
    jitter: float64
    sym: bool
    joseph_cov: bool
    alpha: float | None
    beta: float | None
    kappa: float | None
    # Allocation dimensions for filters
    T: int
    n_obs: int


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


def build_param_components(
    *,
    param_names: Sequence[str],
    param_index: Mapping[str, int],
    matrix_member_names: set[str],
    param_transforms: Mapping[str, Any],
    calib_index: Mapping[str, int],
) -> tuple[NDI, NDI, NDI, NDF]:
    """Flatten the estimated *scalar* params into indexed lookup tables.

    Walks ``param_names`` in theta order, skipping CPC block members (their
    correlation is built by the cov-spec ``corr_from_block`` regime, not the
    scalar scatter). Each scalar's transform is the role-resolved object already
    on ``param_transforms`` (Log for a std, Tanh for a standalone corr, the
    prior's transform or Identity for a plain param); ``_pack_transform`` maps it
    to the native ``(code, params)``. ``param_slot`` comes from ``calib_index``
    (name -> slot in ``calib_params`` order), built once by the caller and shared
    with ``base_params`` so the ordering has a single origin.

    Two boundary invariants are asserted here because the native path has no
    fallback: every estimated scalar is a calibrated parameter (so its slot
    exists), and its transform packs to a native code (never ``None``).
    """
    theta_idx = []
    param_slot = []
    transform_code = []
    transform_params = []
    for name in param_names:
        if name in matrix_member_names:
            continue
        if name not in calib_index:
            raise ValueError(
                f"Estimated scalar '{name}' is not a calibrated parameter; its slot "
                f"in the native parameter vector cannot be resolved."
            )
        transform = param_transforms[name]
        code, params = _pack_transform(transform)
        if code is None:
            raise ValueError(
                f"Transform {type(transform).__name__!r} on estimated scalar '{name}' "
                f"has no native transform code."
            )

        theta_idx.append(int(param_index[name]))
        param_slot.append(int(calib_index[name]))
        transform_code.append(int(code))
        transform_params.append(params)

    return (
        np.asarray(theta_idx, dtype=np.int64),
        np.asarray(param_slot, dtype=np.int64),
        np.asarray(transform_code, dtype=np.int64),
        np.asarray(transform_params, dtype=np.float64).reshape(-1, N_TRANSFORM_PARAMS),
    )


@dataclass(frozen=True, slots=True)
class PyParamMap:
    """Mirror of ``sdsge_param_map``: theta->params resolution tables.

    ``scalars`` is the array-of-structs the C ``scalars`` pointer addresses
    (``n_scalars`` = ``len(scalars)``). ``base_params`` and every slot index
    (``scalars`` ``param_slot``, and the cov specs' ``std_slots``/``pair_slot``)
    are in ``calib_params`` order, so ``params`` doubles as the residual argument
    vector with no gather step.
    """

    base_params: NDF  # n_par, calib_params order
    theta_idx: NDI  # n_scalars, theta -> params scatter
    param_slot: NDI  # n_scalars, theta -> params scatter
    transform_code: NDI  # n_scalars, theta -> params scatter
    transform_params: NDF  # n_scalars x SDSGE_N_TRANSFORM_PARAMS,


def build_calib_index(compiled: CompiledModel) -> dict[str, int]:
    """Name -> slot in ``calib_params`` order.

    The single origin of the calib ordering shared by ``base_params``, the scalar
    scatter's ``param_slot``, and the cov specs' ``std_slots``/``pair_slot``; build
    once at the composer and thread it into each builder.
    """
    return {str(p): i for i, p in enumerate(compiled.calib_params)}


def build_param_map(
    *,
    compiled: CompiledModel,
    param_names: Sequence[str],
    param_index: Mapping[str, int],
    matrix_member_names: set[str],
    param_transforms: Mapping[str, Any],
    calib_index: Mapping[str, int] | None = None,
) -> PyParamMap:
    """Assemble the theta->params resolution tables (``sdsge_param_map``).

    ``base_params`` (the calibrated baseline the estimated slots scatter over at
    eval time) and the scalar scatter's ``param_slot`` share one ``calib_index``
    so the calib ordering has a single origin. Pass ``calib_index`` from the
    composer when other builders (the cov specs) need the same map; it defaults
    to :func:`build_calib_index`. Both outputs are in ``calib_params`` order.
    """
    if calib_index is None:
        calib_index = build_calib_index(compiled)
    base_dict = extract_base_params(compiled)
    base_params = np.empty(len(calib_index), dtype=float64)
    for name, slot in calib_index.items():
        base_params[slot] = base_dict[name]
    (
        theta_idx,
        param_slot,
        transform_code,
        transform_params,
    ) = build_param_components(
        param_names=param_names,
        param_index=param_index,
        matrix_member_names=matrix_member_names,
        param_transforms=param_transforms,
        calib_index=calib_index,
    )
    return PyParamMap(
        base_params=base_params,
        theta_idx=theta_idx,
        param_slot=param_slot,
        transform_code=transform_code,
        transform_params=transform_params,
    )


class PyCovSpec(NamedTuple):
    """Mirror of ``sdsge_cov_spec``: a Q or R covariance build spec.

    ``is_constant`` picks a loop-invariant ``constant`` (K*K, resolved once in
    prep) over the per-eval rebuild. When rebuilt, ``std_slots`` gives the K
    diagonal param slots; the correlation comes either from a CPC block
    (``corr_from_block`` with ``block_theta_off``/``block_theta_len`` into theta)
    or from the ``pair_i``/``pair_j``/``pair_slot`` triples
    (``n_pairs`` = ``len(pair_i)``).
    """

    K: int  # n_exog (Q) or n_obs (R)
    is_constant: bool
    constant: NDF | None  # K*K, or None
    std_slots: NDI  # K
    corr_from_block: bool
    block_theta_off: int
    block_theta_len: int
    pair_i: NDI  # n_pairs
    pair_j: NDI  # n_pairs
    pair_slot: NDI  # n_pairs


def _build_q_spec(
    *,
    compiled: CompiledModel,
    calib_index: Mapping[str, int],
    param_index: Mapping[str, int],
    matrix_blocks: Mapping[str, MatrixPriorBlock],
    base_dict: Mapping[str, float64],
) -> PyCovSpec:
    """Covariance spec for Q (shock covariance), mirroring :func:`build_Q`.

    Members are the shocks in ``shocks`` order; each std is
    ``shock_std[shock]`` and each off-diagonal correlation is the ``shock_corr``
    symbol for that shock pair (absent pairs stay zero). A ``Q_corr`` CPC block
    takes the ``corr_from_block`` regime.
    """
    calib = compiled.config.calibration
    shock_std = calib.shock_std
    shock_corr = calib.shock_corr
    std_names = [shock_std[s] for s in compiled.config.shocks]

    corr_pairs: list[tuple[int, int, str]] = []
    for pair, pname in shock_corr.items():
        if pname is None:
            continue
        i, j = (compiled.shock_idx[s.name] for s in pair)
        corr_pairs.append((i, j, pname))

    block = matrix_blocks.get("Q_corr")
    is_constant = not (
        block is not None
        or any(name in param_index for name in std_names)
        or any(pname in param_index for _, _, pname in corr_pairs)
    )
    theta_slice = block.theta_slice if block is not None else slice(0, 0)

    return PyCovSpec(
        K=compiled.n_exog,
        is_constant=is_constant,
        constant=(
            _shock_covariance(compiled, params=base_dict) if is_constant else None
        ),
        std_slots=np.asarray([calib_index[n] for n in std_names], dtype=np.int64),
        corr_from_block=block is not None,
        block_theta_off=int(theta_slice.start),
        block_theta_len=int(theta_slice.stop - theta_slice.start),
        pair_i=np.asarray([i for i, _, _ in corr_pairs], dtype=np.int64),
        pair_j=np.asarray([j for _, j, _ in corr_pairs], dtype=np.int64),
        pair_slot=np.asarray(
            [calib_index[pn] for _, _, pn in corr_pairs], dtype=np.int64
        ),
    )


def _build_r_spec(
    *,
    compiled: CompiledModel,
    observables: Sequence[str],
    calib_index: Mapping[str, int],
    param_index: Mapping[str, int],
    matrix_blocks: Mapping[str, Any],
    base_dict: Mapping[str, float64],
    R_override: NDF | None = None,
) -> PyCovSpec:
    """Covariance spec for R (measurement covariance), mirroring :func:`_build_R`.

    Members are the active ``observables``; each std is ``R_std_param_map[obs]``
    and each off-diagonal correlation is the ``R_corr_param_map`` name for that
    observable pair. An ``R_corr`` CPC block takes the ``corr_from_block``
    regime. An ``R_override`` outranks both: it forces the constant regime
    whatever the config names, which is the precedence the branch order used to
    carry. A config with no named std map has nothing to estimate, so it reaches
    the constant regime on the same test, and ``_build_R`` resolves all three
    constant sources (override, named map at base calibration, fixed
    ``kalman.R``) behind one call.
    """
    n_obs = len(observables)
    obs_list = list(observables)

    kalman = compiled.kalman
    std_map = None if kalman is None else kalman.R_std_param_map
    corr_map: Mapping[Any, str | None] | None = (
        None if kalman is None else kalman.R_corr_param_map
    )

    std_names: list[str] = [] if std_map is None else [std_map[o] for o in obs_list]
    corr_pairs: list[tuple[int, int, str]] = []
    if std_map is not None and corr_map is not None:
        for i in range(n_obs):
            for j in range(i + 1, n_obs):
                pname = corr_map[obs_list[i], obs_list[j]]
                if pname is not None:
                    corr_pairs.append((i, j, pname))

    block = matrix_blocks.get("R_corr")
    is_constant = R_override is not None or not (
        block is not None
        or any(name in param_index for name in std_names)
        or any(pname in param_index for _, _, pname in corr_pairs)
    )
    theta_slice = block.theta_slice if block is not None else slice(0, 0)

    return PyCovSpec(
        K=n_obs,
        is_constant=is_constant,
        constant=(
            _build_R(compiled, obs_list, base_dict, R_override=R_override)
            if is_constant
            else None
        ),
        std_slots=np.asarray([calib_index[n] for n in std_names], dtype=np.int64),
        corr_from_block=block is not None,
        block_theta_off=int(theta_slice.start),
        block_theta_len=int(theta_slice.stop - theta_slice.start),
        pair_i=np.asarray([i for i, _, _ in corr_pairs], dtype=np.int64),
        pair_j=np.asarray([j for _, j, _ in corr_pairs], dtype=np.int64),
        pair_slot=np.asarray(
            [calib_index[pn] for _, _, pn in corr_pairs], dtype=np.int64
        ),
    )


class SolveDTO(NamedTuple):
    residual_addr: int
    bc_residual_addr: int
    ss_seed: NDF
    incidence: NDArray[np.int8]
    n_var: int
    n_state: int
    n_ctrl: int
    n_exog: int
    n_par: int


class EstimDTO(NamedTuple):
    """Mirror of ``sdsge_obj_common``: the mode-independent objective inputs.

    Runtime addresses arrive as ``int`` (cfunc ``.address`` / capsule pointer);
    ``zgges`` is absent because the composer pulls it from the scipy cython_lapack
    capsule, not from Python. The scratch fields on the C struct (``params``,
    ``Q``, ``R``, ``corr_q``, ``corr_r``, ``std_q``, ``std_r``) and the
    ``bk_violations`` output are composer-owned and omitted here.
    """

    solve_ctx: SolveDTO
    filter_ctx: FilterDTO
    pmap: PyParamMap
    q_spec: PyCovSpec
    r_spec: PyCovSpec
    prior: PyPriorTables
    n_theta: int


def build_dto(
    *,
    compiled: CompiledModel,
    prepared: FilterDTO,
    param_names: Sequence[str],
    param_index: Mapping[str, int],
    matrix_member_names: set[str],
    matrix_blocks: Mapping[str, MatrixPriorBlock],
    param_transforms: Mapping[str, Any],
    priors: Mapping[str, Prior] | None,
    ss_seed: Any,
    R_override: NDF | None,
) -> EstimDTO:
    """Assemble the mode-independent objective inputs (``sdsge_obj_common``).

    The single orchestration point for the input tables: it builds one
    ``calib_index`` and threads it through the param map and both cov specs so the
    calib ordering has one origin, and lowers ``priors`` to the packed log-prior
    program here rather than taking pre-built tables, so every table the native
    objective reads is produced in one place. Runtime addresses: ``residual`` from the
    objective cfunc, ``bc_residual`` only for the unscented (second-order) path,
    ``meas``/``jac`` off the prepared run. ``ss_seed`` is resolved to canonical
    variable order by the solver's authority. Scratch buffers and the
    ``bk_violations`` output are the composer's job, not here.
    """
    calib_index = build_calib_index(compiled)
    base_dict = extract_base_params(compiled)

    bc_residual_addr = (
        compiled.construct_objective_cfunc_bicomplex().address
        if prepared.mode == "unscented"
        else 0
    )
    ss_seed_vec = DSGESolver._resolve_ss_seed(ss_seed, compiled)
    prior_tables = build_packed_logprior(
        priors=priors,
        param_index=param_index,
        matrix_blocks=matrix_blocks,
        matrix_member_names=matrix_member_names,
    )

    solve = SolveDTO(
        residual_addr=int(compiled.construct_objective_cfunc().address),
        bc_residual_addr=int(bc_residual_addr),
        ss_seed=ss_seed_vec,
        incidence=compiled._incidence,
        n_var=compiled.n_var,
        n_state=compiled.n_state,
        n_ctrl=compiled.n_ctrl,
        n_exog=compiled.n_exog,
        n_par=compiled.n_par,
    )

    return EstimDTO(
        solve_ctx=solve,
        filter_ctx=prepared,
        pmap=build_param_map(
            compiled=compiled,
            param_names=param_names,
            param_index=param_index,
            matrix_member_names=matrix_member_names,
            param_transforms=param_transforms,
            calib_index=calib_index,
        ),
        q_spec=_build_q_spec(
            compiled=compiled,
            calib_index=calib_index,
            param_index=param_index,
            matrix_blocks=matrix_blocks,
            base_dict=base_dict,
        ),
        r_spec=_build_r_spec(
            compiled=compiled,
            observables=prepared.observables,
            calib_index=calib_index,
            param_index=param_index,
            matrix_blocks=matrix_blocks,
            base_dict=base_dict,
            R_override=R_override,
        ),
        prior=prior_tables if prior_tables is not None else PyPriorTables.empty(),
        n_theta=len(param_names),
    )


def extract_base_params(compiled: CompiledModel) -> dict[str, float64]:
    params = compiled.config.calibration.parameters
    return {str(k): float64(v) for k, v in params.items()}


def reorder_observables(
    compiled: CompiledModel,
    observables: Sequence[str] | None,
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


def _build_R(
    compiled: CompiledModel,
    observables: list[str],
    params: Mapping[str, float64],
    *,
    R_override: NDF | None = None,
) -> NDF:
    """Assemble the measurement covariance for a likelihood eval, mirroring :func:`build_Q`.

    Priority: a user-supplied ``R_override`` wins (validated to the observable
    count); else, if the config carries parser-generated std/correlation maps, R is
    rebuilt from the current ``params`` every eval exactly as Q is; else a fixed
    ``kalman.R`` (a directly-configured constant with no named params) is sliced to
    the observables as-is.
    """
    if R_override is not None:
        R = asarray(R_override, dtype=float64)
        m = len(observables)
        if R.shape != (m, m):
            raise ValueError(f"Provided R has shape {R.shape}, expected ({m}, {m}).")
        return R

    kalman = compiled.kalman
    if kalman is None:
        raise ValueError(
            "KalmanConfig is required to build R from config parameters."
            "Supply an R override or add a kalman config to the model specification."
        )

    if kalman.R_std_param_map is not None:
        return _measurement_covariance(
            compiled,
            params,
            observables=observables,
        )
    if kalman.R is None:
        raise ValueError("R is not available. Provide `R` or a KalmanConfig with R.")
    obs_idx = {name: i for i, name in enumerate(compiled.observable_names)}
    mat_idx = [obs_idx[name] for name in observables]
    return kalman.R[np.ix_(mat_idx, mat_idx)]


def resolve_filter_options(
    jitter: float | float64 | None,
    symmetrize: bool,
) -> tuple[float64, bool]:
    kf_jitter = float64(0.0) if jitter is None else float64(jitter)
    kf_sym = False if symmetrize is None else bool(symmetrize)
    return kf_jitter, kf_sym


def prepare_filter_run(
    *,
    compiled: CompiledModel,
    y: NDF | pd.DataFrame,
    observables: Sequence[str] | None,
    filter_mode: str,
    jitter: float | float64 | None,
    symmetrize: bool,
    joseph_cov: bool = False,
    P0: NDF | None = None,
) -> FilterDTO:
    obs, y_reordered = reorder_observables(compiled, observables, y)
    mode = filter_mode

    kf_jitter, kf_sym = resolve_filter_options(jitter, symmetrize)
    return FilterDTO(
        observables=obs,
        y_reordered=y_reordered,
        mode=mode,
        meas_addr=compiled.construct_measurement_cfunc(obs).address,
        jac_addr=compiled.construct_measurement_jacobian_cfunc(obs).address,
        P0=_resolve_P0(FilterMode(mode), compiled.n_state, compiled.n_var, P0),
        jitter=kf_jitter,
        sym=kf_sym,
        joseph_cov=bool(joseph_cov),
        alpha=1.0,
        beta=2.0,
        kappa=1.0,
        T=y_reordered.shape[0],
        n_obs=y_reordered.shape[1],
    )


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
