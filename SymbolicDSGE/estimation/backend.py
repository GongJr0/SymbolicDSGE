from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import sympy as sp
from numpy import asarray, float64
from numpy.typing import NDArray
from scipy import optimize
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

NDF = NDArray[np.float64]


@dataclass(frozen=True)
class PreparedFilterRun:
    observables: list[str]
    y_reordered: NDF
    mode: str
    meas_addr: int
    jac_addr: int
    zero_state: NDF
    P0: NDF | None
    kf_jitter: float64
    kf_sym: bool


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
    zero_state: NDF,
    calib_params: NDF,
    n_obs: int,
) -> tuple[NDF, NDF]:
    n_var = zero_state.shape[0]
    d = measurement_eval(meas_addr, zero_state, calib_params, n_obs)
    C = jacobian_eval(jac_addr, zero_state, calib_params, n_obs, n_var)
    return C, d


def _build_named_P0(
    var_names: Sequence[str],
    kalman: KalmanConfig | None,
    p0_mode: str | None,
    p0_scale: float | float64 | None,
) -> NDF | None:
    """Build a P0 covariance over exactly ``var_names`` (eye/diag from config or
    overrides). Returns ``None`` when no P0 is configured or requested, matching
    the filter's 'use its own default' contract."""
    n = len(var_names)
    if kalman is None and p0_mode is None:
        return None

    if kalman is None:
        mode = p0_mode
        scale = float64(1.0 if p0_scale is None else p0_scale)
        diag = None
    else:
        mode = p0_mode if p0_mode is not None else kalman.P0.mode
        scale = float64(kalman.P0.scale) if p0_scale is None else float64(p0_scale)
        diag = kalman.P0.diag

    if mode == "eye":
        return np.eye(n, dtype=float64) * scale

    if mode == "diag":
        if diag is None:
            raise ValueError("P0 diag mode requires diagonal entries.")
        mat = np.zeros((n, n), dtype=float64)
        for i, var in enumerate(var_names):
            if var not in diag:
                raise ValueError(f"Missing P0 diagonal entry for variable '{var}'.")
            mat[i, i] = float64(diag[var]) * scale
        return mat

    raise ValueError(f"Unrecognized p0_mode: {mode}")


def build_P0(
    compiled: CompiledModel,
    kalman: KalmanConfig | None,
    p0_mode: str | None,
    p0_scale: float | float64 | None,
) -> NDF | None:
    """P0 over all model variables (the linear/extended state space)."""
    return _build_named_P0(compiled.var_names, kalman, p0_mode, p0_scale)


def build_unscented_P0(
    compiled: CompiledModel,
    kalman: KalmanConfig | None,
    p0_mode: str | None,
    p0_scale: float | float64 | None,
) -> NDF | None:
    """Augmented ``(2*n_state, 2*n_state)`` block-diagonal P0 for unscented
    filtering: the state-variable P0 in the top-left block, an identity in the
    bottom-right (the second-order augmentation channel), zeros off-diagonal.
    Returns ``None`` when no state P0 is configured, matching :func:`build_P0`."""
    n_state = compiled.n_state
    state_P0 = _build_named_P0(
        list(compiled.var_names[:n_state]), kalman, p0_mode, p0_scale
    )
    if state_P0 is None:
        return None
    out = np.zeros((2 * n_state, 2 * n_state), dtype=float64)
    out[:n_state, :n_state] = state_P0
    out[n_state:, n_state:] = np.eye(n_state, dtype=float64)
    return out


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
    y: NDF | pd.DataFrame,
    observables: list[str] | None,
    filter_mode: str,
    p0_mode: str | None,
    p0_scale: float | float64 | None,
    jitter: float | float64 | None,
    symmetrize: bool | None,
) -> PreparedFilterRun:
    kalman = compiled.kalman
    obs, y_reordered = reorder_observables(compiled, observables, y)
    mode = filter_mode
    if mode == "unscented":
        P0 = build_unscented_P0(compiled, kalman, p0_mode, p0_scale)
    else:
        P0 = build_P0(compiled, kalman, p0_mode, p0_scale)
    kf_jitter, kf_sym = resolve_filter_options(jitter, symmetrize)

    return PreparedFilterRun(
        observables=obs,
        y_reordered=y_reordered,
        mode=mode,
        meas_addr=compiled.construct_measurement_cfunc(obs).address,
        jac_addr=compiled.construct_observable_jacobian_cfunc(obs).address,
        zero_state=np.zeros((len(compiled.cur_syms),), dtype=float64),
        P0=P0,
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
    steady_state: NDF | dict[str, float] | None,
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
        steady_state=steady_state,
        raise_on_bk_violation=raise_on_bk_violation,
    )


def _prepare_filter_loglik(
    *,
    sol: Any,
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
            prepared.zero_state,
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
            "hx": sol.policy.p,
            "gx": sol.policy.f,
            "bx": asarray(sol.B[:n_state, :], dtype=float64),
            "hxx": sol.policy.hxx,
            "gxx": sol.policy.gxx,
            "hss": sol.policy.hss,
            "gss": sol.policy.gss,
            "steady_state": sol.policy.steady_state,
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
    steady_state: NDF | dict[str, float] | None,
    x0: NDF | None,
    p0_mode: str | None,
    p0_scale: float | float64 | None,
    jitter: float | float64 | None,
    symmetrize: bool | None,
    R: NDF | None,
    prepared: PreparedFilterRun | None = None,
    q_corr: NDF | None = None,
) -> float64:
    prepared_run = (
        prepared
        if prepared is not None
        else prepare_filter_run(
            compiled=compiled,
            y=y,
            observables=observables,
            filter_mode=filter_mode,
            p0_mode=p0_mode,
            p0_scale=p0_scale,
            jitter=jitter,
            symmetrize=symmetrize,
        )
    )
    sol = _get_solution(
        solver=solver,
        compiled=compiled,
        params=params,
        mode=prepared_run.mode,
        steady_state=steady_state,
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
