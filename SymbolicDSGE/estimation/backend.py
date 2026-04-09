from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import sympy as sp
from numpy import asarray, float64
from numpy.typing import NDArray
from scipy import optimize
from sympy import Symbol
from numba import njit

from ..bayesian.distributions.lkj_chol import LKJChol
from ..bayesian.priors import Prior
from ..core.compiled_model import CompiledModel
from ..core.config import SymbolGetterDict
from ..core.solver import DSGESolver
from ..kalman.config import KalmanConfig
from ..kalman.filter import KalmanFilter

NDF = NDArray[np.float64]


@dataclass(frozen=True)
class PreparedFilterRun:
    observables: list[str]
    y_reordered: NDF
    mode: str
    measurement_func: Callable[[NDF, NDF], NDF]
    measurement_jac: Callable[[NDF, NDF], NDF]
    zero_state: NDF
    P0: NDF | None
    kf_jitter: float64
    kf_sym: bool


def _name_of(p: str | Symbol) -> str:
    return p if isinstance(p, str) else p.name


def extract_base_params(compiled: CompiledModel) -> dict[str, float64]:
    params = compiled.config.calibration.parameters
    return {_name_of(k): float64(v) for k, v in params.items()}


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
    names = [_name_of(p) for p in compiled.calib_params]
    return asarray([float64(params[name]) for name in names], dtype=float64)


def reorder_observables(
    compiled: CompiledModel,
    kalman: KalmanConfig | None,
    observables: list[str] | None,
    y: NDF | pd.DataFrame,
) -> tuple[list[str], NDF]:
    canon = compiled.observable_names
    canon_idx = {name: i for i, name in enumerate(canon)}

    if observables is None:
        if kalman is not None and kalman.y_names:
            obs_given = list(kalman.y_names)
        else:
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
        y_reordered = y.loc[:, obs_canonical].to_numpy(dtype=float64)
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


def infer_filter_mode(
    compiled: CompiledModel,
    observables: list[str] | None,
) -> str:
    kalman = getattr(compiled, "kalman", None)
    canon = getattr(compiled, "observable_names", [])
    if observables is None:
        if kalman is not None and getattr(kalman, "y_names", None):
            obs = list(kalman.y_names)
        else:
            obs = list(canon)
    else:
        obs = list(observables)

    eqs = getattr(getattr(compiled, "config", None), "equations", None)
    if eqs is None or not hasattr(eqs, "obs_is_affine") or len(obs) == 0:
        return "linear"
    is_affine = eqs.obs_is_affine
    all_affine = all(bool(is_affine[name]) for name in obs)
    return "linear" if all_affine else "extended"


def build_Q(compiled: CompiledModel, params: Mapping[str, float64]) -> NDF:
    shock_map = compiled.config.shock_map
    shock_std = compiled.config.calibration.shock_std
    shock_corr = compiled.config.calibration.shock_corr

    exogs = compiled.var_names[: compiled.n_exog]
    rev: SymbolGetterDict[Symbol, Symbol] = SymbolGetterDict(
        {exo: shock for shock, exo in shock_map.items()}
    )
    shocks = [rev[exo] for exo in exogs]

    stds = asarray([float64(params[shock_std[s].name]) for s in shocks], dtype=float64)
    corr = np.eye(len(exogs), dtype=float64)

    n = len(stds)
    for i in range(n):
        for j in range(i + 1, n):
            pair = frozenset({shocks[i], shocks[j]})
            corr_sym = shock_corr.get(pair, None)
            corr_ij = float64(params[corr_sym.name]) if corr_sym is not None else 0.0
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


def build_C_d(
    compiled: CompiledModel,
    params: Mapping[str, float64],
    observables: list[str],
) -> tuple[NDF, NDF]:
    return compiled.build_affine_measurement_matrices(params, observables)


def build_C_d_from_dispatchers(
    measurement_func: Callable[[NDF, NDF], NDF],
    measurement_jac: Callable[[NDF, NDF], NDF],
    zero_state: NDF,
    calib_params: NDF,
) -> tuple[NDF, NDF]:
    d = np.asarray(measurement_func(zero_state, calib_params), dtype=float64).reshape(
        -1
    )
    C = np.asarray(measurement_jac(zero_state, calib_params), dtype=float64)
    return (
        np.ascontiguousarray(C, dtype=float64),
        np.ascontiguousarray(d, dtype=float64),
    )


def build_P0(
    compiled: CompiledModel,
    kalman: KalmanConfig | None,
    p0_mode: str | None,
    p0_scale: float | float64 | None,
) -> NDF | None:
    n = len(compiled.var_names)
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
        for i, var in enumerate(compiled.var_names):
            if var not in diag:
                raise ValueError(f"Missing P0 diagonal entry for variable '{var}'.")
            mat[i, i] = float64(diag[var]) * scale
        return mat

    raise ValueError(f"Unrecognized p0_mode: {mode}")


def resolve_filter_options(
    kalman: KalmanConfig | None,
    jitter: float | float64 | None,
    symmetrize: bool | None,
) -> tuple[float64, bool]:
    if jitter is None:
        if kalman is not None and getattr(kalman, "jitter", None) is not None:
            kf_jitter = float64(kalman.jitter)
        else:
            kf_jitter = float64(0.0)
    else:
        kf_jitter = float64(jitter)

    if symmetrize is None:
        if kalman is not None and getattr(kalman, "symmetrize", None) is not None:
            kf_sym = bool(kalman.symmetrize)
        else:
            kf_sym = False
    else:
        kf_sym = bool(symmetrize)

    return kf_jitter, kf_sym


def resolve_R(
    compiled: CompiledModel,
    kalman: KalmanConfig | None,
    observables: list[str],
    R: NDF | None,
) -> NDF:
    m = len(observables)
    if R is not None:
        if R.shape != (m, m):
            raise ValueError(f"Provided R has shape {R.shape}, expected ({m}, {m}).")
        return asarray(R, dtype=float64)

    if kalman is None or kalman.R is None:
        raise ValueError("R is not available. Provide `R` or a KalmanConfig with R.")

    obs_idx = {name: i for i, name in enumerate(compiled.observable_names)}
    mat_idx = [obs_idx[name] for name in observables]
    return asarray(kalman.R[np.ix_(mat_idx, mat_idx)], dtype=float64)


def prepare_filter_run(
    *,
    compiled: CompiledModel,
    y: NDF | pd.DataFrame,
    observables: list[str] | None,
    filter_mode: str | None,
    p0_mode: str | None,
    p0_scale: float | float64 | None,
    jitter: float | float64 | None,
    symmetrize: bool | None,
) -> PreparedFilterRun:
    kalman = compiled.kalman
    obs, y_reordered = reorder_observables(compiled, kalman, observables, y)
    mode = infer_filter_mode(compiled, obs) if filter_mode is None else filter_mode
    P0 = build_P0(compiled, kalman, p0_mode, p0_scale)
    kf_jitter, kf_sym = resolve_filter_options(kalman, jitter, symmetrize)

    return PreparedFilterRun(
        observables=obs,
        y_reordered=y_reordered,
        mode=mode,
        measurement_func=compiled.construct_measurement_array_func(obs),
        measurement_jac=compiled.construct_observable_jacobian_array_func(obs),
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
    builder = getattr(kalman, "R_builder", None)
    arg_names = getattr(kalman, "R_param_names", None)
    if builder is None or arg_names is None:
        raise ValueError("KalmanConfig does not expose symbolic R builder metadata.")

    vals = []
    for name in arg_names:
        if name not in params:
            raise KeyError(f"Missing R-builder parameter '{name}' in params.")
        vals.append(float64(params[name]))

    R_full = asarray(builder(*vals), dtype=float64)
    n_all = len(compiled.observable_names)
    if R_full.shape != (n_all, n_all):
        raise ValueError(
            f"R builder returned shape {R_full.shape}, expected ({n_all}, {n_all})."
        )

    obs_idx = {name: i for i, name in enumerate(compiled.observable_names)}
    mat_idx = [obs_idx[name] for name in observables]
    return asarray(R_full[np.ix_(mat_idx, mat_idx)], dtype=float64)


def evaluate_loglik(
    *,
    solver: DSGESolver,
    compiled: CompiledModel,
    y: NDF | pd.DataFrame,
    params: Mapping[str, float64],
    filter_mode: str | None,
    observables: list[str] | None,
    steady_state: NDF | dict[str, float] | None,
    x0: NDF | None,
    p0_mode: str | None,
    p0_scale: float | float64 | None,
    jitter: float | float64 | None,
    symmetrize: bool | None,
    R: NDF | None,
    prepared: PreparedFilterRun | None = None,
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
    sol = solver.solve(
        compiled=compiled,
        parameters={k: float(v) for k, v in params.items()},
        steady_state=steady_state,
    )
    Q = build_Q(compiled, params)
    R_mat = resolve_R(compiled, compiled.kalman, prepared_run.observables, R)
    calib_params = build_calib_param_vector(compiled, params)

    if prepared_run.mode == "linear":
        C, d = build_C_d_from_dispatchers(
            prepared_run.measurement_func,
            prepared_run.measurement_jac,
            prepared_run.zero_state,
            calib_params,
        )
        result = KalmanFilter.run(
            A=sol.A,
            B=sol.B,
            C=C,
            d=d,
            Q=Q,
            R=R_mat,
            y=prepared_run.y_reordered,
            x0=x0,
            P0=prepared_run.P0,
            jitter=float(prepared_run.kf_jitter),
            symmetrize=prepared_run.kf_sym,
            return_shocks=False,
        )
        return float64(result.loglik)

    if prepared_run.mode == "extended":
        result = KalmanFilter.run_extended(
            A=sol.A,
            B=sol.B,
            h=prepared_run.measurement_func,
            H_jac=prepared_run.measurement_jac,
            calib_params=calib_params,
            Q=Q,
            R=R_mat,
            y=prepared_run.y_reordered,
            x0=x0,
            P0=prepared_run.P0,
            jitter=float(prepared_run.kf_jitter),
            symmetrize=prepared_run.kf_sym,
            return_shocks=False,
            compute_y_filt=False,
        )
        return float64(result.loglik)

    raise ValueError(f"Unrecognized filter_mode: {prepared_run.mode}")


def estimate_R_diag(
    *,
    solver: DSGESolver,
    compiled: CompiledModel,
    y: NDF | pd.DataFrame,
    params: Mapping[str, float64],
    observables: list[str] | None,
    steady_state: NDF | dict[str, float] | None,
    x0: NDF | None,
    p0_mode: str | None,
    p0_scale: float | float64 | None,
    jitter: float | float64 | None,
    symmetrize: bool | None,
) -> NDF:
    prepared = prepare_filter_run(
        compiled=compiled,
        y=y,
        observables=observables,
        filter_mode=None,
        p0_mode=p0_mode,
        p0_scale=p0_scale,
        jitter=jitter,
        symmetrize=symmetrize,
    )
    m = prepared.y_reordered.shape[1]
    eps = float64(1e-9)
    eta0 = np.log(
        np.asarray(
            [max(0.1 * np.var(prepared.y_reordered[:, i]), eps) for i in range(m)],
            dtype=float64,
        )
    )
    try:
        sol = solver.solve(
            compiled=compiled,
            parameters={k: float(v) for k, v in params.items()},
            steady_state=steady_state,
        )
    except BaseException:
        return np.diag(np.exp(eta0))

    Q = build_Q(compiled, params)
    bounds = [(-30.0, 10.0)] * m
    calib_params = build_calib_param_vector(compiled, params)

    if prepared.mode == "linear":
        C, d = build_C_d_from_dispatchers(
            prepared.measurement_func,
            prepared.measurement_jac,
            prepared.zero_state,
            calib_params,
        )

        def obj(eta: NDF) -> float64:
            R = np.diag(np.exp(eta))
            run = KalmanFilter.run(
                A=sol.A,
                B=sol.B,
                C=C,
                d=d,
                Q=Q,
                R=R,
                y=prepared.y_reordered,
                x0=x0,
                P0=prepared.P0,
                jitter=float(prepared.kf_jitter),
                symmetrize=prepared.kf_sym,
                return_shocks=False,
            )
            return float64(-run.loglik)

    else:

        def obj(eta: NDF) -> float64:
            R = np.diag(np.exp(eta))
            run = KalmanFilter.run_extended(
                A=sol.A,
                B=sol.B,
                h=prepared.measurement_func,
                H_jac=prepared.measurement_jac,
                calib_params=calib_params,
                Q=Q,
                R=R,
                y=prepared.y_reordered,
                x0=x0,
                P0=prepared.P0,
                jitter=float(prepared.kf_jitter),
                symmetrize=prepared.kf_sym,
                return_shocks=False,
                compute_y_filt=False,
            )
            return float64(-run.loglik)

    opt = optimize.minimize(
        obj,
        x0=eta0,
        bounds=bounds,
        method="L-BFGS-B",
    )
    if not bool(opt.success):
        return np.diag(np.exp(eta0))

    return np.diag(np.exp(opt.x))


@njit(cache=True)
def _corr_chol_from_unconstrained_backend(z: NDF, K: int) -> NDF:
    cpc: NDF = np.tanh(z)
    L: NDF = np.zeros((K, K), dtype=float64)
    L[0, 0] = 1.0
    idx: int = 0
    for k in range(1, K):
        rem: float64 = float64(1.0)
        for j in range(k):
            v = float64(np.sqrt(max(rem, 1e-14)))
            L[k, j] = float64(cpc[idx] * v)
            rem = float64(rem - L[k, j] * L[k, j])
            idx += 1
        L[k, k] = float64(np.sqrt(max(rem, 1e-14)))
    return L


def _corr_chol_from_unconstrained(z: NDF, K: int) -> NDF:
    """Map unconstrained z in R^(K(K-1)/2) -> valid corr Cholesky factor."""
    expected = (K * (K - 1)) // 2
    if z.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} unconstrained CPC elements, got {z.shape[0]}."
        )
    return cast(NDF, _corr_chol_from_unconstrained_backend(z, K))


@njit(cache=True)
def _unconstrained_from_corr_chol_backend(L: NDF) -> NDF:
    K = L.shape[0]
    n_cpc = (K * (K - 1)) // 2
    z = np.empty((n_cpc,), dtype=float64)
    idx = 0
    for k in range(1, K):
        rem = float64(1.0)
        for j in range(k):
            v = float64(np.sqrt(max(rem, 1e-14)))
            cpc = float64(L[k, j] / v) if v > 0.0 else float64(0.0)
            if cpc < (-1.0 + 1e-14):
                cpc = float64(-1.0 + 1e-14)
            elif cpc > (1.0 - 1e-14):
                cpc = float64(1.0 - 1e-14)
            z[idx] = float64(np.arctanh(cpc))
            rem = float64(rem - L[k, j] * L[k, j])
            idx += 1
    return z


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
    return cast(NDF, _unconstrained_from_corr_chol_backend(L))


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


@njit(cache=True)
def _R_from_unconstrained_backend(u: NDF, K: int) -> tuple[NDF, NDF, NDF]:
    log_std = u[:K]
    z = u[K:]
    std = np.exp(log_std).astype(float64)
    Lcorr: NDF = _corr_chol_from_unconstrained_backend(z.astype(float64), K)
    LcorrT: NDF = np.ascontiguousarray(Lcorr.T)
    corr: NDF = Lcorr @ LcorrT
    std_col = std.reshape((K, 1))
    std_row = std.reshape((1, K))
    R = corr * std_col * std_row
    return (R.astype(float64), std, Lcorr)


def _R_from_unconstrained(u: NDF, K: int) -> tuple[NDF, NDF, NDF]:
    """u = [log stds (K), unconstrained CPC values] -> (R, stds, Lcorr)."""
    n_cpc = (K * (K - 1)) // 2
    if u.shape[0] != K + n_cpc:
        raise ValueError(f"Expected length {K + n_cpc}, got {u.shape[0]}.")
    return cast(tuple[NDF, NDF, NDF], _R_from_unconstrained_backend(u, K))


def estimate_R(
    *,
    solver: DSGESolver,
    compiled: CompiledModel,
    y: NDF | pd.DataFrame,
    params: Mapping[str, float64],
    observables: list[str] | None,
    steady_state: NDF | dict[str, float] | None,
    x0: NDF | None,
    p0_mode: str | None,
    p0_scale: float | float64 | None,
    jitter: float | float64 | None,
    symmetrize: bool | None,
    lkj_eta: float = 2.0,
) -> NDF:
    """
    Estimate full measurement covariance R with MAP first (LKJ prior on correlation),
    then fallback to plain MLE if MAP optimization fails.
    """
    prepared = prepare_filter_run(
        compiled=compiled,
        y=y,
        observables=observables,
        filter_mode=None,
        p0_mode=p0_mode,
        p0_scale=p0_scale,
        jitter=jitter,
        symmetrize=symmetrize,
    )
    m = prepared.y_reordered.shape[1]
    n_cpc = (m * (m - 1)) // 2

    eps = float64(1e-9)
    log_std0 = np.log(
        np.asarray(
            [max(0.1 * np.var(prepared.y_reordered[:, i]), eps) for i in range(m)],
            dtype=float64,
        )
    )
    u0 = np.concatenate([log_std0, np.zeros((n_cpc,), dtype=float64)])
    try:
        sol = solver.solve(
            compiled=compiled,
            parameters={k: float(v) for k, v in params.items()},
            steady_state=steady_state,
        )
    except BaseException:
        return np.diag(np.exp(log_std0))

    Q = build_Q(compiled, params)
    bounds: list[tuple[float, float]] = [(-30.0, 10.0)] * m + [(-5.0, 5.0)] * n_cpc
    lkj = LKJChol(eta=float(lkj_eta), K=m, random_state=None)
    calib_params = build_calib_param_vector(compiled, params)

    if prepared.mode == "linear":
        C, d = build_C_d_from_dispatchers(
            prepared.measurement_func,
            prepared.measurement_jac,
            prepared.zero_state,
            calib_params,
        )

    def loglik_for_R(R: NDF) -> float64:
        if prepared.mode == "linear":
            run = KalmanFilter.run(
                A=sol.A,
                B=sol.B,
                C=C,
                d=d,
                Q=Q,
                R=R,
                y=prepared.y_reordered,
                x0=x0,
                P0=prepared.P0,
                jitter=float(prepared.kf_jitter),
                symmetrize=prepared.kf_sym,
                return_shocks=False,
            )
            return float64(run.loglik)

        run = KalmanFilter.run_extended(
            A=sol.A,
            B=sol.B,
            h=prepared.measurement_func,
            H_jac=prepared.measurement_jac,
            calib_params=calib_params,
            Q=Q,
            R=R,
            y=prepared.y_reordered,
            x0=x0,
            P0=prepared.P0,
            jitter=float(prepared.kf_jitter),
            symmetrize=prepared.kf_sym,
            return_shocks=False,
            compute_y_filt=False,
        )
        return float64(run.loglik)

    def nlogpost(u: NDF) -> float64:
        try:
            R, _std, Lcorr = _R_from_unconstrained(asarray(u, dtype=float64), m)
            ll = loglik_for_R(R)
            lp_corr = float64(lkj.logpdf(Lcorr))
            # Weak scale regularization in log-space.
            lp_scale = float64(-0.5 * np.sum((u[:m] / 2.0) ** 2))
            val = ll + lp_corr + lp_scale
            return float64(-val) if np.isfinite(val) else float64(np.inf)
        except BaseException:
            return float64(np.inf)

    def nloglik(u: NDF) -> float64:
        try:
            R, _std, _Lcorr = _R_from_unconstrained(asarray(u, dtype=float64), m)
            ll = loglik_for_R(R)
            return float64(-ll) if np.isfinite(ll) else float64(np.inf)
        except BaseException:
            return float64(np.inf)

    map_opt = optimize.minimize(
        nlogpost,
        x0=u0,
        bounds=bounds,
        method="L-BFGS-B",
    )
    if bool(map_opt.success):
        R_map, _, _ = _R_from_unconstrained(asarray(map_opt.x, dtype=float64), m)
        return R_map

    mle_opt = optimize.minimize(
        nloglik,
        x0=u0,
        bounds=bounds,
        method="L-BFGS-B",
    )
    if bool(mle_opt.success):
        R_mle, _, _ = _R_from_unconstrained(asarray(mle_opt.x, dtype=float64), m)
        return R_mle

    # Final fallback: diagonal estimate from initial variance heuristic.
    return np.diag(np.exp(log_std0))


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
