"""Resolution of Kalman filter arguments from a solved model.

Every caller that runs a filter (a :class:`SolvedModel`, an estimator, a Monte
Carlo pipeline) starts from the same three inputs: the model, the observations,
and the observable selection. The resolvers here turn those into the complete
argument set of the matching ``KalmanFilter.run_*_raw`` runner. A caller that
runs the filter in Python splats the result; a caller that lowers to C reads the
individual resolvers and stages the arrays itself.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Sequence, TypedDict

import numpy as np
import pandas as pd
from numpy import asarray, float64, int64
from numpy.typing import NDArray
from sympy import Symbol

from .._ckernels.kalman import stationary_covariance
from .config import make_R

if TYPE_CHECKING:
    from ..core.solved_model import SolvedModel

NDF = NDArray[float64]
Float64Like = float | float64 | int | int64


class FilterMode(StrEnum):
    LINEAR = "linear"
    EXTENDED = "extended"
    UNSCENTED = "unscented"


class LinearRunArgs(TypedDict):
    A: NDF
    B: NDF
    C: NDF
    d: NDF
    Q: NDF
    R: NDF
    y: NDF
    x0: NDF
    P0: NDF
    return_shocks: bool
    symmetrize: bool
    joseph_cov: bool
    jitter: float


class ExtendedRunArgs(TypedDict):
    meas_addr: int
    jac_addr: int
    A: NDF
    B: NDF
    calib_params: NDF
    Q: NDF
    R: NDF
    y: NDF
    x0: NDF
    P0: NDF
    return_shocks: bool
    symmetrize: bool
    joseph_cov: bool
    jitter: float


class UnscentedRunArgs(TypedDict):
    meas_addr: int
    hx: NDF
    gx: NDF
    bu: NDF
    hxx: NDF
    gxx: NDF
    hxu: NDF
    gxu: NDF
    huu: NDF
    guu: NDF
    hss: NDF
    gss: NDF
    steady_state: NDF
    calib_params: NDF
    Q: NDF
    R: NDF
    y: NDF
    z0: NDF
    P0: NDF
    alpha: float
    beta: float
    kappa: float
    symmetrize: bool
    jitter: float


def resolve_linear_args(
    model: SolvedModel,
    y: NDF | pd.DataFrame,
    observables: Sequence[str] | None = None,
    *,
    x0: NDF | None = None,
    P0: NDF | None = None,
    R: NDF | None = None,
    jitter: Float64Like | None = None,
    symmetrize: bool = True,
    joseph_cov: bool = True,
    return_shocks: bool = False,
) -> LinearRunArgs:
    """Complete argument set for ``KalmanFilter.run_raw``."""
    obs, y_canonical = _resolve_observations(model, observables, y)
    C, d = model._build_C_d_from_obs(obs)
    A, B = model.policy.A, model.policy.B
    return LinearRunArgs(
        A=A,
        B=B,
        C=C,
        d=d,
        Q=model._build_Q(),
        R=_build_constant_R(model, R, obs),
        y=y_canonical,
        x0=_default_x0(A) if x0 is None else x0,
        P0=_build_P0(model, FilterMode.LINEAR, P0),
        return_shocks=bool(return_shocks),
        symmetrize=bool(symmetrize),
        joseph_cov=bool(joseph_cov),
        jitter=_jitter(jitter),
    )


def resolve_extended_args(
    model: SolvedModel,
    y: NDF | pd.DataFrame,
    observables: Sequence[str] | None = None,
    *,
    x0: NDF | None = None,
    P0: NDF | None = None,
    R: NDF | None = None,
    jitter: Float64Like | None = None,
    symmetrize: bool = True,
    joseph_cov: bool = True,
    return_shocks: bool = False,
) -> ExtendedRunArgs:
    """Complete argument set for ``KalmanFilter.run_extended_raw``."""
    obs, y_canonical = _resolve_observations(model, observables, y)
    A, B = model.policy.A, model.policy.B
    return ExtendedRunArgs(
        meas_addr=int(model.compiled.construct_measurement_cfunc(obs).address),
        jac_addr=int(model.compiled.construct_observable_jacobian_cfunc(obs).address),
        A=A,
        B=B,
        calib_params=_calib_params(model),
        Q=model._build_Q(),
        R=_build_constant_R(model, R, obs),
        y=y_canonical,
        x0=_default_x0(A) if x0 is None else x0,
        P0=_build_P0(model, FilterMode.EXTENDED, P0),
        return_shocks=bool(return_shocks),
        symmetrize=bool(symmetrize),
        joseph_cov=bool(joseph_cov),
        jitter=_jitter(jitter),
    )


def resolve_unscented_args(
    model: SolvedModel,
    y: NDF | pd.DataFrame,
    observables: Sequence[str] | None = None,
    *,
    x0: NDF | None = None,
    P0: NDF | None = None,
    R: NDF | None = None,
    jitter: Float64Like | None = None,
    symmetrize: bool = True,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 1.0,
) -> UnscentedRunArgs:
    """Complete argument set for ``KalmanFilter.run_unscented_raw``."""
    if model.policy.order != 2:
        raise ValueError("Unscented Kalman Filter requires a second order solution.")

    obs, y_canonical = _resolve_observations(model, observables, y)
    policy = model.policy
    return UnscentedRunArgs(
        meas_addr=int(model.compiled.construct_measurement_cfunc(obs).address),
        hx=policy.p,
        gx=policy.f,
        bu=policy.B,
        hxx=_ukf_array(model, "hxx"),
        gxx=_ukf_array(model, "gxx"),
        hxu=_ukf_array(model, "hxu"),
        gxu=_ukf_array(model, "gxu"),
        huu=_ukf_array(model, "huu"),
        guu=_ukf_array(model, "guu"),
        hss=_ukf_array(model, "hss"),
        gss=_ukf_array(model, "gss"),
        steady_state=policy.steady_state,
        calib_params=_calib_params(model),
        Q=model._build_Q(),
        R=_build_constant_R(model, R, obs),
        y=y_canonical,
        z0=_build_unscented_z0(model, x0),
        P0=_build_P0(model, FilterMode.UNSCENTED, P0),
        alpha=float(alpha),
        beta=float(beta),
        kappa=float(kappa),
        symmetrize=bool(symmetrize),
        jitter=_jitter(jitter),
    )


def _jitter(jitter: Float64Like | None) -> float:
    return 0.0 if jitter is None else float(jitter)


def _default_x0(A: NDF) -> NDF:
    return np.zeros((A.shape[0],), dtype=float64)


def _calib_params(model: SolvedModel) -> NDF:
    params = model.config.calibration.parameters
    return asarray(
        [params[name] for name in model.compiled.calib_params], dtype=float64
    )


def _ukf_array(model: SolvedModel, name: str) -> NDF:
    value: NDF | None = getattr(model.policy, name, None)
    if value is None:
        raise ValueError(f"Unscented filtering requires policy.{name}.")
    return value


def _resolve_P0(
    mode: FilterMode, n_state: int, n_var: int, P0: NDF | None
) -> NDF | None:
    if P0 is None:
        return None
    if mode != FilterMode.UNSCENTED:
        return P0

    out = np.zeros((2 * n_state, 2 * n_state), dtype=float64)
    out[:n_state, :n_state] = P0[:n_state, :n_state]
    return out


def _resolve_obs_names(
    model: SolvedModel, obs: Sequence[str] | None
) -> tuple[str, ...]:
    """Validate ``obs`` against the model and return it in canonical order.

    Canonical order is ``model.compiled.observable_names``. ``None`` selects all
    of them.
    """
    canon = model.compiled.observable_names
    canon_idx = {name: i for i, name in enumerate(canon)}

    if obs is None:
        obs_given = list(canon)  # default: all observables in canonical order
    else:
        obs_given = list(obs)

    if len(obs_given) == 0:
        raise ValueError("Observable list is empty.")

    if len(set(obs_given)) != len(obs_given):
        dupes = [n for n in obs_given if obs_given.count(n) > 1]
        raise ValueError(f"Duplicate observables provided: {sorted(set(dupes))}")

    missing = [n for n in obs_given if n not in canon_idx]
    if missing:
        raise ValueError(
            f"Unknown observables not in model.compiled.observable_names: {missing}"
        )

    obs_canonical = sorted(obs_given, key=lambda n: canon_idx[n])

    return tuple(obs_canonical)


def _resolve_observations(
    model: SolvedModel, obs: Sequence[str] | None, y: NDF | pd.DataFrame
) -> tuple[tuple[str, ...], NDF]:
    """Canonical observable names paired with ``y``'s columns reordered to match.

    An ndarray's columns are read in the order ``obs`` names them. A DataFrame is
    aligned by column label instead, so its own column order is irrelevant.
    """
    obs_given = tuple(model.compiled.observable_names) if obs is None else tuple(obs)
    obs_canonical = _resolve_obs_names(model, obs_given)

    if isinstance(y, pd.DataFrame):
        missing_cols = [name for name in obs_given if name not in y.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing observable columns: {missing_cols}")
        y_reordered = y.loc[:, list(obs_canonical)].to_numpy(dtype=float64)
    else:
        y_arr = asarray(y, dtype=float64)
        if y_arr.ndim != 2:
            raise ValueError(
                f"Observation data must be 2D. Shape (T,m) expected, got {y_arr.shape}."
            )
        m = y_arr.shape[1]
        if m != len(obs_given):
            raise ValueError(
                f"y has {m} columns but obs list has {len(obs_given)} names."
            )
        pos_in_given = {name: j for j, name in enumerate(obs_given)}
        y_reordered = y_arr[:, [pos_in_given[name] for name in obs_canonical]]

    if np.isnan(y_reordered).any():
        raise ValueError("Observation data contains NaN values.")

    return obs_canonical, y_reordered


def _validate_user_R(R: NDF | None, observables: Sequence[str]) -> NDF | None:
    if R is None:
        return None

    given_shape = R.shape
    implied_shape = (len(observables), len(observables))
    if given_shape != implied_shape:
        raise ValueError(
            f"Provided R matrix has shape {given_shape} but expected {implied_shape} based on number of observables."
        )

    return R


def _build_constant_R(
    model: SolvedModel, R: NDF | None, observables: Sequence[str]
) -> NDF:
    validated_R = _validate_user_R(R, observables)
    if validated_R is not None:
        return validated_R

    conf = model.kalman_config
    if conf is None:
        raise ValueError(
            "R must be provided in symbolic or scalar form, either through the "
            "model's Kalman configuration or as a parameter override."
        )

    obs_idx = {name: i for i, name in enumerate(model.compiled.observable_names)}

    std_map = conf.R_std_param_map
    corr_map = conf.R_corr_param_map
    if std_map is not None:
        # Assemble the constant R from the CURRENT calibration (which may have
        # moved since parse, e.g. a re-solved model). The name->position maps
        # fix the layout at parse; only the values are read live here.
        calib = model.config.calibration.parameters
        params_by_name = {
            (k if isinstance(k, str) else k.name): float64(v) for k, v in calib.items()
        }

        def _param(name: str) -> float64:
            if name not in params_by_name:
                raise KeyError(f"Missing R parameter '{name}' in calibration.")
            return params_by_name[name]

        all_obs = model.compiled.observable_names
        y_syms = [Symbol(name) for name in all_obs]
        std_vals = {Symbol(name): _param(std_map[name]) for name in all_obs}
        corr_vals = {
            frozenset(Symbol(n) for n in pair): _param(param_name)
            for pair, param_name in (corr_map or {}).items()
            if param_name is not None
        }

        R_full = make_R(y_syms, std_vals, corr_vals)

        mat_idx = [obs_idx[name] for name in observables]
        return asarray(R_full[np.ix_(mat_idx, mat_idx)], dtype=float64)

    R = conf.R
    if R is None:
        raise ValueError("Constant R matrix not specified in configuration.")

    # Get included observables
    mat_idx = [obs_idx[name] for name in observables]
    R_subset: NDF = asarray(R[np.ix_(mat_idx, mat_idx)], dtype=float64)
    return R_subset


def _build_default_P0(model: SolvedModel, filter_mode: FilterMode) -> NDF:
    if filter_mode == FilterMode.UNSCENTED:
        n_state = model.compiled.n_state
        err, state_P0 = stationary_covariance(
            model.policy.p, model.policy.B[:n_state, :], model._build_Q()
        )
        if err != 0:
            state_P0 = np.eye(n_state, dtype=float64)
        P0 = np.zeros((2 * n_state, 2 * n_state), dtype=float64)
        P0[:n_state, :n_state] = state_P0
        return P0

    n_var = model.compiled.n_var
    err, P0 = stationary_covariance(model.policy.A, model.policy.B, model._build_Q())
    if err != 0:
        return np.eye(n_var, dtype=float64)
    return P0


def _build_P0(model: SolvedModel, filter_mode: FilterMode | str, P0: NDF | None) -> NDF:
    mode = FilterMode(filter_mode)
    resolved = _resolve_P0(mode, model.compiled.n_state, model.compiled.n_var, P0)
    if resolved is not None:
        return resolved
    return _build_default_P0(model, mode)


def _build_unscented_z0(model: SolvedModel, x0: NDF | None) -> NDF:
    n_state = model.compiled.n_state
    n_var = model.compiled.n_var
    if x0 is None:
        x0_state = np.zeros((n_state,), dtype=float64)
    else:
        raw = asarray(x0, dtype=float64)
        if raw.ndim != 1:
            raise ValueError("x0 must be a 1D array.")
        if raw.shape[0] == n_state:
            x0_state = raw.copy()
        elif raw.shape[0] == n_var:
            x0_state = raw[:n_state].copy()
        else:
            raise ValueError(
                f"x0 must have length {n_state} or {n_var}, got {raw.shape[0]}."
            )

    z0 = np.zeros((2 * n_state,), dtype=float64)
    z0[:n_state] = x0_state
    return z0
