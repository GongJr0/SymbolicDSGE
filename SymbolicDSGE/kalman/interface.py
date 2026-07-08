from .filter import KalmanFilter, FilterResult, UnscentedFilterResult
from .config import KalmanConfig
from .validator import validate_kf_inputs, _KalmanDebugInfo, FilterMode
from typing import TYPE_CHECKING, Any, Tuple, Literal, Callable
import warnings

if TYPE_CHECKING:
    from ..core.solved_model import SolvedModel

from ..core.config import ModelConfig, SymbolGetterDict


from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy import asarray, float64, int64
from numpy.typing import NDArray
from scipy import optimize

from sympy import Symbol

import pandas as pd

NDF = NDArray[float64]
Float64Like = float | float64 | int | int64
FilterOutput = FilterResult | UnscentedFilterResult


@dataclass
class _KFMatrices:
    """Cached model-constant Kalman matrices for one cache key.

    Stored on the :class:`SolvedModel` (see ``SolvedModel._kf_cache``) and shared
    across repeated filter calls so the per-call interface stops rebuilding
    ``C``/``d``/``Q``/``P0``/``R`` for a fixed calibration. ``R_const`` is filled
    lazily on the first constant-R call for the key (a key first seen via a
    user-supplied or estimated ``R`` leaves it ``None``). ``validated`` flips to
    ``True`` once the constant covariances pass the symmetry / non-negative-
    diagonal checks, letting subsequent constant-R calls skip them.
    """

    C: NDF | None
    d: NDF | None
    Q: NDF
    P0: NDF
    R_const: NDF | None = None
    validated: bool = False


class KalmanInterface(KalmanFilter):
    def __init__(
        self,
        model: "SolvedModel",
        observables: list[str] | None,
        y: NDF | pd.DataFrame,
        filter_mode: Literal["linear", "extended", "unscented"] = "linear",
        *,
        h_func: Callable[..., NDF] | None = None,
        H_jac: Callable[..., NDF] | None = None,
        meas_addr: int | None = None,
        calib_params: NDF | None = None,
        R: NDF | None = None,
        p0_mode: Literal["diag", "eye"] | None = None,
        p0_scale: Float64Like | None = None,
        jitter: Float64Like | None = None,
        symmetrize: bool | None = None,
        return_shocks: bool = False,
        estimate_R_diag: bool = False,
    ) -> None:

        self.model = model
        self.mode = FilterMode(filter_mode)
        self.estimate_R_diag = bool(estimate_R_diag)

        obs, y = self._reorder_obs(observables, y)
        if obs is None:
            raise ValueError("Reordering of observables failed.")

        self.observables = obs  # reorder to canonical order
        self.y = y

        self.A, self.B = model.A, model.B

        # C/d/Q/P0 (and, on the default path, R) are model-constant: rebuilt
        # identically on every call for a fixed calibration. Resolve them through
        # the model's cache so repeated filtering (Monte Carlo, MLE/MCMC that
        # revisits a parameter vector) skips the rebuild. The key carries every
        # dependency; the calibration fingerprint turns a parameter change into a
        # cache miss, so estimation that varies parameters stays correct.
        self._uses_const_R = R is None and not self.estimate_R_diag
        cache_key = (
            self.mode,
            tuple(self.observables),
            p0_mode,
            None if p0_scale is None else float(p0_scale),
            model._calibration_fingerprint(),
        )
        record = model._kf_cache_get(cache_key)
        if record is None:
            if self.mode == FilterMode.LINEAR:
                C, d = self._get_C_d()
            else:
                C, d = None, None
            record = _KFMatrices(
                C=C,
                d=d,
                Q=self._build_Q(),
                P0=self._build_P0(p0_mode=p0_mode, p0_scale=p0_scale),
            )
            model._kf_cache_put(cache_key, record)
        self._cache_record = record

        self.C, self.d = record.C, record.d
        self.Q = record.Q
        self.P0 = record.P0
        if self._uses_const_R:
            # Fill R lazily so a key first seen with a user/estimated R still
            # gets a constant R when later reused on the default path.
            if record.R_const is None:
                record.R_const = self._build_constant_R(None)
            self.R = record.R_const
        elif R is not None:
            self.R = self._build_constant_R(R)
        else:
            self.R = self._initial_R_diag_guess()

        self.h_func = h_func
        self.H_jac = H_jac
        self.meas_addr = meas_addr
        self.calib_params = calib_params
        self.ukf_alpha = 1.0
        self.ukf_beta = 2.0
        self.ukf_kappa = 1.0

        self._validate_mode_and_inputs()

        self.jitter = self._get_jitter(jitter)
        self.symmetrize = self._get_symmetrize(symmetrize)
        self.return_shocks = bool(return_shocks)

        self._debug_info: _KalmanDebugInfo | None = None

    def filter(
        self,
        x0: NDF | None = None,
        _debug: bool = False,
        _arg_overrides: dict[str, Any] | None = None,
    ) -> FilterOutput:
        if _arg_overrides is None:
            _arg_overrides = {}

        if self.mode == FilterMode.LINEAR:
            return self.filter_linear(
                x0=x0,
                _debug=_debug,
                _arg_overrides=_arg_overrides,
            )
        if self.mode == FilterMode.EXTENDED:
            return self.filter_extended(
                x0=x0,
                _debug=_debug,
                _arg_overrides=_arg_overrides,
            )
        if self.mode == FilterMode.UNSCENTED:
            return self.filter_unscented(
                x0=x0,
                _debug=_debug,
                _arg_overrides=_arg_overrides,
            )
        raise ValueError(f"Unrecognized filter mode: {self.mode}")

    def filter_linear(
        self,
        x0: NDF | None = None,
        _debug: bool = False,
        _arg_overrides: dict[str, Any] | None = None,
    ) -> FilterResult:
        if _arg_overrides is None:
            _arg_overrides = {}
        if x0 is None:
            x0 = np.zeros((self.A.shape[0],), dtype=float64)
        base_args = self._linear_validated_args
        run_args = base_args | _arg_overrides
        self._validate_linear_extended_run(
            run_args,
            x0=x0,
            probe_measurement=False,
            arg_overrides=_arg_overrides,
        )

        run = self.run(
            **run_args,
            x0=x0,
            jitter=self.jitter,
            symmetrize=self.symmetrize,
            return_shocks=self.return_shocks,
        )
        if _debug:
            self._set_debug_info(x0=x0, run_args=run_args)
        return run

    def filter_extended(
        self,
        x0: NDF | None = None,
        _debug: bool = False,
        _arg_overrides: dict[str, Any] | None = None,
    ) -> FilterResult:
        if _arg_overrides is None:
            _arg_overrides = {}
        if x0 is None:
            x0 = np.zeros((self.A.shape[0],), dtype=float64)
        base_args = self._extended_validated_args
        run_args = base_args | _arg_overrides
        self._validate_linear_extended_run(
            run_args,
            x0=x0,
            probe_measurement=True,
            arg_overrides=_arg_overrides,
        )

        run = self.run_extended(
            **run_args,
            x0=x0,
            jitter=self.jitter,
            symmetrize=self.symmetrize,
            return_shocks=self.return_shocks,
        )
        if _debug:
            self._set_debug_info(x0=x0, run_args=run_args)
        return run

    def _validate_linear_extended_run(
        self,
        run_args: dict[str, Any],
        *,
        x0: NDF,
        probe_measurement: bool,
        arg_overrides: dict[str, Any],
    ) -> None:
        const_validated = (
            self._cache_record.validated and self._uses_const_R and not arg_overrides
        )
        run_const_checks = not const_validated
        validate_kf_inputs(
            **run_args,
            x0=x0,
            filter_mode=self.mode,
            check_nonneg_diag=run_const_checks,
            check_symmetry=run_const_checks,
            probe_measurement=probe_measurement,
            probe_state="x0",
        )
        if run_const_checks and self._uses_const_R and not arg_overrides:
            self._cache_record.validated = True

    def filter_unscented(
        self,
        x0: NDF | None = None,
        _debug: bool = False,
        _arg_overrides: dict[str, Any] | None = None,
    ) -> UnscentedFilterResult:
        if _arg_overrides is None:
            _arg_overrides = {}
        if self.return_shocks:
            raise ValueError("return_shocks is not supported for unscented filtering.")

        z0 = self._build_unscented_z0(x0)
        base_args = self._unscented_validated_args
        run_args = base_args | _arg_overrides
        self._validate_unscented_covariances(run_args, arg_overrides=_arg_overrides)

        run = self.run_unscented(
            **run_args,
            z0=z0,
            alpha=self.ukf_alpha,
            beta=self.ukf_beta,
            kappa=self.ukf_kappa,
            jitter=float(self.jitter),
            symmetrize=self.symmetrize,
        )
        if _debug:
            self._set_debug_info(x0=x0, z0=z0, run_args=run_args)
        return run

    def _set_debug_info(
        self,
        *,
        x0: NDF | None,
        run_args: dict[str, Any],
        z0: NDF | None = None,
    ) -> None:
        self._debug_info = _KalmanDebugInfo(
            A=run_args.get("A", self.A),
            B=run_args.get("B", self.B),
            C=run_args.get("C", self.C),
            d=run_args.get("d", self.d),
            h_func=run_args.get("h", self.h_func),
            H_jac=run_args.get("H_jac", self.H_jac),
            Q=run_args.get("Q", self.Q),
            R=run_args.get("R", self.R),
            y=run_args.get("y", self.y),
            x0=x0,
            P0=run_args.get("P0", self.P0),
            meas_addr=run_args.get("meas_addr", self.meas_addr),
            hx=run_args.get("hx"),
            gx=run_args.get("gx"),
            bx=run_args.get("bx"),
            hxx=run_args.get("hxx"),
            gxx=run_args.get("gxx"),
            hss=run_args.get("hss"),
            gss=run_args.get("gss"),
            steady_state=run_args.get("steady_state"),
            calib_params=run_args.get("calib_params", self.calib_params),
            z0=z0,
            alpha=self.ukf_alpha if self.mode == FilterMode.UNSCENTED else None,
            beta=self.ukf_beta if self.mode == FilterMode.UNSCENTED else None,
            kappa=self.ukf_kappa if self.mode == FilterMode.UNSCENTED else None,
        )

    def _get_symmetrize(self, symmetrize_arg: bool | None) -> bool:
        return bool(symmetrize_arg) if symmetrize_arg is not None else False

    def _get_jitter(self, jitter_arg: Float64Like | None) -> float64:
        return float64(jitter_arg) if jitter_arg is not None else float64(0.0)

    def _validate_user_R(self, R: NDF | None) -> NDF | None:
        if R is None:
            return None

        given_shape = R.shape
        implied_shape = (len(self.observables), len(self.observables))
        if given_shape != implied_shape:
            raise ValueError(
                f"Provided R matrix has shape {given_shape} but expected {implied_shape} based on number of observables."
            )

        return R

    def _build_constant_R(self, R: NDF | None) -> NDF:
        validated_R = self._validate_user_R(R)
        if validated_R is not None:
            return validated_R

        conf = self.kalman_config

        builder = getattr(conf, "R_builder", None)
        arg_names = getattr(conf, "R_param_names", None)
        if builder is not None and arg_names is not None:
            calib = self.model.config.calibration.parameters
            params_by_name = {
                (k if isinstance(k, str) else k.name): float64(v)
                for k, v in calib.items()
            }

            vals = []
            for name in arg_names:
                if name not in params_by_name:
                    raise KeyError(
                        f"Missing R-builder parameter '{name}' in calibration."
                    )
                vals.append(params_by_name[name])

            R_full = asarray(builder(*vals), dtype=float64)
            n_all = len(self.model.compiled.observable_names)
            if R_full.shape != (n_all, n_all):
                raise ValueError(
                    f"R builder returned shape {R_full.shape}, expected ({n_all}, {n_all})."
                )

            obs_idx = self._obs_idx
            mat_idx = [obs_idx[name] for name in self.observables]
            return asarray(R_full[np.ix_(mat_idx, mat_idx)], dtype=float64)

        R = getattr(conf, "R", None)
        if R is None:
            raise ValueError("Constant R matrix not specified in configuration.")

        # Get included observables
        obs_idx = self._obs_idx
        mat_idx = [obs_idx[name] for name in self.observables]
        R_subset: NDF = asarray(R[np.ix_(mat_idx, mat_idx)], dtype=float64)
        return R_subset

    def _initial_R_diag_guess(self, eps: float64 = float64(1e-6)) -> NDF:
        diag = np.asarray(
            [
                np.maximum(0.1 * np.var(self.y[:, i]), eps)
                for i in range(self.y.shape[1])
            ],
            dtype=float64,
        )
        return np.diag(diag).astype(float64)

    def _ML_estimate_R_diag(
        self,
        scale_factor: float = 1.0,
    ) -> None:
        n = len(self.observables)
        R_0 = self._initial_R_diag_guess()
        eta_0: NDF = np.log(np.diag(R_0))
        bounds = [(-30, 10)] * n  # e^(-30) ~ 9e-14, e^(10) ~ 22026

        def obj(eta: NDF) -> float64:
            R_diag = np.exp(eta)
            R = np.diag(R_diag)
            result: FilterResult | UnscentedFilterResult = self.filter(
                x0=np.zeros((self.A.shape[0],), dtype=float64),
                _debug=False,
                _arg_overrides={"R": R},
            )

            return np.float64(-1.0 * result.loglik)

        opt = optimize.minimize(
            obj,
            x0=eta_0,
            bounds=bounds,
            method="L-BFGS-B",
        )
        if not opt.success:
            warnings.warn(
                f"R estimation optimization did not converge: {opt.message}",
                UserWarning,
            )
            self.R = R_0 * scale_factor
            print(f"Using initial diagonal R guess:\n {self.R}")
            return

        estimated_R = np.diag(np.exp(opt.x)) * scale_factor
        print(
            f"R estimation optimization successful.\nUsing estimated R matrix:\n {estimated_R}\nLog-likelihood: {-opt.fun}"
        )
        self.R = estimated_R

    def _build_P0(
        self,
        p0_mode: Literal["diag", "eye"] | None = None,
        p0_scale: Float64Like | None = None,
    ) -> NDF:
        mode = getattr(self, "mode", FilterMode.LINEAR)
        if mode == FilterMode.UNSCENTED:
            return self._build_unscented_P0(p0_mode=p0_mode, p0_scale=p0_scale)
        return self._build_full_P0(p0_mode=p0_mode, p0_scale=p0_scale)

    def _build_full_P0(
        self,
        p0_mode: Literal["diag", "eye"] | None = None,
        p0_scale: Float64Like | None = None,
    ) -> NDF:
        return self._build_named_P0(
            self.model.compiled.var_names,
            p0_mode=p0_mode,
            p0_scale=p0_scale,
            required_scope="model variables",
        )

    def _build_unscented_P0(
        self,
        p0_mode: Literal["diag", "eye"] | None = None,
        p0_scale: Float64Like | None = None,
    ) -> NDF:
        n_state = self.model.compiled.n_state
        state_vars = self.model.compiled.var_names[:n_state]
        state_P0 = self._build_named_P0(
            state_vars,
            p0_mode=p0_mode,
            p0_scale=p0_scale,
            required_scope="state variables",
        )

        out = np.zeros((2 * n_state, 2 * n_state), dtype=float64)
        out[:n_state, :n_state] = state_P0
        out[n_state:, n_state:] = np.eye(n_state, dtype=float64)
        return out

    def _build_named_P0(
        self,
        vars_ordered: list[str],
        p0_mode: Literal["diag", "eye"] | None = None,
        p0_scale: Float64Like | None = None,
        required_scope: str = "model variables",
    ) -> NDF:
        conf = self.kalman_config
        n = len(vars_ordered)

        if (P0 := getattr(conf, "P0", None)) is not None:
            mode = p0_mode if p0_mode is not None else P0.mode
            scale = (
                float64(p0_scale)
                if p0_scale is not None
                else float64(getattr(P0, "scale", 1.0))
            )
            if mode == "diag":
                if (diag_dict := getattr(P0, "diag", None)) is not None:
                    if not all(var in diag_dict for var in vars_ordered):
                        raise ValueError(
                            f"P0 diagonal specification must include all {required_scope}."
                        )

                    mat = np.zeros((n, n), dtype=float64)
                    for i, var in enumerate(vars_ordered):
                        mat[i, i] = float64(diag_dict.get(var, 1.0)) * scale
                    return mat
                else:
                    raise ValueError(
                        "P0 diagonal specification missing in configuration."
                    )
            elif mode == "eye":
                return np.eye(n, dtype=float64) * scale
            else:
                raise ValueError(
                    f"Unrecognized P0 mode: {mode}. Expected 'diag' or 'eye'."
                )

        else:
            if p0_mode is None or p0_scale is None:
                raise ValueError(
                    "P0 configuration not found in KalmanConfig. "
                    "Both p0_mode and p0_scale must be provided as overrides."
                )

            if p0_mode == "diag":
                raise ValueError(
                    "P0 diagonal specification must be provided in configuration when p0_mode is 'diag'."
                )
            elif p0_mode == "eye":
                return np.eye(n, dtype=float64) * float64(p0_scale)
            else:
                raise ValueError(
                    f"Unrecognized p0_mode: {p0_mode}. Expected 'diag' or 'eye'."
                )

    def _build_Q(self) -> NDF:
        params = self.model_config.calibration.parameters
        shock_map = self.model.config.shock_map
        shock_std = self.model.config.calibration.shock_std
        shock_corr = self.model.config.calibration.shock_corr

        var_order = self.model.compiled.var_names
        exogs = var_order[: self.model.compiled.n_exog]

        rev: SymbolGetterDict[Symbol, Symbol] = SymbolGetterDict(
            {exo: shock for shock, exo in shock_map.items()}
        )
        shocks = [rev[exo] for exo in exogs]
        stds = asarray(
            [float64(params[shock_std[shock]]) for shock in shocks], dtype=float64
        )

        corr = np.eye(len(exogs), dtype=float64)
        n = len(stds)
        for i in range(n):
            for j in range(i + 1, n):
                pair = frozenset({shocks[i], shocks[j]})
                corr_sym = shock_corr.get(pair, None)
                if corr_sym is not None and corr_sym in params:
                    corr_ij = params[corr_sym]
                else:
                    corr_ij = 0.0

                corr[i, j] = corr_ij
                corr[j, i] = corr_ij

        return np.outer(stds, stds) * corr

    def _get_C_d(self) -> Tuple[NDF, NDF]:
        return self.model._build_C_d_from_obs(self.observables)

    def _reorder_obs(
        self, obs: list[str] | None, y: NDF | pd.DataFrame | None
    ) -> Tuple[list[str], NDF]:
        """
        Return (obs_canonical, y_reordered)

        Canonical order is model.compiled.observable_names.
        - If y is ndarray: assume columns are in the *provided* obs order (or config/default if obs is None)
        - If y is DataFrame: require it contains the provided obs names and align by column labels
        """
        # Canonical order (source of truth)
        canon = self.model.compiled.observable_names
        canon_idx = self._obs_idx

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

        if isinstance(y, pd.DataFrame):
            if any(n not in y.columns for n in obs_given):
                missing_cols = [n for n in obs_given if n not in y.columns]
                raise ValueError(
                    f"DataFrame is missing observable columns: {missing_cols}"
                )

            y_reordered = y.loc[:, obs_canonical].to_numpy(dtype=float64)

        else:
            # ndarray path: assume current column order is obs_given
            y_arr = asarray(y, dtype=float64)
            if y_arr.ndim != 2:
                raise ValueError(
                    f"Observation data must be 2D. Shape (T,m) expected, got {y_arr.shape}."
                )
            T, m = y_arr.shape
            if m != len(obs_given):
                raise ValueError(
                    f"y has {m} columns but obs list has {len(obs_given)} names."
                )

            # Map canonical names to their position in obs_given (current column positions)
            pos_in_given = {name: j for j, name in enumerate(obs_given)}
            y_reordered = y_arr[:, [pos_in_given[name] for name in obs_canonical]]

        if np.isnan(y_reordered).any():
            raise ValueError("Observation data contains NaN values.")

        return obs_canonical, y_reordered

    def _validate_mode_and_inputs(self) -> None:
        if self.mode == FilterMode.LINEAR:
            if (self.C is None) or self.d is None:
                raise ValueError(
                    "C and d matrices are required for linear Kalman Filter."
                )
            # if (self.h_func is not None) or (self.H_jac is not None):
            #     warnings.warn(
            #         "h_func and H_jac are ignored in linear filter mode.",
            #         UserWarning,
            #     )

        elif self.mode == FilterMode.EXTENDED:
            if (self.h_func is None) or (self.H_jac is None):
                raise ValueError(
                    "h_func and H_jac are required for extended Kalman Filter."
                )
            # if (self.C is not None) or (self.d is not None):
            #     warnings.warn(
            #         "C and d matrices are ignored in extended filter mode.",
            #         UserWarning,
            #     )

        elif self.mode == FilterMode.UNSCENTED:
            if self.meas_addr is None or self.meas_addr == 0:
                raise ValueError("meas_addr is required for unscented Kalman Filter.")
            if self.calib_params is None:
                raise ValueError(
                    "calib_params is required for unscented Kalman Filter."
                )
            if getattr(getattr(self.model, "policy", None), "order", None) != 2:
                raise ValueError(
                    "Unscented Kalman Filter requires a second order solution."
                )

        else:
            raise ValueError(f"Unrecognized filter mode: {self.mode}")

    @staticmethod
    def _check_covariance_matrix(name: str, value: NDF) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray.")
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError(f"{name} must be a square 2D matrix.")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values.")
        if not np.allclose(value, value.T):
            raise ValueError(f"{name} must be symmetric.")
        if np.any(np.diag(value) < 0.0):
            raise ValueError(f"{name} must have non-negative diagonal entries.")

    def _validate_unscented_covariances(
        self,
        run_args: dict[str, Any],
        *,
        arg_overrides: dict[str, Any],
    ) -> None:
        const_validated = (
            self._cache_record.validated and self._uses_const_R and not arg_overrides
        )
        if const_validated:
            return

        self._check_covariance_matrix("Q", run_args["Q"])
        self._check_covariance_matrix("R", run_args["R"])
        self._check_covariance_matrix("P0", run_args["P0"])
        if self._uses_const_R and not arg_overrides:
            self._cache_record.validated = True

    def _build_unscented_z0(self, x0: NDF | None) -> NDF:
        n_state = self.model.compiled.n_state
        n_var = len(self.model.compiled.var_names)
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

    def _policy_array(self, name: str) -> NDF:
        value = getattr(self.model.policy, name, None)
        if value is None:
            raise ValueError(f"Unscented filtering requires policy.{name}.")
        return asarray(np.real_if_close(value), dtype=float64)

    @cached_property
    def _obs_idx(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.model.compiled.observable_names)}

    @cached_property
    def model_config(self) -> ModelConfig:
        return self.model.config

    @cached_property
    def kalman_config(self) -> KalmanConfig:
        config = self.model.kalman_config
        if config is None:
            raise ValueError(
                "Kalman Filter configuration with the R matrix is required."
            )
        return config

    @property
    def _linear_validated_args(self) -> dict:
        return {
            # State Space Definition
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "d": self.d,
            "Q": self.Q,
            "R": self.R,
            "P0": self.P0,
            # Initial State
            # "x0": x0,  # provided at filter() call time
            # Data
            "y": self.y,
        }

    @property
    def _extended_validated_args(self) -> dict:
        return {
            # State Space Definition
            "A": self.A,
            "B": self.B,
            "h": self.h_func,
            "H_jac": self.H_jac,
            "calib_params": self.calib_params,
            "Q": self.Q,
            "R": self.R,
            "P0": self.P0,
            # Initial State
            # "x0": x0,  # provided at filter() call time
            # Data
            "y": self.y,
        }

    @property
    def _unscented_validated_args(self) -> dict:
        n_state = self.model.compiled.n_state
        return {
            "meas_addr": self.meas_addr,
            "hx": self._policy_array("p"),
            "gx": self._policy_array("f"),
            "bx": asarray(self.B[:n_state, :], dtype=float64),
            "hxx": self._policy_array("hxx"),
            "gxx": self._policy_array("gxx"),
            "hss": self._policy_array("hss"),
            "gss": self._policy_array("gss"),
            "steady_state": self._policy_array("steady_state"),
            "calib_params": self.calib_params,
            "Q": self.Q,
            "R": self.R,
            "y": self.y,
            "P0": self.P0,
        }
