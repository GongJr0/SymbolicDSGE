from .filter import KalmanFilter, FilterResult
from .config import KalmanConfig
from .validator import validate_kf_inputs, _KalmanDebugInfo
from typing import TYPE_CHECKING, Tuple, Literal

if TYPE_CHECKING:
    from ..solver import SolvedModel

from ..model_config import ModelConfig, SymbolGetterDict


from functools import cached_property

import numpy as np
from numpy import asarray, float64, int64
from numpy.typing import NDArray

from sympy import Symbol

import pandas as pd

NDF = NDArray[float64]
Float64Like = float | float64 | int | int64


class KalmanInterface(KalmanFilter):
    def __init__(
        self,
        model: "SolvedModel",
        observables: list[str] | None,
        y: NDF | pd.DataFrame,
        *,
        p0_mode: Literal["diag", "eye"] | None = None,
        p0_scale: Float64Like | None = None,
        jitter: Float64Like | None = None,
        symmetrize: bool | None = None,
        return_shocks: bool = False,
    ) -> None:
        self.model = model

        obs, y = self._reorder_obs(observables, y)
        self.observables = obs
        self.y = y

        self.A, self.B = model.A, model.B
        self.C, self.d = self._get_C_d()
        self.Q = self._build_Q()
        self.P0 = self._build_P0(p0_mode=p0_mode, p0_scale=p0_scale)
        self.R = self._build_constant_R()

        self.jitter = self._get_jitter(jitter)
        self.symmetrize = self._get_symmetrize(symmetrize)
        self.return_shocks = bool(return_shocks)

        self._debug_info: _KalmanDebugInfo | None = None

    def filter(self, x0: NDF | None = None, _debug: bool = False) -> FilterResult:

        if x0 is None:
            x0 = np.zeros((self.A.shape[0],), dtype=float64)

        validated_args = {
            # State Space Definition
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "d": self.d,
            "Q": self.Q,
            "R": self.R,
            "P0": self.P0,
            # Initial State
            "x0": x0,
            # Data
            "y": self.y,
        }
        validate_kf_inputs(
            **validated_args, check_nonneg_diag=True, check_symmetry=True  # type: ignore
        )

        run = self.run(
            **validated_args,
            # Options
            jitter=self.jitter,
            symmetrize=self.symmetrize,
            return_shocks=self.return_shocks,
        )
        if _debug:
            self._debug_info = _KalmanDebugInfo(
                A=self.A,
                B=self.B,
                C=self.C,
                d=self.d,
                Q=self.Q,
                R=self.R,
                y=self.y,
                x0=x0,
                P0=self.P0,
            )
        return run

    def _get_symmetrize(self, symmetrize_arg: bool | None) -> bool:
        if symmetrize_arg is not None:
            return bool(symmetrize_arg)

        conf = self.kalman_config
        if (sym_conf := getattr(conf, "symmetrize", None)) is not None:
            return bool(sym_conf)
        return False

    def _get_jitter(self, jitter_arg: Float64Like | None) -> float64:
        if jitter_arg is not None:
            return float64(jitter_arg)

        conf = self.kalman_config
        if (jitter_conf := getattr(conf, "jitter", None)) is not None:
            return float64(jitter_conf)

        return float64(0.0)

    def _build_constant_R(self) -> NDF:
        conf = self.kalman_config
        R = getattr(conf, "R", None)
        if R is None:
            raise ValueError("Constant R matrix not specified in configuration.")

        # Get included observables
        obs_idx = self._obs_idx
        mat_idx = [obs_idx[name] for name in self.observables]
        R_subset: NDF = R[np.ix_(mat_idx, mat_idx)]
        return R_subset

    def _build_P0(
        self,
        p0_mode: Literal["diag", "eye"] | None = None,
        p0_scale: Float64Like | None = None,
    ) -> NDF:
        conf = self.kalman_config
        vars_ordered = self.model.compiled.var_names
        n = len(vars_ordered)

        # Branch 0: Has P0 Config
        if (P0 := getattr(conf, "P0", None)) is not None:
            mode = p0_mode if p0_mode is not None else P0.mode
            scale = (
                float64(p0_scale)
                if p0_scale is not None
                else float64(getattr(P0, "scale", 1.0))
            )
            if mode == "diag":
                if (diag_dict := getattr(P0, "diag", None)) is not None:
                    # Check all variables have value
                    if not all(var in diag_dict for var in vars_ordered):
                        raise ValueError(
                            "P0 diagonal specification must include all model variables."
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

        # Branch 1: No P0 Config, Check for overrides
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
                if p0_scale is None:
                    raise ValueError("p0_scale must be provided when p0_mode is 'eye'.")
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
        self, obs: list[str] | None, y: NDF | pd.DataFrame
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
            if (obs_ls := getattr(self.kalman_config, "y_names", None)) is not None:
                obs_given = list(obs_ls)
            else:
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
