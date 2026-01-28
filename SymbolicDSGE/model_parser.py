from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml
import sympy as sp
from sympy import Symbol, Function, Eq, Expr
from sympy.core.relational import Relational
from numpy import float64

from .model_config import (
    ModelConfig,
    Equations,
    Calib,
    SymbolGetterDict,
    PairGetterDict,
)
from .kalman.config import KalmanConfig, make_R, P0Config


@dataclass(frozen=True)
class ParsedConfig:
    model: ModelConfig
    kalman: KalmanConfig | None

    def __iter__(self) -> Iterable[Any]:
        yield from (self.model, self.kalman)


class ModelParser:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.parsed: ParsedConfig = self.from_yaml()
        self.__post_init__()

    def __post_init__(self) -> None:
        conf = self.parsed.model
        self.validate_constraints(conf)
        self.validate_calib(conf)

    def get(self) -> ModelConfig:
        return self.parsed.model

    def get_all(self) -> ParsedConfig:
        return self.parsed

    # --- existing validators unchanged ---
    @classmethod
    def validate_constraints(cls, conf: ModelConfig) -> None:
        is_constrained = conf.constrained
        doesnt_exist = []
        no_constraint = []
        for var, constrained in is_constrained.items():
            if var not in conf.variables:
                doesnt_exist.append(var)
            if constrained and var not in conf.equations.constraint:
                no_constraint.append(var)

        if doesnt_exist and no_constraint:
            raise ValueError(
                f"The following variables are marked as constrained but do not exist: {doesnt_exist}, "
                f"and the following constrained variables have no constraint equations: {no_constraint}"
            )
        elif doesnt_exist:
            raise ValueError(
                f"The following variables are marked as constrained but do not exist: {doesnt_exist}"
            )
        elif no_constraint:
            raise ValueError(
                f"The following constrained variables have no constraint equations: {no_constraint}"
            )

    @classmethod
    def validate_calib(cls, conf: ModelConfig) -> None:
        nf_param = []
        for param in conf.calibration.parameters:
            if param not in conf.parameters:
                nf_param.append(param)
        if nf_param:
            raise ValueError(f"Calibration contains unknown parameters: {nf_param}")

    # --- refactored from_yaml ---
    def from_yaml(self) -> ParsedConfig:
        data = self._load_yaml(self.config_path)
        self._require_calibrated_params(data)  # your strict contract

        ns = self._build_namespace(data)
        (
            _LOCALS,
            variables,
            constrained,
            params,
            observables,
            shock_map,
            shock_syms,
        ) = ns

        # SymPy parsing helpers bound to this namespace
        _get_expr, _get_relational, _get_eq = self._sympy_parsers(_LOCALS)

        equations = self._parse_equations(
            data, _LOCALS, _get_eq, _get_relational, _get_expr
        )
        parameters = self._parse_parameters(data, _LOCALS)

        shock_std, shock_corr = self._parse_shock_calibration(data, _LOCALS, shock_syms)
        calibration = Calib(
            parameters=parameters,
            shock_std=shock_std,
            shock_corr=PairGetterDict(shock_corr),
        )

        mdl_cfg = ModelConfig(
            name=data.get("name", "Unnamed"),
            variables=variables,
            constrained=constrained,
            parameters=params,
            shock_map=shock_map,
            observables=observables,
            equations=equations,
            calibration=calibration,
        )

        kalman_cfg = self._parse_kalman_if_present(
            data, _LOCALS, parameters, observables
        )
        return ParsedConfig(model=mdl_cfg, kalman=kalman_cfg)

    # ---------------- helpers ----------------

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise TypeError("YAML root must be a mapping/dict.")
        return data

    @staticmethod
    def _build_namespace(
        data: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        list[Function],
        dict[Function, bool],
        list[Symbol],
        list[Symbol],
        SymbolGetterDict[Symbol, Symbol],
        list[Symbol],
    ]:
        t = sp.symbols("t", integer=True)

        variables: list[Function] = list(map(Function, data["variables"]))
        constrained: dict[Function, bool] = dict(
            zip(variables, [data["constrained"][var] for var in data["variables"]])
        )

        params: list[Symbol] = list(sp.symbols(data["parameters"]))
        observables: list[Symbol] = list(sp.symbols(data["observables"]))

        shock_map: SymbolGetterDict[Symbol, Symbol] = SymbolGetterDict(
            {sp.Symbol(k): sp.Symbol(v) for k, v in data["shock_map"].items()}
        )
        shock_syms: list[Symbol] = list(shock_map.keys())

        _LOCALS: dict[str, Any] = {
            "t": t,
            **{var.name: var for var in variables},
            **{param.name: param for param in params},
            **{shock.name: shock for shock in shock_syms},
            **{obs.name: obs for obs in observables},
        }
        return (
            _LOCALS,
            variables,
            constrained,
            params,
            observables,
            shock_map,
            shock_syms,
        )

    @staticmethod
    def _sympy_parsers(
        _LOCALS: dict[str, Any],
    ) -> tuple[
        Callable[[str], Expr],
        Callable[[str], Relational],
        Callable[[str], Eq],
    ]:
        def _get_expr(expr: str) -> Expr:
            out = sp.parse_expr(expr, local_dict=_LOCALS, evaluate=False)
            if not isinstance(out, Expr):
                raise TypeError(f"Expression is not a valid SymPy Expr: {expr!r}")
            return out

        def _get_relational(expr: str) -> Relational:
            out = sp.parse_expr(expr, local_dict=_LOCALS, evaluate=False)
            if not isinstance(out, Relational):
                raise TypeError(f"Constraint is not a valid SymPy Relational: {expr!r}")
            return out

        def _get_eq(expr: str) -> Eq:
            parts = [p.strip() for p in expr.split("=", maxsplit=2)]
            if len(parts) != 2:
                raise ValueError(f"Equation must contain exactly one '=': {expr!r}")
            lhs = sp.parse_expr(parts[0], local_dict=_LOCALS, evaluate=False)
            rhs = sp.parse_expr(parts[1], local_dict=_LOCALS, evaluate=False)
            out = sp.Eq(lhs, rhs)
            if not isinstance(out, Eq):
                raise TypeError(f"Not a valid equality: {expr!r}")
            return out

        return _get_expr, _get_relational, _get_eq

    @staticmethod
    def _parse_equations(
        data: dict[str, Any],
        _LOCALS: dict[str, Any],
        _get_eq: Callable[[str], Eq],
        _get_relational: Callable[[str], Relational],
        _get_expr: Callable[[str], Expr],
    ) -> Equations:
        eq_data = data["equations"]

        model: list[Eq] = [_get_eq(eq) for eq in eq_data["model"]]

        # preserve variable order
        ordered_var_names: list[str] = list(data["variables"])
        constraint_raw = eq_data.get("constraint", {}) or {}
        constraint: dict[Symbol, Relational] = {
            _LOCALS[var_name]: _get_relational(constraint_raw[var_name])
            for var_name in ordered_var_names
            if var_name in constraint_raw
        }

        observables_raw = eq_data.get("observables", {}) or {}
        observables_eq: dict[Symbol, Expr] = {
            _LOCALS[obs_name]: _get_expr(observables_raw[obs_name])
            for obs_name in data["observables"]
            if obs_name in observables_raw
        }

        return Equations(
            model=model,
            constraint=SymbolGetterDict(constraint),
            observable=SymbolGetterDict(observables_eq),
        )

    @staticmethod
    def _parse_parameters(
        data: dict[str, Any], _LOCALS: dict[str, Any]
    ) -> SymbolGetterDict[Symbol, float64]:
        calib = data.get("calibration", {}).get("parameters", {}) or {}
        return SymbolGetterDict(
            {
                _LOCALS[param_name]: float64(calib[param_name])
                for param_name in data["parameters"]
                if param_name in calib
            }
        )

    @staticmethod
    def _parse_shock_calibration(
        data: dict[str, Any],
        _LOCALS: dict[str, Any],
        shock_syms: list[Symbol],
    ) -> tuple[
        SymbolGetterDict[Symbol, Symbol | None], dict[frozenset[Symbol], Symbol | None]
    ]:
        shocks = data.get("calibration", {}).get("shocks", {}) or {}
        std_map = shocks.get("std", {}) or {}
        corr_map = shocks.get("corr", {}) or {}

        # std: map shock symbol -> parameter Symbol (or None)
        shock_std: SymbolGetterDict[Symbol, Symbol | None] = SymbolGetterDict(
            {
                s: (sp.Symbol(std_map[s.name]) if s.name in std_map else None)
                for s in shock_syms
            }
        )

        # corr: map unordered pair(shock_i, shock_j) -> parameter Symbol (or None)
        shock_corr: dict[frozenset[Symbol], Symbol | None] = {}

        # fill explicitly provided
        for pair_str, param_name in corr_map.items():
            names = [x.strip() for x in pair_str.split(",")]
            if len(names) != 2:
                raise ValueError(
                    f"Correlation pair must contain exactly two shocks: {pair_str!r}"
                )
            a = _LOCALS[names[0]]
            b = _LOCALS[names[1]]
            shock_corr[frozenset((a, b))] = sp.Symbol(param_name)

        # fill missing with None (all unordered pairs i<j)
        for i in range(len(shock_syms)):
            for j in range(i + 1, len(shock_syms)):
                key = frozenset((shock_syms[i], shock_syms[j]))
                shock_corr.setdefault(key, None)

        return shock_std, shock_corr

    @staticmethod
    def _parse_kalman_if_present(
        data: dict[str, Any],
        _LOCALS: dict[str, Any],
        parameters: SymbolGetterDict[Symbol, float64],
        observables: list[Symbol],
    ) -> KalmanConfig | None:
        kalman_data = data.get("kalman")
        if not kalman_data:
            return None

        y_order = [_LOCALS[o] for o in data["observables"]]
        y_cfg = kalman_data.get("y", [])
        if y_cfg:
            y_str = [o.name for o in y_order]
        else:
            y_str = y_cfg

        jit = kalman_data.get("jitter", None)
        symm = kalman_data.get("symmetrize", None)

        P0 = kalman_data.get("P0", {}) or {}
        P0_mode = P0.get("mode", "diag")
        P0_scale = float64(P0.get("scale", 1.0))
        P0_diag = P0.get("diag", None)

        R_data = kalman_data.get("R")
        if R_data:
            std_map = R_data.get("std", {}) or {}
            obs_sig: SymbolGetterDict[Symbol, float64] = SymbolGetterDict(
                {
                    _LOCALS[obs_name]: parameters[_LOCALS[param_name]]
                    for obs_name, param_name in std_map.items()
                }
            )

            corr_map = R_data.get("corr", {}) or {}
            obs_corr: dict[frozenset[Symbol], float64] = {}
            for pair_str, param_name in corr_map.items():
                names = [x.strip() for x in pair_str.split(",")]
                if len(names) != 2:
                    raise ValueError(
                        f"Correlation pair must contain exactly two observables: {pair_str!r}"
                    )
                a = _LOCALS[names[0]]
                b = _LOCALS[names[1]]
                obs_corr[frozenset((a, b))] = parameters[_LOCALS[param_name]]

            R = make_R(y_order, obs_sig, PairGetterDict(obs_corr))
        else:
            R = None

        P0_cfg = P0Config(
            mode=P0_mode,
            scale=P0_scale,
            diag=P0_diag,
        )

        return KalmanConfig(
            y_names=y_str,
            R=R,
            jitter=jit,
            symmetrize=symm,
            P0=P0_cfg,
        )

    # --- your strict contract kept (minor tidy only) ---
    @staticmethod
    def _require_calibrated_params(data: dict[str, Any]) -> None:
        declared = set(data.get("parameters", []))
        calib = data.get("calibration", {}).get("parameters", {}) or {}
        calibrated = set(calib.keys())

        missing_declared = sorted(declared - calibrated)
        if missing_declared:
            raise ValueError(
                "Missing calibration values for declared parameter(s): "
                + ", ".join(missing_declared)
            )

        referenced: set[str] = set()

        shocks = data.get("calibration", {}).get("shocks", {}) or {}
        referenced.update((shocks.get("std", {}) or {}).values())
        referenced.update((shocks.get("corr", {}) or {}).values())

        kal = data.get("kalman", {}) or {}
        R = kal.get("R", {}) or {}
        referenced.update((R.get("std", {}) or {}).values())
        referenced.update((R.get("corr", {}) or {}).values())

        referenced = {p for p in referenced if isinstance(p, str)}

        unknown = sorted(referenced - declared)
        if unknown:
            raise ValueError(
                "Config references parameter(s) not declared in `parameters`: "
                + ", ".join(unknown)
            )

        missing_ref = sorted(referenced - calibrated)
        if missing_ref:
            raise ValueError(
                "Config references parameter(s) without calibration values: "
                + ", ".join(missing_ref)
            )
