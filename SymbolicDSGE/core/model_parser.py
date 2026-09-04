from __future__ import annotations

import copy
import re
import sys
from dataclasses import dataclass
from io import StringIO
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import FrameType
from typing import Any, Callable, Iterator, TypeAlias
import warnings

from sympy.core.basic import Basic
from sympy.core.symbol import AppliedUndef
import yaml
import sympy as sp
from sympy import Symbol, Function, Eq, Expr
from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Or, Not
from sympy.parsing.sympy_parser import standard_transformations, convert_xor
from numpy import float64, ndarray

from .config import (
    ModelConfig,
    Constraint,
    Regime,
    Equations,
    Calib,
    RegimeGetterDict,
    Variables,
    SymbolGetterDict,
    PairGetterDict,
    FunctionGetterDict,
)
from ..kalman.config import KalmanConfig, make_R
from .linearization import LinearizationMethod

_GLOBAL_TRANSFORMATIONS = standard_transformations + (convert_xor,)

#: The only fields permitted at the top level of a model config. Any other key
#: is rejected at parse time (see ``_validate_schema``).
_DEPRECATED_TOP_LEVEL_KEYS = frozenset({"parameters"})
_ALLOWED_TOP_LEVEL_KEYS = (
    frozenset(
        {
            "name",
            "variables",
            "observables",
            "shocks",
            "equations",
            "calibration",
            "kalman",
        }
    )
    | _DEPRECATED_TOP_LEVEL_KEYS
)

#: Allowed sub-keys for the nested config blocks
_ALLOWED_EQUATION_KEYS = frozenset({"model", "constraint", "regime", "observables"})
_ALLOWED_CONSTRAINT_KEYS = frozenset({"bind", "relax"})

#: Sympy types accepted as a regime entry/exit condition.
_REGIME_SHIFT_CONDITIONAL: TypeAlias = Relational | And | Or | Not

_ALLOWED_CALIBRATION_KEYS = frozenset({"parameters", "shocks"})
_ALLOWED_SHOCK_KEYS = frozenset({"std", "corr"})
_ALLOWED_KALMAN_KEYS = frozenset({"R"})
_ALLOWED_P0_KEYS = frozenset({"mode", "diag"})
_ALLOWED_R_KEYS = frozenset({"std", "corr"})


def _caller_stacklevel() -> int:
    """``warnings.warn`` stacklevel of the first frame outside this module.

    Entry depth varies: ``from_string`` re-enters through ``__init__``, so it
    sits one frame deeper than path-based construction.
    """
    frame: FrameType | None = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename == __file__:
        frame = frame.f_back
        level += 1
    return level


def _parse_in(text: str, _LOCALS: dict[str, Any]) -> Basic:
    """Parse ``text`` against the config namespace.

    ``evaluate=False`` preserves the authored form rather than SymPy's
    canonical rearrangement of it.
    """
    return sp.parse_expr(
        text,
        local_dict=_LOCALS,
        evaluate=False,
        transformations=_GLOBAL_TRANSFORMATIONS,
    )


def _check_connective_parens(expr: str) -> None:
    """Require each ``&``/``|`` in a condition to join parenthesized relations.

    Python binds ``&``/``|`` tighter than the comparisons, so ``x > 0 | y < 1``
    parses as ``And(x > y, y < 1)``: a valid condition that tests something the
    user never wrote. No inspection of the parsed tree can detect that, so the
    raw text is the only place to catch it.
    """
    for match in re.finditer(r"[&|]", expr):
        before = expr[: match.start()].rstrip()
        after = expr[match.end() :].lstrip()
        if not before.endswith(")") or not after.startswith("("):
            raise ValueError(
                f"Condition {expr!r} uses '{match.group()}' outside parentheses. "
                f"'&' and '|' bind tighter than the comparisons, so each relation "
                f"must be parenthesized: '(x > 0) {match.group()} (y < 1)'."
            )


def _list_representer(dumper: yaml.Dumper, data: list[Any]) -> yaml.Node:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


class InlineList(list):
    pass


@dataclass(frozen=True)
class ParsedConfig:
    model: ModelConfig
    kalman: KalmanConfig | None

    def __iter__(self) -> Iterator[Any]:
        yield from (self.model, self.kalman)


class ModelParser:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.raw_data, self.parsed = self.from_yaml()
        self.parsed.model.source_yaml = self.config_path.read_text(encoding="utf-8")
        self.__post_init__()

    def __post_init__(self) -> None:
        conf = self.parsed.model
        self.validate_equations(conf)
        self.validate_constraints(conf)
        self.validate_regimes(conf)  # reads validated bind conditions
        self.validate_ss_seed(conf)

    def get(self) -> ModelConfig:
        return self.parsed.model

    def get_all(self) -> ParsedConfig:
        return self.parsed

    @classmethod
    def from_string(cls, text: str) -> "ModelParser":
        """Construct a parser from YAML *text* (e.g. a bundle config member).

        Mirrors path-based construction by routing the text through a temporary
        file, so the full parse pipeline (including the Kalman block) runs
        unchanged.
        """
        with NamedTemporaryFile(
            "w", suffix=".yaml", encoding="utf-8", delete=False
        ) as handle:
            handle.write(text)
            tmp_path = Path(handle.name)
        try:
            parser = cls(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        # Preserve the caller's original text verbatim (the temp-file round-trip
        # is a no-op today, but pin it here so callers can rely on equality).
        parser.parsed.model.source_yaml = text
        return parser

    @classmethod
    def _validate_condition(cls, name: str, kind: str, cond: Any) -> None:
        """Reject a condition that is not relations joined by connectives.

        ``Symbol`` subclasses ``Boolean``, so ``And(x > 0, y)`` builds without
        complaint and the root type gate accepts it. Only a positive
        ``Relational`` check at every leaf catches the bare operand, and a
        positive ``Expr`` check on each side catches a relation compared against
        another relation.
        """
        if isinstance(cond, (And, Or, Not)):
            for arg in cond.args:
                cls._validate_condition(name, kind, arg)
            return

        if not isinstance(cond, Relational):
            raise TypeError(
                f"{kind} condition for constraint '{name}' is not a valid "
                f"SymPy Relational: {cond!r}"
            )

        for side in cond.args:
            if not isinstance(side, Expr):
                raise TypeError(
                    f"{kind} condition for constraint '{name}' compares the truth "
                    f"value {side!r} rather than a numeric expression, in {cond!r}. "
                    f"Combine relations with '&' or '|' between parenthesized "
                    f"relations, not with a comparison."
                )

    @staticmethod
    def _shock_atoms(conf: ModelConfig, *exprs: Basic) -> set[Symbol]:
        """Shocks referenced by *exprs*.

        Conditions are tested on a realized path where the shocks have already
        been absorbed into the variables, so a shock cannot appear in one.
        """
        shocks = set(conf.shocks)
        found: set[Symbol] = set()
        for expr in exprs:
            found |= expr.free_symbols & shocks  # pyright: ignore
        return found

    @staticmethod
    def _unknown_atoms(conf: ModelConfig, *exprs: Basic) -> set[Any]:
        """Symbols referenced by *exprs* that the model declares nowhere.

        Declared means a variable, a parameter, or a shock: regime replacements
        are ordinary model equations, so a shock is as legitimate there as it is
        in the equation being replaced. Conditions reject shocks earlier, with
        their own message, so nothing reaches here relying on the old behavior.

        Time symbols come from the free symbols of each applied function's
        arguments, so an offset term like ``x(t+1)`` clears ``t`` the way
        ``x(t)`` does.
        """
        applied: set[Any] = set()
        free: set[Symbol] = set()
        for expr in exprs:
            applied |= expr.atoms(AppliedUndef)
            free |= expr.free_symbols  # pyright: ignore

        time_syms = {s for c in applied for a in c.args for s in a.free_symbols}
        var_funcs = {c.func for c in applied}

        declared = set(conf.parameters) | set(conf.shocks)
        return (var_funcs - set(conf.variables.variables)) | (
            (free - time_syms) - declared
        )

    @classmethod
    def validate_equations(cls, conf: ModelConfig) -> None:
        """Reject a model equation naming a symbol the model declares nowhere.

        The same check regime replacements get, on the equations they replace: a
        typo would otherwise survive parse as a live ``Symbol`` and only fail in
        the printer, where the name is no longer attached to an equation.
        """
        for name, eq in conf.equations.model.items():
            unknown_atoms = cls._unknown_atoms(conf, eq)
            if unknown_atoms:
                raise ValueError(
                    f"Equation '{name}' references unknown symbols: "
                    f"{sorted(str(a) for a in unknown_atoms)}"
                )

    @classmethod
    def validate_constraints(cls, conf: ModelConfig) -> None:
        if not conf.equations.constraint:
            return

        constraints: dict[str, Constraint] = conf.equations.constraint

        if len(constraints) > 2:
            raise NotImplementedError(
                "SymbolicDSGE currently supports the 1- and 2-constraint OccBin equivalent to that of Dynare."
            )

        for name, constraint in constraints.items():

            binds, relaxes = constraint.bind, constraint.relax
            cls._validate_condition(name, "Binding", binds)
            cls._validate_condition(name, "Relaxing", relaxes)

            # Named before _unknown_atoms, which would report a shock as an
            # unknown symbol: shocks are neither variables nor parameters.
            shocks = cls._shock_atoms(conf, binds, relaxes)
            if shocks:
                raise ValueError(
                    f"Constraint '{name}' references shock(s) "
                    f"{sorted(s.name for s in shocks)}; conditions may only "
                    f"reference model variables and parameters."
                )

            # Check if inequalities refer to uninitialized variables
            unknown_atoms = cls._unknown_atoms(conf, binds, relaxes)
            if unknown_atoms:
                raise ValueError(
                    f"Constraint '{name}' references unknown symbols: {unknown_atoms}"
                )

    @classmethod
    def validate_regimes(cls, conf: ModelConfig) -> None:
        constraints = conf.equations.constraint
        regimes = conf.equations.regime
        if not constraints and not regimes:
            return
        if not constraints or not regimes:
            raise ValueError(
                "equations.constraint and equations.regime must be declared together; "
                f"got constraint={'set' if constraints else 'empty'}, "
                f"regime={'set' if regimes else 'empty'}"
            )

        declared = set(constraints)
        unknown_members = {n for key in regimes for n in key} - declared
        if unknown_members:
            raise ValueError(
                f"Regime keys name undeclared constraints: {sorted(unknown_members)}; "
                f"declared: {sorted(declared)}"
            )

        # Every combination of binding constraints needs its own equation set.
        expected = {
            frozenset(combo)
            for size in range(1, len(declared) + 1)
            for combo in combinations(sorted(declared), size)
        }
        missing = expected - set(regimes)
        if missing:
            raise ValueError(
                "equations.regime is missing entries for binding combinations: "
                f"{sorted(', '.join(sorted(key)) for key in missing)}"
            )

        for key, regime in regimes.items():
            cls._validate_regime(conf, key, regime, constraints)

    @classmethod
    def _validate_regime(
        cls,
        conf: ModelConfig,
        key: frozenset[str],
        regime: Regime,
        constraints: dict[str, Constraint],
    ) -> None:
        label = ", ".join(sorted(key))

        if sp.simplify(sp.And(*(constraints[n].bind for n in key))) is sp.false:
            raise ValueError(
                f"Regime '{label}' can never bind: its members' conditions are mutually exclusive"
            )

        if not regime:
            raise ValueError(f"Regime '{label}' replaces no model equations")

        unknown_targets = set(regime) - set(conf.equations.model)
        if unknown_targets:
            raise ValueError(
                f"Regime '{label}' replaces undeclared model equations: "
                f"{sorted(unknown_targets)}; declared: {sorted(conf.equations.model)}"
            )

        for target, replacement in regime.items():
            if not isinstance(replacement, Eq):
                raise TypeError(
                    f"Replacement for '{target}' in regime '{label}' is not a valid SymPy Eq: {replacement!r}"
                )
            unknown_atoms = cls._unknown_atoms(conf, replacement)
            if unknown_atoms:
                raise ValueError(
                    f"Replacement for '{target}' in regime '{label}' "
                    f"references unknown symbols: {unknown_atoms}"
                )

    @classmethod
    def validate_ss_seed(cls, conf: ModelConfig) -> None:
        params = set(conf.parameters)
        for var, expr in conf.variables.ss_seed.items():
            if expr is None:
                continue
            applied = expr.atoms(AppliedUndef)
            if applied:
                raise ValueError(
                    f"ss_seed for variable '{var.__name__}' must resolve to a number "
                    f"but references model variables: {applied}"
                )
            unknown = expr.free_symbols - params
            if unknown:
                raise ValueError(
                    f"ss_seed for variable '{var.__name__}' references unknown symbols: {unknown}"
                )

    def from_yaml(self) -> tuple[dict, ParsedConfig]:
        data = self._load_yaml(self.config_path)
        self._validate_schema(data)

        ns = self._build_namespace(data)
        (
            _LOCALS,
            ordered_var_names,
            variable_funcs,
            params,
            observables,
            shocks,
            shock_syms,
        ) = ns

        # Locals resolve before any field is parsed, so the helpers below can
        # eliminate them from everything they build.
        parameters, local_subs = self._resolve_calibration_locals(data, _LOCALS)
        self._require_calibrated_params(data, local_subs)
        params = [param for param in params if param not in local_subs]

        _get_expr, _get_relational, _get_eq = self._sympy_parsers(_LOCALS, local_subs)

        variables = self._parse_variables(
            data, _LOCALS, ordered_var_names, variable_funcs, _get_expr
        )
        equations = self._parse_equations(
            data, _LOCALS, ordered_var_names, _get_eq, _get_relational, _get_expr
        )

        shock_std, shock_corr = self._parse_shock_calibration(data, _LOCALS, shock_syms)
        calibration = Calib(
            parameters=parameters,
            shock_std=shock_std,  # pyright: ignore
            shock_corr=PairGetterDict(shock_corr),
        )

        mdl_cfg = ModelConfig(
            name=data.get("name", "Unnamed"),
            variables=variables,
            parameters=params,
            shocks=shocks,
            observables=observables,
            equations=equations,
            calibration=calibration,
            symbolically_linearized=False,
        )

        kalman_cfg = self._parse_kalman_if_present(data, _LOCALS, parameters)
        return data, ParsedConfig(model=mdl_cfg, kalman=kalman_cfg)

    def to_yaml(
        self,
        config: ModelConfig | None = None,
        *,
        digits: int = 3,
    ) -> str:
        """Emit the authored configuration as canonical YAML text.

        Re-dumps the retained ``raw_data`` so the Kalman block and the
        ``(t)``/``(t+1)`` equation grammar round-trip through :class:`ModelParser`
        unchanged. When ``config`` is given, its calibration is baked in (rounded
        to ``digits``), capturing e.g. estimated parameter values.
        """
        data_out = copy.deepcopy(self.raw_data)

        if config is not None:
            # Update in place: derived entries carry no value to write back and
            # the equations dumped alongside them still reference their names.
            params_out = data_out.setdefault("calibration", {}).setdefault(
                "parameters", {}
            )
            params_out.update(
                {
                    str(k): round(float(v), digits)
                    for k, v in config.calibration.parameters.items()
                }
            )

        for key in ("variables", "observables"):
            if isinstance(data_out.get(key), list):
                data_out[key] = InlineList(data_out[key])

        yaml.add_representer(InlineList, _list_representer)
        buffer = StringIO()
        yaml.dump(data_out, buffer, sort_keys=False)
        return buffer.getvalue()

    def update_calibration_parameters(
        self,
        new_config: ModelConfig,
        digits: int = 3,
        output_path: str | Path | None = None,
    ) -> StringIO:
        text = self.to_yaml(new_config, digits=digits)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        return StringIO(text)

    # ---------------- helpers ----------------

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise TypeError("YAML root must be a mapping/dict.")
        return data

    @staticmethod
    def _reject_unknown_keys(mapping: Any, allowed: frozenset[str], where: str) -> None:
        if not isinstance(mapping, dict):
            return
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown field(s) under '{where}': {unknown}. "
                f"Allowed: {sorted(allowed)}."
            )

    @staticmethod
    def _require_mapping(value: Any, where: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError(
                f"'{where}' must be a mapping, got {type(value).__name__}: {value!r}"
            )
        return value

    @classmethod
    def _validate_schema(cls, data: dict[str, Any]) -> None:
        cls._check_deprecated(data)
        cls._reject_unknown_keys(data, _ALLOWED_TOP_LEVEL_KEYS, "<root>")
        cls._reject_unknown_keys(
            (eq := data.get("equations")), _ALLOWED_EQUATION_KEYS, "equations"
        )
        if isinstance(eq, dict):
            for name, spec in (eq.get("constraint") or {}).items():
                cls._reject_unknown_keys(
                    spec,
                    _ALLOWED_CONSTRAINT_KEYS,
                    f"equations.constraint.{name}",
                )
        calib = data.get("calibration")
        cls._reject_unknown_keys(calib, _ALLOWED_CALIBRATION_KEYS, "calibration")
        if isinstance(calib, dict):
            cls._reject_unknown_keys(
                calib.get("shocks"), _ALLOWED_SHOCK_KEYS, "calibration.shocks"
            )

        kal = data.get("kalman")
        cls._reject_unknown_keys(kal, _ALLOWED_KALMAN_KEYS, "kalman")
        if isinstance(kal, dict):
            cls._reject_unknown_keys(kal.get("P0"), _ALLOWED_P0_KEYS, "kalman.P0")
            cls._reject_unknown_keys(kal.get("R"), _ALLOWED_R_KEYS, "kalman.R")

    @staticmethod
    def _build_namespace(
        data: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        list[str],
        list[Function],
        list[Symbol],
        list[Symbol],
        list[Symbol],
        list[Symbol],
    ]:
        ordered_var_names, _ = ModelParser._coerce_variable_data(data)
        t = sp.symbols("t", integer=True)

        variables: list[Function] = list(
            map(Function, ordered_var_names)
        )  # pyright: ignore

        params: list[Symbol] = list(
            sp.symbols(list(data.get("calibration", {}).get("parameters", {}).keys()))
        )
        observables: list[Symbol] = list(sp.symbols(data["observables"]))

        shocks: list[Symbol] = [sp.Symbol(name) for name in data["shocks"]]
        shock_syms: list[Symbol] = list(shocks)

        _LOCALS: dict[str, Any] = {
            "t": t,
            **{var.name: var for var in variables},  # pyright: ignore
            **{param.name: param for param in params},
            **{shock.name: shock for shock in shock_syms},
            **{obs.name: obs for obs in observables},
        }
        return (
            _LOCALS,
            ordered_var_names,
            variables,
            params,
            observables,
            shocks,
            shock_syms,
        )

    @staticmethod
    def _coerce_variable_data(
        data: dict[str, Any],
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        raw_variables = data["variables"]
        if isinstance(raw_variables, list):
            ordered_var_names = list(raw_variables)
            return ordered_var_names, {name: {} for name in ordered_var_names}
        if not isinstance(raw_variables, dict):
            raise TypeError("`variables` must be either a list or a mapping.")

        ordered_var_names = list(raw_variables.keys())
        variable_data: dict[str, dict[str, Any]] = {}
        allowed_keys = {"ss_seed", "linearization"}
        for name, spec in raw_variables.items():
            if spec is None:
                variable_data[name] = {}
            elif isinstance(spec, dict):
                unknown_keys = sorted(set(spec).difference(allowed_keys))
                if unknown_keys:
                    raise ValueError(
                        f"Variable '{name}' has unsupported metadata keys: {unknown_keys}. "
                        f"Supported keys are: {sorted(allowed_keys)}."
                    )
                variable_data[name] = spec
            else:
                raise TypeError(
                    "Each variable entry must be a mapping or null when `variables` is a mapping."
                )
        return ordered_var_names, variable_data

    @staticmethod
    def _sympy_parsers(
        _LOCALS: dict[str, Any],
        local_subs: dict[Symbol, Expr],
    ) -> tuple[
        Callable[[str], Expr],
        Callable[[str], _REGIME_SHIFT_CONDITIONAL],
        Callable[[str], Eq],
    ]:
        """Parsing helpers bound to the config namespace.

        ``local_subs`` maps each derived calibration entry to its formula over
        base parameters. Applying it here is what keeps derived names out of
        the parsed config: every symbolic field the parser builds passes
        through one of these three.
        """

        def _get_expr(expr: str) -> Expr:
            out = _parse_in(expr, _LOCALS).xreplace(local_subs)
            if not isinstance(out, Expr):
                raise TypeError(f"Expression is not a valid SymPy Expr: {expr!r}")
            return out

        def _get_relational(expr: str) -> _REGIME_SHIFT_CONDITIONAL:
            _check_connective_parens(expr)
            out = _parse_in(expr, _LOCALS).xreplace(local_subs)
            if not isinstance(out, _REGIME_SHIFT_CONDITIONAL):
                raise TypeError(f"Constraint is not a valid SymPy Relational: {expr!r}")
            return out

        def _get_eq(expr: str | None) -> Eq:
            if expr is None:
                raise ValueError("Equation string cannot be None.")

            parts = [p.strip() for p in expr.split("=", maxsplit=2)]
            if len(parts) != 2:
                raise ValueError(f"Equation must contain exactly one '=': {expr!r}")
            lhs = _parse_in(parts[0], _LOCALS).xreplace(local_subs)
            rhs = _parse_in(parts[1], _LOCALS).xreplace(local_subs)
            out = sp.Eq(lhs, rhs)
            if not isinstance(out, Eq):
                raise TypeError(f"Not a valid equality: {expr!r}")
            return out

        return _get_expr, _get_relational, _get_eq

    @classmethod
    def _parse_equations(
        cls,
        data: dict[str, Any],
        _LOCALS: dict[str, Any],
        ordered_var_names: list[str],
        _get_eq: Callable[[str], Eq],
        _get_relational: Callable[[str], _REGIME_SHIFT_CONDITIONAL],
        _get_expr: Callable[[str], Expr],
    ) -> Equations:
        eq_data = data["equations"]

        model: dict[str, Eq] = {
            name: _get_eq(eq) for name, eq in eq_data["model"].items()
        }

        constraint_raw = eq_data.get("constraint", {}) or {}
        if len(constraint_raw) > 2:
            raise NotImplementedError(
                "OBCs are solved via OccBin of Guirreiri and Iacoviello (2015), which explicitly supports one or two constraints. "
                "Use at most two constraints in your model configuration."
            )

        constraint: dict[str, Constraint] = {}
        for name, raw_spec in constraint_raw.items():
            spec = cls._require_mapping(raw_spec, f"equations.constraint.{name}")
            if (bind := spec.get("bind")) is None:
                raise ValueError(f"Constraint '{name}' is missing a 'bind' condition.")
            bind = _get_relational(bind)
            if (relax := spec.get("relax")) is None:
                relax = Not(bind)  # Default relax for symmetric constraint
            else:
                relax = _get_relational(relax)

            constraint[name] = Constraint(
                bind=bind,
                relax=relax,
            )

        regime_raw = eq_data.get("regime", {}) or {}

        regime = RegimeGetterDict({})

        for raw_key, v in regime_raw.items():
            if raw_key in regime:
                raise ValueError(
                    f"Duplicate regime key '{raw_key}' encountered "
                    "in equations.regime. Each regime key must be unique."
                )
            regime[raw_key] = {
                name: _get_eq(eq)
                for name, eq in cls._require_mapping(
                    v, f"equations.regime.{raw_key}"
                ).items()
            }

        observables_raw = eq_data.get("observables", {}) or {}
        observables_eq: dict[Symbol, Expr] = {
            _LOCALS[obs_name]: _get_expr(observables_raw[obs_name])
            for obs_name in data["observables"]
            if obs_name in observables_raw
        }

        is_affine = ModelParser._derive_observable_structure(
            observables_eq=observables_eq,
            ordered_var_names=ordered_var_names,
            _LOCALS=_LOCALS,
        )

        return Equations(
            model=model,
            constraint=constraint if constraint else None,
            regime=regime if regime else None,
            observable=SymbolGetterDict(observables_eq),
            obs_is_affine=SymbolGetterDict(is_affine),
        )

    @staticmethod
    def _derive_observable_structure(
        *,
        observables_eq: dict[Symbol, Expr],
        ordered_var_names: list[str],
        _LOCALS: dict[str, Any],
    ) -> dict[Symbol, bool]:
        t = _LOCALS["t"]

        state_funcs = [_LOCALS[var_name] for var_name in ordered_var_names]
        state_atoms = [sf(t) for sf in state_funcs]

        state_sym_subs = {
            atom: Symbol(name) for atom, name in zip(state_atoms, ordered_var_names)
        }
        state_syms = list(state_sym_subs.values())
        state_set = set(state_syms)

        is_affine = {obs: False for obs in observables_eq}
        for obs, expr in observables_eq.items():
            expr_symbolized = expr.xreplace(state_sym_subs)
            grads = [expr_symbolized.diff(s) for s in state_syms]
            if all((g.free_symbols & state_set) == set() for g in grads):
                is_affine[obs] = True

        return is_affine

    @staticmethod
    def _parse_variables(
        data: dict[str, Any],
        _LOCALS: dict[str, Any],
        ordered_var_names: list[str],
        variable_funcs: list[Function],
        _get_expr: Callable[[str], Expr],
    ) -> Variables:
        _, variable_data = ModelParser._coerce_variable_data(data)

        ss_seed: dict[Function, Expr | None] = {}
        linearization: dict[Function, LinearizationMethod] = {}

        for var_name, var_func in zip(ordered_var_names, variable_funcs):
            spec = variable_data[var_name]

            ss_raw = spec.get("ss_seed", None)
            if ss_raw is None:
                ss_seed[var_func] = None
            else:
                ss_seed[var_func] = _get_expr(str(ss_raw))

            method_raw = spec.get("linearization", LinearizationMethod.NONE.value)
            if isinstance(method_raw, str):
                method_raw = method_raw.strip().lower()
            try:
                linearization[var_func] = LinearizationMethod(method_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid linearization method '{method_raw}' for variable '{var_name}'."
                ) from exc

        return Variables(
            variables=variable_funcs,
            ss_seed=FunctionGetterDict(ss_seed),
            linearization=FunctionGetterDict(linearization),
        )

    @staticmethod
    def _resolve_calibration_locals(
        data: dict[str, Any], _LOCALS: dict[str, Any]
    ) -> tuple[SymbolGetterDict[float64], dict[Symbol, Expr]]:
        """Split ``calibration.parameters`` into values and derived locals.

        An entry carrying free symbols names a formula over other parameters
        rather than a value, in the manner of a Dynare ``#`` definition. Such
        an entry never reaches the model: the returned map rewrites it away
        wherever it is referenced, leaving the base parameters behind it, so
        estimating those base parameters moves the formula with them.
        """
        calib = data.get("calibration", {}).get("parameters", {}) or {}
        param_syms = {_LOCALS[name] for name in calib}

        values: dict[Symbol, float64] = {}
        local_subs: dict[Symbol, Expr] = {}
        for name, raw in calib.items():
            expr = _parse_in(str(raw), _LOCALS)
            if not isinstance(expr, Expr):
                raise TypeError(
                    f"Calibration entry '{name}' is not a valid SymPy Expr: {raw!r}"
                )
            if applied := expr.atoms(AppliedUndef):
                raise ValueError(
                    f"Calibration entry '{name}' references model variable(s): "
                    f"{sorted(map(str, applied))}"
                )
            if unknown := expr.free_symbols - param_syms:
                raise ValueError(
                    f"Calibration entry '{name}' references undeclared "
                    f"parameter(s): {sorted(map(str, unknown))}"
                )
            if expr.free_symbols:
                local_subs[_LOCALS[name]] = expr
            else:
                values[_LOCALS[name]] = float64(expr)

        # A local may cite another. Each pass composes one level of nesting, so
        # their count bounds how many it can take to reach base parameters.
        derived = set(local_subs)
        for _ in range(len(local_subs)):
            if not any(e.free_symbols & derived for e in local_subs.values()):
                break
            local_subs = {s: e.xreplace(local_subs) for s, e in local_subs.items()}

        if cyclic := {s for s, e in local_subs.items() if e.free_symbols & derived}:
            raise ValueError(
                "Calibration entries reference each other in a cycle: "
                + ", ".join(sorted(s.name for s in cyclic))
            )

        return SymbolGetterDict(values), local_subs

    @staticmethod
    def _parse_shock_calibration(
        data: dict[str, Any],
        _LOCALS: dict[str, Any],
        shock_syms: list[Symbol],
    ) -> tuple[SymbolGetterDict[Symbol | None], PairGetterDict[Symbol | None]]:
        shocks = data.get("calibration", {}).get("shocks", {}) or {}
        std_map = shocks.get("std", {}) or {}
        corr_map = shocks.get("corr", {}) or {}

        # std: map shock symbol -> parameter Symbol (or None)
        shock_std: SymbolGetterDict[Symbol | None] = SymbolGetterDict(
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

        return shock_std, PairGetterDict(shock_corr)

    @staticmethod
    def _validate_P0(
        mode: str,
        diag: dict[str, float] | None,
        declared_var_names: list[str],
    ) -> None:
        if mode == "diag":
            if diag is None:
                raise ValueError("P0 diagonal specification missing in configuration.")
            declared_set = set(declared_var_names)
            diag_set = set(diag)
            if declared_set != diag_set:
                missing = sorted(declared_set - diag_set)
                unknown = sorted(diag_set - declared_set)
                raise ValueError(
                    "P0 diagonal specification must list exactly the model variables; "
                    f"missing {missing}, unknown {unknown}."
                )
            if any(v < 0 for v in diag.values()):
                raise ValueError("P0 diagonal entries must be non-negative.")
        elif mode != "eye":
            raise ValueError(f"Unrecognized P0 mode: {mode}. Expected 'diag' or 'eye'.")

    @staticmethod
    def _parse_kalman_if_present(
        data: dict[str, Any],
        _LOCALS: dict[str, Any],
        parameters: SymbolGetterDict[float64],
    ) -> KalmanConfig | None:
        kalman_data = data.get("kalman")
        if not kalman_data:
            return None

        y_order = [_LOCALS[o] for o in data["observables"]]
        obs_names = [o.name for o in y_order]

        declared_var_names, _ = ModelParser._coerce_variable_data(data)

        P0_cfg = kalman_data.get("P0", {}) or {}
        P0_mode = P0_cfg.get("mode", "eye")
        P0_diag = P0_cfg.get("diag", None)
        ModelParser._validate_P0(P0_mode, P0_diag, declared_var_names)

        R: ndarray | None
        r_param_symbols: list[Symbol] | None
        R_param_names: list[str] | None

        R_data = kalman_data.get("R")
        if R_data:
            std_map = R_data.get("std", {}) or {}
            corr_map = R_data.get("corr", {}) or {}
            R_std_param_map = {
                obs_name: param_name for obs_name, param_name in std_map.items()
            }
            R_corr_param_map: dict[frozenset[str], str | None] = {}
            obs_sig_sym: SymbolGetterDict[Symbol] = SymbolGetterDict(
                {
                    _LOCALS[obs_name]: _LOCALS[param_name]
                    for obs_name, param_name in std_map.items()
                }
            )
            obs_corr_sym_dict: dict[frozenset[Symbol], Symbol] = {}
            for pair_str, param_name in corr_map.items():
                names = [x.strip() for x in pair_str.split(",")]
                if len(names) != 2:
                    raise ValueError(
                        f"Correlation pair must contain exactly two observables: {pair_str!r}"
                    )
                a = _LOCALS[names[0]]
                b = _LOCALS[names[1]]
                obs_corr_sym_dict[frozenset((a, b))] = _LOCALS[param_name]
                R_corr_param_map[frozenset((names[0], names[1]))] = param_name
            obs_corr_sym: PairGetterDict[Symbol] = PairGetterDict(obs_corr_sym_dict)

            for i in range(len(obs_names)):
                for j in range(i + 1, len(obs_names)):
                    pair = frozenset((obs_names[i], obs_names[j]))
                    R_corr_param_map.setdefault(pair, None)

            std_vals = {y: parameters[obs_sig_sym[y]] for y in y_order}
            corr_vals = {pair: parameters[sym] for pair, sym in obs_corr_sym.items()}
            R = make_R(y_order, std_vals, corr_vals)

            r_param_symbols_local: list[Symbol] = []
            seen: set[Symbol] = set()
            for param_name in std_map.values():
                sym = _LOCALS[param_name]
                if sym not in seen:
                    seen.add(sym)
                    r_param_symbols_local.append(sym)
            for param_name in corr_map.values():
                sym = _LOCALS[param_name]
                if sym not in seen:
                    seen.add(sym)
                    r_param_symbols_local.append(sym)

            r_param_symbols = r_param_symbols_local

            R_param_names = [sym.name for sym in r_param_symbols]
        else:
            R = None
            R_param_names = None
            R_std_param_map = None
            R_corr_param_map = {}

        return KalmanConfig(
            R=R,
            R_param_names=R_param_names,
            R_std_param_map=R_std_param_map,
            R_corr_param_map=R_corr_param_map,
        )

    @staticmethod
    def _require_calibrated_params(
        data: dict[str, Any], local_subs: dict[Symbol, Expr]
    ) -> None:
        """Check the blocks that reference a parameter by name.

        Unlike an equation, these carry a bare name rather than an expression,
        so the name has to be one a value can be read from: a derived entry
        resolves to a formula and leaves nothing to build a covariance out of.
        """
        calib = data.get("calibration", {}).get("parameters", {}) or {}

        referenced: set[str] = set()

        shocks = data.get("calibration", {}).get("shocks", {}) or {}
        referenced.update((shocks.get("std", {}) or {}).values())
        referenced.update((shocks.get("corr", {}) or {}).values())

        kal = data.get("kalman", {}) or {}
        R = kal.get("R", {}) or {}
        referenced.update((R.get("std", {}) or {}).values())
        referenced.update((R.get("corr", {}) or {}).values())

        referenced = {p for p in referenced if isinstance(p, str)}

        unknown = sorted(referenced - calib.keys())
        if unknown:
            raise ValueError(
                "Config references parameter(s) not declared in `calibration.parameters`: "
                + ", ".join(unknown)
            )

        named_locals = sorted(referenced & {sym.name for sym in local_subs})
        if named_locals:
            raise ValueError(
                "Shock and measurement blocks reference a parameter by name, so "
                "they cannot name a derived calibration entry: "
                + ", ".join(named_locals)
            )

    @staticmethod
    def _check_deprecated(data: dict[str, Any]) -> None:
        for key in _DEPRECATED_TOP_LEVEL_KEYS:
            if key in data:
                warnings.warn(
                    f"The key '{key}' is deprecated and will be ignored. Please refer to the documentation (Model Configuration Guide) for the current configuration format.",
                    FutureWarning,
                    stacklevel=_caller_stacklevel(),
                )

        if isinstance(data.get("equations"), dict):
            if isinstance(data["equations"].get("model", {}), list):
                raise NotImplementedError(
                    "Equations as a list have been deprecated and removed. Please use a mapping/dictionary format. "
                    "Refer to the documentation (Model Configuration Guide) for the updated format."
                )
