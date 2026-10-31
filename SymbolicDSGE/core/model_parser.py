"""Parser for model configurations from YAML into :class:`ModelConfig`."""

from __future__ import annotations

import io
import re
import sys
from itertools import combinations
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Mapping, TypeAlias, Sequence
import warnings

from sympy.core.basic import Basic
from sympy.core.symbol import AppliedUndef
import yaml
from yaml.constructor import ConstructorError
import sympy as sp
from sympy import Symbol, Function, Eq, Expr
from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Or, Not
from sympy.parsing.sympy_parser import (
    standard_transformations,
    convert_xor,
    factorial_notation,
)
from numpy import float64
from .types import (
    LockableDict,
    SymbolGetterDict,
    FunctionGetterDict,
    PairGetterDict,
    RegimeGetterDict,
    Regime,
)
from .config import (
    CovarianceSpec,
    ModelConfig,
    Constraint,
    Equations,
    Calib,
)


def parse_config(path: str | Path) -> ModelConfig:
    """Parse a model configuration from YAML into a :class:`ModelConfig` object.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    ModelConfig
        Configuration object holding symbolic model equations, variables, parameters, shocks, and calibration.

    """
    return ModelParser(path).get()


_GLOBAL_TRANSFORMATIONS = tuple(
    t for t in standard_transformations if t is not factorial_notation
) + (convert_xor,)

# fmt: off
_GLOBALS = {
    # User symbols
    "Symbol":   sp.Symbol,

    # Numbers
    "Integer":  sp.Integer,
    "Float":    sp.Float,
    "Rational": sp.Rational,

    # Variables: fn(time)
    "Function": sp.Function,

    # Printer supported operators
    "exp":      sp.exp,
    "log":      sp.log,
    "sqrt":     sp.sqrt,

    # Relationals for conditions
    "Lt":       sp.Lt, "Le": sp.Le,
    "Gt":       sp.Gt, "Ge": sp.Ge,

    # Logicals for conditions
    "And":      sp.And,
    "Or":       sp.Or,
    "Not":      sp.Not,

    # block builtins. `eval=True` injects it to globals.
    # Changes nothing for the allowlist, having the key here
    # prevents the dict from mutating mid-parse.
    "__builtins__": {},
}
# fmt: on

#: The only fields permitted at the top level of a model config. Any other key
#: is rejected at parse time (see ``_validate_schema``).

# "parameters": list of param names, re-declared in the calibration.parameters mapping
# "observables": list of observable names, re-declared in the equations.observables mapping
# "kalman": Used to store many kalman filter knobs that are now arguments.
# R is the only remaining symbol map and it's folded into the model config.
_DEPRECATED_TOP_LEVEL_KEYS = frozenset({"parameters", "observables", "kalman"})
_ALLOWED_TOP_LEVEL_KEYS = (
    frozenset(
        {
            "name",
            "variables",
            "ss_seed",
            "shocks",
            "equations",
            "calibration",
        }
    )
    | _DEPRECATED_TOP_LEVEL_KEYS
)

#: Allowed sub-keys for the nested config blocks
_ALLOWED_EQUATION_KEYS = frozenset({"model", "constraints", "regimes", "observables"})
_ALLOWED_CONSTRAINT_KEYS = frozenset({"bind", "relax"})

#: Sympy types accepted as a regime entry/exit condition.
_REGIME_SHIFT_CONDITIONAL: TypeAlias = Relational | And | Or | Not

_ALLOWED_CALIBRATION_KEYS = frozenset({"parameters", "shocks", "measurement_noise"})
_ALLOWED_COV_KEYS = frozenset({"std", "corr"})


def _caller_stacklevel() -> int:
    """``warnings.warn`` stacklevel of the first frame outside this module.

    Entry depth varies with which constructor was used, so the frames are
    walked rather than counted.
    """
    frame: FrameType | None = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename == __file__:
        frame = frame.f_back
        level += 1
    return level


def _parse_in(text: str, _LOCALS: dict[str, Any]) -> Basic:
    """Parse ``text`` against the config namespace and safe globals."""
    return sp.parse_expr(
        text,
        local_dict=_LOCALS,
        global_dict=_GLOBALS,
        evaluate=True,
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


class ModelParser:
    """Parse a model configuration from YAML into a :class:`ModelConfig` object.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Attributes
    ----------
    config_path : str | Path
        Path the configuration was read from, or the source name when it was
        built from text.
    """

    def __init__(self, config_path: str | Path) -> None:
        path = Path(config_path)
        self._build(path.read_text(encoding="utf-8"), str(path), path)

    def _build(self, text: str, name: str, config_path: str | Path) -> None:
        self.config_path = config_path
        self._text = text
        self._name = name
        self.config = self.from_yaml()
        self.__post_init__()

    def __post_init__(self) -> None:
        conf = self.config
        self.validate_equations(conf)
        self.validate_constraints(conf)
        self.validate_regimes(conf)  # reads validated bind conditions
        self.validate_ss_seed(conf)

    def get(self) -> ModelConfig:
        """Return the parsed model config.

        Returns
        -------
        ModelConfig
            Configuration object holding symbolic model equations, variables, parameters, shocks, and calibration.

        """
        return self.config

    @classmethod
    def from_string(cls, text: str, name: str = "<yaml string>") -> "ModelParser":
        """Construct a parser from YAML *text* (e.g. a bundle config member).

        ``name`` is what a parse error names as its source, so a caller holding
        one (a bundle member, say) should pass it.
        """
        parser = cls.__new__(cls)
        parser._build(text, name, name)
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

        declared = set(conf.calibration.parameters) | set(conf.shocks)
        return (var_funcs - set(conf.variables)) | ((free - time_syms) - declared)

    @classmethod
    def validate_equations(cls, conf: ModelConfig) -> None:
        """Validate model and observable equations for unknown or unpermitted symbols.

        Model equations are checked for unknown symbols only.
        Observables extend the check to reject references to shocks.
        Finally, the specification is checked for a one-to-one mapping between
        variables and model equations.
        """
        for name, eq in conf.equations.model.items():
            unknown_atoms = cls._unknown_atoms(conf, eq)
            raise_if_unk("Equation", name, unknown_atoms)

        for name, eq in conf.equations.observables.items():
            unknown_atoms = cls._unknown_atoms(conf, eq)
            raise_if_unk("Observable", name, unknown_atoms)

            shock_refs = cls._shock_atoms(conf, eq)
            if shock_refs:
                raise ValueError(
                    f"Observable {str(name)!r} references shock(s) "
                    f"{set_sort(shock_refs)!r}; observables may only "
                    f"reference model variables and parameters."
                )

        defining = list(conf.equations.model.values())
        for replacements in (conf.equations.regimes or {}).values():
            defining.extend(replacements.values())
        used = {call.func for eq in defining for call in eq.atoms(AppliedUndef)}

        absent = [v for v in conf.variables if v not in used]
        if absent:
            raise ValueError(
                f"Variable(s) {sorted(str(v) for v in absent)} occur in no model "
                "equation."
            )

        n_var, n_eq = len(conf.variables), len(conf.equations.model)
        if n_var != n_eq:
            raise ValueError(
                f"The model block has {n_eq} equation(s) for {n_var} variable(s); "
                f"each variable needs exactly one equation."
            )

    @classmethod
    def validate_constraints(cls, conf: ModelConfig) -> None:
        """Validate the constraints specified on a model configuration.

        Parameters
        ----------
        conf : ModelConfig
            The configuration object containing the model equations and constraints.

        """
        if not conf.equations.constraints:
            return

        constraints: Mapping[str, Constraint] = conf.equations.constraints

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
            raise_if_unk("Constraint", name, unknown_atoms)

    @classmethod
    def validate_regimes(cls, conf: ModelConfig) -> None:
        """Validate the regimes specified on a model configuration.

        Parameters
        ----------
        conf : ModelConfig
            The configuration object containing the model equations and regimes.

        """
        constraints = conf.equations.constraints
        regimes = conf.equations.regimes
        if not constraints and not regimes:
            return
        if not constraints or not regimes:
            raise ValueError(
                "equations.constraints and equations.regimes must be declared together; "
                f"got constraints={'set' if constraints else 'empty'}, "
                f"regimes={'set' if regimes else 'empty'}"
            )

        declared = set(constraints)
        unknown_members = {n for key in regimes for n in key} - declared
        if unknown_members:
            raise ValueError(
                f"Regime keys name undeclared constraints: {set_sort(unknown_members)}; "
                f"declared: {set_sort(declared)}"
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
                "equations.regimes is missing entries for binding combinations: "
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
        constraints: Mapping[str, Constraint],
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
                    f"references unknown symbol(s): {set_sort(unknown_atoms)!r}"
                )

    @classmethod
    def validate_ss_seed(cls, conf: ModelConfig) -> None:
        """Validate the steady-state seed expressions for model variables.

        Parameters
        ----------
        conf : ModelConfig
            The configuration object containing the model variables and their steady-state seed expressions.

        """
        params = set(conf.calibration.parameters)
        declared = set(conf.variables)
        for var, expr in conf.ss_seed.items():
            if expr is None:
                continue
            applied = expr.atoms(AppliedUndef)
            # An undeclared call is an unsupported function, not a variable: it
            # arrives applied because the parse namespace admits no other.
            unknown_calls = {call.func for call in applied} - declared
            raise_if_unk("ss_seed for variable", var.__name__, unknown_calls)
            if applied:
                raise ValueError(
                    f"ss_seed for variable '{var.__name__}' must resolve to a number "
                    f"but references model variable(s): {set_sort(applied)!r}"
                )
            unknown = expr.free_symbols - params
            raise_if_unk("ss_seed for variable", var.__name__, unknown)

    def from_yaml(self) -> ModelConfig:
        """Parse a YAML configuration file into a :class:`ModelConfig` object.

        Returns
        -------
        tuple[dict, ModelConfig]
            Object containing model configuration. The first element is the raw YAML data as a dictionary,
            and the second element is the parsed configuration.

        """
        data = _load_yaml(self._text, self._name)
        _validate_schema(data)

        ns = _build_namespace(data)
        (
            _LOCALS,
            ordered_var_names,
            variable_funcs,
            params,
            observables,
            shocks,
        ) = ns

        # Locals resolve before any field is parsed, so the helpers below can
        # eliminate them from everything they build.
        parameters, local_subs = _resolve_calibration_locals(data, _LOCALS)
        _require_calibrated_params(data, local_subs)
        params = [param for param in params if param not in local_subs]

        _get_expr, _get_relational, _get_eq = _sympy_parsers(_LOCALS, local_subs)

        variables, ss_seed = _parse_variables(
            data, ordered_var_names, variable_funcs, _get_expr
        )

        equations = _parse_equations(
            data,
            _LOCALS,
            observables,
            ordered_var_names,
            _get_eq,
            _get_relational,
            _get_expr,
        )

        shock_map = _parse_shock_calibration(data, _LOCALS, shocks)
        measurement_noise = _parse_measurement_noise(data, observables, parameters)
        calibration = Calib(
            parameters=parameters,
            shocks=shock_map,
            measurement_noise=measurement_noise,
        )

        mdl_cfg = ModelConfig(
            name=data.get("name", "Unnamed"),
            variables=variables,
            ss_seed=ss_seed,
            shocks=shocks,
            observables=observables,
            equations=equations,
            calibration=calibration,
        )

        return mdl_cfg


# ---------------- helpers ----------------


class _StrictLoader(yaml.SafeLoader):
    """``SafeLoader`` that rejects a mapping with a repeated key.

    A repeat is legal YAML and resolves to the last value, so a parameter or an
    equation written twice would otherwise resolve to the last one.
    """

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        seen: set[Any] = set()
        duplicates: list[Any] = []
        mark = node.start_mark
        for key_node, _ in node.value:
            # Merge keys are removed by ``flatten_mapping`` below and have no
            # constructor of their own, so they are never built here.
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen and key not in duplicates:
                duplicates.append(key)
                if len(duplicates) == 1:
                    mark = key_node.start_mark
            seen.add(key)
        if duplicates:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key(s): {sorted(str(k) for k in duplicates)}",
                mark,
            )
        return super().construct_mapping(node, deep=deep)


class _NamedStringIO(io.StringIO):
    """``StringIO`` that reports a name, which YAML marks cite as the source."""

    def __init__(self, text: str, name: str) -> None:
        super().__init__(text)
        self.name = name


def _load_yaml(text: str, name: str) -> dict[str, Any]:
    # A fresh buffer per call: PyYAML reads the stream to EOF, and `name` is
    # what puts the source in every mark it raises.
    data = yaml.load(_NamedStringIO(text, name), Loader=_StrictLoader)
    if not isinstance(data, dict):
        raise TypeError("YAML root must be a mapping/dict.")
    return data


def _reject_unknown_keys(mapping: Any, allowed: frozenset[str], where: str) -> None:
    if not isinstance(mapping, dict):
        return
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown field(s) under '{where}': {unknown}. "
            f"Allowed: {sorted(allowed)}."
        )


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(
            f"'{where}' must be a mapping, got {type(value).__name__}: {value!r}"
        )
    return value


def _validate_schema(data: dict[str, Any]) -> None:
    _check_deprecated(data)
    _reject_unknown_keys(data, _ALLOWED_TOP_LEVEL_KEYS, "<root>")
    _reject_unknown_keys(
        (eq := data.get("equations")), _ALLOWED_EQUATION_KEYS, "equations"
    )
    if isinstance(eq, dict):
        for name, spec in (eq.get("constraints") or {}).items():
            _reject_unknown_keys(
                spec,
                _ALLOWED_CONSTRAINT_KEYS,
                f"equations.constraints.{name}",
            )
    calib = data.get("calibration")
    _reject_unknown_keys(calib, _ALLOWED_CALIBRATION_KEYS, "calibration")
    if isinstance(calib, dict):
        _reject_unknown_keys(
            calib.get("shocks"), _ALLOWED_COV_KEYS, "calibration.shocks"
        )
        _reject_unknown_keys(
            calib.get("measurement_noise"),
            _ALLOWED_COV_KEYS,
            "calibration.measurement_noise",
        )


def _build_namespace(
    data: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[str],
    list[Function],
    list[Symbol],
    list[Symbol],
    list[Symbol],
]:
    ordered_var_names = _get_variables(data)

    t = sp.symbols("t", integer=True)

    _raise_if_not_unique(ordered_var_names, "variables must be unique.")
    _raise_if_not_unique(data.get("shocks", []), "shocks must be unique.")

    def nested_keys(mapping: dict[str, Any], l1: str, l2: str) -> list[str]:
        return list(mapping.get(l1, {}).get(l2, {}).keys())

    # Params and observables come from mappings.
    # The yaml loader is modified to reject duplicate keys already.
    params: list[Symbol] = [
        sp.Symbol(name) for name in nested_keys(data, "calibration", "parameters")
    ]
    observables: list[Symbol] = [
        sp.Symbol(name) for name in nested_keys(data, "equations", "observables")
    ]
    variables: list[Function] = list(
        map(Function, ordered_var_names)
    )  # pyright: ignore
    shocks: list[Symbol] = [sp.Symbol(name) for name in data.get("shocks", [])]

    _LOCALS: dict[str, Any] = {
        "t": t,
        **{var.name: var for var in variables},  # pyright: ignore
        **{param.name: param for param in params},
        **{shock.name: shock for shock in shocks},
        **{obs.name: obs for obs in observables},
    }
    return (
        _LOCALS,
        ordered_var_names,
        variables,
        params,
        observables,
        shocks,
    )


def _get_variables(data: dict[str, Any]) -> list[str]:
    variables = data.get("variables")
    if variables is None:
        raise ValueError("Missing required 'variables' field in the configuration.")
    if not isinstance(variables, list):
        raise TypeError(f"'variables' must be a list, got {type(variables).__name__}")
    if not all(isinstance(v, str) for v in variables):
        raise TypeError("'variables' must be a list of strings")
    return variables


def _get_ss_seed(data: dict[str, Any]) -> dict[str, Any]:
    ss_seed = data.get("ss_seed")
    if ss_seed is None:
        return {}
    _require_mapping(ss_seed, "ss_seed")
    _reject_unknown_keys(ss_seed, frozenset(data["variables"]), "ss_seed")
    return dict(ss_seed)


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


def _parse_equations(
    data: dict[str, Any],
    _LOCALS: dict[str, Any],
    observables: list[Symbol],
    ordered_var_names: list[str],
    _get_eq: Callable[[str], Eq],
    _get_relational: Callable[[str], _REGIME_SHIFT_CONDITIONAL],
    _get_expr: Callable[[str], Expr],
) -> Equations:
    eq_data = data["equations"]

    model: dict[str, Eq] = {name: _get_eq(eq) for name, eq in eq_data["model"].items()}

    constraints_raw = eq_data.get("constraints", {}) or {}
    if len(constraints_raw) > 2:
        raise NotImplementedError(
            "OBCs are solved via OccBin of Guirreiri and Iacoviello (2015), which explicitly supports one or two constraints. "
            "Use at most two constraints in your model configuration."
        )

    constraints: dict[str, Constraint] = {}
    for name, raw_spec in constraints_raw.items():
        spec = _require_mapping(raw_spec, f"equations.constraints.{name}")
        if (bind := spec.get("bind")) is None:
            raise ValueError(f"Constraint '{name}' is missing a 'bind' condition.")
        bind = _get_relational(bind)
        if (relax := spec.get("relax")) is None:
            relax = Not(bind)  # Default relax for symmetric constraint
        else:
            relax = _get_relational(relax)

        constraints[name] = Constraint(
            bind=bind,
            relax=relax,
        )

    regimes_raw = eq_data.get("regimes", {}) or {}

    regimes: dict[str, Regime] = {}
    for raw_key, v in regimes_raw.items():
        if raw_key in regimes:
            raise ValueError(
                f"Duplicate regime key '{raw_key}' encountered "
                "in equations.regimes. Each regime key must be unique."
            )
        regimes[raw_key] = LockableDict(
            {
                name: _get_eq(eq)
                for name, eq in _require_mapping(
                    v, f"equations.regimes.{raw_key}"
                ).items()
            }
        )

    observables_raw = eq_data.get("observables", {}) or {}
    observables_eq: dict[Symbol, Expr] = {
        obs: _get_expr(observables_raw[obs.name]) for obs in observables
    }

    is_affine = _derive_observable_structure(
        observables_eq=observables_eq,
        ordered_var_names=ordered_var_names,
        _LOCALS=_LOCALS,
    )

    return Equations(
        model=LockableDict(model),
        constraints=LockableDict(constraints) if constraints else None,
        regimes=RegimeGetterDict(regimes) if regimes else None,
        observables=SymbolGetterDict(observables_eq),
        obs_is_affine=SymbolGetterDict(is_affine),
    )


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
        grads = [expr_symbolized.diff(s) for s in state_syms]  # pyright: ignore
        if all((g.free_symbols & state_set) == set() for g in grads):
            is_affine[obs] = True

    return is_affine


def _parse_variables(
    data: dict[str, Any],
    ordered_var_names: list[str],
    variable_funcs: list[Function],
    _get_expr: Callable[[str], Expr],
) -> tuple[list[Function], FunctionGetterDict[Expr | None]]:
    ss_seed_raw = _get_ss_seed(data)

    ss_seed: dict[Function, Expr | None] = {}
    for var_name, var_func in zip(ordered_var_names, variable_funcs):
        ss_raw = ss_seed_raw.get(var_name)
        if ss_raw is None:
            ss_seed[var_func] = None
        else:
            ss_seed[var_func] = _get_expr(str(ss_raw))

    return (variable_funcs, FunctionGetterDict(ss_seed))


def _resolve_calibration_locals(
    data: dict[str, Any], _LOCALS: dict[str, Any]
) -> tuple[SymbolGetterDict[float], dict[Symbol, Expr]]:
    """Split ``calibration.parameters`` into values and derived locals.

    An entry carrying free symbols names a formula over other parameters
    rather than a value, in the manner of a Dynare ``#`` definition. Such
    an entry never reaches the model: the returned map rewrites it away
    wherever it is referenced, leaving the base parameters behind it, so
    estimating those base parameters moves the formula with them.
    """
    calib = data.get("calibration", {}).get("parameters", {}) or {}
    param_syms = {_LOCALS[name] for name in calib}
    declared_vars = set(_get_variables(data))

    values: dict[Symbol, float64] = {}
    local_subs: dict[Symbol, Expr] = {}
    for name, raw in calib.items():
        expr = _parse_in(str(raw), _LOCALS)
        if not isinstance(expr, Expr):
            raise TypeError(
                f"Calibration entry '{name}' is not a valid SymPy Expr: {raw!r}"
            )
        applied = expr.atoms(AppliedUndef)
        # See validate_ss_seed: an undeclared call is an unsupported function.
        unknown_calls = {call.func.__name__ for call in applied} - declared_vars
        raise_if_unk("Calibration entry", name, unknown_calls)

        if applied:
            raise ValueError(
                f"Calibration entry '{name}' references model variable(s): "
                f"{set_sort(applied)!r}"
            )
        if unknown := expr.free_symbols - param_syms:
            raise ValueError(
                f"Calibration entry '{name}' references undeclared "
                f"parameter(s): {set_sort(unknown)!r}"
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
        local_subs = {
            s: e.xreplace(local_subs) for s, e in local_subs.items()
        }  # pyright: ignore

    if cyclic := {s for s, e in local_subs.items() if e.free_symbols & derived}:
        raise ValueError(
            "Calibration entries reference each other in a cycle: "
            f"{set_sort(cyclic)!r}"
        )

    return SymbolGetterDict(values), local_subs


def _parse_shock_calibration(
    data: dict[str, Any],
    _LOCALS: dict[str, Any],
    shock_syms: list[Symbol],
) -> CovarianceSpec:
    shocks = data.get("calibration", {}).get("shocks", {}) or {}
    std_map = shocks.get("std", {}) or {}
    corr_map = shocks.get("corr", {}) or {}

    # std: map shock symbol -> parameter Symbol (or None)
    if missing := {shock.name for shock in shock_syms} - set(std_map):
        raise ValueError(
            "Missing standard deviation parameter(s) in calibration.shocks.std "
            f"for shock(s): {set_sort(missing)!r}"
        )
    shock_std: dict[Symbol, str] = {shock: std_map[shock.name] for shock in shock_syms}

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
        shock_corr[frozenset((a, b))] = param_name

    # fill missing with None (all unordered pairs i<j)
    for i in range(len(shock_syms)):
        for j in range(i + 1, len(shock_syms)):
            key = frozenset((shock_syms[i], shock_syms[j]))
            shock_corr.setdefault(key, None)
    _raise_if_not_unique(
        list(shock_corr.values()), "calibration.shocks.corr values must be unique."
    )

    cross_vals = list(set(shock_std.values())) + list(set(shock_corr.values()))
    _raise_if_not_unique(
        cross_vals,
        "calibration.shocks cannot share parameter names between std and corr.",
    )

    return CovarianceSpec(
        SymbolGetterDict(shock_std),
        PairGetterDict(shock_corr),
    )


def _parse_measurement_noise(
    data: dict[str, Any],
    observables: list[Symbol],
    parameters: SymbolGetterDict[float],
) -> CovarianceSpec | None:
    mcov_data = data.get("calibration", {}).get("measurement_noise")
    if not mcov_data:
        return None

    std_param_map: SymbolGetterDict[str] | None
    corr_param_map: PairGetterDict[str | None]

    std_map = mcov_data.get("std", {}) or {}
    corr_map = mcov_data.get("corr", {}) or {}

    obs_syms = {obs.name: obs for obs in observables}

    def _obs(name: str, field: str) -> Symbol:
        if (obs := obs_syms.get(name)) is None:
            raise ValueError(
                f"calibration.measurement_noise.{field} names '{name}', which is not an observable. "
                "Declare it under equations.observables or remove the entry."
            )
        return obs

    std_param_map = SymbolGetterDict(
        {_obs(obs_name, "std"): param_name for obs_name, param_name in std_map.items()}
    )
    if missing := set(observables) - set(std_param_map):
        raise ValueError(
            "Missing standard deviation parameter(s) in calibration.measurement_noise.std: "
            f"{set_sort(missing)!r}"
        )

    corr_param_map = PairGetterDict({})

    for pair_str, param_name in corr_map.items():
        names = [x.strip() for x in pair_str.split(",")]
        if len(names) != 2:
            raise ValueError(
                f"Correlation pair must contain exactly two observables: {pair_str!r}"
            )
        a = _obs(names[0], "corr")
        b = _obs(names[1], "corr")
        corr_param_map[frozenset((a, b))] = param_name

    for a, b in combinations(observables, 2):
        corr_param_map.setdefault(frozenset((a, b)), None)
    _raise_if_not_unique(
        list(corr_param_map.values()),
        "calibration.measurement_noise.corr values must be unique.",
    )

    cross_vals = list(set(std_param_map.values())) + list(set(corr_param_map.values()))
    _raise_if_not_unique(
        cross_vals,
        "calibration.measurement_noise cannot share parameter names between std and corr.",
    )

    return CovarianceSpec(std_param_map, corr_param_map)


def _require_calibrated_params(
    data: dict[str, Any], local_subs: dict[Symbol, Expr]
) -> None:
    """Check the blocks that reference a parameter by name.

    Unlike an equation, these carry a bare name rather than an expression,
    so the name has to be one a value can be read from: a derived entry
    resolves to a formula and leaves nothing to build a covariance out of.
    """
    calib = data.get("calibration", {})
    params = calib.get("parameters", {})

    referenced: set[str] = set()

    shocks = calib.get("shocks", {}) or {}
    referenced.update((shocks.get("std", {}) or {}).values())
    referenced.update((shocks.get("corr", {}) or {}).values())

    meas_noise = calib.get("measurement_noise", {}) or {}
    referenced.update((meas_noise.get("std", {}) or {}).values())
    referenced.update((meas_noise.get("corr", {}) or {}).values())

    referenced = {p for p in referenced if isinstance(p, str)}

    unknown = referenced - params.keys()
    if unknown:
        raise ValueError(
            "Config references parameter(s) not declared in "
            f"`calibration.parameters`: {set_sort(unknown)!r}"
        )

    named_locals = referenced & {sym.name for sym in local_subs}
    if named_locals:
        raise ValueError(
            "Shock and measurement cannot reference derived calibration entries: "
            f"{set_sort(named_locals)!r}"
        )


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


def _raise_if_not_unique(values: Sequence[Any], preamble: str) -> None:
    """Raise if any value appears more than once in the sequence."""
    unique = set(values)
    if len(unique) != len(values):
        duplicates = sorted(
            [v for v in unique if values.count(v) > 1 and (v is not None)]
        )
        if duplicates:
            raise ValueError(f"{preamble}" f"\nDuplicate entries: {duplicates}.")


def set_sort(s: set[Any]) -> list[Any]:
    """Return a sorted list of the set's elements."""
    return sorted(map(str, s))


def raise_if_unk(place: str, name: str | Symbol, unk: set[Any]) -> None:
    if unk:
        raise ValueError(
            f"{place} {str(name)!r} references unknown symbol(s): {set_sort(unk)}"
        )
