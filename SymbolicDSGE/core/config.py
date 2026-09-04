from dataclasses import dataclass, asdict
from typing import Any, TypeAlias, TypeVar, Dict
from collections import UserDict
from sympy import Symbol, Function, Eq, Expr, And, Or, Not
from sympy.core.relational import Relational
from numpy import float64
import pickle

from .linearization import LinearizationMethod

V = TypeVar("V")


def _symbolify(key: str | Symbol) -> Symbol:
    if isinstance(key, str):
        return Symbol(key)
    return key


def _frozensetify(
    key: frozenset[Symbol] | tuple[Symbol, Symbol] | tuple[str, str],
) -> frozenset[Symbol]:
    if isinstance(key, tuple):
        return frozenset(Symbol(k) if isinstance(k, str) else k for k in key)
    return key


def _functionify(key: str | Function) -> Function:
    if isinstance(key, str):
        return Function(key)  # pyright: ignore
    return key


class SymbolGetterDict(UserDict[Symbol, V]):
    def __init__(self, inp: Any) -> None:
        super().__init__(inp)

    def __getitem__(self, key: str | Symbol) -> Any:
        key = _symbolify(key)
        return self.data[key]

    def __setitem__(self, key: str | Symbol, value: Any) -> None:
        key = _symbolify(key)
        self.data[key] = value

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, str):
            key = Symbol(key)
        return self.data.__contains__(key)

    def __delitem__(self, key: str | Symbol) -> None:
        key = _symbolify(key)
        del self.data[key]


class PairGetterDict(UserDict[frozenset[Symbol], V]):
    def __init__(self, inp: Any) -> None:
        super().__init__(inp)

    def __getitem__(
        self, key: frozenset[Symbol] | tuple[Symbol, Symbol] | tuple[str, str]
    ) -> Any:
        key = _frozensetify(key)
        return self.data[key]

    def __setitem__(
        self,
        key: frozenset[Symbol] | tuple[Symbol, Symbol] | tuple[str, str],
        value: Any,
    ) -> None:
        key = _frozensetify(key)
        self.data[key] = value

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, tuple):
            key = _frozensetify(key)
        return self.data.__contains__(key)

    def __delitem__(
        self, key: frozenset[Symbol] | tuple[Symbol, Symbol] | tuple[str, str]
    ) -> None:
        key = _frozensetify(key)
        del self.data[key]


class FunctionGetterDict(UserDict[Function, V]):
    def __init__(self, inp: Any) -> None:
        super().__init__(inp)

    def __getitem__(self, key: str | Function) -> Any:
        fmt_key = _functionify(key)
        return self.data[fmt_key]

    def __setitem__(self, key: str | Function, value: Any) -> None:
        fmt_key = _functionify(key)
        self.data[fmt_key] = value

    def __contains__(self, key: Any) -> bool:
        key = _functionify(key)
        return self.data.__contains__(key)

    def __delitem__(self, key: str | Function) -> None:
        fmt_key = _functionify(key)
        del self.data[fmt_key]


@dataclass
class Base:
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def serialize(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)


@dataclass
class Constraint(Base):
    bind: Relational | And | Or | Not
    relax: Relational | And | Or | Not


Regime: TypeAlias = Dict[str, Eq]  # {model_equation_name: replacement}


@dataclass
class Equations(Base):
    model: Dict[str, Eq]
    constraint: Dict[str, Constraint] | None  # {constraint_name: Constraint}
    regime: Dict[frozenset[str], Regime] | None  # {binding_set: Regime}
    observable: SymbolGetterDict[Expr]
    obs_is_affine: SymbolGetterDict[bool]


@dataclass
class Calib(Base):
    parameters: SymbolGetterDict[float64]
    shock_std: SymbolGetterDict[Symbol]
    shock_corr: PairGetterDict[Symbol]

    def get_param(self, name: str | Symbol, default: float | None = None) -> float:
        """A calibrated parameter's value, or ``default`` when it is absent.

        Raises :class:`KeyError` with no default, since a missing parameter with
        no fallback is a model-authoring error rather than a zero.
        """
        sym = Symbol(name) if isinstance(name, str) else name
        if sym in self.parameters:
            return float64(self.parameters[sym])
        elif default is not None:
            return float64(default)
        raise KeyError(f"Parameter '{name}' not found in calibration parameters.")

    def get_rho(
        self, var1: str | Symbol, var2: str | Symbol, default: float = 0.0
    ) -> float:
        """The correlation between two shocks, 1.0 for a shock with itself."""
        if var1 == var2:
            return 1.0

        corr = self.shock_corr[var1, var2]  # pyright: ignore # Overloaded __getitem__
        if corr is not None:
            return self.get_param(corr, default=default)

        return float64(default)

    def fingerprint(self) -> int:
        """Hashable snapshot of the parameter values, for keying caches."""
        return hash(
            (
                tuple(self.parameters.keys()),
                tuple(float(v) for v in self.parameters.values()),
            )
        )


@dataclass
class Variables(Base):
    variables: list[Function]
    # None == 0 seed newton.
    ss_seed: FunctionGetterDict[Expr | None]
    linearization: FunctionGetterDict[LinearizationMethod]


@dataclass(repr=False)
class ModelConfig(Base):
    name: str
    variables: Variables
    parameters: list[Symbol]
    #: Innovation symbols. A shock reaches the residual as a bare symbol and
    #: may drive any number of equations, so it names no target variable.
    shocks: list[Symbol]
    observables: list[Symbol]
    equations: Equations
    calibration: Calib
    symbolically_linearized: bool = False

    #: Source YAML text the config was parsed from, retained so a model can be
    #: round-tripped into a ``.sdsge`` bundle without re-reading from disk
    #: (avoiding the staleness window between solve and save). ``None`` for
    #: programmatic construction.
    source_yaml: str | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
