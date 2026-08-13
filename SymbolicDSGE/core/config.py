from dataclasses import dataclass, asdict
from typing import Any, TypeAlias, TypeVar, Dict
from sympy import Symbol, Function, Eq, Expr, And, Or, Not
from sympy.core.relational import Relational
from numpy import float64
import pickle

from .linearization import LinearizationMethod

K = TypeVar("K", bound=Symbol)
F = TypeVar("F", bound=Function)
V = TypeVar("V")


class SymbolGetterDict(Dict[K, V]):
    def __init__(self, inp: Any) -> None:
        super().__init__(inp)

    def __getitem__(self, key: str | Symbol) -> Any:
        if isinstance(key, str):
            key = Symbol(key)
        return super().__getitem__(key)  # pyright: ignore


class PairGetterDict(Dict[frozenset[Symbol], V]):
    def __init__(self, inp: Any) -> None:
        super().__init__(inp)

    def __getitem__(
        self, key: frozenset[Symbol] | tuple[Symbol, Symbol] | tuple[str, str]
    ) -> Any:

        if isinstance(key, tuple):
            fmt_key = frozenset(Symbol(k) if isinstance(k, str) else k for k in key)
        else:
            fmt_key = key
        return super().__getitem__(fmt_key)


class FunctionGetterDict(Dict[F, V]):
    def __init__(self, inp: Any) -> None:
        super().__init__(inp)

    def __getitem__(self, key: str | F) -> Any:
        if isinstance(key, str):
            fmt_key = Function(key)
        else:
            fmt_key = key
        return super().__getitem__(fmt_key)  # pyright: ignore

    def get(self, key: str | F, default: Any = None) -> Any:
        if isinstance(key, str):
            fmt_key = Function(key)
        else:
            fmt_key = key
        return super().get(fmt_key, default)  # pyright: ignore


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
    observable: SymbolGetterDict[Symbol, Expr]
    obs_is_affine: SymbolGetterDict[Symbol, bool]


@dataclass
class Calib(Base):
    parameters: SymbolGetterDict[Symbol, float64]
    shock_std: SymbolGetterDict[Symbol, Symbol]
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
    ss_seed: FunctionGetterDict[Function, Expr | None]
    linearization: FunctionGetterDict[Function, LinearizationMethod]


@dataclass
class ModelConfig(Base):
    name: str
    variables: Variables
    parameters: list[Symbol]
    shock_map: dict[Symbol, Symbol]
    observables: list[Symbol]
    equations: Equations
    calibration: Calib
    symbolically_linearized: bool = False

    #: Source YAML text the config was parsed from, retained so a model can be
    #: round-tripped into a ``.sdsge`` bundle without re-reading from disk
    #: (avoiding the staleness window between solve and save). ``None`` for
    #: programmatic construction.
    source_yaml: str | None = None
