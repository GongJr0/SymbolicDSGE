from dataclasses import dataclass, asdict
from typing import AbstractSet, Any, TypeAlias, TypeVar, Dict, Sequence
from collections import UserDict
from sympy import Symbol, Function, Eq, Expr, And, Or, Not
from sympy.core.relational import Relational
from numpy import float64
import pickle

from .linearization import LinearizationMethod

KT = TypeVar("KT")
V = TypeVar("V")

Regime: TypeAlias = Dict[str, Eq]  # {model_equation_name: replacement}


class _NormalizedKeyDict(UserDict[KT, V]):
    """A mapping whose key accepts more than one spelling.

    A subclass supplies ``_key``, resolving any accepted spelling to the one the
    underlying dict is keyed by. The three primitives below route through it and
    ``UserDict`` builds ``get``, ``pop``, ``setdefault``, ``update`` and the rest
    on top of them, so the coercion reaches the whole mapping API. Subclassing
    ``dict`` instead would leave every method not overridden by hand reading the
    hash table directly, which is silently wrong rather than loud.
    """

    def __init__(self, inp: Any) -> None:
        super().__init__(inp)

    @staticmethod
    def _key(key: Any) -> Any:
        raise NotImplementedError

    def __getitem__(self, key: Any) -> Any:
        return self.data[self._key(key)]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[self._key(key)] = value

    def __delitem__(self, key: Any) -> None:
        del self.data[self._key(key)]

    def __contains__(self, key: Any) -> bool:
        return self._key(key) in self.data


class SymbolGetterDict(_NormalizedKeyDict[Symbol, V]):
    @staticmethod
    def _key(key: Any) -> Any:
        return Symbol(key) if isinstance(key, str) else key


class PairGetterDict(_NormalizedKeyDict[frozenset[Symbol], V]):
    @staticmethod
    def _key(key: Any) -> Any:
        if isinstance(key, (Sequence, AbstractSet)) and not isinstance(key, str):
            return frozenset(Symbol(k) if isinstance(k, str) else k for k in key)
        return key


class FunctionGetterDict(_NormalizedKeyDict[Function, V]):
    @staticmethod
    def _key(key: Any) -> Any:
        return Function(key) if isinstance(key, str) else key


class RegimeGetterDict(_NormalizedKeyDict[frozenset[str], Regime]):
    @staticmethod
    def _key(key: Any) -> Any:
        if isinstance(key, str):
            return frozenset(s.strip() for s in key.split(","))
        if isinstance(key, (Sequence, AbstractSet)):
            return frozenset(str(k) for k in key)
        return key


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


@dataclass
class Equations(Base):
    model: Dict[str, Eq]
    constraint: Dict[str, Constraint] | None  # {constraint_name: Constraint}
    regime: RegimeGetterDict | None  # {binding_set: Regime}
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
