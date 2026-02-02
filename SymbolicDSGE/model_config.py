from dataclasses import dataclass, asdict
from typing import Any, TypeVar, Dict
from sympy import Symbol, Function, Eq, Expr, Matrix
from sympy.core.relational import Relational
from numpy import float64
import pickle

K = TypeVar("K", bound=Symbol)
V = TypeVar("V")


class SymbolGetterDict(Dict[K, V]):
    def __init__(self, inp: Any) -> None:
        super().__init__(inp)

    def __getitem__(self, key: str | Symbol) -> Any:
        if isinstance(key, str):
            key = Symbol(key)
        return super().__getitem__(key)


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
class Equations(Base):
    model: list[Eq]
    constraint: SymbolGetterDict[Symbol, Relational]
    observable: SymbolGetterDict[Symbol, Expr]
    obs_is_affine: SymbolGetterDict[Symbol, bool]
    obs_jacobian: Matrix


@dataclass
class Calib(Base):
    parameters: SymbolGetterDict[Symbol, float64]
    shock_std: SymbolGetterDict[Symbol, Symbol]
    shock_corr: PairGetterDict[Symbol]


@dataclass
class ModelConfig(Base):
    name: str
    variables: list[Function]
    constrained: dict[Function, bool]
    parameters: list[Symbol]
    shock_map: dict[Symbol, Symbol]
    observables: list[Symbol]
    equations: Equations
    calibration: Calib
