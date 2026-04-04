from dataclasses import dataclass, asdict
from typing import Any, TypeVar, Dict
from sympy import Symbol, Function, Eq, Expr, Matrix
from sympy.core.relational import Relational
from numpy import float64
import pickle

from ..linearization.linearizer import LinearizationMethod

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
class Variables(Base):
    variables: list[Function]
    steady_state: FunctionGetterDict[Function, Expr | None]
    linearization: FunctionGetterDict[Function, LinearizationMethod]


@dataclass
class ModelConfig(Base):
    name: str
    variables: Variables
    constrained: dict[Function, bool]
    parameters: list[Symbol]
    shock_map: dict[Symbol, Symbol]
    observables: list[Symbol]
    equations: Equations
    calibration: Calib
