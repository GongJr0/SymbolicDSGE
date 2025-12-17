from dataclasses import dataclass, asdict
from typing import Any
from sympy import Symbol, Function, Eq, Expr  # type: ignore
from sympy.core.relational import Relational  # type: ignore
from numpy import float64
import pickle


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
    constraint: dict[Symbol, Relational]
    observable: dict[Symbol, Expr]


@dataclass
class Calib(Base):
    parameters: dict[Symbol, float64]


@dataclass
class ModelConfig(Base):
    name: str
    variables: list[Function]
    constrained: dict[Function, bool]
    parameters: list[Symbol]
    shocks: list[Symbol]
    observables: list[Symbol]
    equations: Equations
    calibration: Calib
