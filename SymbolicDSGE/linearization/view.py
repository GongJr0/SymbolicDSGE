from __future__ import annotations
from dataclasses import dataclass
from sympy import Expr
from ..core.config import SymbolGetterDict


@dataclass(frozen=True)
class PointLinearView:
    model: dict[str, Expr]
    observables: dict[str, Expr]
    parameters: SymbolGetterDict[float]
    ss: dict[str, float]
