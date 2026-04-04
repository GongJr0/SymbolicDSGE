import sympy as sp

from numpy import ndarray, float64
from numpy.typing import NDArray
from sympy import Function, Symbol, Expr, Eq
from sympy.core.function import UndefinedFunction

from typing import TypeAlias
from enum import StrEnum

_VAR_FUNC: TypeAlias = Function | UndefinedFunction
NDF: TypeAlias = NDArray[float64]


class LinearizationMethod(StrEnum):
    LOG = "log"
    TAYLOR = "taylor"
    NONE = "none"


class Linearizer:
    def __init__(
        self,
        method_dict: dict[_VAR_FUNC, str | LinearizationMethod],
        steady_state: dict[_VAR_FUNC, Expr | float64 | None],
        equations: list[Eq],
    ):
        self._method_dict = self._parse_methods(method_dict)
        self._steady_state = {
            varname: None if val is None else sp.sympify(val)
            for varname, val in steady_state.items()
        }
        self._equations = equations
        self._residuals: list[Expr] = [
            eq.lhs - eq.rhs for eq in equations  # pyright: ignore
        ]

    def _get_perturbations(self) -> dict[_VAR_FUNC, Expr]:
        perturbations = {}
        for var, method in self.method_dict.items():
            if method == LinearizationMethod.NONE:
                perturbations[var] = var
                continue

            ss = self.steady_state[var]
            if ss is None:
                raise ValueError(
                    f"Variable '{var}' is missing a steady state required for {method.value} linearization."
                )
            if method == LinearizationMethod.LOG:
                perturbations[var] = ss + sp.exp(var)
            elif method == LinearizationMethod.TAYLOR:
                perturbations[var] = ss + var
        return perturbations

    @staticmethod
    def _parse_methods(
        method_dict: dict[_VAR_FUNC, str | LinearizationMethod],
    ) -> dict[_VAR_FUNC, LinearizationMethod]:
        parsed_methods: dict[_VAR_FUNC, LinearizationMethod] = {}
        for var, method in method_dict.items():
            if isinstance(method, LinearizationMethod):
                parsed_methods[var] = method
                continue

            normalized = method.strip().lower()
            try:
                parsed_methods[var] = LinearizationMethod(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid method '{method}' for variable '{var}'. Valid methods are: {[member.value for member in LinearizationMethod]}"
                ) from exc
        return parsed_methods

    @property
    def method_dict(self) -> dict[_VAR_FUNC, LinearizationMethod]:
        return self._method_dict

    @property
    def steady_state(self) -> dict[_VAR_FUNC, Expr | None]:
        return self._steady_state

    @property
    def equations(self) -> list[Eq]:
        return self._equations

    @property
    def residuals(self) -> list[Expr]:
        return self._residuals
