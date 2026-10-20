from __future__ import annotations

from typing import TypeAlias

from dataclasses import dataclass
from enum import StrEnum

import sympy as sp
from sympy import Expr
from sympy.core.function import FunctionClass, UndefinedFunction

VAR_FUNC: TypeAlias = FunctionClass | UndefinedFunction


class LinearizationMethod(StrEnum):
    """Linearization methods for transforming variables in a symbolic model.

    Attributes
    ----------
    LOG : Literal["log"]
        Log-linearization
    TAYLOR : Literal["taylor"]
        First-order Taylor expansion linearization
    NONE : Literal["none"]
        No linearization; the variable is used as-is in the linearized model.

    """

    LOG = "log"
    TAYLOR = "taylor"
    NONE = "none"


@dataclass(frozen=True)
class VariableTransformSpec:
    """Specification of a variable transformation for symbolic linearization.

    Attributes
    ----------
    original : VAR_FUNC
        Original variable function (e.g., a sympy.Function) before transformation.
    transformed : VAR_FUNC
        Transformed variable function after applying the specified linearization method.
    method : LinearizationMethod
        Method of linearization applied to the variable (e.g., log-linearization, Taylor expansion, or none).
    steady_state : Expr | None
        Steady state value of the variable, required for certain linearization methods (e.g., log-linearization).

    """

    original: VAR_FUNC
    transformed: VAR_FUNC
    method: LinearizationMethod
    steady_state: Expr | None

    def reconstruct(self, transformed_call: Expr) -> Expr:
        """Reconstruct the original variable expression from the transformed variable expression.

        Parameters
        ----------
        transformed_call : Expr
            Transformed variable expression (e.g., the result of applying the linearization method).

        Returns
        -------
        Expr
            Original variable expression reconstructed from the transformed expression,
            using the specified linearization method and steady state.

        """
        if self.method == LinearizationMethod.NONE:
            return transformed_call

        if self.steady_state is None:
            raise ValueError(
                f"Variable '{self.original}' is missing a steady state required for {self.method.value} linearization."
            )

        if self.method == LinearizationMethod.LOG:
            return self.steady_state * sp.exp(transformed_call)

        if self.method == LinearizationMethod.TAYLOR:
            return self.steady_state + transformed_call

        raise ValueError(f"Unsupported linearization method '{self.method}'.")
