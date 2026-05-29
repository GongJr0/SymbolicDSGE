from .config import HessianMode
from .model_parametrizer import ModelParametrizer

import sympy as sp
from typing import Callable


class HessianValidator:
    """
    Symbolic Regression class with expression candidate selection based on linearity constraints.
    The class respects the restrictions defined in the TemplateConfig and returns both qualified and disqualified expressions.
    """

    def __init__(self, parametrizer: ModelParametrizer) -> None:
        self._parametrizer = parametrizer

    def hessian_compliant(self, expr: sp.Expr) -> bool:
        """
        Check if the expression satisfies the Hessian-based linearity constraints defined in the TemplateConfig.

        :param expr: The expression to check.
        :type expr: sp.Expr
        :return: True if the expression satisfies the Hessian constraints, False otherwise.
        :rtype: bool
        """
        check_fn = self.hessian_check
        return check_fn(expr)

    @staticmethod
    def _hessian(expr: sp.Expr) -> sp.Matrix:
        var_syms = [
            sym for sym in expr.free_symbols if not isinstance(sym, sp.Dummy)
        ]  # Excludes Dummy variables (parametrized constants are represented as Dummies)
        if len(var_syms) == 0:
            return sp.Matrix([[0]])  # Hessian of a constant is zero
        return sp.hessian(expr, var_syms)

    @staticmethod
    def _hessian_full(expr: sp.Expr) -> bool:
        hessian = HessianValidator._hessian(expr)
        out: bool = hessian.is_zero_matrix
        return out if out else False

    @staticmethod
    def _hessian_diag(expr: sp.Expr) -> bool:
        hessian = HessianValidator._hessian(expr)
        diag = hessian.diagonal()
        out: bool = diag.is_zero_matrix
        return out if out else False

    @staticmethod
    def _hessian_free(expr: sp.Expr) -> bool:
        return True

    @property
    def parametrizer(self) -> ModelParametrizer:
        return self._parametrizer

    @property
    def hessian_mode(self) -> HessianMode:
        return HessianMode(self.parametrizer.config.hessian_restriction)

    @property
    def hessian_check(self) -> Callable[[sp.Expr], bool]:
        match self.hessian_mode:
            case HessianMode.DIAG:
                return self._hessian_diag

            case HessianMode.FULL:
                return self._hessian_full

            case HessianMode.FREE:
                return self._hessian_free

            case _:
                raise ValueError(f"Unsupported Hessian mode: {self.hessian_mode}")
